"""Adaptive enrichment of the fixed-basis primal Sliced Normal model."""

import time

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from .features import Features, _normalize_sample_weight
from .optimization import _PrimalObjective, estimate_primal_B


def _weighted_cdf_reference(values, weights):
    """Inverse empirical-CDF thresholds and reference CDFs, including ties."""
    thresholds, observed = [], []
    positive = weights > 0
    for column in values.T:
        support, inverse = np.unique(column[positive], return_inverse=True)
        # Accumulate ties in extended precision and tolerate rounding at exact
        # quantile boundaries. Integer weights and repeated rows should choose
        # the same thresholds even if their normalization rounds differently.
        mass = np.zeros(len(support), dtype=np.longdouble)
        np.add.at(mass, inverse, weights[positive].astype(np.longdouble))
        cdf = np.asarray(np.cumsum(mass) / mass.sum(), dtype=float)
        cdf[-1] = 1.0
        quantiles = np.linspace(0.02, 0.98, 49) - 8 * np.finfo(float).eps
        indices = np.searchsorted(cdf, quantiles, side="left")
        indices = np.unique(indices)
        thresholds.append(support[indices])
        observed.append(cdf[indices])
    return thresholds, observed


def _extra_polynomials(H, directions):
    return directions[0] + H @ directions[1:]


def _energy(H, model):
    count = H.shape[1]
    result = (H ** 2) @ model["coefficients"][:count]
    if model["directions"].shape[1]:
        result += (_extra_polynomials(H, model["directions"]) ** 2) @ model["coefficients"][count:]
    return result


def _psi_second_moment(H, probabilities, chunk_size):
    result = np.zeros((H.shape[1] + 1, H.shape[1] + 1))
    result[0, 0] = probabilities.sum()
    result[0, 1:] = result[1:, 0] = probabilities @ H
    for start in range(0, len(H), chunk_size):
        block = H[start:start + chunk_size]
        result[1:, 1:] += block.T @ (probabilities[start:start + chunk_size, None] * block)
    return (result + result.T) / 2


def _projected_gradient(coefficients, gradient, lower):
    return np.where(gradient > 0, np.minimum(gradient, coefficients - lower), gradient)


class _AdaptiveProblem:
    """Per-fit numerical state; no notebook globals or cached data between calls."""

    def __init__(self, values, weights, degree, basis, bounds, sampler, rng,
                 chunk_size, options):
        self.values, self.weights, self.degree = values, weights, degree
        self.basis, self.bounds = basis, bounds
        self.sampler, self.rng, self.chunk_size = sampler, rng, chunk_size
        self.options = options
        self.count = basis.shape[1]
        self.volume = float(np.prod(bounds[1] - bounds[0]))
        features = Features(pd.DataFrame(values), degree).to_numpy(dtype=float)
        self.feature_mean = weights @ features[:, 1:]
        self.H_data = (features[:, 1:] - self.feature_mean) @ basis[1:]
        self.data_moment = _psi_second_moment(self.H_data, weights, chunk_size)
        self.thresholds, self.observed_cdfs = _weighted_cdf_reference(values, weights)

    def draw(self, count):
        if self.sampler is None:
            points = self.rng.uniform(*self.bounds, size=(count, self.values.shape[1]))
            return points, np.full(count, -np.log(self.volume))
        points, log_q = self.sampler(self.rng, count)
        points, log_q = np.asarray(points, dtype=float), np.asarray(log_q, dtype=float)
        if points.shape != (count, self.values.shape[1]) or log_q.shape != (count,):
            raise ValueError("integration_sampler must return (count, n_variables) points and count log densities")
        return points, log_q

    def polynomials(self, points):
        H = np.empty((len(points), self.count))
        for start in range(0, len(points), self.chunk_size):
            z = Features(pd.DataFrame(points[start:start + self.chunk_size]), self.degree).to_numpy(dtype=float)
            H[start:start + self.chunk_size] = (z[:, 1:] - self.feature_mean) @ self.basis[1:]
        if not np.isfinite(H).all():
            raise ValueError("Nonfinite polynomials; rescale the data or reduce degree")
        return H

    def cache(self, points, log_q):
        points, log_q = np.asarray(points, dtype=float), np.asarray(log_q, dtype=float)
        if (points.ndim != 2 or not len(points) or points.shape[1] != self.values.shape[1]
                or not np.isfinite(points).all() or log_q.shape != (len(points),)
                or not np.isfinite(log_q).all()):
            raise ValueError("Need finite integration points and one finite proposal log density per draw")
        inside = np.all((points >= self.bounds[0]) & (points <= self.bounds[1]), axis=1)
        if not inside.any():
            raise ValueError("No integration points lie inside the fitting box")
        kept = points[inside]
        bins = [np.searchsorted(t, kept[:, j], side="left") for j, t in enumerate(self.thresholds)]
        return {"H": self.polynomials(kept), "log_q": log_q[inside],
                "total_draws": len(points), "bins": bins}

    def fresh_cache(self, count):
        return self.cache(*self.draw(count))

    def evaluate(self, cache, model):
        log_terms = -_energy(cache["H"], model) - cache["log_q"]
        total = logsumexp(log_terms)
        probabilities = np.exp(log_terms - total)
        log_Z = float(total - np.log(cache["total_draws"]))
        errors = []
        for bins, reference in zip(cache["bins"], self.observed_cdfs):
            mass = np.bincount(bins, weights=probabilities, minlength=len(reference) + 1)
            errors.append(np.max(np.abs(np.cumsum(mass)[:-1] - reference)))
        likelihood = float(-self.weights @ _energy(self.H_data, model) - log_Z)
        if not np.isfinite(likelihood) or not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite adaptive likelihood; rescale data or improve integration")
        return {"log_partition": log_Z, "mean_log_likelihood": likelihood,
                "integration_ess": float(1 / (probabilities @ probabilities)),
                "marginal_cdf_error": np.asarray(errors), "probabilities": probabilities}

    def context(self, model, train_cache, screens):
        return {"training": self.evaluate(train_cache, model),
                "screens": [self.evaluate(cache, model) for cache in screens]}

    def refit(self, data_squares, grid_squares, log_measure, initial, lower):
        objective = _PrimalObjective(data_squares, grid_squares, self.volume,
                                     self.weights, log_grid_weights=log_measure)
        _, gradient = objective(initial)
        tolerance = self.options["min_gradient_norm"]
        if np.linalg.norm(_projected_gradient(initial, gradient, lower), ord=np.inf) <= tolerance:
            result = OptimizeResult(x=initial.copy(), success=True, status=0, nit=0,
                                    message="Initial coefficients meet projected-gradient tolerance.")
        elif self.options["max_iterations"] == 0:
            result = OptimizeResult(x=initial.copy(), success=False, status=1, nit=0,
                                    message="Iteration limit reached at initialization.")
        else:
            result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                              bounds=list(zip(lower, [None] * len(lower))),
                              options={"maxiter": self.options["max_iterations"], "gtol": tolerance,
                                       "ftol": 0.0, "maxls": 40})
        result.fun, result.jac = objective(result.x)
        result.projected_gradient = _projected_gradient(result.x, result.jac, lower)
        result.projected_gradient_norm = float(np.linalg.norm(result.projected_gradient, ord=np.inf))
        result.gradient_tolerance_met = result.projected_gradient_norm <= tolerance
        result.integration_ess = objective.integration_ess
        result.log_partition = objective.log_partition
        return result

    def try_addition(self, state, candidate, context, baseline_screen, train_cache,
                     screens, log_measure, *, round_number, attempt_number, mode, candidate_ids):
        started = time.perf_counter()
        model = state["model"]
        count = candidate.shape[1]
        previous_count = model["directions"].shape[1]
        row = {"attempt": attempt_number, "round": round_number, "mode": mode,
               "candidate_ids": list(candidate_ids), "candidate_additions": count,
               "accepted": False, "accepted_additions": 0,
               "total_accepted_additions": previous_count, "status": "screened_out",
               "refit_seconds": 0.0}

        def rejected(reason):
            row.update(reason=reason, attempt_seconds=time.perf_counter() - started)
            return state, row

        data_extra = _extra_polynomials(self.H_data, candidate) ** 2
        grid_extra = _extra_polynomials(train_cache["H"], candidate) ** 2
        data_mean = self.weights @ data_extra
        training_scores = data_mean - context["training"]["probabilities"] @ grid_extra
        fresh_scores = np.asarray([
            data_mean - stats["probabilities"] @ (_extra_polynomials(cache["H"], candidate) ** 2)
            for cache, stats in zip(screens, context["screens"])])
        row.update(training_derivatives=training_scores.tolist(), fresh_derivatives=fresh_scores.tolist())
        if (not np.all(training_scores < -self.options["direction_tolerance"])
                or not np.all(fresh_scores < -self.options["direction_tolerance"])):
            return rejected("Direction failed training or fresh derivative checks")

        directions = np.column_stack((model["directions"], candidate))
        data_squares = np.column_stack((state["data_squares"], data_extra))
        grid_squares = np.column_stack((state["grid_squares"], grid_extra))
        initial = np.r_[model["coefficients"], np.zeros(count)]
        lower = np.r_[np.full(self.count, self.options["min_coefficient"]),
                      np.zeros(directions.shape[1])]
        row["status"] = "rejected"
        refit_started = time.perf_counter()
        try:
            result = self.refit(data_squares, grid_squares, log_measure, initial, lower)
        except ValueError as error:
            if "unbounded" not in str(error) or "likelihood" not in str(error):
                raise
            row["refit_seconds"] = time.perf_counter() - refit_started
            return rejected(str(error))
        row["refit_seconds"] = time.perf_counter() - refit_started
        trial = {"coefficients": result.x.copy(), "directions": directions}
        new_screen = [self.evaluate(cache, trial) for cache in screens]
        old_screen = context["screens"]
        gains = np.asarray([new["mean_log_likelihood"] - old["mean_log_likelihood"]
                            for new, old in zip(new_screen, old_screen)])
        new_error = np.mean([s["marginal_cdf_error"] for s in new_screen], axis=0)
        old_error = np.mean([s["marginal_cdf_error"] for s in old_screen], axis=0)
        base_error = np.mean([s["marginal_cdf_error"] for s in baseline_screen], axis=0)
        drift = float(max(np.max(new_error - old_error), np.max(new_error - base_error)))
        minimum_ess = min([s["integration_ess"] for s in old_screen + new_screen]
                          + [result.integration_ess])
        reasons = []
        if not result.gradient_tolerance_met:
            reasons.append("Returned gradient above tolerance")
        if not result.fun < state["fun"]:
            reasons.append("Training NLL did not decrease")
        if not np.all(gains > self.options["min_likelihood_gain"]):
            reasons.append("Paired likelihood gain did not pass every fresh batch")
        if minimum_ess < self.options["min_integration_ess"]:
            reasons.append("Integration ESS below screening threshold")
        if self.options["marginal_guard"] and drift > self.options["marginal_cdf_tolerance"]:
            reasons.append("Marginal CDF deterioration exceeds tolerance")
        row.update(training_nll=float(result.fun), screen_likelihood_gains=gains.tolist(),
                   minimum_integration_ess=float(minimum_ess), worst_marginal_error_increase=drift,
                   projected_gradient_norm=result.projected_gradient_norm, iterations=int(result.nit))
        if reasons:
            return rejected("; ".join(reasons))
        row.update(accepted=True, accepted_additions=count, status="accepted", reason="Accepted",
                   total_accepted_additions=previous_count + count,
                   attempt_seconds=time.perf_counter() - started)
        return {"model": trial, "fun": float(result.fun), "result": result,
                "data_squares": data_squares, "grid_squares": grid_squares}, row

    def search(self, initial, baseline_info, train_cache):
        state = {"model": initial, "fun": float(baseline_info.fun), "result": baseline_info,
                 "data_squares": self.H_data ** 2, "grid_squares": train_cache["H"] ** 2}
        log_measure = -train_cache["log_q"] - np.log(train_cache["total_draws"])
        history, rounds = [], []
        stop = "Reached the configured round cap"
        if self.options["max_added"] == 0:
            return state, history, rounds, "No adaptive additions requested"
        if not baseline_info.gradient_tolerance_met:
            return state, history, rounds, "Baseline gradient tolerance not met; increase its fitting budget"
        if baseline_info.integration_ess < self.options["min_integration_ess"]:
            return state, history, rounds, "Baseline training integration ESS below screening threshold"

        for round_number in range(1, self.options["max_rounds"] + 1):
            started = time.perf_counter()
            old_count = state["model"]["directions"].shape[1]
            slots = min(self.options["add_per_round"], self.options["max_added"] - old_count)
            training = self.evaluate(train_cache, state["model"])
            G = self.data_moment - _psi_second_moment(
                train_cache["H"], training["probabilities"], self.chunk_size)
            eigenvalues, eigenvectors = np.linalg.eigh((G + G.T) / 2)
            chosen = np.flatnonzero(eigenvalues < -self.options["direction_tolerance"])[:slots]
            attempts_before = len(history)
            screened = 0
            if not len(chosen):
                stop = "No sufficiently negative training moment-gradient direction"
            else:
                candidate = eigenvectors[:, chosen].copy()
                for j in range(candidate.shape[1]):
                    anchor = np.argmax(np.abs(candidate[:, j]))
                    if candidate[anchor, j] < 0:
                        candidate[:, j] *= -1
                data_mean = self.weights @ (_extra_polynomials(self.H_data, candidate) ** 2)
                screens = [self.fresh_cache(self.options["n_screen"])
                           for _ in range(self.options["screen_batches"])]
                current_screen = [self.evaluate(cache, state["model"]) for cache in screens]
                scores = np.asarray([
                    data_mean - stats["probabilities"] @ (_extra_polynomials(cache["H"], candidate) ** 2)
                    for cache, stats in zip(screens, current_screen)])
                keep = np.all(scores < -self.options["direction_tolerance"], axis=0)
                pool = candidate[:, keep]
                ids = (np.flatnonzero(keep) + 1).tolist()
                screened = pool.shape[1]
                if not screened:
                    stop = "Training directions did not repeat on every fresh batch"
                else:
                    baseline_screen = (current_screen if old_count == 0 else
                                       [self.evaluate(cache, initial) for cache in screens])
                    context = {"training": training, "screens": current_screen}

                    def attempt(indices, mode):
                        nonlocal state, context
                        if context is None:
                            context = self.context(state["model"], train_cache, screens)
                        state, row = self.try_addition(
                            state, pool[:, indices], context, baseline_screen, train_cache,
                            screens, log_measure, round_number=round_number,
                            attempt_number=len(history) + 1, mode=mode,
                            candidate_ids=[ids[i] for i in indices])
                        history.append(row)
                        if row["accepted"]:
                            context = None
                        return row["accepted"]

                    accepted = attempt(list(range(screened)), "batch" if screened > 1 else "individual")
                    if not accepted and screened > 1:
                        for index in range(screened):
                            attempt([index], "individual")
                    total = state["model"]["directions"].shape[1]
                    if total >= self.options["max_added"]:
                        stop = "Reached the target number of accepted additions"
                    elif total == old_count:
                        stop = ("Batch and all individual retries were rejected" if screened > 1
                                else "The single screened candidate was rejected")
                    else:
                        stop = "Reached the configured round cap"
                # Release this round's polynomial caches before allocating
                # fresh screens in the next round.
                screens.clear()
            total = state["model"]["directions"].shape[1]
            rounds.append({"round": round_number, "screened_candidates": screened,
                           "attempts": len(history) - attempts_before,
                           "accepted_additions": total - old_count, "total_accepted_additions": total,
                           "round_seconds": time.perf_counter() - started, "reason": stop})
            if self.options["verbosity"]:
                print(f"Adaptive round {round_number}: {total} accepted additions; "
                      f"{len(history) - attempts_before} attempts.", flush=True)
            if total >= self.options["max_added"] or total == old_count:
                break
        return state, history, rounds, stop


def estimate_adaptive_primal_B(
        data, degree=2, n_grid=10000, *, sample_weight=None,
        basis="eigen", covariance_regularization=1e-8, min_coefficient=1e-6,
        max_iterations=1000, min_gradient_norm=1e-6,
        max_added=30, add_per_round=5, max_rounds=30,
        n_screen=None, screen_batches=2, direction_tolerance=1e-3,
        min_likelihood_gain=1e-3, min_integration_ess=200,
        marginal_guard=True, marginal_cdf_tolerance=0.02,
        n_audit=None, audit_batches=3, integration_sampler=None,
        integration_points=None, proposal_log_density=None,
        random_state=None, chunk_size=8192, verbosity=1, return_info=False):
    """Fit a primal SN and adaptively add weighted squared-polynomial interactions.

    Equal observation weights are used unless sample_weight is supplied. One
    finite nonnegative positional weight per row is required, with positive
    total. Normalized weights enter the baseline FMLE, coefficient likelihood,
    direction moments, screening scores, marginal reference CDFs and final audit.
    They do not reweight integration draws or subsequent MCMC samples. Every
    input row, including zero-weight rows, still defines the fitting box.

    Starting from estimate_primal_B, set psi=(1,h_1,...,h_r) and add up to
    max_added terms beta*(a.T@psi)**2, beta>=0, without increasing degree.
    Negative eigenvectors of the weighted data-minus-model moment matrix
    propose additions. Each round screens at most add_per_round directions on
    screen_batches fresh integration batches, then refits ALL coefficients.
    Rejected batches are retried individually against the latest accepted
    model; rejected trials never replace it. Original coefficients retain
    min_coefficient; added coefficients may be exactly zero.

    Acceptance requires returned projected-gradient tolerance, decreased
    training NLL, a paired mean-log-likelihood gain > min_likelihood_gain on
    EVERY screening batch, and integration ESS >= min_integration_ess. With
    marginal_guard=True, the mean screening CDF error in each coordinate may
    worsen by at most marginal_cdf_tolerance relative to BOTH the original
    baseline and the previous accepted model. This checks 49 weighted empirical
    quantile thresholds (fewer with ties); it does not preserve marginals exactly.
    A baseline failing gradient tolerance or training ESS is returned without
    adaptation, with the reason recorded in info.search_stopping_reason.

    Integration defaults to independent uniform draws from the same data box
    as estimate_primal_B. For narrow/high-dimensional data supply
    integration_sampler(rng, count) -> (points, log_q), where rng is a NumPy
    Generator. The sampler must produce count independent draws from a known
    normalized density covering the entire box and finite log densities at
    every draw. ALL draws, including outside-box draws, must be returned.
    This callback is used separately for training, screening and final audit.
    Caller-supplied integration_points and proposal_log_density may instead
    override training draws only; the callback (or uniform default) is still
    used for fresh batches. Caller-supplied draws must be independent of them.
    n_screen and n_audit default to n_grid. audit_batches must be >=2, or 0
    to explicitly disable the independent final audit. screen_batches>=2.

    random_state is a seed or Generator for integration; NumPy's global RNG
    is not used. Baseline options match estimate_primal_B. max_added=0 returns
    the baseline. max_iterations limits EACH coefficient fit; max_rounds limits
    adaptive rounds. chunk_size bounds feature and moment working blocks;
    training/screen polynomial caches still require O(n_grid * r) memory.

    Returns B, or (B, info) when return_info=True. B uses the existing monomial
    order and is accepted by the existing likelihood/MCMC helpers. info includes
    coefficients, directions in psi coordinates, baseline_basis, polynomials
    (all final atoms in monomial coordinates), baseline_B, baseline_info,
    history, round_history and audit. B=(polynomials*coefficients)@polynomials.T.
    info.fun/jac/gradient diagnostics always describe the last ACCEPTED model.
    info.success/gradient_tolerance_met concern coefficient optimization only;
    search_stopping_reason explains adaptive stopping. Audit records compare
    baseline and adaptive weighted in-sample likelihood, integration ESS and
    marginal errors on common fresh draws, never used for acceptance.

    This remains a restricted model, not full matrix optimization. Convergence,
    screening ESS and audit variability do not certify integration accuracy,
    predictive performance or MCMC mixing. The existing general-purpose MCMC
    helper is unbounded, whereas this fitter normalizes on a finite box.
    """
    n_screen = n_grid if n_screen is None else n_screen
    n_audit = n_grid if n_audit is None else n_audit
    integers = {"degree": (degree, 1), "n_grid": (n_grid, 1), "max_iterations": (max_iterations, 0),
                "max_added": (max_added, 0), "add_per_round": (add_per_round, 1),
                "max_rounds": (max_rounds, 0), "n_screen": (n_screen, 1),
                "screen_batches": (screen_batches, 2), "n_audit": (n_audit, 1),
                "audit_batches": (audit_batches, 0), "chunk_size": (chunk_size, 1)}
    for name, (value, minimum) in integers.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if audit_batches == 1:
        raise ValueError("audit_batches must be zero or at least two")
    for name, value in {"direction_tolerance": direction_tolerance,
                        "min_likelihood_gain": min_likelihood_gain,
                        "min_integration_ess": min_integration_ess,
                        "marginal_cdf_tolerance": marginal_cdf_tolerance}.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not isinstance(marginal_guard, (bool, np.bool_)):
        raise ValueError("marginal_guard must be boolean")
    if integration_sampler is not None and not callable(integration_sampler):
        raise ValueError("integration_sampler must be callable")
    if (integration_points is None) != (proposal_log_density is None):
        raise ValueError("Supply integration_points and proposal_log_density together")
    values = np.asarray(data, dtype=float)
    if (values.ndim != 2 or len(values) < 2 or not values.shape[1] or not np.isfinite(values).all()):
        raise ValueError("data must contain at least two finite rows and one variable")
    weights = _normalize_sample_weight(sample_weight, len(values))
    if weights is None:
        weights = np.full(len(values), 1.0 / len(values))
    bounds = np.stack((values.min(axis=0) - 0.01 * np.abs(values.min(axis=0)),
                       values.max(axis=0) + 0.01 * np.abs(values.max(axis=0))))
    volume = float(np.prod(bounds[1] - bounds[0]))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("data must define a positive finite integration box")
    rng = np.random.default_rng(random_state)
    if integration_points is None:
        if integration_sampler is None:
            integration_points = rng.uniform(*bounds, size=(n_grid, values.shape[1]))
            proposal_log_density = np.full(n_grid, -np.log(volume))
        else:
            integration_points, proposal_log_density = integration_sampler(rng, n_grid)
            if (np.shape(integration_points) != (n_grid, values.shape[1])
                    or np.shape(proposal_log_density) != (n_grid,)):
                raise ValueError("integration_sampler returned incorrect training shapes")
    started = time.perf_counter()
    baseline_B, baseline = estimate_primal_B(
        pd.DataFrame(values), degree=degree, n_grid=n_grid, sample_weight=weights,
        basis=basis, covariance_regularization=covariance_regularization,
        min_coefficient=min_coefficient, max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm, integration_points=integration_points,
        proposal_log_density=proposal_log_density, verbosity=0, return_info=True)
    baseline_seconds = time.perf_counter() - started
    options = dict(max_iterations=max_iterations, min_gradient_norm=min_gradient_norm,
                   min_coefficient=min_coefficient, max_added=max_added, add_per_round=add_per_round,
                   max_rounds=max_rounds, n_screen=n_screen, screen_batches=screen_batches,
                   direction_tolerance=direction_tolerance, min_likelihood_gain=min_likelihood_gain,
                   min_integration_ess=min_integration_ess, marginal_guard=marginal_guard,
                   marginal_cdf_tolerance=marginal_cdf_tolerance, verbosity=verbosity)
    started = time.perf_counter()
    problem = _AdaptiveProblem(values, weights, degree, baseline.basis.copy(), bounds,
                               integration_sampler, rng, chunk_size, options)
    train_cache = problem.cache(integration_points, proposal_log_density)
    initial = {"coefficients": baseline.coefficients.copy(),
               "directions": np.empty((problem.count + 1, 0))}
    state, history, rounds, stop = problem.search(initial, baseline, train_cache)
    adaptation_seconds = time.perf_counter() - started
    final = state["model"]
    transform = np.column_stack((np.eye(problem.count + 1)[:, 0], baseline.basis))
    atoms = np.column_stack((baseline.basis, transform @ final["directions"]))
    if not final["directions"].shape[1]:
        B = baseline_B.copy()
    else:
        B = (atoms * final["coefficients"]) @ atoms.T
        B = (B + B.T) / 2
    train_stats = problem.evaluate(train_cache, final)
    info = OptimizeResult(
        x=final["coefficients"].copy(), coefficients=final["coefficients"].copy(),
        directions=final["directions"].copy(), baseline_basis=baseline.basis.copy(),
        polynomials=atoms, feature_mean=problem.feature_mean.copy(),
        baseline_B=baseline_B.copy(), baseline_info=baseline, initial_B=baseline.initial_B.copy(),
        fun=state["fun"], jac=state["result"].jac.copy(),
        projected_gradient_norm=state["result"].projected_gradient_norm,
        gradient_tolerance_met=bool(state["result"].gradient_tolerance_met),
        success=bool(state["result"].gradient_tolerance_met), message=stop,
        search_stopping_reason=stop, accepted_additions=final["directions"].shape[1],
        history=history, round_history=rounds, integration_bounds=bounds,
        integration_ess=train_stats["integration_ess"], log_partition=train_stats["log_partition"],
        integration_sample_count=train_cache["total_draws"],
        integration_in_bounds_count=len(train_cache["H"]),
        effective_sample_size=float(1 / (weights @ weights)),
        marginal_cdf_thresholds=[t.copy() for t in problem.thresholds],
        observed_marginal_cdfs=[cdf.copy() for cdf in problem.observed_cdfs],
        baseline_seconds=baseline_seconds, adaptation_seconds=adaptation_seconds,
        fitting_seconds=baseline_seconds + adaptation_seconds, audit=[])
    # Release the largest training arrays before drawing the independent audit.
    del train_cache, state, integration_points, proposal_log_density
    started = time.perf_counter()
    for batch in range(audit_batches):
        cache = problem.fresh_cache(n_audit)
        base_stats, final_stats = problem.evaluate(cache, initial), problem.evaluate(cache, final)
        for stats in (base_stats, final_stats):
            stats.pop("probabilities")
        info.audit.append({"batch": batch + 1, "baseline": base_stats, "adaptive": final_stats,
                           "likelihood_gain": final_stats["mean_log_likelihood"] - base_stats["mean_log_likelihood"]})
    info.audit_seconds = time.perf_counter() - started
    if verbosity:
        print(f"Adaptive primal Sliced Normal: {len(info.coefficients)} coefficients, "
              f"{info.accepted_additions} accepted additions.\n{stop}")
    return (B, info) if return_info else B
