# Sliced-Normal

The objective is to model the density of dependent variables using polynomials. The sliced normal density is parameterized by $3$ quantities: $(d,\mu,\Sigma)$. Suppose we have a $m-$variate data: $x = (x_1,...,x_m)$. Let $Z$ denote the vector of all monomials for $x$ with degree $\leq d$. For example, with $m = 3$ and $d=2$ we have:
$x = (x_1,x_2,x_3)$ and $Z_x = (x_1,x_2,x_3,x_1^2,x_1x_2,x_1x_3,x_2^2,x_2x_3,x_3^2)$. The sliced normal density is defined as:   

$$f(x) \propto \phi(Z_x| \mu,\Sigma) \propto e^{-\frac 12 (Z_x-\mu)^T\Sigma^{-1}(Z_x-\mu)}$$  

Here, $\phi(Z_x |\mu,\Sigma)$ denote the density of a multivariate Gaussian at $Z_x$ with parameters $\mu$ and $\Sigma$. In particular, we fix a region $\delta$, where we are interested to model the data. We define:  

$$c(\mu,\Sigma) = \int_{\Delta} \phi(Z_x| \mu,\Sigma) dx$$

Thus, our sliced normal density becomes:
$f(x) = \frac{1}{c}\phi(Z_x| \mu,\Sigma)$

## Parameter Estimation
Let $P:=\Sigma^{-1}$. We search for the MLE of $(\mu,P)$ given the data $(X_1,...,X_n)$:
$$(\hat\mu,\hat P) := \arg\min_{\mu,P} \ \prod_{i=1}^n \frac{1}{c(\mu,P)}e^{-\frac 12 (Z_{X_i}-\mu)^TP(Z_{X_i}-\mu)}$$

It is equivalent to maximize the log-likelihood:
$$-n\log c(\mu,P) -\sum_{i=1}^n \frac 12 (Z_{X_i}-\mu)^TP(Z_{X_i}-\mu) $$

To ensure $P$ is Positive Definite, we consider an upper triangular matrix $V$ and set $P = V^TV$. We maximize the likelihood w.r.t the variables $\mu,V$. 

## Reference:
Luis G. Crespo, Brendon K. Colbert, Sean P. Kenny, Daniel P. Giesy,
On the quantification of aleatory and epistemic uncertainty using Sliced-Normal distributions,
Systems & Control Letters.
