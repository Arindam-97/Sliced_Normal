"""Run the full demo from a checkout or after installing the package."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
from workflow import fit_demo, sample_and_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--grid', type=int, default=5000)
    parser.add_argument('--evaluation-grid', type=int, default=50000)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--burnin', type=int, default=300)
    parser.add_argument('--thin', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'Demo/outputs')
    args = parser.parse_args()
    if not (0 <= args.burnin < args.steps and args.thin > 0):
        parser.error('Require 0 <= burnin < steps and thin > 0')
    data, B_fmle, B, metrics = fit_demo(args.iterations, args.grid, args.evaluation_grid)
    sample_and_plot(data, B_fmle, B, args.output, args.steps, args.burnin, args.thin)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    import numpy as np
    np.save(args.output/'B_fast.npy', B)
    np.save(args.output/'B_fmle.npy', B_fmle)
    print(json.dumps(metrics, indent=2))
    print('Outputs:', args.output.resolve())
