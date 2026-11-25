"""End to end demo: sparse recovery with lasso, elastic net and logistic CV.

Run from the repository root:

    python examples/demo.py

It prints a short report and writes SVG plots next to this file.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enetpath import cv_enet, cv_svg, enet_path, path_svg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def correlated_design(n, p, rho, rng):
    """Block of features with pairwise correlation rho."""
    z = rng.normal(size=(n, 1))
    return np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=(n, p))


def main():
    rng = np.random.default_rng(2026)
    n, p = 150, 40
    X = np.hstack([correlated_design(n, 5, 0.9, rng), rng.normal(size=(n, p - 5))])
    beta = np.zeros(p)
    beta[:5] = 1.0        # a group of five highly correlated true signals
    beta[10:12] = [-1.5, 2.0]
    y = X @ beta + 1.5 * rng.normal(size=n)
    names = [f"g{j}" if j < 5 else f"x{j}" for j in range(p)]

    print("Gaussian: lasso vs elastic net on a correlated group\n")
    for alpha in (1.0, 0.3):
        cv = cv_enet(X, y, alpha=alpha, n_folds=10, seed=0, feature_names=names)
        _, b = cv.coef("1se")
        picked = [names[j] for j in np.flatnonzero(b)]
        group = sum(1 for j in range(5) if b[j] != 0)
        print(f"  alpha={alpha:<4g} lambda_1se={cv.lambda_1se:.4f}  df={len(picked):>2}  "
              f"group members kept={group}/5  selected={picked}")
        tag = "lasso" if alpha == 1.0 else "enet"
        path_svg(cv.path, os.path.join(HERE, f"path_{tag}.svg"))
        cv_svg(cv, os.path.join(HERE, f"cv_{tag}.svg"))

    print("\nBinomial: misclassification CV")
    prob = 1 / (1 + np.exp(-(X @ beta) / 2))
    yb = (rng.random(n) < prob).astype(float)
    cv = cv_enet(X, yb, family="binomial", alpha=0.5, measure="class", n_folds=5, seed=1)
    print(f"  lambda_min={cv.lambda_min:.4f}  CV error={cv.cvm[cv.index_min]:.3f}  df={cv.path.df[cv.index_min]}")

    print("\nPoisson: deviance explained along the path")
    yp = rng.poisson(np.exp(0.2 + 0.3 * (X @ beta) / 2)).astype(float)
    fit = enet_path(X, yp, family="poisson", alpha=1.0)
    for k in range(0, fit.n_lambda, max(fit.n_lambda // 5, 1)):
        print(f"  lambda={fit.lambdas[k]:.4f}  df={fit.df[k]:>2}  %dev={100 * fit.dev_ratio[k]:.1f}")
    print(f"\nKKT violations repaired by the strong rule check: {int(fit.kkt_violations.sum())}")
    print(f"median strong set size: {int(np.median(fit.strong_set_sizes))} of {p} features")
    print(f"\nSVG plots written to {HERE}")


if __name__ == "__main__":
    main()
