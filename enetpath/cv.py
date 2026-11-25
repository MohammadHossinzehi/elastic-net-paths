"""K fold cross validation over a shared lambda sequence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .path import ElasticNetPath, enet_path

MEASURES = ("deviance", "mse", "mae", "class")


@dataclass
class CVResult:
    path: ElasticNetPath
    measure: str
    cvm: np.ndarray
    cvsd: np.ndarray
    fold_losses: np.ndarray
    folds: np.ndarray
    lambda_min: float
    lambda_1se: float
    index_min: int
    index_1se: int

    def coef(self, which: str = "1se"):
        k = self.index_1se if which == "1se" else self.index_min
        return self.path.intercepts[k], self.path.coefs[:, k].copy()

    def predict(self, X, which: str = "1se", kind: str = "response"):
        lam = self.lambda_1se if which == "1se" else self.lambda_min
        return self.path.predict(X, lam=lam, kind=kind)


def _loss(measure, family, y, mu):
    if measure == "deviance":
        return family.unit_deviance(y[:, None], mu)
    if measure == "mse":
        return (y[:, None] - mu) ** 2
    if measure == "mae":
        return np.abs(y[:, None] - mu)
    if measure == "class":
        if family.name != "binomial":
            raise ValueError("measure='class' only makes sense for the binomial family")
        return ((mu > 0.5).astype(float) != y[:, None]).astype(float)
    raise ValueError(f"unknown measure {measure!r}; choose from {MEASURES}")


def make_folds(n: int, n_folds: int, seed: int = 0) -> np.ndarray:
    if not 2 <= n_folds <= n:
        raise ValueError("n_folds must be between 2 and n")
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % n_folds)


def cv_enet(X, y, family="gaussian", n_folds: int = 10, measure: str = "deviance", seed: int = 0,
            folds=None, weights=None, **path_kwargs) -> CVResult:
    """Cross validate an elastic net path, glmnet style.

    The full data path fixes the lambda sequence; each fold is refit on that
    exact sequence (with early stopping off) so losses line up column for
    column.  ``lambda_1se`` is the largest lambda whose mean CV loss is within
    one standard error of the minimum.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    full = enet_path(X, y, family=family, weights=w, **path_kwargs)
    lambdas = full.lambdas
    folds = make_folds(n, n_folds, seed) if folds is None else np.asarray(folds)
    fold_ids = np.unique(folds)
    K = len(fold_ids)
    kw = {k: v for k, v in path_kwargs.items() if k not in ("lambdas", "n_lambda", "lambda_min_ratio")}
    losses = np.zeros((K, len(lambdas)))
    fold_w = np.zeros(K)
    for i, f in enumerate(fold_ids):
        test = folds == f
        fit = enet_path(X[~test], y[~test], family=family, weights=w[~test], lambdas=lambdas, **kw)
        mu = fit.predict(X[test], kind="response")
        wt = w[test]
        losses[i] = wt @ _loss(measure, full.family, y[test], mu) / wt.sum()
        fold_w[i] = wt.sum()
    cvm = fold_w @ losses / fold_w.sum()
    var = fold_w @ (losses - cvm) ** 2 / fold_w.sum()
    cvsd = np.sqrt(var / max(K - 1, 1))
    i_min = int(np.argmin(cvm))
    within = np.flatnonzero(cvm <= cvm[i_min] + cvsd[i_min])
    i_1se = int(within.min())
    return CVResult(full, measure, cvm, cvsd, losses, folds, float(lambdas[i_min]), float(lambdas[i_1se]), i_min, i_1se)
