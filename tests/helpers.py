import numpy as np

from enetpath.oracle import standardize


def make_data(family, n=120, p=12, seed=0, k=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(0.4 + 0.6 * np.eye(p)).T  # correlated features
    X = X * rng.uniform(0.5, 4.0, size=p) + rng.normal(size=p)
    beta = np.zeros(p)
    beta[:k] = rng.choice([-1, 1], size=k) * rng.uniform(0.5, 1.5, size=k)
    Xs = (X - X.mean(0)) / X.std(0)
    eta = Xs @ beta
    if family == "gaussian":
        y = 2.0 + eta + rng.normal(size=n)
    elif family == "binomial":
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(0.5 + 0.4 * eta)).astype(float)
    return X, y


def std_intercept(fit, k):
    """Intercept on the standardised scale for path index k."""
    return fit.intercepts[k] + fit.x_mean @ fit.coefs[:, k]


def score(fit, X, y, k, weights=None):
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    Xs, _, _, w = standardize(X, w)
    mu = fit.family.inverse_link(std_intercept(fit, k) + Xs @ fit.coefs_std[:, k])
    return Xs.T @ (w * (y - mu))
