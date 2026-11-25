"""Independent reference solvers used by the test suite.

Nothing here shares code with the coordinate descent path: the proximal
gradient solver works on the full objective with a generic gradient and a
backtracking line search, so agreement between the two is meaningful
evidence that both are right.
"""

from __future__ import annotations

import numpy as np

from .families import get_family


def standardize(X, w, fit_intercept=True, standardize=True):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean = w @ X if fit_intercept else np.zeros(X.shape[1])
    Xc = X - mean
    scale = np.sqrt(w @ Xc**2) if standardize else np.ones(X.shape[1])
    return Xc / scale, mean, scale, w


def objective(Xs, y, w, b0, b, lam, alpha, pf, family):
    fam = get_family(family)
    mu = fam.inverse_link(b0 + Xs @ b)
    return 0.5 * fam.deviance(y, mu, w) + lam * np.sum(pf * (alpha * np.abs(b) + 0.5 * (1 - alpha) * b**2))


def fista(Xs, y, w, lam, alpha, pf, family="gaussian", fit_intercept=True, max_iter=100_000, tol=1e-11):
    """Accelerated proximal gradient (FISTA with restarts) on the standardised problem."""
    fam = get_family(family)
    n, p = Xs.shape
    pf = np.asarray(pf, dtype=float)

    def smooth(b0, b):
        mu = fam.inverse_link(b0 + Xs @ b)
        val = 0.5 * fam.deviance(y, mu, w) + 0.5 * lam * (1 - alpha) * np.sum(pf * b * b)
        resid = w * (mu - y)
        g = Xs.T @ resid + lam * (1 - alpha) * pf * b
        g0 = resid.sum() if fit_intercept else 0.0
        return val, g0, g

    def prox(b, step):
        t = step * lam * alpha * pf
        return np.sign(b) * np.maximum(np.abs(b) - t, 0.0)

    def full(b0, b):
        return smooth(b0, b)[0] + lam * alpha * np.sum(pf * np.abs(b))

    b0 = float(fam.link(np.array([w @ y]))[0]) if fit_intercept else 0.0
    b = np.zeros(p)
    yb0, yb = b0, b.copy()
    t = 1.0
    # Start from the Lipschitz bound of the quadratic case; backtracking only
    # ever shrinks it, which is what the Poisson family needs.
    A = np.column_stack([np.ones(n), Xs]) if fit_intercept else Xs
    lip = np.linalg.eigvalsh((A * w[:, None]).T @ A)[-1] + lam * (1 - alpha) * float(np.max(pf, initial=0))
    if fam.name == "binomial":
        lip = 0.25 * lip + lam * (1 - alpha) * float(np.max(pf, initial=0))
    step = 1.0 / lip
    f_old = full(b0, b)
    for iters in range(1, max_iter + 1):
        val, g0, g = smooth(yb0, yb)
        while True:
            nb = prox(yb - step * g, step)
            nb0 = yb0 - step * g0
            d0, d = nb0 - yb0, nb - yb
            nval = smooth(nb0, nb)[0]
            if nval <= val + g0 * d0 + g @ d + (d0 * d0 + d @ d) / (2 * step) + 1e-15 * (1 + abs(val)):
                break
            step *= 0.5
        f_new = full(nb0, nb)
        if f_new > f_old + 1e-15 * (1 + abs(f_old)):
            # Adaptive restart (O'Donoghue and Candes): drop the momentum.
            t = 1.0
            yb0, yb = b0, b.copy()
            continue
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        mom = (t - 1) / t_new
        yb0 = nb0 + mom * (nb0 - b0)
        yb = nb + mom * (nb - b)
        b0, b, t, f_old = nb0, nb, t_new, f_new
        # Stop on the norm of the proximal gradient mapping at the iterate,
        # a genuine first order optimality measure for composite problems.
        _, h0, h = smooth(b0, b)
        gm = np.concatenate([[h0 * fit_intercept], (b - prox(b - step * h, step)) / step])
        if np.max(np.abs(gm)) < tol:
            break
    fista.last_iterations = iters
    return b0, b


def newton_mle(X, y, w, family, fit_intercept=True, iters=100):
    """Unpenalised maximum likelihood by plain Newton's method on the original scale."""
    fam = get_family(family)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    A = np.column_stack([np.ones(len(y)), X]) if fit_intercept else X
    theta = np.zeros(A.shape[1])
    if fit_intercept:
        theta[0] = float(fam.link(np.array([w @ y]))[0])
    for _ in range(iters):
        mu = fam.inverse_link(A @ theta)
        v = fam.variance(mu)
        g = A.T @ (w * (y - mu))
        H = (A * (w * v)[:, None]).T @ A
        step = np.linalg.solve(H, g)
        theta += step
        if np.max(np.abs(step)) < 1e-13:
            break
    if fit_intercept:
        return theta[0], theta[1:]
    return 0.0, theta
