"""Regularisation paths for elastic net penalised GLMs.

The public entry point is :func:`enet_path`, which mirrors the core of R's
glmnet: standardisation with observation weights, an automatically chosen
log spaced lambda sequence starting at the smallest lambda that zeroes every
penalised coefficient, warm starts, sequential strong rules with a full KKT
check, IRLS outer loops for non Gaussian families, and glmnet style early
stopping when the path stops explaining additional deviance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .cd import cd_wls
from .families import Family, get_family

KKT_RTOL = 1e-6
KKT_ATOL = 1e-9


@dataclass
class ElasticNetPath:
    family: Family
    alpha: float
    lambdas: np.ndarray
    intercepts: np.ndarray
    coefs: np.ndarray  # shape (p, n_lambda), original feature scale
    coefs_std: np.ndarray  # shape (p, n_lambda), standardised scale
    dev_ratio: np.ndarray
    null_deviance: float
    df: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    penalty_factor: np.ndarray
    sweeps: np.ndarray
    kkt_violations: np.ndarray
    strong_set_sizes: np.ndarray
    feature_names: list = field(default_factory=list)

    @property
    def n_lambda(self) -> int:
        return len(self.lambdas)

    def coef_at(self, lam: float):
        """Coefficients at an arbitrary lambda by linear interpolation in log lambda.

        This matches what glmnet's ``coef(fit, s = lam)`` does without
        ``exact = TRUE``.  Values outside the path are clamped to its ends.
        """
        lams = self.lambdas
        if lam >= lams[0]:
            return self.intercepts[0], self.coefs[:, 0].copy()
        if lam <= lams[-1]:
            return self.intercepts[-1], self.coefs[:, -1].copy()
        k = int(np.searchsorted(-lams, -lam))  # lams[k-1] > lam >= lams[k]
        hi, lo = lams[k - 1], lams[k]
        t = (np.log(hi) - np.log(lam)) / (np.log(hi) - np.log(lo))
        b0 = (1 - t) * self.intercepts[k - 1] + t * self.intercepts[k]
        b = (1 - t) * self.coefs[:, k - 1] + t * self.coefs[:, k]
        return b0, b

    def predict(self, X, lam=None, kind: str = "link"):
        """Predict on new data.

        ``lam=None`` returns an (n, n_lambda) matrix for every lambda on the
        path; a scalar returns a length n vector.  ``kind`` is ``"link"`` for
        the linear predictor or ``"response"`` for the mean.
        """
        X = np.asarray(X, dtype=float)
        if lam is None:
            eta = X @ self.coefs + self.intercepts[None, :]
        else:
            b0, b = self.coef_at(float(lam))
            eta = X @ b + b0
        if kind == "link":
            return eta
        if kind == "response":
            return self.family.inverse_link(eta)
        raise ValueError("kind must be 'link' or 'response'")

    def nonzero(self, k: int):
        """Feature names (or indices) with non zero coefficients at path index k."""
        idx = np.flatnonzero(self.coefs[:, k])
        if self.feature_names:
            return [self.feature_names[i] for i in idx]
        return idx.tolist()


def penalty(beta: np.ndarray, lam: float, alpha: float, pf: np.ndarray) -> float:
    return float(lam * np.sum(pf * (alpha * np.abs(beta) + 0.5 * (1.0 - alpha) * beta**2)))


def default_lambda_sequence(lambda_max: float, n_lambda: int, min_ratio: float) -> np.ndarray:
    if n_lambda == 1:
        return np.array([lambda_max])
    return lambda_max * np.exp(np.linspace(0.0, np.log(min_ratio), n_lambda))


class _Problem:
    """Standardised data plus the warm start state carried along the path."""

    def __init__(self, X, y, w, family, alpha, pf, fit_intercept, standardize, tol, max_sweeps, irls_tol, max_irls):
        self.family = family
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_sweeps = max_sweeps
        self.irls_tol = irls_tol
        self.max_irls = max_irls
        self.y = y
        self.w = w

        self.x_mean = w @ X if fit_intercept else np.zeros(X.shape[1])
        Xc = X - self.x_mean
        if standardize:
            scale = np.sqrt(w @ (Xc**2))
        else:
            scale = np.ones(X.shape[1])
        self.constant = scale <= 1e-12 * (1.0 + np.abs(self.x_mean))
        scale = np.where(self.constant, 1.0, scale)
        self.x_scale = scale
        self.Xs = np.asfortranarray(Xc / scale)

        p = X.shape[1]
        self.eligible = np.flatnonzero(~self.constant)
        pf = pf.copy()
        pf[self.constant] = 0.0
        s = pf[self.eligible].sum()
        if s <= 0:
            raise ValueError("at least one feature must be penalised (penalty_factor > 0)")
        pf = pf * (len(self.eligible) / s)
        self.pf = pf
        self.unpenalized = self.eligible[pf[self.eligible] == 0.0]
        self.penalized = self.eligible[pf[self.eligible] > 0.0]

        self.beta = np.zeros(p)
        if fit_intercept:
            ybar = float(w @ y)
            self.b0 = float(family.link(np.array([ybar]))[0])
        else:
            self.b0 = 0.0
        self.eta = self.b0 + self.Xs @ self.beta

    def mu(self):
        return self.family.inverse_link(self.eta)

    def score(self):
        """s_j = sum_i w_i x_ij (y_i - mu_i): minus the gradient of the loss."""
        return self.Xs.T @ (self.w * (self.y - self.mu()))

    def objective(self, lam):
        dev = self.family.deviance(self.y, self.mu(), self.w)
        return 0.5 * dev + penalty(self.beta, lam, self.alpha, self.pf)

    def solve(self, lam, candidates):
        """IRLS (one step for the Gaussian family) around coordinate descent."""
        sweeps = 0
        gaussian = self.family.name == "gaussian"
        obj = self.objective(lam)
        for _ in range(1 if gaussian else self.max_irls):
            _, W, z = self.family.working(self.y, self.eta, self.w)
            beta_old, b0_old = self.beta.copy(), self.b0
            b0, beta, s = cd_wls(
                self.Xs, z, W, self.beta.copy(), self.b0, lam, self.alpha, self.pf, candidates,
                fit_intercept=self.fit_intercept, tol=self.tol, max_sweeps=self.max_sweeps,
            )
            sweeps += s
            self.beta, self.b0 = beta, b0
            self.eta = self.b0 + self.Xs @ self.beta
            if gaussian:
                break
            new_obj = self.objective(lam)
            halvings = 0
            # Step halving keeps IRLS monotone when the quadratic model overshoots.
            while new_obj > obj + 1e-12 * (1.0 + abs(obj)) and halvings < 30:
                self.beta = 0.5 * (self.beta + beta_old)
                self.b0 = 0.5 * (self.b0 + b0_old)
                self.eta = self.b0 + self.Xs @ self.beta
                new_obj = self.objective(lam)
                halvings += 1
            converged = abs(new_obj - obj) <= self.irls_tol * (abs(new_obj) + 0.1)
            obj = new_obj
            if converged:
                break
        else:
            if not gaussian:
                warnings.warn(f"IRLS did not converge at lambda={lam:.4g}", RuntimeWarning, stacklevel=3)
        return sweeps

    def kkt_violators(self, lam, considered):
        """Penalised eligible features outside ``considered`` that violate KKT."""
        s = self.score()
        outside = np.setdiff1d(self.penalized, considered, assume_unique=True)
        bound = lam * self.alpha * self.pf[outside]
        bad = np.abs(s[outside]) > bound * (1.0 + KKT_RTOL) + KKT_ATOL
        return outside[bad]


def enet_path(
    X,
    y,
    family="gaussian",
    alpha: float = 1.0,
    lambdas=None,
    n_lambda: int = 100,
    lambda_min_ratio=None,
    weights=None,
    penalty_factor=None,
    standardize: bool = True,
    fit_intercept: bool = True,
    strong_rules: bool = True,
    early_stop: bool = True,
    tol: float = 1e-18,
    max_sweeps: int = 100_000,
    irls_tol: float = 1e-12,
    max_irls: int = 100,
    feature_names=None,
) -> ElasticNetPath:
    """Fit an elastic net regularisation path.

    Minimises, for each lambda,

        1/2 * sum_i w_i d(y_i, mu_i) + lam * sum_j pf_j (alpha |b_j| + (1 - alpha)/2 b_j^2)

    where ``w`` are the observation weights normalised to sum to one, ``d``
    is the family's unit deviance and ``b`` are the coefficients on the
    standardised scale (when ``standardize`` is true).
    """
    fam = get_family(family)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2 dimensional array")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y have different numbers of rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non finite values")
    fam.validate(y)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non negative, length n, and not all zero")
    w = w / w.sum()
    pf = np.ones(p) if penalty_factor is None else np.asarray(penalty_factor, dtype=float).ravel()
    if pf.shape[0] != p or np.any(pf < 0):
        raise ValueError("penalty_factor must be non negative with one entry per feature")

    prob = _Problem(X, y, w, fam, alpha, pf, fit_intercept, standardize, tol, max_sweeps, irls_tol, max_irls)

    # Null model: intercept plus unpenalised features (their penalty is zero
    # whatever lambda is, so any lambda works here).
    prob.solve(1.0, prob.unpenalized)
    s0 = prob.score()
    if fam.name == "gaussian":
        null_dev = fam.deviance(y, np.full(n, w @ y if fit_intercept else 0.0), w)
    else:
        mu0 = (w @ y) if fit_intercept else float(fam.inverse_link(np.zeros(1))[0])
        null_dev = fam.deviance(y, np.full(n, mu0), w)

    user_lambdas = lambdas is not None
    if user_lambdas:
        lambdas = np.sort(np.asarray(lambdas, dtype=float).ravel())[::-1]
        if np.any(lambdas < 0):
            raise ValueError("lambdas must be non negative")
    else:
        a = max(alpha, 1e-3)  # glmnet's convention so ridge paths start somewhere sensible
        pen = prob.penalized
        lambda_max = float(np.max(np.abs(s0[pen]) / (a * prob.pf[pen]))) if pen.size else 1.0
        if lambda_max <= 0:
            lambda_max = 1.0
        if lambda_min_ratio is None:
            lambda_min_ratio = 1e-4 if n > p else 1e-2
        lambdas = default_lambda_sequence(lambda_max, n_lambda, lambda_min_ratio)

    rows = []
    ever_active = set(prob.unpenalized.tolist())
    prev_lam = lambdas[0]
    for k, lam in enumerate(lambdas):
        s_prev = prob.score()
        if strong_rules:
            thresh = alpha * prob.pf * (2.0 * lam - prev_lam)
            strong = set(prob.penalized[np.abs(s_prev[prob.penalized]) >= thresh[prob.penalized]].tolist())
        else:
            strong = set(prob.penalized.tolist())
        work = np.array(sorted(strong | ever_active), dtype=int)
        sweeps = 0
        violations = 0
        # At an automatically chosen lambda_max the null model is optimal by
        # construction (for alpha >= 1e-3); keep it exactly rather than let
        # round off nudge a coefficient off zero.
        at_null = k == 0 and not user_lambdas and alpha >= 1e-3
        while not at_null:
            sweeps += prob.solve(lam, work)
            bad = prob.kkt_violators(lam, work)
            if bad.size == 0:
                break
            violations += bad.size
            work = np.union1d(work, bad)
        ever_active |= set(np.flatnonzero(prob.beta).tolist())
        dev = fam.deviance(y, prob.mu(), w)
        dev_ratio = 1.0 - dev / null_dev if null_dev > 0 else 0.0
        coef = prob.beta / prob.x_scale
        rows.append((lam, prob.b0 - prob.x_mean @ coef, coef, prob.beta.copy(), dev_ratio,
                     int(np.count_nonzero(prob.beta)), sweeps, violations, len(work)))
        prev_lam = lam

        if early_stop and not user_lambdas and k >= 1:
            if dev_ratio > 0.999:
                break
            if k >= 5 and dev_ratio - rows[-2][4] < 1e-5 * max(dev_ratio, 1e-12):
                break

    lam_arr = np.array([r[0] for r in rows])
    return ElasticNetPath(
        family=fam,
        alpha=alpha,
        lambdas=lam_arr,
        intercepts=np.array([r[1] for r in rows]),
        coefs=np.column_stack([r[2] for r in rows]),
        coefs_std=np.column_stack([r[3] for r in rows]),
        dev_ratio=np.array([r[4] for r in rows]),
        null_deviance=null_dev,
        df=np.array([r[5] for r in rows]),
        x_mean=prob.x_mean,
        x_scale=prob.x_scale,
        penalty_factor=prob.pf,
        sweeps=np.array([r[6] for r in rows]),
        kkt_violations=np.array([r[7] for r in rows]),
        strong_set_sizes=np.array([r[8] for r in rows]),
        feature_names=list(feature_names) if feature_names is not None else [],
    )
