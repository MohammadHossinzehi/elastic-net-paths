"""Exponential families with canonical links.

Every family works with observation weights ``w`` that have already been
normalised to sum to one.  With that convention the smooth part of the
objective is always

    L(eta) = 1/2 * sum_i w_i * d(y_i, mu_i)

where ``d`` is the unit deviance.  For the Gaussian family this is
``RSS / (2n)``; for the GLM families it is ``-loglik / n`` plus a constant,
which is exactly the loss glmnet minimises.

``working`` returns the iteratively reweighted least squares (IRLS)
quantities: the working response ``z`` and working weights ``W`` such that

    L(eta + delta) ~= const + 1/2 * sum_i W_i * (z_i - eta_i - delta_i)^2.
"""

from __future__ import annotations

import numpy as np

_EPS_PROB = 1e-9


def _xlogy_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * log(a / b) with the convention 0 * log(0 / b) = 0."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.zeros(a.shape)
    nz = a > 0
    out[nz] = a[nz] * np.log(a[nz] / b[nz])
    return out


class Family:
    name = "base"

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def link(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def deviance(self, y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
        return float(np.dot(w, self.unit_deviance(y, mu)))

    def variance(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def validate(self, y: np.ndarray) -> None:
        if not np.all(np.isfinite(y)):
            raise ValueError("response contains non finite values")

    def working(self, y: np.ndarray, eta: np.ndarray, w: np.ndarray):
        """Return (mu, W, z) for a canonical link family."""
        mu = self.inverse_link(eta)
        v = self.variance(mu)
        return mu, w * v, eta + (y - mu) / v


class Gaussian(Family):
    name = "gaussian"

    def inverse_link(self, eta):
        return np.asarray(eta, dtype=float)

    def link(self, mu):
        return np.asarray(mu, dtype=float)

    def unit_deviance(self, y, mu):
        return (y - mu) ** 2

    def variance(self, mu):
        return np.ones_like(mu, dtype=float)

    def working(self, y, eta, w):
        return np.asarray(eta, dtype=float), w.copy(), np.asarray(y, dtype=float).copy()


class Binomial(Family):
    name = "binomial"

    def inverse_link(self, eta):
        eta = np.asarray(eta, dtype=float)
        mu = np.exp(-np.logaddexp(0.0, -eta))
        return np.clip(mu, _EPS_PROB, 1.0 - _EPS_PROB)

    def link(self, mu):
        mu = np.clip(np.asarray(mu, dtype=float), _EPS_PROB, 1.0 - _EPS_PROB)
        return np.log(mu / (1.0 - mu))

    def unit_deviance(self, y, mu):
        return 2.0 * (_xlogy_ratio(y, mu) + _xlogy_ratio(1.0 - y, 1.0 - mu))

    def variance(self, mu):
        return mu * (1.0 - mu)

    def validate(self, y):
        super().validate(y)
        if np.any((y < 0) | (y > 1)):
            raise ValueError("binomial response must lie in [0, 1]")


class Poisson(Family):
    name = "poisson"

    def inverse_link(self, eta):
        return np.exp(np.clip(np.asarray(eta, dtype=float), -700.0, 700.0))

    def link(self, mu):
        return np.log(np.maximum(np.asarray(mu, dtype=float), 1e-300))

    def unit_deviance(self, y, mu):
        return 2.0 * (_xlogy_ratio(y, mu) - (y - mu))

    def variance(self, mu):
        return np.maximum(mu, 1e-300)

    def validate(self, y):
        super().validate(y)
        if np.any(y < 0):
            raise ValueError("poisson response must be non negative")


FAMILIES = {"gaussian": Gaussian, "binomial": Binomial, "poisson": Poisson}


def get_family(family) -> Family:
    if isinstance(family, Family):
        return family
    try:
        return FAMILIES[str(family).lower()]()
    except KeyError:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}") from None
