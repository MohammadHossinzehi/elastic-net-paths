"""Coordinate descent against an independent FISTA solver, plus KKT checks."""

import unittest

import numpy as np

from enetpath import enet_path
from enetpath.oracle import fista, objective, standardize
from tests.helpers import make_data, score, std_intercept

FAMILIES = ("gaussian", "binomial", "poisson")
ALPHAS = (1.0, 0.5, 0.1)


class AgainstFista(unittest.TestCase):
    def test_coefficients_and_objective(self):
        for family in FAMILIES:
            for alpha in ALPHAS:
                with self.subTest(family=family, alpha=alpha):
                    X, y = make_data(family, n=100, p=10, seed=21)
                    fit = enet_path(X, y, family=family, alpha=alpha, n_lambda=30)
                    Xs, _, _, w = standardize(X, np.ones(len(y)))
                    for k in range(0, fit.n_lambda, 6):
                        lam = fit.lambdas[k]
                        b0, b = fista(Xs, y, w, lam, alpha, fit.penalty_factor, family)
                        np.testing.assert_allclose(fit.coefs_std[:, k], b, atol=1e-8)
                        ours = objective(Xs, y, w, std_intercept(fit, k), fit.coefs_std[:, k], lam, alpha,
                                         fit.penalty_factor, family)
                        ref = objective(Xs, y, w, b0, b, lam, alpha, fit.penalty_factor, family)
                        self.assertLessEqual(ours, ref + 1e-10)


class KKT(unittest.TestCase):
    """Every point on every path satisfies the elastic net optimality conditions."""

    def test_whole_path(self):
        for family in FAMILIES:
            for alpha in ALPHAS:
                with self.subTest(family=family, alpha=alpha):
                    X, y = make_data(family, n=90, p=25, seed=4)
                    fit = enet_path(X, y, family=family, alpha=alpha, n_lambda=50)
                    for k, lam in enumerate(fit.lambdas):
                        s = score(fit, X, y, k)
                        b = fit.coefs_std[:, k]
                        pf = fit.penalty_factor
                        nz = b != 0
                        expect = lam * pf[nz] * (alpha * np.sign(b[nz]) + (1 - alpha) * b[nz])
                        np.testing.assert_allclose(s[nz], expect, atol=1e-6)
                        self.assertTrue(np.all(np.abs(s[~nz]) <= lam * alpha * pf[~nz] + 1e-6))

    def test_high_dimensional(self):
        X, y = make_data("gaussian", n=40, p=200, seed=9, k=5)
        fit = enet_path(X, y, alpha=0.9)
        for k, lam in enumerate(fit.lambdas):
            s = score(fit, X, y, k)
            zero = fit.coefs_std[:, k] == 0
            self.assertTrue(np.all(np.abs(s[zero]) <= lam * 0.9 + 1e-6))
        self.assertLessEqual(fit.df.max(), 40)


if __name__ == "__main__":
    unittest.main()
