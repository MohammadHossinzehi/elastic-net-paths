"""Cases where the elastic net solution is known exactly."""

import unittest

import numpy as np

from enetpath import enet_path, soft_threshold
from enetpath.oracle import newton_mle, standardize
from tests.helpers import make_data, std_intercept


class SoftThreshold(unittest.TestCase):
    def test_values(self):
        self.assertEqual(soft_threshold(3.0, 1.0), 2.0)
        self.assertEqual(soft_threshold(-3.0, 1.0), -2.0)
        self.assertEqual(soft_threshold(0.5, 1.0), 0.0)
        self.assertEqual(soft_threshold(-1.0, 1.0), 0.0)


class OrthonormalDesign(unittest.TestCase):
    """With centred columns satisfying X'X/n = I the Gaussian solution is
    b_j = S(x_j'y/n, lam*alpha) / (1 + lam*(1-alpha)) for every lambda."""

    def setUp(self):
        rng = np.random.default_rng(7)
        n, p = 200, 8
        A = rng.normal(size=(n, p))
        A -= A.mean(0)
        Q, _ = np.linalg.qr(A)
        self.X = Q * np.sqrt(n)
        self.y = self.X @ np.array([3, -2, 1, 0.5, 0, 0, 0, 0.2]) + rng.normal(size=n)
        self.n = n

    def check(self, alpha):
        fit = enet_path(self.X, self.y, alpha=alpha, n_lambda=40, early_stop=False)
        c = self.X.T @ (self.y - self.y.mean()) / self.n
        for k, lam in enumerate(fit.lambdas):
            expect = np.array([soft_threshold(cj, lam * alpha) for cj in c]) / (1 + lam * (1 - alpha))
            np.testing.assert_allclose(fit.coefs_std[:, k], expect, atol=1e-9)
        np.testing.assert_allclose(fit.intercepts, self.y.mean(), atol=1e-9)

    def test_lasso(self):
        self.check(1.0)

    def test_elastic_net(self):
        self.check(0.3)

    def test_ridge(self):
        self.check(0.0)

    def test_lambda_max_is_tight(self):
        fit = enet_path(self.X, self.y, alpha=1.0, n_lambda=20)
        self.assertEqual(fit.df[0], 0)
        self.assertGreater(fit.df[1], 0)
        c = self.X.T @ (self.y - self.y.mean()) / self.n
        self.assertAlmostEqual(fit.lambdas[0], np.max(np.abs(c)), places=12)


class RidgeClosedForm(unittest.TestCase):
    def test_matches_normal_equations(self):
        X, y = make_data("gaussian", n=80, p=15, seed=3)
        Xs, _, _, w = standardize(X, np.ones(len(y)))
        fit = enet_path(X, y, alpha=0.0, n_lambda=25)
        yc = y - y.mean()
        for k, lam in enumerate(fit.lambdas):
            b = np.linalg.solve(Xs.T @ Xs / len(y) + lam * np.eye(X.shape[1]), Xs.T @ yc / len(y))
            np.testing.assert_allclose(fit.coefs_std[:, k], b, atol=1e-8)


class UnpenalisedLimit(unittest.TestCase):
    """As lambda -> 0 with n > p the path reaches the maximum likelihood fit."""

    def check(self, family, atol):
        X, y = make_data(family, n=300, p=6, seed=11)
        fit = enet_path(X, y, family=family, lambdas=[1e-12])
        b0, b = newton_mle(X, y, np.ones(len(y)), family)
        np.testing.assert_allclose(fit.coefs[:, 0], b, atol=atol)
        self.assertAlmostEqual(fit.intercepts[0], b0, delta=atol)

    def test_ols(self):
        self.check("gaussian", 1e-8)

    def test_logistic(self):
        self.check("binomial", 1e-6)

    def test_poisson(self):
        self.check("poisson", 1e-6)


class InterceptOnly(unittest.TestCase):
    def test_null_model_at_lambda_max(self):
        for family in ("gaussian", "binomial", "poisson"):
            X, y = make_data(family, seed=5)
            fit = enet_path(X, y, family=family, n_lambda=10)
            self.assertTrue(np.all(fit.coefs[:, 0] == 0))
            mu0 = fit.family.inverse_link(np.array([std_intercept(fit, 0)]))[0]
            self.assertAlmostEqual(mu0, y.mean(), places=8)
            self.assertAlmostEqual(fit.dev_ratio[0], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
