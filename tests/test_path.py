"""Path level behaviour: strong rules, invariances, weights and penalty factors."""

import unittest
import warnings

import numpy as np

from enetpath import cd_wls, enet_path
from enetpath.oracle import newton_mle
from tests.helpers import make_data, score


class StrongRules(unittest.TestCase):
    def test_same_answer_with_and_without(self):
        for family in ("gaussian", "binomial", "poisson"):
            X, y = make_data(family, n=80, p=60, seed=13)
            a = enet_path(X, y, family=family, alpha=0.7, strong_rules=True)
            b = enet_path(X, y, family=family, alpha=0.7, strong_rules=False, lambdas=a.lambdas)
            np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-7)

    def test_strong_set_is_much_smaller(self):
        X, y = make_data("gaussian", n=100, p=400, seed=2, k=6)
        fit = enet_path(X, y, alpha=1.0)
        self.assertLess(np.median(fit.strong_set_sizes), 400 / 4)

    def test_violations_are_caught(self):
        # A deliberately coarse lambda grid makes the sequential strong rule
        # wrong sometimes; the KKT pass must repair every such case.
        X, y = make_data("gaussian", n=60, p=80, seed=31, k=10)
        dense = enet_path(X, y, alpha=1.0, n_lambda=200)
        coarse_l = dense.lambdas[::40]
        fit = enet_path(X, y, alpha=1.0, lambdas=coarse_l)
        ref = enet_path(X, y, alpha=1.0, lambdas=coarse_l, strong_rules=False)
        np.testing.assert_allclose(fit.coefs, ref.coefs, atol=1e-7)
        for k, lam in enumerate(fit.lambdas):
            zero = fit.coefs_std[:, k] == 0
            self.assertTrue(np.all(np.abs(score(fit, X, y, k)[zero]) <= lam + 1e-6))


class Polishing(unittest.TestCase):
    def test_polish_changes_speed_not_answer(self):
        X, y = make_data("gaussian", n=80, p=40, seed=13)
        Xs = (X - X.mean(0)) / X.std(0)
        W = np.full(len(y), 1.0 / len(y))
        args = (Xs, y, W)
        lam = 0.01
        cand = np.arange(X.shape[1])
        b0a, ba, sa = cd_wls(*args, np.zeros(40), y.mean(), lam, 1.0, np.ones(40), cand, tol=1e-20)
        b0b, bb, sb = cd_wls(*args, np.zeros(40), y.mean(), lam, 1.0, np.ones(40), cand, tol=1e-20,
                             polish_tol=0.0)
        np.testing.assert_allclose(ba, bb, atol=1e-9)
        self.assertAlmostEqual(b0a, b0b, places=9)
        self.assertLess(sa, sb)


class Invariances(unittest.TestCase):
    def test_feature_rescaling(self):
        X, y = make_data("gaussian", seed=6)
        c = np.linspace(0.01, 100, X.shape[1])
        a = enet_path(X, y, alpha=0.5)
        b = enet_path(X * c, y, alpha=0.5, lambdas=a.lambdas)
        np.testing.assert_allclose(b.coefs * c[:, None], a.coefs, rtol=1e-6, atol=1e-9)

    def test_feature_shift(self):
        X, y = make_data("binomial", seed=6)
        a = enet_path(X, y, family="binomial")
        b = enet_path(X + 1000.0, y, family="binomial", lambdas=a.lambdas)
        np.testing.assert_allclose(b.coefs, a.coefs, atol=1e-7)
        np.testing.assert_allclose(a.predict(X), b.predict(X + 1000.0), atol=1e-6)

    def test_column_permutation(self):
        X, y = make_data("poisson", seed=8)
        perm = np.random.default_rng(0).permutation(X.shape[1])
        a = enet_path(X, y, family="poisson")
        b = enet_path(X[:, perm], y, family="poisson", lambdas=a.lambdas)
        np.testing.assert_allclose(b.coefs, a.coefs[perm], atol=1e-7)


class Weights(unittest.TestCase):
    def test_integer_weights_equal_duplicated_rows(self):
        for family in ("gaussian", "binomial", "poisson"):
            X, y = make_data(family, n=60, seed=17)
            w = np.random.default_rng(1).integers(1, 4, size=len(y))
            a = enet_path(X, y, family=family, weights=w, alpha=0.8)
            rep = np.repeat(np.arange(len(y)), w)
            b = enet_path(X[rep], y[rep], family=family, alpha=0.8, lambdas=a.lambdas)
            np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-7)
            np.testing.assert_allclose(a.intercepts, b.intercepts, atol=1e-7)

    def test_zero_weight_rows_are_ignored(self):
        X, y = make_data("gaussian", n=70, seed=19)
        w = np.ones(len(y))
        w[:10] = 0
        a = enet_path(X, y, weights=w)
        b = enet_path(X[10:], y[10:], lambdas=a.lambdas)
        np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-8)


class PenaltyFactors(unittest.TestCase):
    def test_unpenalised_feature_always_in_model(self):
        X, y = make_data("gaussian", seed=23)
        pf = np.ones(X.shape[1])
        pf[-1] = 0.0  # a pure noise feature, but unpenalised
        fit = enet_path(X, y, penalty_factor=pf)
        self.assertTrue(np.all(fit.coefs[-1] != 0))
        self.assertTrue(np.all(fit.coefs[:-1, 0] == 0))

    def test_unpenalised_block_matches_partial_mle_at_lambda_max(self):
        X, y = make_data("binomial", n=200, seed=29)
        pf = np.ones(X.shape[1])
        pf[:3] = 0.0
        fit = enet_path(X, y, family="binomial", penalty_factor=pf)
        b0, b = newton_mle(X[:, :3], y, np.ones(len(y)), "binomial")
        np.testing.assert_allclose(fit.coefs[:3, 0], b, atol=1e-6)
        self.assertAlmostEqual(fit.intercepts[0], b0, places=6)

    def test_rescaling_convention(self):
        X, y = make_data("gaussian", seed=1)
        a = enet_path(X, y, penalty_factor=np.full(X.shape[1], 5.0))
        b = enet_path(X, y)
        np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-10)

    def test_all_zero_rejected(self):
        X, y = make_data("gaussian")
        with self.assertRaises(ValueError):
            enet_path(X, y, penalty_factor=np.zeros(X.shape[1]))


class PathShape(unittest.TestCase):
    def test_lambdas_decrease_and_dev_ratio_grows(self):
        for family in ("gaussian", "binomial", "poisson"):
            X, y = make_data(family, seed=3)
            fit = enet_path(X, y, family=family)
            self.assertTrue(np.all(np.diff(fit.lambdas) < 0))
            self.assertTrue(np.all(np.diff(fit.dev_ratio) >= -1e-10))

    def test_default_ratio_depends_on_shape(self):
        X, y = make_data("gaussian", n=50, p=10)
        fit = enet_path(X, y, early_stop=False)
        self.assertAlmostEqual(fit.lambdas[-1] / fit.lambdas[0], 1e-4)
        X, y = make_data("gaussian", n=20, p=40)
        fit = enet_path(X, y, early_stop=False)
        self.assertAlmostEqual(fit.lambdas[-1] / fit.lambdas[0], 1e-2)

    def test_separable_logistic_stops_early(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = enet_path(X, y, family="binomial")
        self.assertLess(fit.n_lambda, 100)
        self.assertGreater(fit.dev_ratio[-1], 0.99)

    def test_constant_column(self):
        X, y = make_data("gaussian", seed=4)
        X[:, 2] = 3.0
        fit = enet_path(X, y)
        self.assertTrue(np.all(fit.coefs[2] == 0))

    def test_coef_interpolation(self):
        X, y = make_data("gaussian", seed=4)
        fit = enet_path(X, y)
        b0, b = fit.coef_at(fit.lambdas[7])
        np.testing.assert_allclose(b, fit.coefs[:, 7])
        lam = np.sqrt(fit.lambdas[7] * fit.lambdas[8])
        _, b = fit.coef_at(lam)
        np.testing.assert_allclose(b, 0.5 * (fit.coefs[:, 7] + fit.coefs[:, 8]))
        np.testing.assert_allclose(fit.predict(X, lam=fit.lambdas[3]), fit.predict(X)[:, 3])

    def test_no_intercept(self):
        X, y = make_data("gaussian", n=200, p=5, seed=2)
        fit = enet_path(X, y, fit_intercept=False, lambdas=[1e-12])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(fit.coefs[:, 0], b, atol=1e-8)
        self.assertEqual(fit.intercepts[0], 0.0)

    def test_input_validation(self):
        X, y = make_data("gaussian")
        for bad in (dict(alpha=1.5), dict(weights=-np.ones(len(y))), dict(family="gamma")):
            with self.assertRaises(ValueError):
                enet_path(X, y, **bad)
        with self.assertRaises(ValueError):
            enet_path(X, y + 5, family="binomial")
        with self.assertRaises(ValueError):
            enet_path(X, -y, family="poisson")


if __name__ == "__main__":
    unittest.main()
