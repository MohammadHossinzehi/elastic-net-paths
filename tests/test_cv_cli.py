import io
import os
import tempfile
import unittest

import numpy as np

from enetpath import cv_enet, cv_svg, enet_path, path_svg
from enetpath.cli import main
from tests.helpers import make_data


class CrossValidation(unittest.TestCase):
    def test_selects_sparse_true_model(self):
        rng = np.random.default_rng(0)
        n, p = 300, 30
        X = rng.normal(size=(n, p))
        y = X[:, 0] * 2 - X[:, 1] * 1.5 + X[:, 2] + rng.normal(size=n)
        cv = cv_enet(X, y, n_folds=5, seed=1)
        self.assertGreaterEqual(cv.lambda_1se, cv.lambda_min)
        _, b = cv.coef("1se")
        self.assertTrue(np.all(b[:3] != 0))
        self.assertLessEqual(np.count_nonzero(b[3:]), 3)

    def test_fold_losses_match_manual_refit(self):
        X, y = make_data("binomial", n=90, seed=5)
        cv = cv_enet(X, y, family="binomial", n_folds=3, seed=2, measure="class", n_lambda=20)
        test = cv.folds == 1
        fit = enet_path(X[~test], y[~test], family="binomial", lambdas=cv.path.lambdas)
        mu = fit.predict(X[test], kind="response")
        manual = ((mu > 0.5) != y[test][:, None]).mean(0)
        np.testing.assert_allclose(cv.fold_losses[1], manual)

    def test_deterministic_given_seed(self):
        X, y = make_data("poisson", seed=3)
        a = cv_enet(X, y, family="poisson", n_folds=4, seed=9)
        b = cv_enet(X, y, family="poisson", n_folds=4, seed=9)
        np.testing.assert_array_equal(a.cvm, b.cvm)

    def test_bad_measure(self):
        X, y = make_data("gaussian")
        with self.assertRaises(ValueError):
            cv_enet(X, y, measure="class", n_folds=3)


class PlotsAndCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        X, y = make_data("gaussian", n=60, p=5, seed=1)
        self.csv = os.path.join(self.tmp, "d.csv")
        with open(self.csv, "w") as fh:
            fh.write("a,b,c,target,d,e\n")
            for row, t in zip(X, y):
                fh.write(",".join(f"{v:.6f}" for v in row[:3]) + f",{t:.6f}," + ",".join(f"{v:.6f}" for v in row[3:]) + "\n")
        self.X, self.y = X, y

    def test_svg_is_well_formed(self):
        fit = enet_path(self.X, self.y, feature_names=list("abcde"))
        svg = path_svg(fit)
        self.assertTrue(svg.startswith("<svg") and svg.rstrip().endswith("</svg>"))
        self.assertIn("polyline", svg)
        cv = cv_enet(self.X, self.y, n_folds=3)
        self.assertIn("1se", cv_svg(cv))

    def test_fit_command(self):
        out = io.StringIO()
        plot = os.path.join(self.tmp, "p.svg")
        main(["fit", self.csv, "--target", "target", "--alpha", "0.5", "--plot", plot], out=out)
        text = out.getvalue()
        self.assertIn("gaussian elastic net", text)
        self.assertIn("(intercept)", text)
        self.assertTrue(os.path.exists(plot))

    def test_cv_command(self):
        out = io.StringIO()
        main(["cv", self.csv, "--target", "target", "--folds", "3"], out=out)
        self.assertIn("lambda_1se", out.getvalue())

    def test_missing_target(self):
        with self.assertRaises(SystemExit):
            main(["fit", self.csv, "--target", "nope"], out=io.StringIO())


if __name__ == "__main__":
    unittest.main()
