"""Command line interface: ``python -m enetpath fit|cv data.csv --target y``."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .cv import MEASURES, cv_enet
from .path import enet_path
from .plot import cv_svg, path_svg


def read_csv(path, target):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise SystemExit(f"{path}: need a header row and at least one data row")
    header, body = rows[0], [r for r in rows[1:] if r]
    if target not in header:
        raise SystemExit(f"target column {target!r} not found; columns are {header}")
    t = header.index(target)
    names = [h for i, h in enumerate(header) if i != t]
    try:
        data = np.array([[float(v) for v in r] for r in body])
    except ValueError as exc:
        raise SystemExit(f"{path}: every value must be numeric ({exc})") from None
    return np.delete(data, t, axis=1), data[:, t], names


def _print_path(fit, every, out):
    out.write(f"{'step':>4} {'lambda':>12} {'df':>4} {'%dev':>8}\n")
    for k in range(0, fit.n_lambda, max(every, 1)):
        out.write(f"{k:>4} {fit.lambdas[k]:>12.6g} {fit.df[k]:>4d} {100 * fit.dev_ratio[k]:>8.3f}\n")
    if (fit.n_lambda - 1) % max(every, 1):
        k = fit.n_lambda - 1
        out.write(f"{k:>4} {fit.lambdas[k]:>12.6g} {fit.df[k]:>4d} {100 * fit.dev_ratio[k]:>8.3f}\n")


def _print_coefs(b0, b, names, out):
    out.write(f"  {'(intercept)':<20} {b0: .6g}\n")
    for name, v in zip(names, b):
        if v != 0:
            out.write(f"  {name:<20} {v: .6g}\n")


def build_parser():
    ap = argparse.ArgumentParser(prog="enetpath", description="Elastic net regularisation paths for GLMs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "cv"):
        sp = sub.add_parser(name)
        sp.add_argument("csv")
        sp.add_argument("--target", required=True)
        sp.add_argument("--family", default="gaussian", choices=["gaussian", "binomial", "poisson"])
        sp.add_argument("--alpha", type=float, default=1.0)
        sp.add_argument("--n-lambda", type=int, default=100)
        sp.add_argument("--no-standardize", action="store_true")
        sp.add_argument("--plot", help="write an SVG plot to this path")
        if name == "fit":
            sp.add_argument("--every", type=int, default=10, help="print every k-th path step")
            sp.add_argument("--lambda", dest="lam", type=float, help="print coefficients at this lambda")
        else:
            sp.add_argument("--folds", type=int, default=10)
            sp.add_argument("--measure", default="deviance", choices=MEASURES)
            sp.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None, out=None):
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    X, y, names = read_csv(args.csv, args.target)
    common = dict(family=args.family, alpha=args.alpha, n_lambda=args.n_lambda,
                  standardize=not args.no_standardize, feature_names=names)
    if args.cmd == "fit":
        fit = enet_path(X, y, **common)
        out.write(f"{fit.family.name} elastic net, alpha={fit.alpha:g}, n={len(y)}, p={X.shape[1]}, "
                  f"{fit.n_lambda} lambdas\n")
        _print_path(fit, args.every, out)
        lam = args.lam if args.lam is not None else fit.lambdas[-1]
        out.write(f"\ncoefficients at lambda={lam:.6g}:\n")
        _print_coefs(*fit.coef_at(lam), names, out)
        if args.plot:
            path_svg(fit, args.plot)
            out.write(f"\nwrote {args.plot}\n")
    else:
        cv = cv_enet(X, y, n_folds=args.folds, measure=args.measure, seed=args.seed, **common)
        out.write(f"{args.folds} fold CV ({cv.measure}), alpha={args.alpha:g}\n")
        for label, k in (("lambda_min", cv.index_min), ("lambda_1se", cv.index_1se)):
            out.write(f"{label}: {cv.path.lambdas[k]:.6g}  cv={cv.cvm[k]:.6g} +/- {cv.cvsd[k]:.3g}  "
                      f"df={cv.path.df[k]}\n")
        out.write("\ncoefficients at lambda_1se:\n")
        _print_coefs(*cv.coef("1se"), names, out)
        if args.plot:
            cv_svg(cv, args.plot)
            out.write(f"\nwrote {args.plot}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
