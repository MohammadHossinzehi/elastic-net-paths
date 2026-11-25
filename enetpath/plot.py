"""Dependency free SVG plots of coefficient paths and CV curves."""

from __future__ import annotations

import numpy as np

PALETTE = ["#2a6fdb", "#e0533d", "#1f9e6e", "#9b59b6", "#e39b14", "#16a2b8", "#c2185b", "#6d7f2b", "#5d4037", "#455a64"]


def _frame(width, height, title, xlabel, ylabel, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="12">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>\n'
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle">{xlabel}</text>\n'
        f'<text x="14" y="{height / 2}" text-anchor="middle" transform="rotate(-90 14 {height / 2})">{ylabel}</text>\n'
        + body + "</svg>\n"
    )


def _scales(xs, ys, width, height, m):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    pad = 0.05 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad
    sx = lambda x: m + (x1 - x) / (x1 - x0) * (width - 2 * m)  # large lambda on the left
    sy = lambda y: height - m - (y - y0) / (y1 - y0) * (height - 2 * m)
    return sx, sy, (x0, x1, y0, y1)


def _axes(sx, sy, bounds, width, height, m):
    x0, x1, y0, y1 = bounds
    out = [f'<rect x="{m}" y="{m}" width="{width - 2 * m}" height="{height - 2 * m}" fill="none" stroke="#999"/>']
    for x in np.linspace(x0, x1, 5):
        out.append(f'<text x="{sx(x):.1f}" y="{height - m + 16}" text-anchor="middle">{x:.2f}</text>')
    for y in np.linspace(y0, y1, 5):
        out.append(f'<text x="{m - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    if y0 < 0 < y1:
        out.append(f'<line x1="{m}" x2="{width - m}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#ccc" stroke-dasharray="4 3"/>')
    return "\n".join(out) + "\n"


def path_svg(path, filename=None, width=760, height=460, standardized=True):
    """Coefficient profiles against log(lambda)."""
    m = 60
    coefs = path.coefs_std if standardized else path.coefs
    xs = np.log(path.lambdas)
    sx, sy, b = _scales(xs, coefs, width, height, m)
    body = _axes(sx, sy, b, width, height, m)
    names = path.feature_names or [f"x{j}" for j in range(coefs.shape[0])]
    for j in range(coefs.shape[0]):
        if not np.any(coefs[j]):
            continue
        c = PALETTE[j % len(PALETTE)]
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, coefs[j]))
        body += f'<polyline fill="none" stroke="{c}" stroke-width="1.8" points="{pts}"/>\n'
        body += f'<text x="{width - m + 4}" y="{sy(coefs[j, -1]) + 4:.1f}" fill="{c}">{names[j]}</text>\n'
    title = f"{path.family.name} elastic net path (alpha = {path.alpha:g})"
    svg = _frame(width, height, title, "log(lambda)", "standardised coefficient" if standardized else "coefficient", body)
    if filename:
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(svg)
    return svg


def cv_svg(cv, filename=None, width=760, height=460):
    """Mean CV loss with one standard error bars, marking lambda_min and lambda_1se."""
    m = 60
    xs = np.log(cv.path.lambdas)
    lo, hi = cv.cvm - cv.cvsd, cv.cvm + cv.cvsd
    sx, sy, b = _scales(xs, np.concatenate([lo, hi]), width, height, m)
    body = _axes(sx, sy, b, width, height, m)
    for x, a, c in zip(xs, lo, hi):
        body += f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{sy(a):.1f}" y2="{sy(c):.1f}" stroke="#bbb"/>\n'
    pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, cv.cvm))
    body += f'<polyline fill="none" stroke="#e0533d" stroke-width="2" points="{pts}"/>\n'
    for lam, label in ((cv.lambda_min, "min"), (cv.lambda_1se, "1se")):
        x = sx(np.log(lam))
        body += f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{m}" y2="{height - m}" stroke="#2a6fdb" stroke-dasharray="5 4"/>\n'
        body += f'<text x="{x + 4:.1f}" y="{m + 14}" fill="#2a6fdb">{label}</text>\n'
    svg = _frame(width, height, f"{len(cv.fold_losses)} fold cross validation", "log(lambda)", f"CV {cv.measure}", body)
    if filename:
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(svg)
    return svg
