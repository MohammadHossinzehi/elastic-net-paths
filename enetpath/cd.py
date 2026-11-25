"""Cyclic coordinate descent for penalised weighted least squares.

Solves, for fixed working weights ``W`` and working response ``z``,

    minimise  1/2 * sum_i W_i (z_i - b0 - x_i . b)^2
              + lam * sum_j pf_j * (alpha * |b_j| + (1 - alpha) / 2 * b_j^2)

over the coordinates listed in ``candidates`` (all other coefficients are
held at their current values, which the caller guarantees are zero).

The update for coordinate j is the closed form

    b_j <- S(g_j, lam * alpha * pf_j) / (v_j + lam * (1 - alpha) * pf_j)

with ``g_j = sum_i W_i x_ij r_i + v_j b_j``, ``v_j = sum_i W_i x_ij^2`` and
``S`` the soft threshold operator.  The residual is updated in place so a
single coordinate step costs O(n).

Like glmnet we alternate one sweep over every candidate with repeated sweeps
over the currently non zero coefficients only, and declare convergence when
the largest weighted squared change ``v_j * delta_j^2`` in a full sweep falls
below ``tol``.  Once the active set sweeps have nearly settled (change
below ``polish_tol``) we additionally solve the active set optimality
equations exactly (see ``polish``), which turns the slow linear tail of
coordinate descent into a single linear solve.
"""

from __future__ import annotations

import warnings

import numpy as np


class ConvergenceWarning(UserWarning):
    pass


def soft_threshold(g: float, t: float) -> float:
    if g > t:
        return g - t
    if g < -t:
        return g + t
    return 0.0


def cd_wls(
    X: np.ndarray,
    z: np.ndarray,
    W: np.ndarray,
    beta: np.ndarray,
    b0: float,
    lam: float,
    alpha: float,
    pf: np.ndarray,
    candidates: np.ndarray,
    fit_intercept: bool = True,
    tol: float = 1e-10,
    max_sweeps: int = 100_000,
    polish_tol: float = 1e-8,
):
    """Run coordinate descent in place on ``beta``; return (b0, beta, sweeps)."""
    candidates = np.asarray(candidates, dtype=int)
    r = z - b0 - X @ beta
    XW = X * W[:, None]
    v = np.zeros(X.shape[1])
    if candidates.size:
        v[candidates] = np.einsum("ij,ij->j", XW[:, candidates], X[:, candidates])
    l1 = lam * alpha * pf
    l2 = lam * (1.0 - alpha) * pf
    sw = float(W.sum())
    sweeps = 0

    def sweep(idx) -> float:
        nonlocal b0, r
        biggest = 0.0
        for j in idx:
            vj = v[j]
            if vj <= 0.0:
                continue
            bj = beta[j]
            g = float(XW[:, j] @ r) + vj * bj
            new = soft_threshold(g, l1[j]) / (vj + l2[j])
            if new != bj:
                d = new - bj
                r -= d * X[:, j]
                beta[j] = new
                biggest = max(biggest, vj * d * d)
        if fit_intercept and sw > 0.0:
            d = float(W @ r) / sw
            if d != 0.0:
                b0 += d
                r -= d
                biggest = max(biggest, sw * d * d)
        return biggest

    def polish() -> bool:
        """Solve the active set equations exactly, given the current signs.

        Once coordinate descent has found the right support and signs, the
        elastic net optimality conditions restricted to that support are a
        plain linear system.  Solving it removes the slow linear tail of
        coordinate descent on correlated designs.  The step is accepted only
        if no sign flips; the caller then re-verifies with a full sweep.
        """
        nonlocal b0, r
        A = candidates[beta[candidates] != 0.0]
        XA = X[:, A]
        sgn = np.sign(beta[A])
        zA = r + b0 + XA @ beta[A]  # working response net of inactive terms
        G = XA.T @ (W[:, None] * XA) + np.diag(l2[A])
        rhs = XA.T @ (W * zA) - l1[A] * sgn
        if fit_intercept:
            m = W @ XA
            M = np.zeros((len(A) + 1, len(A) + 1))
            M[0, 0], M[0, 1:], M[1:, 0], M[1:, 1:] = sw, m, m, G
            rhs = np.concatenate([[W @ zA], rhs])
        else:
            M = G
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return False
        if not np.all(np.isfinite(sol)):
            return False
        new_b0, new_bA = (sol[0], sol[1:]) if fit_intercept else (b0, sol)
        if A.size and np.any(np.sign(new_bA) != sgn):
            return False
        beta[A] = new_bA
        b0 = float(new_b0)
        r = zA - b0 - XA @ new_bA
        return True

    while sweeps < max_sweeps:
        sweeps += 1
        if sweep(candidates) < tol:
            break
        polish_tried = False
        while sweeps < max_sweeps:
            active = candidates[beta[candidates] != 0.0]
            sweeps += 1
            change = sweep(active)
            if change < tol:
                break
            if change < polish_tol and not polish_tried:
                polish_tried = True
                if polish():
                    break
    else:
        warnings.warn(
            f"coordinate descent hit max_sweeps={max_sweeps} at lambda={lam:.4g}",
            ConvergenceWarning,
            stacklevel=2,
        )
    return b0, beta, sweeps
