"""The exact subspace reduction for curvature-aware bargaining.
property"): the maximiser of the unconstrained program \\eqref{eq:program}
lies in

    V = span{hat_g_0, ..., hat_g_{n-1}} + sum_i range(hat_H_i),

a subspace of dimension at most n + sum_i rank(hat_H_i), independent of the
number of network parameters p. Restricting the program to V is EXACT (not
an approximation), because V is invariant under phi's gradient map.

Pure numpy: no torch. Consumed by E7 (cost/scaling) and used as the
exactness oracle that the cheap v2 training step (streams/methods.py's
Delta = G^T beta reduction) is checked against in tests/test_reduction.py.
"""

from __future__ import annotations

import numpy as np

from cl_bargain.bargain.solver import SolveResult, solve
from cl_bargain.bargain.utilities import BargainProblem


def representer_basis(g: np.ndarray, H: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (p, k) for V = span{g_i} + sum_i range(H_i).

    Built by stacking each g_i and each H_i's nonzero-eigenvalue eigenvectors
    into one (p, k_raw) matrix, then taking an SVD-based orthonormal basis of
    its column space (numerically more stable than Gram-Schmidt on possibly
    near-parallel columns across players).
    """
    n, _p = g.shape
    cols = [g[i] for i in range(n)]
    for i in range(H.shape[0]):
        vals, vecs = np.linalg.eigh(H[i])
        thresh = tol * max(float(vals.max()), 1.0)
        nonzero = vals > thresh
        if np.any(nonzero):
            cols.extend(vecs[:, j] for j in np.flatnonzero(nonzero))
    M = np.stack(cols, axis=1)  # (p, k_raw)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(tol * max(float(S.max()), 1.0) < S)) if S.size else 0
    return U[:, :rank]  # (p, k), orthonormal columns spanning V


def reduce_problem(problem: BargainProblem, V: np.ndarray) -> BargainProblem:
    """Restrict the program to Delta = V z, z in R^k. Because V has
    orthonormal columns (V^T V = I_k), ||V z|| = ||z|| EXACTLY, so the
    robustness terms r_i||Delta|| and zeta_i/2||Delta||^2 reduce without
    approximation alongside the quadratic term -- this is what makes the
    reduction exact rather than merely a first-order approximation (the
    gradient-span restriction Delta = G^T beta used elsewhere in this repo,
    e.g. streams/methods.py, is exact only when H_i's range also lies in
    span{g_i}, which curvature need not respect; see Remark
    "What the reduction is and is not")."""
    g_red = problem.g @ V  # (n, k) = V^T g_i per row
    H_red = np.einsum("pk,ipq,ql->ikl", V, problem.H, V)  # (n, k, k) = V^T H_i V
    return BargainProblem(
        g=g_red, H=H_red, tau=problem.tau, alpha=problem.alpha, r=problem.r, zeta=problem.zeta
    )


def solve_reduced(
    problem: BargainProblem, tol: float = 1e-9, **solve_kwargs
) -> tuple[np.ndarray, np.ndarray, SolveResult]:
    """Solve the exact reduced program and lift the answer back to R^p.

    Returns (Delta, V, result): Delta = V @ result.Delta is the full-space
    update, V is the basis (p, k) used, and result is the SolveResult of the
    k-dimensional solve (result.Delta lives in R^k, NOT R^p)."""
    V = representer_basis(problem.g, problem.H, tol=tol)
    reduced = reduce_problem(problem, V)
    result = solve(reduced, **solve_kwargs)
    Delta = V @ result.Delta
    return Delta, V, result


__all__ = ["reduce_problem", "representer_basis", "solve_reduced"]
