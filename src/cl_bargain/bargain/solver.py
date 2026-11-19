"""Damped Newton solver for the log-slack bargaining program."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cl_bargain.bargain.utilities import BargainProblem

_EPS_NORM = 1e-12


@dataclass
class SolveResult:
    Delta: np.ndarray
    slacks: np.ndarray
    mu: float
    U_max: float
    shadow_prices: np.ndarray
    modulus: float
    converged: bool
    n_iter: int
    grad_norm: float
    feasible_start_ok: bool
    feasible_start_halvings: int


def feasible_start(
    problem: BargainProblem,
    t0: float = 1.0,
    max_halvings: int = 200,
) -> tuple[np.ndarray | None, int]:
    """Delta = -t g_0/||g_0||, halving t until feasible; returns (Delta or None, n_halvings)."""
    zero = np.zeros(problem.p)
    if problem.feasible(zero):
        # feasible at zero whenever every tau_i > 0
        return zero, 0

    g0 = problem.g[0]
    norm_g0 = np.linalg.norm(g0)
    if norm_g0 < _EPS_NORM:
        # zero live-task gradient: use the absent tasks' combined descent direction
        direction = -problem.g[1:].sum(axis=0)
        norm_dir = np.linalg.norm(direction)
        if norm_dir < _EPS_NORM:
            zero_ok = problem.feasible(np.zeros(problem.p))
            return (np.zeros(problem.p), 0) if zero_ok else (None, max_halvings)
        unit = direction / norm_dir
    else:
        unit = -g0 / norm_g0

    t = t0
    for k in range(max_halvings):
        Delta = t * unit
        if problem.feasible(Delta):
            return Delta, k
        t *= 0.5
    return None, max_halvings


def solve(
    problem: BargainProblem,
    Delta0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    max_backtrack: int = 60,
    armijo_c: float = 1e-4,
) -> SolveResult:
    """Damped Newton ascent on phi with a feasibility-preserving line search."""
    fs_halvings = 0
    fs_ok = True
    if Delta0 is None:
        Delta0, fs_halvings = feasible_start(problem)
        if Delta0 is None:
            fs_ok = False
            Delta0 = np.zeros(problem.p)
    elif not problem.feasible(Delta0):
        # infeasible warm start: fall back to a fresh feasible start
        Delta0, fs_halvings = feasible_start(problem)
        if Delta0 is None:
            fs_ok = False
            Delta0 = np.zeros(problem.p)

    Delta = Delta0.copy()
    converged = False
    n_iter = 0
    grad_norm = np.inf
    p = problem.p

    for n_iter in range(1, max_iter + 1):  # noqa: B007 (n_iter reported in the result)
        grad = problem.phi_grad(Delta)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < tol:
            converged = True
            break

        hess = problem.phi_hess(Delta)
        phi0 = problem.phi(Delta)

        step = _newton_step(hess, grad, p)
        if step @ grad <= 0:
            # not an ascent direction: fall back to steepest ascent
            step = grad / grad_norm

        t = 1.0
        accepted = False
        target_slope = armijo_c * (grad @ step)
        for _ in range(max_backtrack):
            Delta_new = Delta + t * step
            if problem.feasible(Delta_new):
                phi_new = problem.phi(Delta_new)
                if phi_new >= phi0 + t * target_slope:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            break
        Delta = Delta_new

    s = problem.slacks(Delta)
    return SolveResult(
        Delta=Delta,
        slacks=s,
        mu=float(s.min()),
        U_max=float(s.max()),
        shadow_prices=problem.shadow_prices(Delta),
        modulus=problem.strong_concavity_modulus(Delta),
        converged=converged,
        n_iter=n_iter,
        grad_norm=grad_norm,
        feasible_start_ok=fs_ok,
        feasible_start_halvings=fs_halvings,
    )


def _newton_step(hess: np.ndarray, grad: np.ndarray, p: int) -> np.ndarray:
    """Newton step -hess^{-1} grad, Levenberg-Marquardt damped until it ascends."""
    damp = 0.0
    eye = np.eye(p)
    for _ in range(30):
        try:
            step = np.linalg.solve(hess - damp * eye, -grad)
        except np.linalg.LinAlgError:
            damp = max(damp * 10, 1e-8)
            continue
        if step @ grad > 0 or damp > 1e6:
            return step
        damp = max(damp * 10, 1e-8)
    return grad / np.linalg.norm(grad)
