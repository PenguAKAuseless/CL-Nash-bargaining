"""First-order Nash bargaining baseline solved over a trust region ||Delta|| <= epsilon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class V1Result:
    Delta: np.ndarray
    slacks: np.ndarray
    K: np.ndarray  # (n, n) Gram matrix G G^T
    lambda_min_K: float
    converged: bool
    n_iter: int
    feasible: bool


def gram_matrix(g: np.ndarray) -> np.ndarray:
    """K = G G^T, the n x n task-gradient Gram matrix."""
    return g @ g.T


def _feasible_start_v1(
    g: np.ndarray, tau: np.ndarray, epsilon: float, max_halvings: int = 60
) -> np.ndarray:
    """Feasible Delta in the ball: mean-gradient direction, then player 0's, then zero."""
    p = g.shape[1]
    candidates = []
    mean_dir = -g.mean(axis=0)
    if np.linalg.norm(mean_dir) > 1e-14:
        candidates.append(mean_dir / np.linalg.norm(mean_dir))
    if np.linalg.norm(g[0]) > 1e-14:
        candidates.append(-g[0] / np.linalg.norm(g[0]))

    for direction in candidates:
        t = epsilon
        for _ in range(max_halvings):
            delta = t * direction
            if np.all(tau - g @ delta > 0):
                return delta
            t *= 0.5
    return np.zeros(p)


def _projected_ascent_v1(g, tau, alpha, epsilon, Delta, tol, max_iter: int = 20000):
    """Projected Armijo ascent on the ball; fallback when the fixed point does not converge."""

    def phi(d):
        return float(alpha @ np.log(tau - g @ d))

    def project(d):
        nd = np.linalg.norm(d)
        return d if nd <= epsilon else epsilon * d / nd

    f = phi(Delta)
    eta = 1.0
    for _ in range(max_iter):
        grad = -((alpha / (tau - g @ Delta)) @ g)
        while True:
            cand = project(Delta + eta * grad)
            s = tau - g @ cand
            if np.all(s > 0):
                f_c = phi(cand)
                if f_c >= f + 1e-4 * grad @ (cand - Delta):
                    break
            eta *= 0.5
            if eta < 1e-16:
                return Delta, False
        step = np.linalg.norm(cand - Delta) / eta
        Delta, f = cand, f_c
        if step < tol * max(1.0, np.linalg.norm(grad)):
            return Delta, True
        eta = min(eta * 2.0, 1e6)
    return Delta, False


def solve_v1_trust_region(
    g: np.ndarray,
    tau: np.ndarray,
    alpha: np.ndarray,
    epsilon: float,
    max_iter: int = 2000,
    tol: float = 1e-9,
    damping: float = 0.5,
) -> V1Result:
    """Damped fixed-point solve of the trust-region-active first-order NBS."""
    Delta = _feasible_start_v1(g, tau, epsilon)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 (n_iter reported in the result)
        s = tau - g @ Delta
        if np.any(s <= 0):
            break
        w = alpha / s
        v = w @ g
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            break
        target = -epsilon * v / norm_v
        # stationarity on the sphere is Delta = target
        if np.linalg.norm(target - Delta) < tol * max(epsilon, 1.0):
            converged = True
            break
        # halve the damped step until every slack stays positive
        t = damping
        Delta_new = None
        while t > 1e-8:
            cand = t * target + (1 - t) * Delta
            norm_c = np.linalg.norm(cand)
            if norm_c > 1e-14:
                cand = epsilon * cand / norm_c
            if np.all(tau - g @ cand > 0):
                Delta_new = cand
                break
            t *= 0.5
        if Delta_new is None:
            break
        Delta = Delta_new

    s = tau - g @ Delta
    if not converged and np.all(s > 0):
        Delta, converged = _projected_ascent_v1(g, tau, alpha, epsilon, Delta, tol)
        s = tau - g @ Delta
    K = gram_matrix(g)
    lam_min_K = float(np.linalg.eigvalsh(K).min())
    return V1Result(
        Delta=Delta,
        slacks=s,
        K=K,
        lambda_min_K=lam_min_K,
        converged=converged,
        n_iter=n_iter,
        feasible=bool(np.all(s > 0)),
    )
