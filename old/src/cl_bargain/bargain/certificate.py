"""The computable damage certificate, admission control, and telemetry
calibration for the computable damage certificate.
"Admission control: enforcing the budget with one scalar", and section
"Fitting the certificate constants from replay telemetry").

Everything here is pure numpy: no torch, no autodiff. `rho` (the fitted
Hessian-Lipschitz constant) and the robust radii `r`, `zeta` are supplied by
the caller -- in the real-stream experiments (E-cert, E-probe) they come from
`calibrate_lp` below, fitted on the probe slice of each task's buffer
(streams/buffer.py's `ReservoirBuffer.probe`).
"""

from __future__ import annotations

import numpy as np

from cl_bargain.bargain.utilities import BargainProblem


def certified_damage(problem: BargainProblem, Delta: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """C_i(Delta) = tau_i - s_i^R(Delta) + rho_i/6 ||Delta||^3
    (eq:direct-forgetting). Every term is evaluated at the COMPUTED Delta,
    independent of optimality or existence of an oracle maximiser -- the
    certificate holds for any strictly feasible Delta under the stated
    derivative-error bounds.

    rho: (n,) fitted cubic (Hessian-Lipschitz) constants, one per player.
    """
    s = problem.slacks(Delta)
    norm3 = float(np.linalg.norm(Delta)) ** 3
    return problem.tau - s + (np.asarray(rho, dtype=float) / 6.0) * norm3


def calibrated_certificate(predicted, q, rzr) -> np.ndarray:
    """C_i = <hat g_i, Delta> + 1/2 Delta^T hat H_i Delta + r q + zeta/2 q^2 + rho/6 q^3
    with q = ||Delta||: the certificate of Theorem "Computable damage
    certificate" written in the quantities `calibrate_lp` fits against, so
    that on the calibration window it covers every step by construction
    (Proposition "Certificate calibration").

    predicted: the quadratic predicted damage <hat g_i, Delta> + 1/2 Delta^T
    hat H_i Delta, the same term `telemetry_residual` subtracts.
    q: ||Delta|| in the space the update is applied in, the same q the fit used.
    rzr: (r, zeta, rho), or an array of shape (..., 3).
    """
    rzr = np.asarray(rzr, dtype=float)
    q = np.asarray(q, dtype=float)
    return (
        np.asarray(predicted, dtype=float)
        + rzr[..., 0] * q
        + 0.5 * rzr[..., 1] * q**2
        + (1.0 / 6.0) * rzr[..., 2] * q**3
    )


def admission_scale(
    problem: BargainProblem, Delta: np.ndarray, rho: np.ndarray, skip_indices: tuple[int, ...] = ()
) -> float:
    """beta* = min(1, min_i{tau_i/C_i(Delta) : C_i(Delta) > tau_i})
    (Proposition "Certified damage under shrinkage"): the one-scalar backtrack
    that admits beta*Delta with a certified F_i <= tau_i for every i, at the
    cost of one extra evaluation of s^R and no extra solve
    (Remark "Admission control is cheap, not optimal").

    `skip_indices` lets a caller exclude players from the scan (e.g. the live
    task, index 0, whose tau_0=0 convention means ANY positive certified
    damage forces beta*=0 -- a legitimate but very strict reading of "the
    live task must not lose"; callers that want the live task's own
    feasibility handled by the solver instead of this backtrack should pass
    skip_indices=(0,))."""
    C = certified_damage(problem, Delta, rho)
    mask = np.ones(problem.n, dtype=bool)
    for idx in skip_indices:
        mask[idx] = False
    over = mask & (problem.tau < C)
    if not np.any(over):
        return 1.0
    ratios = problem.tau[over] / C[over]
    return float(min(1.0, float(ratios.min())))


def telemetry_residual(
    loss_deltas: np.ndarray, g: np.ndarray, H: np.ndarray, Deltas: np.ndarray
) -> np.ndarray:
    """epsilon_{i,t} = [L_i^Probe(theta_{t+1}) - L_i^Probe(theta_t)]
    - [<hat_g_i, Delta_t> + 1/2 Delta_t^T hat_H_i Delta_t]
    (eq:telemetry-residual). Aggregates gradient-proxy error, curvature-proxy
    error, and the quadratic model's own failure over the step actually
    taken -- exactly what `calibrate_lp` below fits (r_i, zeta_i, rho_i) to
    absorb.

    loss_deltas: (T,) measured probe-loss increases.
    g: (p,) a fixed proxy gradient used at every step in the window, OR
       (T, p) one per step.
    H: (p, p) a fixed curvature operator, OR (T, p, p) one per step.
    Deltas: (T, p) the applied updates.
    """
    Deltas = np.asarray(Deltas, dtype=float)
    T, p = Deltas.shape
    g = np.asarray(g, dtype=float)
    g = np.broadcast_to(g, (T, p)) if g.ndim == 1 else g
    H = np.asarray(H, dtype=float)
    H = np.broadcast_to(H, (T, p, p)) if H.ndim == 2 else H

    lin = np.einsum("tp,tp->t", g, Deltas)
    quad = np.einsum("tp,tpq,tq->t", Deltas, H, Deltas)
    predicted = lin + 0.5 * quad
    return np.asarray(loss_deltas, dtype=float) - predicted


def calibrate_lp(q: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, bool]:
    """Proposition "Certificate calibration": the tightest (r, zeta, rho) >= 0
    such that, for every step t in the window,
        r*q_t + zeta/2*q_t^2 + rho/6*q_t^3 >= residual_t   (eq:calibration-lp)
    where q_t = ||Delta_t||. This is a 3-variable linear program (the
    objective and constraints are linear in (r, zeta, rho) for fixed q_t);
    solved here with scipy's HiGHS backend (scipy is already a dependency).

    Conservative by construction (a one-sided constraint, not a least-squares
    fit): a certificate that is exceeded is worthless, a loose one merely
    weak
    calibration"). It is also IN-SAMPLE: validity on the window does not
    imply validity at the next step -- E-cert measures the out-of-sample
    coverage this induces.

    Returns ((r, zeta, rho), success). On infeasibility (only possible if
    some q_t == 0 while residual_t > 0, since q_t > 0 always admits a large
    enough r), returns (nan, nan, nan), False.
    """
    from scipy.optimize import linprog

    q = np.asarray(q, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    # Columns of A are the coefficients of (r, zeta, rho) in the LHS of
    # eq:calibration-lp; the objective (the same LHS summed over the window)
    # is therefore exactly A.sum(axis=0) . (r, zeta, rho).
    A = np.stack([q, 0.5 * q**2, (1.0 / 6.0) * q**3], axis=1)  # (T, 3)
    c = A.sum(axis=0)
    result = linprog(c, A_ub=-A, b_ub=-residuals, bounds=[(0, None)] * 3, method="highs")
    if not result.success:
        return np.full(3, np.nan), False
    return result.x, True


__all__ = [
    "admission_scale",
    "calibrate_lp",
    "calibrated_certificate",
    "certified_damage",
    "telemetry_residual",
]
