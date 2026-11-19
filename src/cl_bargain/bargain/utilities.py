"""Curvature utilities, robust slacks s_i, and the log-slack objective sum_i alpha_i log s_i."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS_NORM = 1e-12


def assert_psd(H: np.ndarray, tol: float = 1e-9, name: str = "H") -> None:
    """Raise if any H_i has an eigenvalue below -tol."""
    Hs = H if H.ndim == 3 else H[None]
    for i, Hi in enumerate(Hs):
        lam_min = np.linalg.eigvalsh(Hi).min()
        if lam_min < -tol:
            raise ValueError(
                f"{name}[{i}] is not PSD: min eigenvalue {lam_min:.3e} < -{tol:.1e}"
            )


@dataclass
class BargainProblem:
    """Bargaining instance: g (n, p), PSD H (n, p, p), budgets tau >= 0, alpha, radii r, zeta."""

    g: np.ndarray
    H: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray | None = None
    r: np.ndarray | None = None
    zeta: np.ndarray | None = None

    n: int = field(init=False)
    p: int = field(init=False)

    def __post_init__(self) -> None:
        self.g = np.asarray(self.g, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.tau = np.asarray(self.tau, dtype=float)
        self.n, self.p = self.g.shape
        if self.H.shape != (self.n, self.p, self.p):
            raise ValueError(f"H shape {self.H.shape} != {(self.n, self.p, self.p)}")
        if self.tau.shape != (self.n,):
            raise ValueError(f"tau shape {self.tau.shape} != {(self.n,)}")
        if np.any(self.tau < 0):
            raise ValueError("tau (tolerable-forgetting budgets) must be >= 0")
        # tau[0] is unconstrained here; CL callers set tau[0] = 0
        if self.alpha is None:
            self.alpha = np.full(self.n, 1.0 / self.n)
        else:
            self.alpha = np.asarray(self.alpha, dtype=float)
        self.r = np.zeros(self.n) if self.r is None else np.asarray(self.r, dtype=float)
        self.zeta = np.zeros(self.n) if self.zeta is None else np.asarray(self.zeta, dtype=float)

    def slacks(self, Delta: np.ndarray) -> np.ndarray:
        """s_i(Delta) for all i, shape (n,)."""
        lin = self.g @ Delta  # (n,)
        quad = np.einsum("p,ipq,q->i", Delta, self.H, Delta)
        norm = np.linalg.norm(Delta)
        return self.tau - lin - 0.5 * quad - self.r * norm - 0.5 * self.zeta * norm**2

    def feasible(self, Delta: np.ndarray) -> bool:
        return bool(np.all(self.slacks(Delta) > 0))

    def phi(self, Delta: np.ndarray) -> float:
        s = self.slacks(Delta)
        if np.any(s <= 0):
            return -np.inf
        return float(np.sum(self.alpha * np.log(s)))

    def _slack_grad_each(self, Delta: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(Delta)
        norm_safe = max(norm, _EPS_NORM)
        Hd = np.einsum("ipq,q->ip", self.H, Delta)  # (n,p)
        norm_term = (self.r / norm_safe)[:, None] * Delta[None, :] if norm > _EPS_NORM else 0.0
        return -self.g - Hd - norm_term - self.zeta[:, None] * Delta[None, :]

    def _slack_hess_each(self, Delta: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(Delta)
        norm_safe = max(norm, _EPS_NORM)
        eye = np.eye(self.p)
        if norm > _EPS_NORM:
            outer = np.outer(Delta, Delta) / norm_safe**3
            norm_hess = self.r[:, None, None] * (eye[None] / norm_safe - outer[None])
        else:
            norm_hess = 0.0
        return -self.H - norm_hess - self.zeta[:, None, None] * eye[None]

    def phi_grad(self, Delta: np.ndarray) -> np.ndarray:
        s = self.slacks(Delta)
        gs = self._slack_grad_each(Delta)
        return np.einsum("i,ip->p", self.alpha / s, gs)

    def phi_hess(self, Delta: np.ndarray) -> np.ndarray:
        s = self.slacks(Delta)
        gs = self._slack_grad_each(Delta)
        hs = self._slack_hess_each(Delta)
        term1 = np.einsum("i,ipq->pq", self.alpha / s, hs)
        term2 = np.einsum("i,ip,iq->pq", self.alpha / s**2, gs, gs)
        return term1 - term2

    def shadow_prices(self, Delta: np.ndarray) -> np.ndarray:
        s = self.slacks(Delta)
        return self.alpha / s

    def strong_concavity_modulus(self, Delta: np.ndarray) -> float:
        """m = alpha_min * (lambda_min(H_bar) + Z) / U_max."""
        s = self.slacks(Delta)
        H_bar = self.H.sum(axis=0)
        Z = self.zeta.sum()
        lam_min = np.linalg.eigvalsh(H_bar).min()
        U_max = s.max()
        return float(self.alpha.min() * (lam_min + Z) / U_max)

    def elasticity(self, Delta: np.ndarray) -> np.ndarray:
        """epsilon_i = r_i ||Delta|| / (2 s_i(Delta)), the shadow-price elasticity."""
        s = self.slacks(Delta)
        norm = np.linalg.norm(Delta)
        return self.r * norm / (2.0 * np.maximum(s, 1e-300))


def joint_rescale(
    problem: BargainProblem, c: np.ndarray | None = None
) -> tuple[BargainProblem, np.ndarray]:
    """Rescale (g_i, H_i, tau_i, r_i, zeta_i) by c_i, leaving Delta* unchanged."""
    if c is None:
        c = 1.0 / np.maximum(np.linalg.norm(problem.g, axis=1), 1e-12)
    c = np.asarray(c, dtype=float)
    rescaled = BargainProblem(
        g=problem.g * c[:, None],
        H=problem.H * c[:, None, None],
        tau=problem.tau * c,
        alpha=problem.alpha,
        r=problem.r * c,
        zeta=problem.zeta * c,
    )
    return rescaled, c
