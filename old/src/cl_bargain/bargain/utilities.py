"""Second-order (curvature) utilities, robust slacks, and the log-slack objective.

Notation follows the implementation's mathematical conventions:

    u_i(Delta)  = -<g_i, Delta> - 1/2 Delta^T H_i Delta            (utility)
    s_i(Delta)  = tau_i - <g_i, Delta> - 1/2 Delta^T H_i Delta
                  - r_i ||Delta|| - zeta_i/2 ||Delta||^2            (robust slack)
    phi(Delta)  = sum_i alpha_i log s_i(Delta)                      (objective)

In the CL application, player 0 is the live task: d_0 = 0, so by convention
tau[0] == 0 (a zero tolerable-forgetting budget -- the live task must not
lose) while absent players 1..n-1 have tau_i > 0. That convention is applied
by the instance generators in streams/sandbox.py; this class itself only
requires tau >= 0 (the general program of Claims C1/C2 is symmetric in all
players and allows tau_i > 0 for every i, including the live task).

Everything here is pure numpy: no torch, no autodiff. H_i is supplied by the
caller (closed form in the sandbox, an estimator in bargain/curvature.py for
real data) and MUST be PSD -- see `assert_psd`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS_NORM = 1e-12


def assert_psd(H: np.ndarray, tol: float = 1e-9, name: str = "H") -> None:
    """Raise if any H_i has an eigenvalue below -tol.

    A curvature estimate with negative eigenvalues makes the utility set
    non-convex and voids the bargaining interpretation.
    """
    Hs = H if H.ndim == 3 else H[None]
    for i, Hi in enumerate(Hs):
        lam_min = np.linalg.eigvalsh(Hi).min()
        if lam_min < -tol:
            raise ValueError(
                f"{name}[{i}] is not PSD: min eigenvalue {lam_min:.3e} < -{tol:.1e}"
            )


@dataclass
class BargainProblem:
    """A single bargaining instance: n players over p parameters.

    g: (n, p) proxy gradients
    H: (n, p, p) PSD curvature operators
    tau: (n,) tolerable-forgetting budgets, tau >= 0 (CL callers set tau[0] = 0)
    alpha: (n,) priority weights, default uniform
    r: (n,) robust gradient-proxy radii, default 0
    zeta: (n,) robust curvature radii, default 0
    """

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
        # No constraint here on tau[0]: the general program (Claims C1/C2) is
        # symmetric in all players and allows tau_i > 0 for every i, including
        # the live task. The CL-specific convention d_0 = 0 (tau[0] == 0,
        # "the live task must not lose") is applied by the instance
        # generators in streams/sandbox.py, not enforced by this class.
        if self.alpha is None:
            self.alpha = np.full(self.n, 1.0 / self.n)
        else:
            self.alpha = np.asarray(self.alpha, dtype=float)
        self.r = np.zeros(self.n) if self.r is None else np.asarray(self.r, dtype=float)
        self.zeta = np.zeros(self.n) if self.zeta is None else np.asarray(self.zeta, dtype=float)

    # ---- slacks and derivatives -----------------------------------------

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
        """grad s_i(Delta) for each i, shape (n, p)."""
        norm = np.linalg.norm(Delta)
        norm_safe = max(norm, _EPS_NORM)
        Hd = np.einsum("ipq,q->ip", self.H, Delta)  # (n,p)
        norm_term = (self.r / norm_safe)[:, None] * Delta[None, :] if norm > _EPS_NORM else 0.0
        return -self.g - Hd - norm_term - self.zeta[:, None] * Delta[None, :]

    def _slack_hess_each(self, Delta: np.ndarray) -> np.ndarray:
        """Hessian of s_i(Delta) for each i, shape (n, p, p)."""
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

    # ---- diagnostics used downstream by the solver / experiments --------

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
        """epsilon_i = r_i ||Delta|| / (2 s_i^R(Delta)) for each i
        (the shadow-price elasticity relation):
        the negative of d(log lambda_i)/d(log m_i) when r_i = sigma_i/sqrt(m_i)
        (Corollary "Stability of damped one-shot substitution" builds its
        stability threshold a < 3/(2 max_i epsilon_i) directly from this)."""
        s = self.slacks(Delta)
        norm = np.linalg.norm(Delta)
        return self.r * norm / (2.0 * np.maximum(s, 1e-300))


def joint_rescale(
    problem: BargainProblem, c: np.ndarray | None = None
) -> tuple[BargainProblem, np.ndarray]:
    """Lemma "Per-player scale invariance": choose c_i > 0 and replace
    (hat_g_i, hat_H_i, tau_i, r_i, zeta_i) -> c_i * (that same tuple), for
    EVERY player, together. This leaves Delta* exactly unchanged and maps
    lambda_i -> lambda_i/c_i (proved by substitution into phi; see
    numerical preconditioning without breaking the
    units"). Default c_i = 1/||hat_g_i|| is that Remark's own choice: it
    brings every player's raw gradient onto a common (unit) scale so tau/eps
    bind comparably across players with very different native gradient
    norms, WITHOUT normalising g_i alone -- normalising only g_i and leaving
    tau unscaled silently destroys tau_i's denomination in the units of L_i
    and voids the certificate (this is the bug the Remark warns about, and
    the one streams/methods.py's bargain_step used to have before this fix).

    Any certified damage C_i computed on the RESCALED problem must be
    divided by c_i to be reported in the original units (Remark, final
    sentence) -- see bargain/certificate.py's `certified_damage`, which takes
    `c` for exactly this.
    """
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
