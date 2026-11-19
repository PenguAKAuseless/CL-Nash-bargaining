"""Quadratic sandbox where every quantity
in the theory -- true g_i, true sigma_i, true forgetting -- is observable in
closed form.

Task i: PSD H_i = A^T A / rank with rank ~ r << p; N examples
z_ij ~ N(c_i, S_i); loss L_i(theta) = 1/(2N) sum_j (theta-z_ij)^T H_i (theta-z_ij).
Then g_i(theta) = H_i (theta - zbar_i) exactly, sigma_i = sqrt(tr(H_i S_i H_i))
is theta-independent, and forgetting L_i(theta+Delta) - L_i(theta) is closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cl_bargain.bargain.curvature import random_psd
from cl_bargain.bargain.utilities import BargainProblem


@dataclass
class QuadraticTask:
    H: np.ndarray  # (p, p) PSD curvature
    zbar: np.ndarray  # (p,) empirical mean of stored examples
    S: np.ndarray  # (p, p) covariance used to draw examples (population, not empirical)
    sigma: float  # sqrt(tr(H S H)), theta-independent
    Z: np.ndarray  # (N, p) stored examples

    def g(self, theta: np.ndarray) -> np.ndarray:
        """Exact gradient of L_i at theta."""
        return self.H @ (theta - self.zbar)

    def loss(self, theta: np.ndarray) -> float:
        diff = theta[None, :] - self.Z
        return float(0.5 / self.Z.shape[0] * np.einsum("jp,pq,jq->", diff, self.H, diff))

    def forgetting(self, theta: np.ndarray, delta: np.ndarray) -> float:
        """L_i(theta+delta) - L_i(theta), closed form."""
        return self.loss(theta + delta) - self.loss(theta)


def make_task(
    rng: np.random.Generator,
    p: int,
    rank: int,
    N: int,
    center_scale: float = 1.0,
    cov_scale: float = 0.3,
) -> QuadraticTask:
    H = random_psd(rng, p, rank)
    c = rng.standard_normal(p) * center_scale
    B = rng.standard_normal((p, p)) * cov_scale
    S = B @ B.T / p + 1e-2 * np.eye(p)
    Z = rng.multivariate_normal(c, S, size=N)
    zbar = Z.mean(axis=0)
    sigma = float(np.sqrt(np.trace(H @ S @ H)))
    return QuadraticTask(H=H, zbar=zbar, S=S, sigma=sigma, Z=Z)


def make_random_instance(
    rng: np.random.Generator,
    n: int,
    p: int,
    rank: int,
    N: int,
    tau: np.ndarray | float,
    alpha: np.ndarray | None = None,
    r: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
    theta_scale: float = 1.0,
) -> tuple[BargainProblem, list[QuadraticTask], np.ndarray]:
    """n tasks (task 0 is the live task), a random theta, and the induced
    BargainProblem g_i(theta), H_i. tau is broadcast to all absent tasks;
    tau[0] is forced to 0 (live task, d_0 = 0)."""
    tasks = [make_task(rng, p, rank, N) for _ in range(n)]
    theta = rng.standard_normal(p) * theta_scale

    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), (n,)).copy()
    tau_arr[0] = 0.0

    g = np.stack([t.g(theta) for t in tasks])
    H = np.stack([t.H for t in tasks])
    problem = BargainProblem(g=g, H=H, tau=tau_arr, alpha=alpha, r=r, zeta=zeta)
    return problem, tasks, theta


def make_degenerate_instance(
    rng: np.random.Generator,
    n: int,
    p: int,
    rank: int,
    kappa: float,
    tau: np.ndarray | float,
    alpha: np.ndarray | None = None,
    r: np.ndarray | None = None,
    zeta: np.ndarray | None = None,
    g0_scale: float = 1.0,
) -> BargainProblem:
    """Synthetic instance for the degeneracy sweep (E0.4 / E1 C4): absent
    tasks (i>=1) have ||g_i|| = kappa exactly ("old tasks approaching their
    optimum"), while curvature H_i is unchanged -- the physical picture of
    SGD having converged on an old task while its local curvature stays put.
    H_i is generated exactly as in `make_task` (PSD, rank ~ rank); g_i is set
    directly rather than derived from a shared theta, since the point of this
    sweep is the effect of ||g_old|| on the solver/certificate, not sandbox
    realism.
    """
    H = np.stack([random_psd(rng, p, rank) for _ in range(n)])
    g = np.empty((n, p))
    g[0] = rng.standard_normal(p) * g0_scale
    for i in range(1, n):
        u = rng.standard_normal(p)
        u /= np.linalg.norm(u)
        g[i] = kappa * u

    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), (n,)).copy()
    tau_arr[0] = 0.0
    return BargainProblem(g=g, H=H, tau=tau_arr, alpha=alpha, r=r, zeta=zeta)
