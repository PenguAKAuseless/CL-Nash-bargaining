"""Reduced bargaining instance shared by the bargain and non-bargain proposers."""

from __future__ import annotations

import numpy as np
import torch

from cl_bargain.bargain.utilities import BargainProblem
from cl_bargain.streams.methods import (
    _flat_gram_and_curvature,
    sample_batch,
    stream_loss,
)
from cl_bargain.streams.paramvec import flatten_grad
from cl_certify.timing import section


def reduced_problem(
    state,
    knob: float,
    chunk_size: int,
    max_active_players: int | None,
    jvp_batched: bool,
    with_curvature: bool,
    zeta_floor: float = 0.0,
):
    """Build the reduced BargainProblem and a `to_full` map back to parameter space."""
    s = state
    active = sorted(s.buffer.task_sizes.keys()) if len(s.buffer) > 0 else []
    active = [t for t in active if t != s.task_index]
    if max_active_players is not None and len(active) > max_active_players:
        perm = torch.randperm(len(active), device=s.device, generator=s.generator).tolist()
        active = sorted(active[i] for i in perm[:max_active_players])

    with section("gradient"):
        x0 = s.stream.augment(s.stream.x_train[s.idx], s.generator)
        y0 = s.stream.y_train[s.idx]
        s.model.zero_grad(set_to_none=True)
        stream_loss(s.stream, s.model(x0), y0).backward()
        g_list = [flatten_grad(s.model).clone()]
        x_list = [s.stream.normalise(s.stream.x_train[s.idx])]

        for t in active:
            idx_i = sample_batch(s.buffer.held(t), s.minibatch_size, s.generator)
            xi = s.stream.augment(s.stream.x_train[idx_i], s.generator)
            s.model.zero_grad(set_to_none=True)
            stream_loss(s.stream, s.model(xi), s.stream.y_train[idx_i]).backward()
            g_list.append(flatten_grad(s.model).clone())
            x_list.append(s.stream.normalise(s.stream.x_train[idx_i]))
        s.model.zero_grad(set_to_none=True)

    n = len(g_list)
    if n == 1:
        g0 = g_list[0]
        return {
            "degenerate": True,
            "delta_fallback": -knob * g0 / (g0.norm() + 1e-12),
            "active": [],
        }

    # per-player rescale c_i = 1/||g_i||, applied jointly to (g_i, H_i, tau_i, zeta_i)
    norms = torch.stack([g.norm() for g in g_list])
    c_np = (1.0 / norms.clamp_min(1e-12)).detach().cpu().numpy().astype(np.float64)
    g_list = [g / norms[i].clamp_min(1e-12) for i, g in enumerate(g_list)]

    with section("curvature"):
        k_mat, m_stack = _flat_gram_and_curvature(
            s.model, s.stream, g_list, x_list, s.stream.n_classes, chunk_size,
            need_curvature=with_curvature, jvp_batched=jvp_batched,
            class_masks=(
                None
                if getattr(s.stream, "class_mask", None) is None
                else [
                    s.stream.class_mask[s.stream.task_range(t)[0]]
                    for t in [s.task_index, *active]
                ]
            ),
        )
    if m_stack is not None:
        m_stack = m_stack * c_np[:, None, None]

    tau_original = np.zeros(n)
    tau_original[1:] = knob
    tau = tau_original * c_np
    alpha = np.full(n, 1.0 / n)

    lam, eigvec = np.linalg.eigh(k_mat)
    keep = lam > 1e-10 * lam.max()
    if not keep.any():
        g0 = g_list[0]
        return {
            "degenerate": True,
            "delta_fallback": -knob * g0 / (g0.norm() + 1e-12),
            "active": active,
        }
    to_beta = eigvec[:, keep] / np.sqrt(lam[keep])
    g_red = eigvec[:, keep] * np.sqrt(lam[keep])
    k_dim = int(keep.sum())

    if with_curvature:
        m_red = np.einsum("ab,jbc,cd->jad", to_beta.T, m_stack, to_beta)
        m_red = 0.5 * (m_red + np.transpose(m_red, (0, 2, 1)))
    else:
        m_red = np.zeros((n, k_dim, k_dim))
    zeta = np.full(n, zeta_floor) * c_np if zeta_floor else None
    problem = BargainProblem(g=g_red, H=m_red, tau=tau, alpha=alpha, zeta=zeta)

    stacked = torch.stack(g_list)

    def to_full(gamma: np.ndarray) -> torch.Tensor:
        beta = to_beta @ np.asarray(gamma, dtype=np.float64)
        beta_t = torch.as_tensor(beta, dtype=stacked.dtype, device=stacked.device)
        return (beta_t @ stacked).detach()

    return {
        "degenerate": False,
        "problem": problem,
        "to_full": to_full,
        "to_beta": to_beta,
        "c": c_np,
        "active": active,
        "k_dim": k_dim,
        "g_list": g_list,
        "x_list": x_list,
    }


__all__ = ["reduced_problem"]
