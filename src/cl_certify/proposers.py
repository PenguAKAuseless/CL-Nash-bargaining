"""Continual-learning methods as proposers of a flat parameter update Delta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from cl_bargain.bargain.utilities import BargainProblem
from cl_bargain.streams.methods import (
    RunState,
    agem_step,
    bargain_step,
    er_step,
    ewc_step,
    sample_batch,
    stream_loss,
)
from cl_bargain.streams.paramvec import (
    add_flat_,
    flat_to_dict,
    flatten_grad,
    flatten_params,
    param_shapes,
)
from cl_certify.timing import section


@dataclass
class ProposalState:
    """Inputs visible to a proposer (excludes the audit and calibration slices)."""

    model: torch.nn.Module
    stream: object
    idx: torch.Tensor
    task_index: int
    buffer: object
    generator: torch.Generator
    minibatch_size: int
    device: torch.device
    step: int
    # per-step allowance from the runner; read only by cumulative-budget proposers
    step_budget: float | None = None


@dataclass
class Proposal:
    delta: torch.Tensor | None
    diagnostics: dict


class Proposer(Protocol):
    name: str

    def propose(self, state: ProposalState) -> Proposal: ...


def _capture(model, write_fn) -> torch.Tensor:
    """Return the Delta written by `write_fn`, with the model restored."""
    theta0 = flatten_params(model).clone()
    write_fn()
    theta1 = flatten_params(model)
    delta = (theta1 - theta0).clone()
    add_flat_(model, -delta)
    return delta


class _OptimiserProposer:
    def __init__(self, name: str, model, lr: float):
        self.name = name
        self.opt = torch.optim.SGD(model.parameters(), lr=lr)
        self.lr = lr

    def _capture_timed(self, model, write_fn) -> torch.Tensor:
        with section("gradient"):
            return _capture(model, write_fn)


class ReplayProposer(_OptimiserProposer):
    """Experience replay (ER): one SGD step on live batch + buffer batch."""

    def __init__(self, model, lr: float = 0.1):
        super().__init__("replay", model, lr)

    def propose(self, s: ProposalState) -> Proposal:
        delta = self._capture_timed(
            s.model,
            lambda: er_step(
                s.model, self.opt, s.stream, s.idx, s.buffer, s.generator,
                s.minibatch_size, s.device,
            ),
        )
        return Proposal(delta, {})


class AGEMProposer(_OptimiserProposer):
    def __init__(self, model, lr: float = 0.1):
        super().__init__("agem", model, lr)

    def propose(self, s: ProposalState) -> Proposal:
        delta = self._capture_timed(
            s.model,
            lambda: agem_step(
                s.model, self.opt, s.stream, s.idx, s.buffer, s.generator,
                s.minibatch_size, s.device,
            ),
        )
        return Proposal(delta, {})


class EWCProposer(_OptimiserProposer):
    def __init__(self, model, lam: float = 3.0, lr: float = 0.1):
        super().__init__("ewc", model, lr)
        self.lam = lam
        self.state = RunState()

    def propose(self, s: ProposalState) -> Proposal:
        delta = self._capture_timed(
            s.model,
            lambda: ewc_step(
                s.model, self.opt, s.stream, s.idx, s.buffer, s.generator,
                s.minibatch_size, s.device, self.state, lam=self.lam,
            ),
        )
        return Proposal(delta, {})


class DERPPProposer(_OptimiserProposer):
    """DER++ (Buzzega et al. 2020): replay plus matching of stored replay-slice logits."""

    def __init__(self, model, lr: float = 0.1, alpha: float = 0.5, beta: float = 0.5):
        super().__init__("derpp", model, lr)
        self.alpha = alpha
        self.beta = beta
        self.logits: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def store_logits(self, model, stream, stream_idx: torch.Tensor) -> None:
        """Called by the runner at each task boundary, on the replay slice."""
        if stream_idx.numel() == 0:
            return
        model.eval()
        for start in range(0, stream_idx.numel(), 256):
            sub = stream_idx[start : start + 256]
            out = model(stream.normalise(stream.x_train[sub])).detach()
            for j, key in enumerate(sub.tolist()):
                self.logits[key] = out[j].clone()
        model.train()

    def _write(self, s: ProposalState) -> None:
        x = s.stream.augment(s.stream.x_train[s.idx], s.generator)
        y = s.stream.y_train[s.idx]
        self.opt.zero_grad(set_to_none=True)
        loss = stream_loss(s.stream, s.model(x), y)

        pool = s.buffer.idx[: s.buffer.filled]
        if pool.numel() > 0 and self.logits:
            b1 = sample_batch(pool, s.minibatch_size, s.generator)
            keys = [k for k in b1.tolist() if k in self.logits]
            if keys:
                kt = torch.tensor(keys, device=b1.device, dtype=b1.dtype)
                z = torch.stack([self.logits[k] for k in keys])
                out1 = s.model(s.stream.augment(s.stream.x_train[kt], s.generator))
                loss = loss + self.alpha * ((out1 - z) ** 2).mean()
            b2 = sample_batch(pool, s.minibatch_size, s.generator)
            out2 = s.model(s.stream.augment(s.stream.x_train[b2], s.generator))
            loss = loss + self.beta * stream_loss(s.stream, out2, s.stream.y_train[b2])
        loss.backward()
        self.opt.step()

    def propose(self, s: ProposalState) -> Proposal:
        return Proposal(self._capture_timed(s.model, lambda: self._write(s)), {})


class ReplayGaussNewtonProposer(_OptimiserProposer):
    """Replay SGD step followed by a Gauss-Newton damping correction on a buffer batch."""

    def __init__(self, model, lr: float = 0.1, mu: float = 1.0):
        super().__init__("replay_ggn", model, lr)
        self.mu = mu
        self._shapes = param_shapes(model)
        self._names = list(self._shapes.keys())

    def _write(self, s: ProposalState) -> None:
        from cl_bargain.bargain.curvature import ggn_vector_product

        x = s.stream.augment(s.stream.x_train[s.idx], s.generator)
        y = s.stream.y_train[s.idx]
        pool = s.buffer.idx[: s.buffer.filled]

        with section("gradient"):
            self.opt.zero_grad(set_to_none=True)
            if pool.numel() > 0:
                b = sample_batch(pool, s.minibatch_size, s.generator)
                bx = s.stream.augment(s.stream.x_train[b], s.generator)
                out = s.model(torch.cat([x, bx]))
                loss = stream_loss(s.stream, out, torch.cat([y, s.stream.y_train[b]]))
            else:
                loss = stream_loss(s.stream, s.model(x), y)
            loss.backward()
            g = flatten_grad(s.model).clone()

        if pool.numel() > 0 and self.mu > 0:
            with section("curvature"):
                b = sample_batch(pool, s.minibatch_size, s.generator)
                bx = s.stream.normalise(s.stream.x_train[b])
                step = -self.lr * g
                v_dict = flat_to_dict(step, self._names, self._shapes)
                hv = ggn_vector_product(
                    s.model, bx, 0, s.stream.n_classes, v_dict, chunk_size=64,
                    weighted=True,
                )
                hv_flat = torch.cat([hv[k].reshape(-1) for k in self._names])
                # (I + mu H)^{-1} step, one Neumann term
                corrected = step - self.mu * hv_flat
            add_flat_(s.model, corrected)
        else:
            add_flat_(s.model, -self.lr * g)

    def propose(self, s: ProposalState) -> Proposal:
        return Proposal(_capture(s.model, lambda: self._write(s)), {})


class BargainProposer:
    """Nash bargaining proposer: "v1" first-order, "v2" Gauss-Newton curvature, "nashmtl"."""

    def __init__(self, method: str, knob: float, zeta_floor: float = 0.01,
                 chunk_size: int = 64, max_active_players: int | None = None,
                 jvp_batched: bool = False, every_k: int = 1,
                 trust_radius: float | None = None):
        self.name = {"v1": "bargain_first_order", "v2": "bargain_curvature"}.get(method, method)
        self.method = method
        self.knob = knob
        self.zeta_floor = zeta_floor if method == "v2" else 0.0
        self.chunk_size = chunk_size
        self.max_active_players = max_active_players
        self.jvp_batched = jvp_batched
        # bargain every k-th step, plain replay otherwise
        self.every_k = every_k
        # step radius, separate from the damage budget `knob`
        self.trust_radius = trust_radius
        self._fallback: ReplayProposer | None = None

    def attach_fallback(self, model, lr: float) -> None:
        self._fallback = ReplayProposer(model, lr)

    def propose(self, s: ProposalState) -> Proposal:
        if self.every_k > 1 and s.step % self.every_k != 0 and self._fallback is not None:
            p = self._fallback.propose(s)
            p.diagnostics["bargain_ran"] = False
            return p

        captured: dict = {}

        def write():
            applied, diag = bargain_step(
                s.model, s.stream, s.idx, s.task_index, s.buffer, s.generator,
                s.minibatch_size, s.device, self.method, self.knob, self.chunk_size,
                zeta_floor=self.zeta_floor,
                max_active_players=self.max_active_players,
                jvp_batched=self.jvp_batched,
                trust_radius=self.trust_radius,
            )
            captured["applied"] = applied
            captured["diag"] = diag

        delta = _capture(s.model, write)
        if not captured["applied"]:
            return Proposal(None, {"bargain_ran": True, "solver_failed": True})
        diag = captured["diag"] or {}
        return Proposal(
            delta,
            {
                "bargain_ran": True,
                "solver_failed": False,
                "n_active": len(diag.get("active", [])),
                "slack_min": float(np.min(diag["slacks"])) if "slacks" in diag else None,
                "problem": diag.get("problem"),
                "beta_reduced": diag.get("beta"),
                "c": diag.get("c"),
                "active": diag.get("active"),
                "radius_binding": diag.get("radius_binding"),
                "gamma_norm_presolve": diag.get("gamma_norm_presolve"),
            },
        )


class ConstrainedQPProposer:
    """Live-task quadratic model minimised s.t. per-absent-task damage <= tau (SLSQP)."""

    name = "constrained_qp"

    def __init__(self, knob: float, chunk_size: int = 64,
                 max_active_players: int | None = None, jvp_batched: bool = False,
                 trust_radius: float | None = None, cumulative_budget: bool = False):
        self.knob = knob
        self.chunk_size = chunk_size
        self.max_active_players = max_active_players
        self.jvp_batched = jvp_batched
        # trust region keeps the problem bounded (H_0 is only PSD)
        self.trust_radius = trust_radius if trust_radius is not None else knob
        # use the runner's tau / steps_remaining as the per-step allowance
        self.cumulative_budget = cumulative_budget

    def propose(self, s: ProposalState) -> Proposal:
        from scipy.optimize import NonlinearConstraint, minimize

        from cl_certify.reduce import reduced_problem

        knob = self.knob
        if self.cumulative_budget and s.step_budget is not None:
            knob = s.step_budget
        red = reduced_problem(
            s, knob, self.chunk_size, self.max_active_players, self.jvp_batched,
            with_curvature=True,
        )
        if red is None:
            return Proposal(None, {"solver_failed": True, "reason": "no_active_players"})
        if red.get("degenerate"):
            return Proposal(red["delta_fallback"], {"bargain_ran": False, "degenerate": True})

        problem: BargainProblem = red["problem"]
        g, h, tau = problem.g, problem.H, problem.tau
        k = g.shape[1]

        def obj(z):
            return float(g[0] @ z + 0.5 * z @ h[0] @ z)

        def obj_grad(z):
            return g[0] + h[0] @ z

        cons = []
        for i in range(1, problem.n):
            cons.append(
                NonlinearConstraint(
                    (lambda z, i=i: float(tau[i] - g[i] @ z - 0.5 * z @ h[i] @ z)),
                    0.0, np.inf,
                    jac=(lambda z, i=i: -(g[i] + h[i] @ z)),
                )
            )
        cons.append(
            NonlinearConstraint(
                lambda z: float(self.trust_radius**2 - z @ z), 0.0, np.inf,
                jac=lambda z: -2.0 * z,
            )
        )

        with section("solver"):
            res = minimize(
                obj, np.zeros(k), jac=obj_grad, constraints=cons, method="SLSQP",
                options={"maxiter": 200, "ftol": 1e-10},
            )
        if not res.success or not np.all(np.isfinite(res.x)):
            return Proposal(None, {"solver_failed": True, "reason": res.message})

        delta = red["to_full"](res.x)
        return Proposal(
            delta,
            {
                "solver_failed": False,
                "n_active": problem.n - 1,
                "objective": float(res.fun),
                "step_tau": float(knob),
                "radius_binding": bool(
                    abs(float(np.linalg.norm(res.x)) - self.trust_radius)
                    < 1e-6 * max(1.0, self.trust_radius)
                ),
                "problem": problem,
                "beta_reduced": res.x,
                "c": red["c"],
                "active": red["active"],
            },
        )


__all__ = [
    "AGEMProposer",
    "BargainProposer",
    "ConstrainedQPProposer",
    "DERPPProposer",
    "EWCProposer",
    "Proposal",
    "ProposalState",
    "Proposer",
    "ReplayGaussNewtonProposer",
    "ReplayProposer",
]
