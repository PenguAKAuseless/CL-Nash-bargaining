"""Admission control: backtracking line search on measured calibration-slice damage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from cl_bargain.streams.paramvec import add_flat_


@dataclass
class AdmissionOutcome:
    beta: float
    n_backtracks: int
    applied: bool
    worst_damage: float
    budget: float
    n_evaluated: int
    task: int = -1


@dataclass
class BacktrackingAdmission:
    """Halve beta until worst per-task calibration damage of beta * Delta is within budget."""

    budget_per_step: float
    max_backtrack: int = 6
    enabled: bool = True
    schedule: str = "constant"
    mode: str = "cumulative"
    history: list[AdmissionOutcome] = field(default_factory=list)

    def admit(
        self,
        model,
        delta: torch.Tensor,
        active_tasks: list[int],
        loss_fn,
        budget_override: float | None = None,
        reference: np.ndarray | None = None,
        task: int = -1,
    ) -> AdmissionOutcome:
        """Apply an admissible beta * delta in place; `reference` is the per-task anchor loss."""
        budget = self.budget_per_step if budget_override is None else budget_override

        if not self.enabled or delta is None:
            if delta is not None:
                add_flat_(model, delta)
            out = AdmissionOutcome(1.0, 0, delta is not None, float("nan"), budget, 0, task)
            self.history.append(out)
            return out

        if not active_tasks:
            add_flat_(model, delta)
            out = AdmissionOutcome(1.0, 0, True, 0.0, budget, 0, task)
            self.history.append(out)
            return out

        if reference is not None:
            before = np.asarray(reference, dtype=float)
            n_eval = 0
        else:
            before = np.asarray(loss_fn(active_tasks), dtype=float)
            n_eval = 1

        beta = 1.0
        applied_shift = torch.zeros_like(delta)
        for k in range(self.max_backtrack + 1):
            target = beta * delta
            add_flat_(model, target - applied_shift)
            applied_shift = target
            after = np.asarray(loss_fn(active_tasks), dtype=float)
            n_eval += 1
            worst = float(np.max(after - before))
            if worst <= budget:
                out = AdmissionOutcome(beta, k, True, worst, budget, n_eval, task)
                self.history.append(out)
                return out
            beta /= 2.0

        # no admissible beta: undo and skip the step
        add_flat_(model, -applied_shift)
        out = AdmissionOutcome(0.0, self.max_backtrack + 1, False, worst, budget, n_eval, task)
        self.history.append(out)
        return out

    def summary(self) -> dict:
        if not self.history:
            return {}
        betas = np.array([h.beta for h in self.history])
        applied = np.array([h.applied for h in self.history])
        bt = np.array([h.n_backtracks for h in self.history])
        return {
            "n_steps": len(self.history),
            "accept_rate": float(applied.mean()),
            "mean_beta": float(betas.mean()),
            "mean_beta_accepted": float(betas[applied].mean()) if applied.any() else 0.0,
            "frac_beta_eq_1": float((betas == 1.0).mean()),
            "mean_backtracks": float(bt.mean()),
            "n_skipped": int((~applied).sum()),
            "calibration_passes": int(sum(h.n_evaluated for h in self.history)),
            "by_task": {
                int(t): {
                    "n_steps": len(g),
                    "accept_rate": float(np.mean([h.applied for h in g])),
                    "mean_beta": float(np.mean([h.beta for h in g])),
                    "mean_worst_damage": float(
                        np.mean([h.worst_damage for h in g if np.isfinite(h.worst_damage)])
                    ) if any(np.isfinite(h.worst_damage) for h in g) else float("nan"),
                    "budget": float(g[0].budget),
                }
                for t, g in _group_by_task(self.history).items()
            },
        }


def _group_by_task(history: list[AdmissionOutcome]) -> dict[int, list[AdmissionOutcome]]:
    out: dict[int, list[AdmissionOutcome]] = {}
    for h in history:
        out.setdefault(h.task, []).append(h)
    return out


def ramp_budget(total_budget: float, live_task: int, oldest_absent: int,
                n_tasks: int) -> float:
    """B * (t - t_i) / (T - 1 - t_i), keyed on the oldest absent task."""
    span = max(1, n_tasks - 1 - oldest_absent)
    return total_budget * (live_task - oldest_absent) / span


def budget_per_step(total_budget: float, steps_remaining: int, schedule: str) -> float:
    """Per-step share of the total budget: B / steps_remaining."""
    if steps_remaining <= 0:
        return total_budget
    return total_budget / steps_remaining if schedule in ("constant", "horizon_amortised") \
        else total_budget


__all__ = ["AdmissionOutcome", "BacktrackingAdmission", "budget_per_step", "ramp_budget"]
