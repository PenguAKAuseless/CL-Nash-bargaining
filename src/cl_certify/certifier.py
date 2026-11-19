"""Paired empirical-Bernstein certification of forgetting on held-out audit slices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cl_certify.bernstein import BernsteinBound, empirical_bernstein_upper


@dataclass
class CheckpointRecord:
    task: int
    checkpoint: int
    step: int
    mode: str
    source: str
    n_audit: int
    bound: float
    mean: float
    var: float
    observed_min: float
    observed_max: float
    clipped_fraction: float
    width_sqrt: float
    width_residual: float
    budget: float | None = None
    violated_budget: bool | None = None


@dataclass
class PairedBernsteinCertifier:
    """Per-task and pooled certification of forgetting on the audit slice."""

    delta: float
    loss_max: float
    n_checkpoints: int
    n_tasks_max: int
    source: str = "audit"
    anchors: dict[int, np.ndarray] = field(default_factory=dict)
    records: list[CheckpointRecord] = field(default_factory=list)

    def set_anchor(self, task: int, per_example_loss: np.ndarray) -> None:
        """Record anchor losses l_j(theta_0) on the audit slice of `task`."""
        self.anchors[task] = np.clip(
            np.asarray(per_example_loss, dtype=float).ravel(), 0.0, self.loss_max
        )

    def has_anchor(self, task: int) -> bool:
        return task in self.anchors

    def _paired(self, task: int, per_example_loss: np.ndarray) -> np.ndarray:
        anchor = self.anchors[task]
        now = np.asarray(per_example_loss, dtype=float).ravel()
        if now.shape != anchor.shape:
            raise ValueError(
                f"task {task}: audit slice changed size between anchor and "
                f"checkpoint ({anchor.size} -> {now.size}). The audit slice must "
                "be frozen at the anchor, otherwise the pairing is meaningless."
            )
        return np.clip(now, 0.0, self.loss_max) - anchor

    def certify_task(
        self,
        task: int,
        per_example_loss: np.ndarray,
        checkpoint: int,
        step: int,
        budget: float | None = None,
    ) -> CheckpointRecord:
        """Certify one task at level delta / n_tasks_max."""
        d = self._paired(task, per_example_loss)
        delta_task = self.delta / self.n_tasks_max
        b = empirical_bernstein_upper(
            d, delta_task, -self.loss_max, self.loss_max, self.n_checkpoints
        )
        rec = self._record(task, checkpoint, step, "per_task", d, b, budget)
        self.records.append(rec)
        return rec

    def certify_pooled_examples(
        self,
        per_task_losses: dict[int, np.ndarray],
        checkpoint: int,
        step: int,
        budget: float | None = None,
    ) -> CheckpointRecord:
        """Certify mean loss drift over all audit examples of the absent tasks."""
        tasks = sorted(per_task_losses)
        parts = [self._paired(t, per_task_losses[t]) for t in tasks]
        # equal sizes make the pooled mean the unweighted per-task average
        sizes = {int(a.size) for a in parts}
        if len(sizes) > 1:
            raise ValueError(
                f"pooled-by-example needs equal audit sizes per task, got {sorted(sizes)}; "
                "use certify_average_of_per_task for unequal slices"
            )
        pooled = np.concatenate(parts) if parts else np.zeros(0)
        b = empirical_bernstein_upper(
            pooled, self.delta, -self.loss_max, self.loss_max, self.n_checkpoints
        )
        rec = self._record(-1, checkpoint, step, "pooled_examples", pooled, b, budget)
        self.records.append(rec)
        return rec

    def certify_pooled(
        self,
        per_task_losses: dict[int, np.ndarray],
        checkpoint: int,
        step: int,
        budget: float | None = None,
    ) -> CheckpointRecord:
        """Certify the unweighted mean drift, using per-task mean differences as the sample."""
        tasks = sorted(per_task_losses)
        per_task_mean = np.array(
            [self._paired(t, per_task_losses[t]).mean() for t in tasks], dtype=float
        )
        b = empirical_bernstein_upper(
            per_task_mean, self.delta, -self.loss_max, self.loss_max, self.n_checkpoints
        )
        rec = self._record(-1, checkpoint, step, "pooled", per_task_mean, b, budget)
        rec.n_audit = int(sum(len(v) for v in per_task_losses.values()))
        self.records.append(rec)
        return rec

    def certify_average_of_per_task(
        self,
        per_task_losses: dict[int, np.ndarray],
        checkpoint: int,
        step: int,
        budget: float | None = None,
        method: str = "bernstein",
    ) -> CheckpointRecord:
        """Average of per-task upper bounds, each at level delta / T; valid for unequal slices."""
        from cl_certify.betting import betting_upper

        tasks = sorted(per_task_losses)
        if not tasks:
            empty = np.zeros(0)
            b = empirical_bernstein_upper(empty, self.delta, -self.loss_max,
                                          self.loss_max, self.n_checkpoints)
            rec = self._record(-1, checkpoint, step, f"avg_per_task_{method}",
                               empty, b, budget)
            self.records.append(rec)
            return rec

        delta_i = self.delta / len(tasks)
        bounds, means, sizes = [], [], []
        for t in tasks:
            d = self._paired(t, per_task_losses[t])
            if method == "betting":
                bi = betting_upper(d, delta_i, -self.loss_max, self.loss_max,
                                   n_checkpoints=self.n_checkpoints)
            else:
                bi = empirical_bernstein_upper(d, delta_i, -self.loss_max,
                                               self.loss_max, self.n_checkpoints)
            bounds.append(float(bi.bound))
            means.append(float(d.mean()))
            sizes.append(int(d.size))

        rec = CheckpointRecord(
            task=-1, source=self.source, checkpoint=checkpoint, step=step,
            mode=f"avg_per_task_{method}", n_audit=int(sum(sizes)),
            bound=float(np.mean(bounds)), mean=float(np.mean(means)),
            var=float(np.var(means, ddof=1)) if len(means) > 1 else 0.0,
            observed_min=float(np.min(bounds)), observed_max=float(np.max(bounds)),
            clipped_fraction=0.0,
            width_sqrt=float(np.mean(bounds) - np.mean(means)), width_residual=0.0,
            budget=budget,
            violated_budget=(None if budget is None else bool(np.mean(bounds) > budget)),
        )
        self.records.append(rec)
        return rec

    def _record(self, task, checkpoint, step, mode, d, b: BernsteinBound, budget):
        clipped = float(np.mean(np.abs(d) > self.loss_max)) if d.size else 0.0
        return CheckpointRecord(
            task=task,
            source=self.source,
            checkpoint=checkpoint,
            step=step,
            mode=mode,
            n_audit=int(d.size),
            bound=float(b.bound),
            mean=float(b.mean),
            var=float(b.var),
            observed_min=float(d.min()) if d.size else float("nan"),
            observed_max=float(d.max()) if d.size else float("nan"),
            clipped_fraction=clipped,
            width_sqrt=float(b.width_sqrt),
            width_residual=float(b.width_residual),
            budget=budget,
            violated_budget=(None if budget is None else bool(b.bound > budget)),
        )

    def certify_betting(
        self,
        per_task_losses: dict[int, np.ndarray],
        checkpoint: int,
        step: int,
        budget: float | None = None,
        task: int | None = None,
    ) -> CheckpointRecord:
        """Certify the pooled (or one task's) drift with a betting confidence bound."""
        from cl_certify.betting import betting_upper

        if task is not None:
            sample = self._paired(task, per_task_losses[task])
            delta = self.delta / self.n_tasks_max
            mode = "betting_per_task"
        else:
            tasks = sorted(per_task_losses)
            sample = np.concatenate(
                [self._paired(t, per_task_losses[t]) for t in tasks]
            ) if tasks else np.zeros(0)
            delta = self.delta
            mode = "betting_pooled"

        b = betting_upper(sample, delta, -self.loss_max, self.loss_max,
                          n_checkpoints=self.n_checkpoints)
        clipped = float(np.mean(np.abs(sample) > self.loss_max)) if sample.size else 0.0
        rec = CheckpointRecord(
            task=-1 if task is None else task,
            source=self.source,
            checkpoint=checkpoint,
            step=step,
            mode=mode,
            n_audit=int(sample.size),
            bound=float(b.bound),
            mean=float(b.mean),
            var=float(sample.var(ddof=1)) if sample.size > 1 else 0.0,
            observed_min=float(sample.min()) if sample.size else float("nan"),
            observed_max=float(sample.max()) if sample.size else float("nan"),
            clipped_fraction=clipped,
            width_sqrt=float("nan"),
            width_residual=float("nan"),
            budget=budget,
            violated_budget=(None if budget is None else bool(b.bound > budget)),
        )
        self.records.append(rec)
        return rec

    def as_rows(self) -> list[dict]:
        return [vars(r) for r in self.records]


__all__ = ["CheckpointRecord", "PairedBernsteinCertifier"]
