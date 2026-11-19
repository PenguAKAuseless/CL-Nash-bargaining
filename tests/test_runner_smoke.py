"""End-to-end wiring tests of propose/admit/certify on a synthetic 32x32 stream."""

from __future__ import annotations

import pytest
import torch

from cl_certify.config import Config
from cl_certify.runner import run

DEV = torch.device("cpu")


class _SynthStream:
    """4 tasks x 3 classes, 32x32, task-masked; same shape contract as SplitCIFAR100."""

    def __init__(self, n_tasks=4, per_class=40, width=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.splits = (width,) * n_tasks
        self.n_classes = n_tasks * width
        n = self.n_classes * per_class
        self.y_train = torch.arange(self.n_classes).repeat_interleave(per_class)
        # class-conditional means, so the task is learnable
        means = torch.randn(self.n_classes, 3, 1, 1, generator=g)
        self.x_train = (
            means[self.y_train] + 0.5 * torch.randn(n, 3, 32, 32, generator=g)
        ).clamp(-3, 3)
        self.y_test = self.y_train.clone()
        self.x_test = self.x_train.clone()

        owner = torch.empty(self.n_classes, dtype=torch.long)
        self.tasks = []
        start = 0
        for i, w in enumerate(self.splits):
            lo, hi = start, start + w
            owner[lo:hi] = i
            self.tasks.append(
                _Task(i, tuple(range(lo, hi)),
                      torch.nonzero((self.y_train >= lo) & (self.y_train < hi))[:, 0],
                      torch.nonzero((self.y_test >= lo) & (self.y_test < hi))[:, 0])
            )
            start = hi
        self.class_mask = owner[:, None] == owner[None, :]

    def task_range(self, t):
        lo = sum(self.splits[:t])
        return lo, lo + self.splits[t]

    def normalise(self, x):
        return x

    def augment(self, x, generator):
        del generator
        return x


class _Task:
    def __init__(self, index, classes, train_idx, test_idx):
        self.index, self.classes = index, classes
        self.train_idx, self.test_idx = train_idx, test_idx

    @property
    def n_examples(self):
        return int(self.train_idx.numel())


def _cfg(**kw):
    base = {
        "epochs": 1, "minibatch_size": 16, "buffer_capacity": 180,
        "backbone_width": 4, "tau": 0.5, "chunk_size": 16, "loss_max": 10.0,
    }
    base.update(kw)
    return Config(**base)


@pytest.mark.parametrize(
    "name", ["replay", "agem", "derpp", "constrained_qp", "bargain_curvature"]
)
def test_run_completes_and_certifies(name):
    stream = _SynthStream()
    res = run(_cfg(proposer=name), DEV, stream=stream, log_steps=True)
    m = res.metrics
    assert 0.0 <= m["A_final"] <= 1.0
    assert m["total_steps"] > 0
    assert m["n_checkpoints_used"] <= m["n_checkpoints_planned"], (
        "more checkpoints were taken than the union bound was sized for"
    )
    assert m["cert_n_scored"] > 0, "nothing was certified"
    assert 0.0 <= m["step_accept_rate"] <= 1.0
    assert res.slice_sizes


def test_audit_slice_is_frozen_across_checkpoints():
    stream = _SynthStream()
    res = run(_cfg(proposer="replay"), DEV, stream=stream)
    per_task = {}
    for r in res.certificate_rows:
        if r["mode"] == "per_task" and r["source"] == "audit" and r["task"] >= 0:
            per_task.setdefault(r["task"], set()).add(r["n_audit"])
    assert per_task
    for task, sizes in per_task.items():
        assert len(sizes) == 1, f"task {task} audit slice changed size: {sizes}"


def test_disabling_admission_raises_acceptance_to_one():
    stream = _SynthStream()
    on = run(_cfg(proposer="replay", admission=True, tau=1e-6), DEV, stream=stream)
    off = run(_cfg(proposer="replay", admission=False, tau=1e-6), DEV, stream=stream)
    assert off.metrics["step_accept_rate"] == 1.0
    assert on.metrics["step_accept_rate"] <= off.metrics["step_accept_rate"]


def test_tighter_budget_reduces_forgetting():
    stream = _SynthStream()
    loose = run(_cfg(proposer="replay", tau=10.0), DEV, stream=stream)
    tight = run(_cfg(proposer="replay", tau=1e-4), DEV, stream=stream)
    assert tight.metrics["forgetting"] <= loose.metrics["forgetting"] + 1e-9


@pytest.mark.parametrize("source", ["heldout_train", "buffer"])
def test_audit_source_switch(source):
    stream = _SynthStream()
    res = run(_cfg(proposer="replay", audit_source=source, audit_heldout_n=30),
              DEV, stream=stream)
    assert res.metrics["cert_n_scored"] > 0
    if source == "heldout_train":
        assert res.metrics["cert_min_audit_m"] == 30
    else:
        assert res.metrics["cert_min_audit_m"] < 30


def test_heldout_audit_is_removed_from_the_training_stream():
    """Withholding k examples per task shortens each epoch by ceil(n/b) - ceil((n-k)/b) steps."""
    stream = _SynthStream()
    k = 20
    base = run(_cfg(proposer="replay", audit_source="buffer"), DEV, stream=stream)
    held = run(_cfg(proposer="replay", audit_source="heldout_train", audit_heldout_n=k),
               DEV, stream=stream)

    n = stream.tasks[0].n_examples
    b = 16
    per_epoch_full = -(-n // b)
    per_epoch_held = -(-(n - k) // b)
    n_tasks, epochs = len(stream.tasks), 1
    assert base.metrics["total_steps"] == per_epoch_full * n_tasks * epochs
    assert held.metrics["total_steps"] == per_epoch_held * n_tasks * epochs
    assert held.metrics["total_steps"] < base.metrics["total_steps"]


def test_heldout_audit_slice_is_identical_across_proposers_and_taus():
    """The held-out audit pool depends only on split_seed."""
    stream = _SynthStream()
    a = run(_cfg(proposer="replay", tau=0.05, audit_heldout_n=25), DEV, stream=stream)
    b = run(_cfg(proposer="agem", tau=5.0, audit_heldout_n=25), DEV, stream=stream)
    anchors_a = {(r["task"], r["n_audit"]) for r in a.certificate_rows
                 if r["source"] == "audit" and r["mode"] == "per_task"}
    anchors_b = {(r["task"], r["n_audit"]) for r in b.certificate_rows
                 if r["source"] == "audit" and r["mode"] == "per_task"}
    assert anchors_a == anchors_b
    assert all(m == 25 for _, m in anchors_a)


def test_replay_capacity_pins_the_replay_slice_not_the_total():
    """replay_capacity sizes the replay slice; the total buffer is larger."""
    from cl_certify.splits import ThreeWayBuffer

    cfg = _cfg(proposer="replay", replay_capacity=2000)
    cap = round(cfg.replay_capacity / cfg.split_fractions[0])
    buf = ThreeWayBuffer(cap, DEV, cfg.split_seed, cfg.split_fractions)
    assert buf.capacities["replay"] == pytest.approx(2000, abs=2)
    assert sum(buf.capacities.values()) > 2000


def test_admission_never_reads_the_audit_slice():
    """No per-example audit accessor is called during admission."""
    import numpy as np

    from cl_certify import runner as runner_mod
    from cl_certify.admission import BacktrackingAdmission

    reads: list[str] = []
    in_admission = {"flag": False}

    real_01 = runner_mod._per_example_01
    real_loss = runner_mod._per_example_loss
    real_admit = BacktrackingAdmission.admit

    def watched_01(*a, **k):
        if in_admission["flag"]:
            reads.append("_per_example_01")
        return real_01(*a, **k)

    def watched_loss(*a, **k):
        if in_admission["flag"]:
            reads.append("_per_example_loss")
        return real_loss(*a, **k)

    def watched_admit(self, *a, **k):
        in_admission["flag"] = True
        try:
            return real_admit(self, *a, **k)
        finally:
            in_admission["flag"] = False

    runner_mod._per_example_01 = watched_01
    runner_mod._per_example_loss = watched_loss
    BacktrackingAdmission.admit = watched_admit
    try:
        res = runner_mod.run(
            _cfg(proposer="replay", admission=True, tau=0.05, admission_budget=0.2),
            DEV, stream=_SynthStream(),
        )
    finally:
        runner_mod._per_example_01 = real_01
        runner_mod._per_example_loss = real_loss
        BacktrackingAdmission.admit = real_admit

    assert not reads, f"admission read per-example audit values: {sorted(set(reads))}"
    assert res.admission["n_steps"] > 0
    assert np.isfinite(res.metrics["A_final"])
