"""reduced_problem builds the same instance that bargain_step solves."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cl_bargain.streams.methods import bargain_step
from cl_certify.proposers import ProposalState
from cl_certify.reduce import reduced_problem
from cl_certify.splits import ThreeWayBuffer

DEV = torch.device("cpu")


class _TinyStream:
    """Minimal stream with identity augmentation, so both paths see identical batches."""

    n_classes = 6

    def __init__(self, n=120, seed=0, masked=False):
        g = torch.Generator().manual_seed(seed)
        self.x_train = torch.rand(n, 3, 8, 8, generator=g)
        self.y_train = torch.randint(0, self.n_classes, (n,), generator=g)
        self.x_test = self.x_train
        self.y_test = self.y_train
        if masked:
            owner = torch.tensor([i // 2 for i in range(self.n_classes)])
            self.class_mask = owner[:, None] == owner[None, :]
        else:
            self.class_mask = None

    def normalise(self, x):
        return x

    def augment(self, x, generator):
        del generator
        return x

    def task_range(self, t):
        return 2 * t, 2 * t + 2


class _TinyNet(torch.nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 12),
            torch.nn.Tanh(), torch.nn.Linear(12, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _fixture(seed=0):
    torch.manual_seed(seed)
    stream = _TinyStream(seed=seed)
    model = _TinyNet(stream.n_classes)
    buf = ThreeWayBuffer(60, DEV, split_seed=5)
    buf.add(torch.arange(0, 40), 0)
    buf.add(torch.arange(40, 80), 1)
    gen = torch.Generator(device=DEV)
    return stream, model, buf, gen


def _state(stream, model, buf, gen, seed):
    gen.manual_seed(seed)
    return ProposalState(
        model=model, stream=stream, idx=torch.arange(80, 96), task_index=2,
        buffer=buf, generator=gen, minibatch_size=8, device=DEV, step=1,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_reduced_problem_matches_bargain_step(seed):
    stream, model, buf, gen = _fixture(seed)
    knob, chunk = 0.05, 32

    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    gen.manual_seed(99)
    applied, diag = bargain_step(
        model, stream, torch.arange(80, 96), 2, buf, gen, 8, DEV, "v2", knob, chunk,
        zeta_floor=0.01,
    )
    assert applied
    assert diag is not None
    # restore so the second path sees the same parameters
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta0[off : off + n].reshape(p.shape))
            off += n

    red = reduced_problem(
        _state(stream, model, buf, gen, 99), knob, chunk, None, False,
        with_curvature=True, zeta_floor=0.01,
    )
    assert not red["degenerate"]

    ref, got = diag["problem"], red["problem"]
    assert got.n == ref.n
    assert np.allclose(got.g, ref.g, atol=1e-6)
    assert np.allclose(got.H, ref.H, atol=1e-5)
    assert np.allclose(got.tau, ref.tau, atol=1e-10)
    assert np.allclose(got.zeta, ref.zeta, atol=1e-10)
    assert np.allclose(red["c"], diag["c"], atol=1e-8)
    assert red["active"] == diag["active"]


def test_to_full_reproduces_the_applied_update():
    stream, model, buf, gen = _fixture(0)
    gen.manual_seed(7)
    diag_delta = None
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    applied, diag = bargain_step(
        model, stream, torch.arange(80, 96), 2, buf, gen, 8, DEV, "v2", 0.05, 32,
        zeta_floor=0.01,
    )
    assert applied
    diag_delta = diag["delta_flat"]
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta0[off : off + n].reshape(p.shape))
            off += n

    red = reduced_problem(
        _state(stream, model, buf, gen, 7), 0.05, 32, None, False,
        with_curvature=True, zeta_floor=0.01,
    )
    rebuilt = red["to_full"](diag["beta"])
    assert torch.allclose(rebuilt, diag_delta, atol=1e-5)


def test_no_absent_players_is_flagged_degenerate():
    stream, model, _, gen = _fixture(0)
    empty = ThreeWayBuffer(60, DEV, split_seed=5)
    red = reduced_problem(
        _state(stream, model, empty, gen, 0), 0.05, 32, None, False, with_curvature=True
    )
    assert red["degenerate"]
    assert red["delta_fallback"].shape[0] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("seed", [0, 1])
def test_reduced_problem_matches_bargain_step_on_a_task_masked_stream(seed):
    """Equivalence holds with task-masked heads on the batched curvature path."""
    torch.manual_seed(seed)
    stream = _TinyStream(seed=seed, masked=True)
    model = _TinyNet(stream.n_classes)
    buf = ThreeWayBuffer(60, DEV, split_seed=5)
    buf.add(torch.arange(0, 40), 0)
    buf.add(torch.arange(40, 80), 1)
    gen = torch.Generator(device=DEV)
    knob, chunk = 0.05, 32

    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    gen.manual_seed(42)
    applied, diag = bargain_step(
        model, stream, torch.arange(80, 96), 2, buf, gen, 8, DEV, "v2", knob, chunk,
        zeta_floor=0.01, jvp_batched=True,
    )
    assert applied
    assert diag is not None
    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta0[off : off + n].reshape(p.shape))
            off += n

    red = reduced_problem(
        _state(stream, model, buf, gen, 42), knob, chunk, None, True,
        with_curvature=True, zeta_floor=0.01,
    )
    ref, got = diag["problem"], red["problem"]
    assert np.allclose(got.g, ref.g, atol=1e-6)
    assert np.allclose(got.H, ref.H, atol=1e-5)
    assert np.allclose(got.tau, ref.tau, atol=1e-10)
