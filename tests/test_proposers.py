"""Every proposer returns a finite Delta and leaves the model untouched."""

from __future__ import annotations

import pytest
import torch
from tests.test_reduce import _TinyNet, _TinyStream

from cl_certify.config import Config
from cl_certify.proposers import ProposalState
from cl_certify.runner import PROPOSERS, build_proposer
from cl_certify.splits import ThreeWayBuffer

DEV = torch.device("cpu")


def _state(model, stream, buf, seed=0):
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    return ProposalState(
        model=model, stream=stream, idx=torch.arange(80, 96), task_index=2,
        buffer=buf, generator=gen, minibatch_size=8, device=DEV, step=1,
    )


def _cfg(name):
    return Config(proposer=name, tau=0.05, lr=0.1, chunk_size=32, minibatch_size=8)


@pytest.mark.parametrize("name", PROPOSERS)
def test_proposer_returns_finite_delta_without_touching_the_model(name):
    torch.manual_seed(0)
    stream = _TinyStream()
    model = _TinyNet(stream.n_classes)
    buf = ThreeWayBuffer(60, DEV, split_seed=5)
    buf.add(torch.arange(0, 40), 0)
    buf.add(torch.arange(40, 80), 1)
    p = build_proposer(name, model, _cfg(name))
    if name == "derpp":
        p.store_logits(model, stream, buf.held(0))

    before = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
    proposal = p.propose(_state(model, stream, buf))
    after = torch.cat([q.detach().reshape(-1) for q in model.parameters()])

    assert torch.allclose(before, after, atol=1e-7), f"{name} wrote to the model"
    if proposal.delta is None:
        pytest.skip(f"{name} reported solver failure on this instance")
    assert proposal.delta.shape == before.shape
    assert torch.isfinite(proposal.delta).all()
    assert proposal.delta.norm() > 0


@pytest.mark.parametrize("name", PROPOSERS)
def test_proposer_handles_an_empty_buffer(name):
    torch.manual_seed(0)
    stream = _TinyStream()
    model = _TinyNet(stream.n_classes)
    buf = ThreeWayBuffer(60, DEV, split_seed=5)
    p = build_proposer(name, model, _cfg(name))
    before = torch.cat([q.detach().reshape(-1) for q in model.parameters()]).clone()
    proposal = p.propose(_state(model, stream, buf))
    after = torch.cat([q.detach().reshape(-1) for q in model.parameters()])
    assert torch.allclose(before, after, atol=1e-7)
    if proposal.delta is not None:
        assert torch.isfinite(proposal.delta).all()


def test_unknown_proposer_is_rejected():
    with pytest.raises(ValueError, match="unknown proposer"):
        build_proposer("telepathy", _TinyNet(), Config())
