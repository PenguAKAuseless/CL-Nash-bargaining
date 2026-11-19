"""The audit slice must never be reachable from the training path."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cl_certify.splits import ThreeWayBuffer

DEV = torch.device("cpu")


def _fill(capacity=600, fractions=(0.70, 0.15, 0.15), seed=7, n_tasks=4, per_task=500):
    buf = ThreeWayBuffer(capacity, DEV, seed, fractions)
    base = 0
    for t in range(n_tasks):
        buf.add(torch.arange(base, base + per_task), t)
        base += per_task
    return buf


def test_slices_are_disjoint():
    _fill().assert_disjoint()


def test_capacities_sum_to_capacity_and_respect_fractions():
    buf = ThreeWayBuffer(2000, DEV, 0, (0.70, 0.15, 0.15))
    assert sum(buf.capacities.values()) == 2000
    assert buf.capacities["audit"] == 300
    assert buf.capacities["calibration"] == 300
    assert buf.capacities["replay"] == 1400


def test_audit_requires_explicit_certification_flag():
    buf = _fill()
    with pytest.raises(RuntimeError, match="for_certification"):
        buf.audit_held(0)
    assert buf.audit_held(0, for_certification=True).numel() > 0


def test_training_accessors_never_return_audit_indices():
    buf = _fill()
    audit = set()
    for t in buf.tasks():
        audit |= set(buf.audit_held(t, for_certification=True).tolist())
    # everything the training path can reach, including the pooled idx[:filled]
    reachable = set(buf.idx[: buf.filled].tolist())
    for t in buf.tasks():
        reachable |= set(buf.held(t).tolist())
        reachable |= set(buf.replay_held(t).tolist())
    assert not (reachable & audit)


def test_calibration_is_disjoint_from_replay_and_audit():
    buf = _fill()
    for t in buf.tasks():
        cal = set(buf.calibration_held(t).tolist())
        assert not (cal & set(buf.held(t).tolist()))
        assert not (cal & set(buf.audit_held(t, for_certification=True).tolist()))


def test_split_is_a_function_of_split_seed_only():
    a = _fill(seed=11)
    b = _fill(seed=11)
    for t in a.tasks():
        assert torch.equal(a.audit_held(t, for_certification=True),
                           b.audit_held(t, for_certification=True))
    c = _fill(seed=12)
    assert not torch.equal(a.audit_held(0, for_certification=True),
                           c.audit_held(0, for_certification=True))


def test_routing_fractions_are_approximately_respected():
    """With non-binding capacity, slice sizes follow the routing fractions."""
    buf = ThreeWayBuffer(100_000, DEV, 3, (0.70, 0.15, 0.15))
    buf.add(torch.arange(20_000), 0)
    sizes = buf.slice_sizes(0)
    total = sum(sizes.values())
    assert total == 20_000
    assert np.isclose(sizes["replay"] / total, 0.70, atol=0.02)
    assert np.isclose(sizes["audit"] / total, 0.15, atol=0.02)


def test_task_sizes_reports_replay_only():
    buf = _fill()
    for t, size in buf.task_sizes.items():
        assert size == buf.held(t).numel()


def test_rejects_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1"):
        ThreeWayBuffer(100, DEV, 0, (0.5, 0.2, 0.2))
    with pytest.raises(ValueError, match="must be >= 0"):
        ThreeWayBuffer(100, DEV, 0, (1.2, -0.1, -0.1))
    with pytest.raises(ValueError, match="need 3 fractions"):
        ThreeWayBuffer(100, DEV, 0, (0.5, 0.5))


def test_zero_audit_fraction_is_allowed_and_empty():
    buf = ThreeWayBuffer(100, DEV, 0, (0.85, 0.15, 0.0))
    buf.add(torch.arange(500), 0)
    assert buf.audit_held(0, for_certification=True).numel() == 0
    assert buf.held(0).numel() > 0
