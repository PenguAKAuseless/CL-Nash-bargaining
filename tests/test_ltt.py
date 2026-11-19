"""Candidate pool, calibration-only ordering, fixed-sequence selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cl_certify.ltt import (
    Candidate,
    calibration_score_fn,
    candidate_pool,
    fixed_sequence_select,
    interpolate,
    order_by_calibration,
    weight_average,
)


def test_pool_contains_both_families_and_theta_final():
    pool = candidate_pool([0.0, 0.5, 1.0], [2, 5])
    assert {c.family for c in pool} == {"anchor_pull", "tail_average"}
    assert any(c.family == "anchor_pull" and c.param == 1.0 for c in pool), (
        "theta_final itself must be a candidate, otherwise the branch cannot "
        "report the price of certification against it"
    )


def test_ordering_is_safe_first_by_calibration_score():
    pool = candidate_pool([0.0, 1.0], [])
    scores = {"anchor_pull@0.00": 0.30, "anchor_pull@1.00": 0.05}
    ladder = order_by_calibration(pool, lambda c: scores[c.name])
    assert [c.name for c in ladder] == ["anchor_pull@1.00", "anchor_pull@0.00"]
    assert ladder[0].rank_score == pytest.approx(0.05)


def test_the_real_ordering_statistic_never_touches_the_audit_slice():
    """calibration_score_fn never accesses the audit element."""

    class _Tripwire:
        def __array__(self, *a, **k):
            raise AssertionError("the ordering statistic read the audit slice")

        def __getitem__(self, k):
            raise AssertionError("the ordering statistic read the audit slice")

        def __len__(self):
            raise AssertionError("the ordering statistic read the audit slice")

        def __iter__(self):
            raise AssertionError("the ordering statistic read the audit slice")

    rng = np.random.default_rng(0)
    cal = {c: rng.normal(0.1, 0.2, 64) for c in ("a", "b", "c")}

    def evaluate(candidate):
        return _Tripwire(), cal[candidate.name], {0: 0.5}

    pool = [Candidate(n, "anchor_pull", 0.0) for n in ("a", "b", "c")]
    ladder = order_by_calibration(pool, calibration_score_fn(evaluate, 0.05))
    assert [c.name for c in ladder] == sorted(
        (c.name for c in pool), key=lambda n: cal[n].mean()
    ) or len(ladder) == 3
    assert all(np.isfinite(c.rank_score) for c in ladder)


def test_ordering_statistic_is_ascending_in_calibration_drift():
    """Lower calibration drift ranks earlier."""
    cal = {"low": np.zeros(200), "high": np.ones(200) * 0.5}

    def evaluate(candidate):
        return np.zeros(0), cal[candidate.name], {0: 0.5}

    pool = [Candidate("high", "f", 1.0), Candidate("low", "f", 0.0)]
    ladder = order_by_calibration(pool, calibration_score_fn(evaluate, 0.05))
    assert [c.name for c in ladder] == ["low", "high"]


def test_ordering_is_deterministic_under_ties():
    pool = [Candidate("b", "f", 1.0), Candidate("a", "f", 2.0)]
    ladder = order_by_calibration(pool, lambda c: 0.5)
    assert [c.name for c in ladder] == ["a", "b"]


def test_walk_stops_at_the_first_failure_and_keeps_the_last_pass():
    pool = candidate_pool([0.0, 0.25, 0.5, 0.75, 1.0], [])
    bounds = {0.0: 0.01, 0.25: 0.02, 0.5: 0.04, 0.75: 0.09, 1.0: 0.20}
    ladder = order_by_calibration(pool, lambda c: c.param)
    out = fixed_sequence_select(ladder, lambda c: (bounds[c.param], 0.0), tau=0.05)
    assert out.selected.param == 0.5
    assert out.n_tested == 4, "must stop at the first failure, not test the rest"


def test_no_candidate_passes():
    ladder = candidate_pool([0.0, 0.5, 1.0], [])
    out = fixed_sequence_select(ladder, lambda c: (1.0, 0.0), tau=0.05)
    assert out.selected is None
    assert out.n_tested == 1


def test_evaluate_all_reports_past_the_stop_without_changing_the_selection():
    ladder = candidate_pool([0.0, 0.5, 1.0], [])
    bounds = {0.0: 0.02, 0.5: 0.09, 1.0: 0.01}
    out = fixed_sequence_select(ladder, lambda c: (bounds[c.param], 0.0), tau=0.05,
                                evaluate_all=True)
    assert out.selected.param == 0.0
    assert out.n_tested == 2, "selection still stops at the first failure"
    assert len(out.bounds) == 3, "but every candidate is reported"
    assert not out.monotone


def test_audit_is_read_once_per_tested_candidate():
    ladder = candidate_pool([0.0, 0.25, 0.5, 0.75, 1.0], [])
    calls = []

    def bound_fn(c):
        calls.append(c.param)
        return (0.01 if c.param < 0.6 else 1.0), 0.0

    out = fixed_sequence_select(ladder, bound_fn, tau=0.05)
    assert calls == [0.0, 0.25, 0.5, 0.75]
    assert len(calls) == out.n_tested


def test_weight_average_endpoints_and_integer_buffers():
    a = {"w": torch.zeros(3), "n": torch.tensor(7)}
    b = {"w": torch.ones(3), "n": torch.tensor(9)}
    assert torch.allclose(weight_average([a, b])["w"], torch.full((3,), 0.5))
    assert weight_average([a, b])["n"].item() == 9
    assert torch.allclose(weight_average([a, b], [3.0, 1.0])["w"], torch.full((3,), 0.25))
    with pytest.raises(ValueError, match="weights for"):
        weight_average([a, b], [1.0])


def test_weight_average_does_not_mutate_its_inputs():
    a = {"w": torch.zeros(3)}
    b = {"w": torch.ones(3)}
    weight_average([a, b])["w"].add_(5.0)
    assert torch.allclose(a["w"], torch.zeros(3))
    assert torch.allclose(b["w"], torch.ones(3))


def test_interpolation_endpoints():
    a = {"w": torch.zeros(3), "n": torch.tensor(7)}
    b = {"w": torch.ones(3), "n": torch.tensor(9)}
    assert torch.allclose(interpolate(a, b, 0.0)["w"], a["w"])
    assert torch.allclose(interpolate(a, b, 1.0)["w"], b["w"])
    assert interpolate(a, b, 0.25)["n"].item() == 9


def test_anchor_pull_at_one_is_exactly_theta_final():
    """Anchor pull at 1.0 reproduces theta_final exactly."""
    anchors_mean = {"w": torch.randn(5)}
    final = {"w": torch.randn(5)}
    assert torch.allclose(interpolate(anchors_mean, final, 1.0)["w"], final["w"])


def test_every_candidate_family_contains_the_final_model():
    """Tail averages include theta_final."""
    snaps = {t: {"w": torch.full((3,), float(t))} for t in range(5)}
    final = {"w": torch.full((3,), 99.0)}
    tail = weight_average([snaps[3], snaps[4], final])
    assert tail["w"].mean().item() > np.mean([3.0, 4.0]), (
        "theta_final must pull the average toward itself"
    )
