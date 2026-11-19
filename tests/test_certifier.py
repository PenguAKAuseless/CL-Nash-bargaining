"""The certifier against ground truth on logistic regression."""

from __future__ import annotations

import numpy as np
import pytest

from cl_certify.certifier import PairedBernsteinCertifier
from cl_certify.sandbox import certifier_coverage_experiment


@pytest.mark.parametrize(("m_audit", "k"), [(30, 10), (100, 20)])
def test_logistic_sandbox_coverage_holds(m_audit, k):
    out = certifier_coverage_experiment(
        m_audit=m_audit, n_checkpoints=k, delta=0.05, trials=300, m_population=100_000
    )
    assert out["failure_rate"] <= out["delta"], out


def test_logistic_sandbox_bound_is_not_vacuous_at_large_m():
    """At m=2000 the slack is small relative to the loss scale."""
    out = certifier_coverage_experiment(
        m_audit=2000, n_checkpoints=5, delta=0.05, trials=60, m_population=100_000
    )
    assert out["failure_rate"] == 0.0
    assert out["mean_slack"] < 1.0, out


def _cert(delta=0.05, loss_max=10.0, k=4, n_tasks=3):
    return PairedBernsteinCertifier(
        delta=delta, loss_max=loss_max, n_checkpoints=k, n_tasks_max=n_tasks
    )


def test_bound_exceeds_the_realised_mean_it_bounds():
    c = _cert()
    rng = np.random.default_rng(0)
    anchor = rng.uniform(0, 1, 50)
    c.set_anchor(0, anchor)
    rec = c.certify_task(0, anchor + 0.3, checkpoint=1, step=10, budget=0.5)
    assert rec.mean == pytest.approx(0.3)
    assert rec.bound > rec.mean
    assert rec.n_audit == 50


def test_zero_change_gives_zero_mean_and_zero_variance_term():
    c = _cert()
    anchor = np.linspace(0, 2, 40)
    c.set_anchor(0, anchor)
    rec = c.certify_task(0, anchor.copy(), checkpoint=1, step=1)
    assert rec.mean == pytest.approx(0.0)
    assert rec.var == pytest.approx(0.0)
    assert rec.width_sqrt == pytest.approx(0.0)
    assert rec.width_residual > 0


def test_audit_slice_must_not_change_size_between_anchor_and_checkpoint():
    c = _cert()
    c.set_anchor(0, np.zeros(20))
    with pytest.raises(ValueError, match="changed size"):
        c.certify_task(0, np.zeros(19), checkpoint=1, step=1)


def test_anchor_is_clipped_to_loss_max():
    c = _cert(loss_max=1.0)
    c.set_anchor(0, np.array([0.5, 5.0, -2.0]))
    assert np.allclose(c.anchors[0], [0.5, 1.0, 0.0])


def test_pooled_mode_certifies_the_mean_over_tasks_not_the_pooled_examples():
    """Pooled mean is over tasks (0.5), not example-weighted."""
    c = _cert()
    c.set_anchor(0, np.zeros(100))
    c.set_anchor(1, np.zeros(4))
    rec = c.certify_pooled({0: np.zeros(100), 1: np.ones(4)}, checkpoint=1, step=1)
    assert rec.mean == pytest.approx(0.5)
    assert rec.mode == "pooled"


def test_budget_violation_flag():
    c = _cert()
    c.set_anchor(0, np.zeros(200))
    lo = c.certify_task(0, np.zeros(200), checkpoint=1, step=1, budget=100.0)
    hi = c.certify_task(0, np.full(200, 5.0), checkpoint=2, step=2, budget=0.01)
    assert lo.violated_budget is False
    assert hi.violated_budget is True


def test_per_task_mode_spends_delta_over_tasks():
    """A larger n_tasks_max gives a strictly wider bound on identical data."""
    data = np.linspace(0, 1, 30)
    narrow = _cert(n_tasks=1)
    wide = _cert(n_tasks=20)
    for c in (narrow, wide):
        c.set_anchor(0, np.zeros(30))
    b1 = narrow.certify_task(0, data, 1, 1).bound
    b2 = wide.certify_task(0, data, 1, 1).bound
    assert b2 > b1
