"""Backtracking admission: the beta it picks, and that it leaves no residue."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cl_certify.admission import BacktrackingAdmission


class _Toy(torch.nn.Module):
    """Task a has calibration loss a * theta, so a step d raises it by exactly a * d."""

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(1))


def _setup(budget, max_backtrack=6):
    m = _Toy()
    adm = BacktrackingAdmission(budget_per_step=budget, max_backtrack=max_backtrack)
    return m, adm, (lambda ts, m=m: [a * m.theta.item() for a in ts])


def test_accepts_unshrunk_step_within_budget():
    m, adm, loss = _setup(budget=1.0)
    out = adm.admit(m, torch.tensor([0.5]), [1], loss)
    assert out.beta == 1.0
    assert out.n_backtracks == 0
    assert out.applied
    assert m.theta.item() == pytest.approx(0.5)


def test_halves_until_within_budget():
    # damage = 0.8 * beta; budget 0.25 -> need beta <= 0.3125 -> beta = 0.25
    m, adm, loss = _setup(budget=0.25)
    out = adm.admit(m, torch.tensor([0.8]), [1], loss)
    assert out.beta == pytest.approx(0.25)
    assert out.n_backtracks == 2
    assert m.theta.item() == pytest.approx(0.2)


def test_skips_and_leaves_parameters_untouched_when_no_beta_qualifies():
    m, adm, loss = _setup(budget=1e-9, max_backtrack=3)
    out = adm.admit(m, torch.tensor([1.0]), [1], loss)
    assert not out.applied
    assert out.beta == 0.0
    assert out.n_backtracks == 4
    assert m.theta.item() == pytest.approx(0.0, abs=1e-12)


def test_worst_case_over_tasks_binds_not_the_mean():
    m, adm, loss = _setup(budget=0.5)
    # task 10's damage is 10x task 1's, so it binds
    out = adm.admit(m, torch.tensor([0.4]), [1, 10], loss)
    assert out.worst_damage == pytest.approx(10 * m.theta.item())
    assert out.worst_damage <= 0.5


def test_negative_damage_is_always_admitted():
    """A step that improves the old tasks is never shrunk."""
    m, adm, loss = _setup(budget=0.0)
    out = adm.admit(m, torch.tensor([-1.0]), [1], loss)
    assert out.beta == 1.0
    assert out.applied


def test_disabled_admission_applies_the_raw_step():
    m = _Toy()
    adm = BacktrackingAdmission(budget_per_step=1e-9, enabled=False)
    out = adm.admit(m, torch.tensor([3.0]), [1], lambda ts: [a * m.theta.item() for a in ts])
    assert out.beta == 1.0
    assert m.theta.item() == pytest.approx(3.0)


def test_no_active_tasks_means_no_calibration_evaluations():
    m, adm, loss = _setup(budget=0.0)
    out = adm.admit(m, torch.tensor([7.0]), [], loss)
    assert out.applied
    assert out.n_evaluated == 0
    assert m.theta.item() == pytest.approx(7.0)


def test_calibration_passes_are_bounded_by_max_backtrack():
    m, adm, loss = _setup(budget=1e-9, max_backtrack=4)
    out = adm.admit(m, torch.tensor([1.0]), [1, 2, 3], loss)
    assert out.n_evaluated == 1 + 5


def test_summary_aggregates_history():
    m, adm, loss = _setup(budget=0.25)
    adm.admit(m, torch.tensor([0.1]), [1], loss)
    adm.admit(m, torch.tensor([-0.1]), [1], loss)
    s = adm.summary()
    assert s["n_steps"] == 2
    assert s["accept_rate"] == 1.0
    assert np.isclose(s["mean_beta"], 1.0)


def test_cumulative_mode_measures_drift_from_the_anchor_not_the_previous_step():
    """Drift is measured from the anchor: beta * 0.4 <= 0.1 gives beta = 0.25."""
    m, adm, loss = _setup(budget=0.5)
    with torch.no_grad():
        m.theta.fill_(0.4)
    out = adm.admit(m, torch.tensor([0.4]), [1], loss, reference=np.array([0.0]))
    assert out.beta == pytest.approx(0.25)
    assert m.theta.item() == pytest.approx(0.5)


def test_cumulative_mode_costs_no_pre_step_forward_pass():
    m, adm, loss = _setup(budget=10.0)
    out = adm.admit(m, torch.tensor([0.1]), [1], loss, reference=np.array([0.0]))
    assert out.n_evaluated == 1


def test_cumulative_mode_skips_once_the_budget_is_already_spent():
    """Drift already above the budget skips the step and leaves parameters unchanged."""
    m, adm, loss = _setup(budget=0.1, max_backtrack=3)
    with torch.no_grad():
        m.theta.fill_(5.0)
    out = adm.admit(m, torch.tensor([1.0]), [1], loss, reference=np.array([0.0]))
    assert not out.applied
    assert m.theta.item() == pytest.approx(5.0)


def test_outcomes_are_tagged_with_the_live_task_and_summarised_per_task():
    """Outcomes carry the live task index and acceptance is summarised per task."""
    import numpy as np
    import torch

    adm = BacktrackingAdmission(budget_per_step=0.1, enabled=True, mode="cumulative")
    model = torch.nn.Linear(3, 2)
    delta = torch.zeros(sum(p.numel() for p in model.parameters()))
    for t, damage in ((0, 0.0), (0, 0.0), (1, 10.0), (1, 10.0), (1, 10.0)):
        adm.admit(model, delta, [0], lambda _t, d=damage: np.array([d]),
                  reference=np.array([0.0]), task=t)
    by = adm.summary()["by_task"]
    assert set(by) == {0, 1}
    assert by[0]["n_steps"] == 2
    assert by[1]["n_steps"] == 3
    assert by[0]["accept_rate"] == 1.0
    assert by[1]["accept_rate"] == 0.0


def test_ramp_budget_grows_from_zero_at_the_task_boundary_to_full_at_the_end():
    from cl_certify.admission import ramp_budget

    # full budget is reached only on the last task of a 20-task stream
    assert ramp_budget(0.2, live_task=1, oldest_absent=0, n_tasks=20) == pytest.approx(0.2 / 19)
    assert ramp_budget(0.2, live_task=10, oldest_absent=0, n_tasks=20) == pytest.approx(
        0.2 * 10 / 19
    )
    assert ramp_budget(0.2, live_task=19, oldest_absent=0, n_tasks=20) == pytest.approx(0.2)


def test_ramp_is_keyed_on_the_oldest_absent_task_not_the_live_one():
    """The ramp span is set by the oldest absent task."""
    from cl_certify.admission import ramp_budget

    early = ramp_budget(0.2, live_task=19, oldest_absent=0, n_tasks=20)
    late = ramp_budget(0.2, live_task=19, oldest_absent=17, n_tasks=20)
    assert early == pytest.approx(0.2)
    assert late == pytest.approx(0.2)
    mid = ramp_budget(0.2, live_task=18, oldest_absent=17, n_tasks=20)
    assert mid == pytest.approx(0.1)
