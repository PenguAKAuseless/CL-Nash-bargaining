"""Monte-Carlo coverage tests for the empirical-Bernstein bound."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from cl_certify.bernstein import bound_width, empirical_bernstein_upper, required_m


def _coverage(rng, dist, true_mean, lo, hi, m, delta, trials, n_checkpoints=1):
    fails = 0
    for _ in range(trials):
        x = dist(rng, m)
        b = empirical_bernstein_upper(x, delta, lo, hi, n_checkpoints)
        if b.bound < true_mean:
            fails += 1
    return fails / trials


@pytest.mark.parametrize(
    ("name", "dist", "true_mean", "lo", "hi"),
    [
        ("uniform", lambda r, m: r.uniform(-1.0, 1.0, m), 0.0, -1.0, 1.0),
        ("bernoulli", lambda r, m: r.binomial(1, 0.3, m).astype(float), 0.3, 0.0, 1.0),
        ("beta", lambda r, m: r.beta(2.0, 5.0, m), 2.0 / 7.0, 0.0, 1.0),
        (
            "two_point_skew",
            lambda r, m: np.where(r.random(m) < 0.02, 1.0, 0.0),
            0.02, 0.0, 1.0,
        ),
    ],
)
def test_coverage_at_least_one_minus_delta(name, dist, true_mean, lo, hi):
    rng = np.random.default_rng(0)
    for m in (20, 200):
        for delta in (0.05, 0.01):
            rate = _coverage(rng, dist, true_mean, lo, hi, m, delta, trials=4000)
            assert rate <= delta, f"{name}: m={m} delta={delta} failure rate {rate:.4f}"


def test_coverage_with_union_bound_over_checkpoints():
    """With the union bound over K checkpoints, all K bounds hold simultaneously."""
    rng = np.random.default_rng(1)
    k, delta, m, trials = 8, 0.05, 50, 3000
    fails = 0
    for _ in range(trials):
        if any(
            empirical_bernstein_upper(rng.uniform(-1, 1, m), delta, -1, 1, k).bound < 0.0
            for _ in range(k)
        ):
            fails += 1
    assert fails / trials <= delta


def test_matches_maurer_pontil_formula_on_unit_interval():
    x = np.array([0.1, 0.4, 0.2, 0.9, 0.5, 0.3])
    b = empirical_bernstein_upper(x, 0.05, 0.0, 1.0)
    log_term = math.log(2.0 / 0.05)
    expected = (
        x.mean()
        + math.sqrt(2 * x.var(ddof=1) * log_term / x.size)
        + 7 * log_term / (3 * (x.size - 1))
    )
    assert b.bound == pytest.approx(expected)
    assert b.log_term == pytest.approx(log_term)


def test_range_enters_residual_term_as_full_width_not_half():
    """The residual term uses the full width 2R of [-R, R]."""
    x = np.zeros(10)
    b = empirical_bernstein_upper(x, 0.05, -1.0, 1.0)
    log_term = math.log(2.0 / 0.05)
    assert b.width_residual == pytest.approx(7 * 2.0 * log_term / (3 * 9))


def test_affine_equivariance():
    """Shifting and scaling the data shifts and scales the bound exactly."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 40)
    a, c = 3.0, -0.7
    b0 = empirical_bernstein_upper(x, 0.05, 0.0, 1.0)
    b1 = empirical_bernstein_upper(a * x + c, 0.05, c, a + c)
    assert b1.bound == pytest.approx(a * b0.bound + c)


def test_bound_degrades_gracefully_below_two_samples():
    for m in (0, 1):
        b = empirical_bernstein_upper(np.zeros(m), 0.05, -1.0, 1.0)
        assert b.n == m
        assert math.isinf(b.width_sqrt)
        assert b.bound == pytest.approx(1.0)


def test_width_monotone_in_m_and_delta():
    w = [bound_width(0.05, -1, 1, m, 0.25) for m in (10, 50, 200, 1000)]
    assert all(a > b for a, b in pairwise(w))
    assert bound_width(0.01, -1, 1, 100, 0.25) > bound_width(0.05, -1, 1, 100, 0.25)


def test_required_m_is_the_smallest_sufficient_m():
    m = required_m(0.05, -1.0, 1.0, var=0.25, target_width=0.05)
    assert bound_width(0.05, -1, 1, m, 0.25) <= 0.05
    assert bound_width(0.05, -1, 1, m - 1, 0.25) > 0.05


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="delta must be in"):
        empirical_bernstein_upper(np.zeros(5), 0.0, -1, 1)
    with pytest.raises(ValueError, match="need hi > lo"):
        empirical_bernstein_upper(np.zeros(5), 0.05, 1, 1)
    with pytest.raises(ValueError, match="n_checkpoints must be"):
        empirical_bernstein_upper(np.zeros(5), 0.05, -1, 1, n_checkpoints=0)
