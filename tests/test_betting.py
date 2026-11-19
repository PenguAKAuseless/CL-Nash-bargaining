"""Betting confidence sequence: fixed-time and every-prefix coverage."""

from __future__ import annotations

import numpy as np
import pytest

from cl_certify.bernstein import empirical_bernstein_upper
from cl_certify.betting import betting_upper

DISTS = [
    ("uniform", lambda r, m: r.uniform(-1.0, 1.0, m), 0.0, -1.0, 1.0),
    ("bernoulli_p30", lambda r, m: r.binomial(1, 0.3, m).astype(float), 0.3, 0.0, 1.0),
    ("beta_2_5", lambda r, m: r.beta(2.0, 5.0, m), 2.0 / 7.0, 0.0, 1.0),
    ("skew_2pct", lambda r, m: (r.random(m) < 0.02).astype(float), 0.02, 0.0, 1.0),
]


@pytest.mark.parametrize(("name", "dist", "mu", "lo", "hi"), DISTS)
def test_fixed_time_coverage(name, dist, mu, lo, hi):
    rng = np.random.default_rng(0)
    for m in (30, 200):
        for delta in (0.05, 0.01):
            fails = sum(
                betting_upper(dist(rng, m), delta, lo, hi).bound < mu
                for _ in range(1500)
            )
            rate = fails / 1500
            assert rate <= delta, f"{name}: m={m} delta={delta} rate={rate:.4f}"


@pytest.mark.parametrize(("name", "dist", "mu", "lo", "hi"), DISTS)
def test_anytime_coverage_over_every_prefix(name, dist, mu, lo, hi):
    """Coverage holds when one stream is checked at every prefix length."""
    rng = np.random.default_rng(1)
    delta, n_max, trials = 0.05, 120, 600
    fails = 0
    for _ in range(trials):
        x = dist(rng, n_max)
        for t in range(2, n_max + 1):
            if betting_upper(x[:t], delta, lo, hi, grid_size=200).bound < mu:
                fails += 1
                break
    assert fails / trials <= delta, f"{name}: anytime rate {fails / trials:.4f}"


def test_a_fixed_time_bound_can_fail_the_every_prefix_protocol():
    """A normal-approximation interval fails the every-prefix protocol."""
    from scipy.stats import norm

    rng = np.random.default_rng(2)
    delta, n_max, trials = 0.05, 300, 600
    z = norm.ppf(1 - delta)
    fails = 0
    for _ in range(trials):
        x = rng.uniform(-1, 1, n_max)
        for t in range(2, n_max + 1):
            prefix = x[:t]
            ci = prefix.mean() + z * prefix.std(ddof=1) / np.sqrt(t)
            if ci < 0.0:
                fails += 1
                break
    assert fails / trials > delta, (
        f"control is toothless: {fails}/{trials} violations"
    )


def test_bernstein_at_k1_is_too_loose_to_be_caught_by_optional_stopping():
    """A K=1 empirical-Bernstein bound is never violated over prefixes at these sizes."""
    rng = np.random.default_rng(7)
    delta, n_max, trials = 0.05, 120, 200
    fails = 0
    for _ in range(trials):
        x = rng.uniform(-1, 1, n_max)
        if any(
            empirical_bernstein_upper(x[:t], delta, -1, 1, n_checkpoints=1).bound < 0.0
            for t in range(2, n_max + 1)
        ):
            fails += 1
    assert fails == 0


def test_bound_lies_within_the_declared_range():
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = rng.uniform(0, 1, 40)
        b = betting_upper(x, 0.05, 0.0, 1.0)
        assert b.lo <= b.bound <= b.hi
        assert b.bound >= b.mean - 1e-9


def test_bound_tightens_as_n_grows():
    rng = np.random.default_rng(4)
    x = rng.beta(2, 5, 4000)
    widths = [betting_upper(x[:n], 0.05, 0.0, 1.0).bound - x[:n].mean()
              for n in (25, 100, 500, 2000)]
    assert widths[0] > widths[-1]
    assert widths[-1] < 0.05


def test_beats_bernstein_at_the_sizes_this_project_runs():
    """At m=250, betting is tighter than Bernstein with K=120 checkpoints."""
    rng = np.random.default_rng(5)
    x = rng.beta(2, 5, 250)
    bet = betting_upper(x, 0.05, 0.0, 1.0).bound
    bern = empirical_bernstein_upper(x, 0.05, 0.0, 1.0, n_checkpoints=120).bound
    assert bet < bern


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="delta must be in"):
        betting_upper(np.zeros(5), 1.0, 0, 1)
    with pytest.raises(ValueError, match="need hi > lo"):
        betting_upper(np.zeros(5), 0.05, 1, 1)
