"""Simultaneous coverage over K checkpoints re-measured on one fixed audit slice."""

from __future__ import annotations

import numpy as np
import pytest

from cl_certify.bernstein import empirical_bernstein_upper
from cl_certify.betting import betting_upper

M = 80
K = 20
DELTA = 0.1
TRIALS = 400


def _checkpoint_sample(rng, mu: float, m: int) -> np.ndarray:
    """Per-example 0-1 loss differences D_j in {-1, 0, 1} with mean mu."""
    p_up = (1.0 + mu) / 2.0 * 0.5
    p_dn = p_up - mu / 2.0
    u = rng.random(m)
    return np.where(u < p_up, 1.0, np.where(u < p_up + p_dn, -1.0, 0.0))


def _simultaneous_violations(method: str, n_checkpoints_claimed: int, seed: int = 0) -> float:
    """Fraction of trials in which some checkpoint's bound falls below its true mean."""
    rng = np.random.default_rng(seed)
    mus = np.linspace(-0.02, 0.06, K)
    violated = 0
    for _ in range(TRIALS):
        bad = False
        for mu in mus:
            d = _checkpoint_sample(rng, float(mu), M)
            if method == "betting":
                b = betting_upper(d, DELTA, -1.0, 1.0, n_checkpoints=n_checkpoints_claimed)
            else:
                b = empirical_bernstein_upper(d, DELTA, -1.0, 1.0, n_checkpoints_claimed)
            if b.bound < mu:
                bad = True
        violated += bad
    return violated / TRIALS


@pytest.mark.parametrize("method", ["betting", "bernstein"])
def test_union_bound_over_checkpoints_controls_simultaneous_violations(method):
    rate = _simultaneous_violations(method, n_checkpoints_claimed=K)
    assert rate <= DELTA, f"{method} with K={K} spent: simultaneous violation {rate:.3f} > {DELTA}"


def test_betting_without_the_union_bound_does_not_cover_all_checkpoints():
    """Betting with K=1 does not control the simultaneous violation rate."""
    naive = _simultaneous_violations("betting", n_checkpoints_claimed=1)
    paid = _simultaneous_violations("betting", n_checkpoints_claimed=K)
    assert naive > DELTA, (
        f"control test is toothless: naive betting gave {naive:.3f} <= {DELTA}, "
        "so this configuration cannot tell the two apart"
    )
    assert paid < naive


def test_betting_is_still_tighter_than_bernstein_at_matched_delta():
    """Betting is tighter than Bernstein when both pay the union bound over K."""
    rng = np.random.default_rng(7)
    d = _checkpoint_sample(rng, 0.0, 250)
    bet = betting_upper(d, 0.05, -1.0, 1.0, n_checkpoints=120).bound
    ber = empirical_bernstein_upper(d, 0.05, -1.0, 1.0, 120).bound
    assert bet < ber
