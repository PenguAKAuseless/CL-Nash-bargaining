"""Anytime-valid confidence sequence by betting (Waudby-Smith & Ramdas 2023)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BettingBound:
    bound: float
    mean: float
    n: int
    lo: float
    hi: float
    grid_size: int
    saturated: bool


def _log_wealth(x01: np.ndarray, m01: float, c: float = 0.5) -> float:
    n = x01.size
    if n == 0:
        return 0.0
    # predictable bets: moments of x_1..x_{i-1} with a (1/2, 1/4) prior pseudo-count
    idx = np.arange(n, dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(x01)[:-1]))
    csq = np.concatenate(([0.0], np.cumsum(x01**2)[:-1]))
    mu_hat = (0.5 + csum) / (1.0 + idx)
    var_hat = (0.25 + csq) / (1.0 + idx) - mu_hat**2
    var_hat = np.maximum(var_hat, 1e-12)

    diff = mu_hat - m01
    lam = diff / (var_hat + diff**2)
    # truncation keeps 1 + lambda (x - m) positive, so Ville's inequality applies
    hi_cap = float("inf") if m01 <= 0.0 else c / m01
    lo_cap = -c / (1.0 - m01) if m01 < 1.0 else -float("inf")
    lam = np.clip(lam, lo_cap, hi_cap)

    terms = 1.0 + lam * (x01 - m01)
    if np.any(terms <= 0.0):
        return float("inf")
    return float(np.sum(np.log(terms)))


def betting_upper(
    samples: np.ndarray,
    delta: float,
    lo: float,
    hi: float,
    grid_size: int = 400,
    n_checkpoints: int = 1,
) -> BettingBound:
    """One-sided upper betting bound at level delta, union-bounded over `n_checkpoints`."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")

    x = np.asarray(samples, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return BettingBound(hi, float("nan"), 0, lo, hi, grid_size, True)

    width = hi - lo
    x01 = np.clip((x - lo) / width, 0.0, 1.0)
    threshold = math.log(float(n_checkpoints) / delta)

    grid = np.linspace(0.0, 1.0, grid_size + 1)
    mean01 = float(x01.mean())
    # Only candidates at or above the sample mean can be the upper endpoint.
    cands = grid[grid >= mean01]
    if cands.size == 0:
        return BettingBound(hi, float(x.mean()), n, lo, hi, grid_size, True)

    bound01 = 1.0
    saturated = True
    for m01 in cands:
        if _log_wealth(x01, float(m01)) >= threshold:
            # first rejected grid point; rounding up keeps the bound conservative
            bound01 = float(m01)
            saturated = False
            break

    return BettingBound(
        bound=lo + bound01 * width,
        mean=float(x.mean()),
        n=n,
        lo=lo,
        hi=hi,
        grid_size=grid_size,
        saturated=saturated,
    )


__all__ = ["BettingBound", "betting_upper"]
