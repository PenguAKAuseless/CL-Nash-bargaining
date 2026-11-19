"""Paired empirical-Bernstein bound (Maurer & Pontil 2009, Theorem 4)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BernsteinBound:
    """Upper confidence bound and its decomposition."""

    bound: float
    mean: float
    var: float
    n: int
    range_width: float
    log_term: float
    width_sqrt: float
    width_residual: float

    @property
    def width(self) -> float:
        return self.width_sqrt + self.width_residual


def empirical_bernstein_upper(
    samples: np.ndarray,
    delta: float,
    lo: float,
    hi: float,
    n_checkpoints: int = 1,
) -> BernsteinBound:
    """Upper bound on E[X] at level 1 - delta, union-bounded over `n_checkpoints`."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo}, hi={hi}")
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")

    x = np.asarray(samples, dtype=float).ravel()
    m = int(x.size)
    width_range = float(hi - lo)
    log_term = math.log(2.0 * n_checkpoints / delta)

    if m < 2:
        mean = float(x.mean()) if m else float("nan")
        return BernsteinBound(
            bound=width_range + lo, mean=mean, var=float("nan"), n=m,
            range_width=width_range, log_term=log_term,
            width_sqrt=float("inf"), width_residual=float("inf"),
        )

    # clip so an out-of-range observation cannot silently void the bound
    x = np.clip(x, lo, hi)
    mean = float(x.mean())
    var = float(x.var(ddof=1))

    width_sqrt = math.sqrt(2.0 * var * log_term / m)
    # residual constant is 7 * L with L the full range, not the half-range
    width_residual = 7.0 * width_range * log_term / (3.0 * (m - 1))
    return BernsteinBound(
        bound=mean + width_sqrt + width_residual,
        mean=mean,
        var=var,
        n=m,
        range_width=width_range,
        log_term=log_term,
        width_sqrt=width_sqrt,
        width_residual=width_residual,
    )


def bound_width(delta: float, lo: float, hi: float, m: int, var: float,
                n_checkpoints: int = 1) -> float:
    """Width of the bound (bound - mean) for a given m and variance."""
    log_term = math.log(2.0 * n_checkpoints / delta)
    if m < 2:
        return float("inf")
    return math.sqrt(2.0 * var * log_term / m) + 7.0 * (hi - lo) * log_term / (3.0 * (m - 1))


def required_m(delta: float, lo: float, hi: float, var: float, target_width: float,
               n_checkpoints: int = 1, m_max: int = 10_000_000) -> int:
    """Smallest m with bound_width(...) <= target_width, capped at m_max."""
    if bound_width(delta, lo, hi, m_max, var, n_checkpoints) > target_width:
        return m_max
    low, high = 2, m_max
    while low < high:
        mid = (low + high) // 2
        if bound_width(delta, lo, hi, mid, var, n_checkpoints) <= target_width:
            high = mid
        else:
            low = mid + 1
    return low


__all__ = ["BernsteinBound", "bound_width", "empirical_bernstein_upper", "required_m"]
