"""Clopper-Pearson intervals for measured violation rates."""

from __future__ import annotations

from scipy import stats


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial interval for k successes in n trials."""
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


__all__ = ["clopper_pearson"]
