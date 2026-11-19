"""Logistic-regression sandbox checking certifier coverage against known ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cl_certify.bernstein import empirical_bernstein_upper


@dataclass
class LogisticTask:
    w_true: np.ndarray
    rng: np.random.Generator
    dim: int
    noise: float = 0.0

    def sample(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.rng.normal(size=(m, self.dim))
        p = 1.0 / (1.0 + np.exp(-(x @ self.w_true)))
        if self.noise:
            p = (1 - self.noise) * p + self.noise * 0.5
        y = (self.rng.random(m) < p).astype(float)
        return x, y


def per_example_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray, loss_max: float) -> np.ndarray:
    z = x @ w
    # log(1 + exp(z)) - y z, numerically stable
    ll = np.logaddexp(0.0, z) - y * z
    return np.clip(ll, 0.0, loss_max)


def certifier_coverage_experiment(
    dim: int = 5,
    m_audit: int = 30,
    m_population: int = 200_000,
    n_checkpoints: int = 10,
    delta: float = 0.05,
    trials: int = 400,
    loss_max: float = 10.0,
    drift: float = 0.4,
    seed: int = 0,
) -> dict:
    """Rate of trials where the audit bound falls below true forgetting at any checkpoint."""
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=dim)
    task = LogisticTask(w_true=w_true, rng=rng, dim=dim)

    x_pop, y_pop = task.sample(m_population)
    w0 = w_true.copy()
    pop_anchor = per_example_loss(w0, x_pop, y_pop, loss_max)

    ws = [w0 + drift * rng.normal(size=dim) for _ in range(n_checkpoints)]
    true_forgetting = [
        float(per_example_loss(w, x_pop, y_pop, loss_max).mean() - pop_anchor.mean()) for w in ws
    ]

    failures = 0
    slacks = []
    for _ in range(trials):
        xa, ya = task.sample(m_audit)
        anchor = per_example_loss(w0, xa, ya, loss_max)
        trial_failed = False
        for w, truth in zip(ws, true_forgetting, strict=True):
            d = per_example_loss(w, xa, ya, loss_max) - anchor
            b = empirical_bernstein_upper(d, delta, -loss_max, loss_max, n_checkpoints)
            slacks.append(b.bound - truth)
            if b.bound < truth:
                trial_failed = True
        failures += int(trial_failed)

    return {
        "failure_rate": failures / trials,
        "delta": delta,
        "mean_slack": float(np.mean(slacks)),
        "min_slack": float(np.min(slacks)),
        "m_audit": m_audit,
        "n_checkpoints": n_checkpoints,
        "true_forgetting": true_forgetting,
    }


__all__ = ["LogisticTask", "certifier_coverage_experiment", "per_example_loss"]
