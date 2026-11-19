"""Learn-then-Test: fixed-sequence selection of a final model on the audit slice."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    name: str
    family: str
    param: float
    rank_score: float = float("nan")


@dataclass
class LTTOutcome:
    selected: Candidate | None
    selected_index: int
    bounds: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    rejected: list[bool] = field(default_factory=list)
    monotone: bool = True
    n_tested: int = 0


def candidate_pool(betas: list[float], tail_ks: list[int]) -> list[Candidate]:
    """All candidates, unordered; `order_by_calibration` fixes the order."""
    pool = [Candidate(f"anchor_pull@{b:.2f}", "anchor_pull", float(b)) for b in betas]
    pool += [Candidate(f"tail_avg@{k}", "tail_average", float(k)) for k in tail_ks]
    return pool


# positions of each slice in the tuple returned by `evaluate`
_AUDIT = 0
_CALIBRATION = 1


def calibration_score_fn(evaluate, delta: float):
    """Ordering score that reads only the calibration element of `evaluate(candidate)`."""

    from cl_certify.bernstein import empirical_bernstein_upper

    def score(candidate) -> float:
        calibration = evaluate(candidate)[_CALIBRATION]
        return float(empirical_bernstein_upper(
            calibration, delta, 0.0, 1.0, n_checkpoints=1).bound)

    return score


def order_by_calibration(pool: list[Candidate], score_fn) -> list[Candidate]:
    """Sort candidates by ascending calibration score; never reads audit data."""
    scored = []
    for c in pool:
        c.rank_score = float(score_fn(c))
        scored.append(c)
    return sorted(scored, key=lambda c: (c.rank_score, c.name))


def fixed_sequence_select(
    candidates: list[Candidate],
    bound_fn,
    tau: float,
    evaluate_all: bool = False,
) -> LTTOutcome:
    """Fixed-sequence test: select the last candidate before the first bound exceeding tau."""
    out = LTTOutcome(selected=None, selected_index=-1)
    stopped = False
    for i, c in enumerate(candidates):
        b, mu = bound_fn(c)
        out.bounds.append(float(b))
        out.means.append(float(mu))
        passed = b <= tau
        out.rejected.append(passed)
        # bounds past the stopping point are reported only, never used for selection
        if not stopped:
            out.n_tested = i + 1
            if passed:
                out.selected = c
                out.selected_index = i
            else:
                stopped = True
                if not evaluate_all:
                    break
    out.monotone = all(
        out.bounds[i] <= out.bounds[i + 1] + 1e-12 for i in range(len(out.bounds) - 1)
    )
    return out


def weight_average(sds: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted mean of state dicts; integer buffers are taken from the last entry."""
    if not sds:
        raise ValueError("need at least one state dict")
    w = [1.0 / len(sds)] * len(sds) if weights is None else list(weights)
    if len(w) != len(sds):
        raise ValueError(f"{len(w)} weights for {len(sds)} state dicts")
    total = sum(w)
    out = {}
    for k, v in sds[-1].items():
        if not v.dtype.is_floating_point:
            out[k] = v.clone()
            continue
        acc = sds[0][k] * (w[0] / total)
        for sd, wi in zip(sds[1:], w[1:], strict=True):
            acc = acc + sd[k] * (wi / total)
        out[k] = acc
    return out


def interpolate(sd_from: dict, sd_to: dict, beta: float) -> dict:
    """sd_from + beta * (sd_to - sd_from); integer buffers are taken from sd_to."""
    out = {}
    for k, v in sd_from.items():
        w = sd_to[k]
        if not v.dtype.is_floating_point:
            out[k] = w.clone()
        else:
            out[k] = (v + beta * (w - v)).clone()
    return out


__all__ = ["Candidate", "LTTOutcome", "calibration_score_fn", "candidate_pool",
           "fixed_sequence_select",
           "interpolate", "order_by_calibration", "weight_average"]
