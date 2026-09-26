"""TensorBoard helper for long-running experiments (E-cert/E-probe/E4/E6),
plus a plain-text progress heartbeat.

Usage in an experiment script:

    from cl_bargain.tb import writer, heartbeat

    tb = writer("e8_certificate", tag=f"seed{seed}")   # runs/e8_certificate/seed0/
    for step in training_loop:
        ...
        tb.log_scalar("coverage", coverage_value, step)
        heartbeat(HEARTBEAT_PATH, f"step {step}: coverage={coverage_value:.3f}")

`writer` degrades to a no-op stub when the `tensorboard` package is not
installed, so nothing under src/cl_bargain/ outside the already-approved
torch-importing modules (pyproject.toml) gains a HARD tensorboard
dependency -- it stays a dev-only convenience for watching a run live
(`tensorboard --logdir runs/`), never a requirement to reproduce results.

`heartbeat` is the direct fix for the failure mode this project hit once
already: a background run silently going for 37 hours with no one checking
its progress. It appends one timestamped line to a plain text file --
readable with `tail`, no dependency, no viewer needed -- so "is this run
still making progress" is always a one-line check away.
"""

from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"


class _NullWriter:
    """No-op stand-in used when `tensorboard` is not installed. Every method
    silently does nothing, so callers never need an `if tb is not None`
    guard around their logging calls."""

    def log_scalar(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def writer(experiment: str, tag: str = "default"):
    """A small wrapper around torch.utils.tensorboard.SummaryWriter, writing
    to runs/<experiment>/<tag>/. Returns a _NullWriter (same interface, does
    nothing) if the tensorboard package is not installed -- callers do not
    need to branch on availability."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return _NullWriter()

    log_dir = RUNS_DIR / experiment / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    return _RealWriter(SummaryWriter(str(log_dir)))


class _RealWriter:
    def __init__(self, summary_writer) -> None:
        self._sw = summary_writer

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._sw.add_scalar(name, value, step)

    def close(self) -> None:
        self._sw.close()


def heartbeat(path: Path, message: str) -> None:
    """Append one timestamped line to a plain-text progress log. No
    dependency, no viewer -- `tail -f <path>` (or a plain `cat`) is always
    enough to check whether a long background run is actually progressing,
    which is the check that was missing when E2's full-scale sweep ran for
    37 hours before anyone looked at it again."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


__all__ = ["heartbeat", "writer"]
