"""Per-phase wall-clock accounting, enabled by the CL_TIMING environment variable."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

_ACC: defaultdict[str, float] = defaultdict(float)
_COUNT: defaultdict[str, int] = defaultdict(int)
_DEPTH: defaultdict[str, int] = defaultdict(int)
_ENABLED = bool(os.environ.get("CL_TIMING"))


def enabled() -> bool:
    return _ENABLED


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def section(name: str):
    if not _ENABLED:
        yield
        return
    if _DEPTH[name]:
        _DEPTH[name] += 1
        try:
            yield
        finally:
            _DEPTH[name] -= 1
        return
    _sync()
    _DEPTH[name] += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _ACC[name] += time.perf_counter() - t0
        _COUNT[name] += 1
        _DEPTH[name] -= 1


def reset() -> None:
    _ACC.clear()
    _COUNT.clear()
    _DEPTH.clear()


def snapshot() -> dict:
    return {"seconds": dict(_ACC), "calls": dict(_COUNT)}


__all__ = ["enabled", "reset", "section", "snapshot"]
