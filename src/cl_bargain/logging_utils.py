"""Structured run logging: one JSON line per run.

Every experiment script appends to logs/<experiment>.jsonl via `log_run`, so
every table in tables/*.md is regenerable from the log alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def log_run(log_path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line, stamping git commit and wall-clock time."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {
        "git_commit": git_commit(),
        "wall_clock": time.time(),
        **record,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(stamped, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
