"""Append-only JSONL results plus CSV/Markdown table rendering."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def append(path: Path, record: dict) -> None:
    # single O_APPEND write per line so concurrent writers do not interleave
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fingerprints(path: Path) -> set[str]:
    return {r.get("fingerprint") for r in read(path)}


def to_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def to_markdown(rows: list[dict], path: Path, columns: list[str],
                headers: list[str] | None = None, fmt: dict | None = None) -> str:
    headers = headers or columns
    fmt = fmt or {}

    def cell(r, c):
        v = r.get(c)
        if v is None:
            return "--"
        f = fmt.get(c)
        if f is None:
            return str(v)
        try:
            return f.format(v)
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(cell(r, c) for c in columns) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


__all__ = ["append", "fingerprints", "read", "to_csv", "to_markdown"]
