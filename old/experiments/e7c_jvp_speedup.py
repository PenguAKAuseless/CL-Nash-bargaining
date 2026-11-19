"""E7c -- measured cost of the curvature term, and of the exact cheap path.

The curvature block of the reduced problem is the whole cost of the method at
network scale: it is the only part that scales as n^2 per step, and a full
ten-task run was measured at roughly four hours against thirty-eight seconds
for plain replay on the same stream.

Two ways of computing the same n x n x n array are timed here:

  * the default path, one Gauss-Newton-vector product per (j, a) pair, each of
    which is one forward-mode and one reverse-mode pass;
  * the forward-only path, which contracts the directional derivatives J g_a in
    output space and never runs a backward pass
    (streams/methods.py's `_gram_curvature_jvp_only`).

The two are algebraically identical, and tests/test_methods_torch.py gates that
equivalence numerically, so any difference measured here is wall clock alone.

Run:  uv run python experiments/e7c_jvp_speedup.py [--smoke]
Writes: logs/e7c_jvp_speedup.jsonl, tables/e7c_jvp_speedup.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.logging_utils import log_run  # noqa: E402

LOG = ROOT / "logs" / "e7c_jvp_speedup.jsonl"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

PLAYER_COUNTS = [2, 4, 6, 8, 10]


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_path(model, g_list, x_list, n_classes, chunk_size, jvp_only, repeats):
    import torch

    from cl_bargain.streams.methods import _flat_gram_and_curvature

    # one warm-up pass, so kernel autotuning and allocator growth are not timed
    _flat_gram_and_curvature(
        model, None, g_list, x_list, n_classes, chunk_size,
        need_curvature=True, jvp_only=jvp_only,
    )
    _sync(torch)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _, m = _flat_gram_and_curvature(
            model, None, g_list, x_list, n_classes, chunk_size,
            need_curvature=True, jvp_only=jvp_only,
        )
    _sync(torch)
    return (time.perf_counter() - t0) / repeats, m


def main(smoke: bool) -> list[dict]:
    import torch
    import torch.nn.functional as f

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.paramvec import flatten_grad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    nf = 8 if smoke else 20
    n_classes = 100
    batch = 16 if smoke else 64
    chunk_size = 128
    repeats = 1 if smoke else 3
    counts = PLAYER_COUNTS[:2] if smoke else PLAYER_COUNTS

    model = make_backbone(n_classes, device, nf=nf)
    p_params = int(sum(p.numel() for p in model.parameters()))
    print(f"device={device} nf={nf} params={p_params} batch={batch}")

    rows = []
    for n in counts:
        g_list, x_list = [], []
        for _ in range(n):
            x = torch.randn(batch, 3, 32, 32, device=device)
            y = torch.randint(0, n_classes, (batch,), device=device)
            model.zero_grad(set_to_none=True)
            f.cross_entropy(model(x), y).backward()
            g_list.append(flatten_grad(model).clone())
            x_list.append(x)
        model.zero_grad(set_to_none=True)

        t_ref, m_ref = time_path(model, g_list, x_list, n_classes, chunk_size, False, repeats)
        t_fast, m_fast = time_path(model, g_list, x_list, n_classes, chunk_size, True, repeats)
        rel = float(np.max(np.abs(m_fast - m_ref)) / max(np.max(np.abs(m_ref)), 1e-30))

        row = {
            "deliverable": "e7c_jvp_speedup",
            "n_players": n,
            "params": p_params,
            "batch": batch,
            "seconds_default_path": t_ref,
            "seconds_forward_only_path": t_fast,
            "speedup": t_ref / max(t_fast, 1e-12),
            "max_relative_disagreement": rel,
            "smoke": smoke,
        }
        print(row)
        log_run(LOG, row)
        rows.append(row)
        del g_list, x_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def write_table(rows: list[dict]) -> None:
    if not rows:
        return
    lines = [
        "# E7c -- curvature block: default path against the forward-only path\n",
        f"Reduced ResNet-18 ({rows[0]['params']} parameters), {rows[0]['batch']} examples "
        "per player, one curvature block per timing. Both paths compute the same array; "
        "the rightmost column is the measured disagreement between them.\n",
        "| players | default (s) | forward-only (s) | speedup | max relative disagreement |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_players']} | {r['seconds_default_path']:.3f} "
            f"| {r['seconds_forward_only_path']:.3f} | {r['speedup']:.2f}x "
            f"| {r['max_relative_disagreement']:.2e} |"
        )
    med = float(np.median([r["speedup"] for r in rows]))
    worst_gap = max(r["max_relative_disagreement"] for r in rows)
    lines += [
        "",
        f"Median speedup: **{med:.2f}x**, at a worst relative disagreement of "
        f"{worst_gap:.1e} (the two paths are algebraically identical; see "
        "tests/test_methods_torch.py).\n",
    ]
    (TABLES / "e7c_jvp_speedup.md").write_text("\n".join(lines))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    rows = main(smoke=smoke)
    write_table(rows)
    print("Wrote tables/e7c_jvp_speedup.md")
