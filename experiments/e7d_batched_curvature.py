"""E7d -- the curvature block with every player's examples in one batch.

E7c measured the forward-only block (`_gram_curvature_jvp_only`) at about 2x
the two-sided one. That path still runs n JVPs on each player's own batch, n^2
forward passes of a few dozen examples, and at that size a GPU pays launch
latency rather than arithmetic. With GroupNorm every layer acts per example, so
the players' examples can share one batch and n JVPs replace n^2
(`_gram_curvature_jvp_batched`). This file times the three paths on the
configuration the real-stream runs use (width 16, 32 examples per player) and
reports the disagreement of each fast path from the two-sided one, with TF32
off so that the comparison is between algorithms rather than precisions.

Run:  uv run python experiments/e7d_batched_curvature.py [--smoke]
Writes: logs/e7d_batched_curvature.jsonl, tables/e7d_batched_curvature.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.logging_utils import log_run  # noqa: E402

LOG = ROOT / "logs" / "e7d_batched_curvature.jsonl"
TABLES = ROOT / "tables"
PLAYER_COUNTS = [2, 4, 6, 8, 10]
PATHS = {"two-sided": {}, "forward-only": {"jvp_only": True}, "batched": {"jvp_batched": True}}


def _time(fn, repeats: int):
    import torch

    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats, out


def main(smoke: bool) -> list[dict]:
    import torch
    import torch.nn.functional as f

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.methods import _flat_gram_and_curvature
    from cl_bargain.streams.paramvec import flatten_grad

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    torch.manual_seed(0)
    nf, batch, n_classes, chunk_size = 16, 32, 100, 64
    repeats = 1 if smoke else 5
    counts = PLAYER_COUNTS[:2] if smoke else PLAYER_COUNTS
    model = make_backbone(n_classes, device, nf=nf)
    p_params = int(sum(p.numel() for p in model.parameters()))
    gpu = torch.cuda.get_device_name(0)

    rows = []
    for n in counts:
        g_list, x_list = [], []
        for _ in range(n):
            x = torch.randn(batch, 3, 32, 32, device=device)
            y = torch.randint(0, n_classes, (batch,), device=device)
            model.zero_grad(set_to_none=True)
            f.cross_entropy(model(x), y).backward()
            g = flatten_grad(model).clone()
            g_list.append(g / g.norm())
            x_list.append(x)
        model.zero_grad(set_to_none=True)
        timings, blocks = {}, {}
        for name, kw in PATHS.items():
            timings[name], (_, blocks[name]) = _time(
                lambda kw=kw, g_list=g_list, x_list=x_list: _flat_gram_and_curvature(
                    model, None, g_list, x_list, n_classes, chunk_size, True, **kw
                ),
                repeats,
            )
        ref = blocks["two-sided"]
        scale = float(np.abs(ref).max())
        row = {
            "deliverable": "e7d_batched_curvature",
            "gpu": gpu,
            "params": p_params,
            "examples_per_player": batch,
            "n_players": n,
            "seconds": timings,
            "max_rel_disagreement": {
                k: float(np.abs(blocks[k] - ref).max() / scale) for k in PATHS if k != "two-sided"
            },
            "smoke": smoke,
        }
        print(row)
        log_run(LOG, row)
        rows.append(row)
    return rows


def write_table(rows: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    r0 = rows[0]
    lines = [
        "# E7d -- curvature block: per-player against batched forward-only\n",
        f"{r0['gpu']}, reduced ResNet-18 ({r0['params']} parameters, width 16), "
        f"{r0['examples_per_player']} examples per player, FP32 (TF32 off). "
        "Disagreement is the largest entry-wise difference from the two-sided block, "
        "relative to its largest entry.\n",
        "| players | two-sided (s) | forward-only (s) | batched (s) | batched vs forward-only "
        "| batched vs two-sided | disagreement forward-only | disagreement batched |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["seconds"]
        d = r["max_rel_disagreement"]
        lines.append(
            f"| {r['n_players']} | {s['two-sided']:.3f} | {s['forward-only']:.3f} "
            f"| {s['batched']:.3f} | {s['forward-only'] / s['batched']:.1f}x "
            f"| {s['two-sided'] / s['batched']:.1f}x | {d['forward-only']:.1e} "
            f"| {d['batched']:.1e} |"
        )
    (TABLES / "e7d_batched_curvature.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    res = main(smoke="--smoke" in sys.argv)
    write_table(res)
    print("Wrote tables/e7d_batched_curvature.md")
