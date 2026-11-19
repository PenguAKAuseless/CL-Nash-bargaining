"""Build the benchmark summary from logs/e6_benchmark_split20.jsonl.

Reports the two quantities the continual-learning literature reports for a
task-incremental split: average accuracy over all tasks after the last one,
and average forgetting, the drop of each task from the accuracy it had when
its own training finished. The log stores chance-corrected accuracies, so both
are converted back with acc = corrected (1 - 1/k) + 1/k for the k classes of a
task, which is exact for the equal split used here.

Run:  uv run python experiments/make_benchmark_table.py [--log-suffix _split20]
Writes: tables/benchmark_split20.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"

NAMES = {
    "er": "replay",
    "agem": "averaged-gradient replay",
    "ewc": "regularisation (EWC)",
    "v1": "first-order bargain, $d_i=-\\tau_i$",
    "nashmtl": "first-order bargain, $d_i=0$",
    "v2": "this work",
}
ORDER = ["er", "agem", "ewc", "v1", "nashmtl", "v2"]
CLASSES_PER_TASK = 5


def raw(corrected: float, k: int = CLASSES_PER_TASK) -> float:
    return corrected * (1.0 - 1.0 / k) + 1.0 / k


def load(suffix: str) -> dict[str, list[dict]]:
    path = ROOT / "logs" / f"e6_benchmark{suffix}.jsonl"
    by_method: dict[str, list[dict]] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "method" in rec and "acc_final" in rec:
            by_method.setdefault(rec["method"], []).append(rec)
    return by_method


def summarise(recs: list[dict]) -> dict:
    acc, forg, worst, skip, wall = [], [], [], [], []
    for r in recs:
        final = {int(k): raw(v) for k, v in r["acc_final"].items()}
        at_end = {int(k): raw(v) for k, v in r["acc_at_end"].items()}
        n = len(final)
        acc.append(np.mean(list(final.values())))
        forg.append(np.mean([at_end[t] - final[t] for t in range(n - 1)]))
        worst.append(min(final.values()))
        skip.append(r["skipped_steps"] / max(r["total_steps"], 1))
        wall.append(r["wall_seconds"])
    return {
        "n_seeds": len(recs),
        "acc": (100 * np.mean(acc), 100 * np.std(acc)),
        "forgetting": (np.mean(forg), np.std(forg)),
        "worst": (100 * np.mean(worst), 100 * np.std(worst)),
        "skipped": np.mean(skip),
        "wall": np.mean(wall),
    }


def main(suffix: str) -> None:
    by_method = load(suffix)
    rows = [(m, summarise(by_method[m])) for m in ORDER if m in by_method]
    lines = [
        f"# Benchmark ({suffix.lstrip('_')})\n",
        "| method | seeds | A_20 (%) | forgetting | worst-task acc (%) | skipped | wall (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, v in rows:
        lines.append(
            f"| {m} | {v['n_seeds']} | {v['acc'][0]:.1f} +/- {v['acc'][1]:.1f} "
            f"| {v['forgetting'][0]:.3f} +/- {v['forgetting'][1]:.3f} "
            f"| {v['worst'][0]:.1f} +/- {v['worst'][1]:.1f} "
            f"| {v['skipped']:.1%} | {v['wall']:.0f} |"
        )
    (TABLES / f"benchmark{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    print("\nSummary rows:")
    for m, v in rows:
        print(
            f"{NAMES[m]:<34} & ${v['acc'][0]:.1f}\\pm{v['acc'][1]:.1f}$ "
            f"& ${v['forgetting'][0]:+.3f}$ & ${v['worst'][0]:.1f}\\pm{v['worst'][1]:.1f}$ "
            f"& ${v['skipped']:.1%}$".replace("%", "\\%")
            + f" & ${v['wall']:.0f}$ \\\\"
        )


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--log-suffix") + 1] if "--log-suffix" in sys.argv else "_split20")
