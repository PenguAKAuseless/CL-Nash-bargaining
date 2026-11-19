"""Train one cell per proposer and record per-phase wall time."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_certify import results  # noqa: E402
from cl_certify.config import Config  # noqa: E402
from cl_certify.runner import run  # noqa: E402

LOG = ROOT / "logs" / "e5_walltime.jsonl"
RUNS = ROOT / "runs"

# cheapest first, so a partial table is still informative
ORDER = ["replay", "derpp", "replay_ggn", "bargain_first_order",
         "constrained_qp", "bargain_curvature"]
TAU_IN_PROPOSER = frozenset({"constrained_qp", "bargain_first_order", "bargain_curvature"})

TRUST_RADIUS = 0.65
TAU = 0.4

# baselines run at their tuned hyper-parameters, not the dataclass defaults
TUNED = {
    "derpp": {"derpp_alpha": 1.0, "derpp_beta": 1.0},
    "replay_ggn": {"ggn_mu": 0.1},
}

PHASES = ["gradient", "curvature", "solver", "certify", "admission"]
SCHEMA_VERSION = 1

COLUMNS = ["proposer", "wall_min", "sum_phases_min", "sync_overhead_min",
           *[f"{p}_min" for p in PHASES], "other_min",
           *[f"{p}_pct" for p in PHASES], "A_final", "total_steps"]
HEADERS = ["proposer", "wall (min)", "phases (min)", "sync (min)",
           "grad", "curv", "solver", "certify", "admit", "other",
           "grad %", "curv %", "solver %", "certify %", "admit %",
           "A_20", "steps"]
FMT = {c: "{:.1f}" for c in COLUMNS if c.endswith("_min")}
FMT.update({c: "{:.1f}" for c in COLUMNS if c.endswith("_pct")})
FMT["A_final"] = "{:.4f}"


def build_table() -> None:
    rows = [r for r in results.read(LOG) if not r.get("failed")]
    rows.sort(key=lambda r: r.get("wall_min", 0.0))
    if rows:
        print(results.to_markdown(rows, ROOT / "tables" / "e5_walltime.md",
                                  COLUMNS, HEADERS, FMT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", nargs="+", default=ORDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--table-only", action="store_true")
    # timing is env-driven and not in the config, so timed and untimed cells share a fingerprint
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--ggn-mu", type=float, default=None)
    args = ap.parse_args()

    if args.table_only:
        build_table()
        return

    if not args.no_timing and not os.environ.get("CL_TIMING"):
        raise SystemExit("set CL_TIMING=1, or pass --no-timing for a seed replicate")

    from cl_bargain.streams.cifar import SplitCIFAR100
    from cl_certify.runner import _data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    done = {r.get("fingerprint") for r in results.read(LOG)}
    stream = SplitCIFAR100(str(_data_root()), "split20", device=device,
                           class_order_seed=args.seed, download=False, task_masked=True)

    for name in args.proposers:
        tuned = dict(TUNED.get(name, {}))
        if args.ggn_mu is not None and "ggn_mu" in tuned:
            tuned["ggn_mu"] = args.ggn_mu
        cfg = Config(
            proposer=name,
            tau=(TAU if name in TAU_IN_PROPOSER else 0.0),
            seed=args.seed,
            admission=False,
            trust_radius=(TRUST_RADIUS if name in TAU_IN_PROPOSER else None),
            qp_cumulative_budget=(name == "constrained_qp"),
            budget_mode=("per_step" if name == "constrained_qp" else "cumulative"),
            replay_capacity=2000,
            tag="e5",
            extra={"save_snapshots": True, "branch": "ltt"},
            **tuned,
        )
        fp = cfg.fingerprint()
        if fp in done:
            print(f"[e5] skip {name} (done)", flush=True)
            continue
        try:
            res = run(cfg, device, stream=stream)
        except Exception:
            traceback.print_exc()
            results.append(LOG, {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fp, "proposer": name, "failed": True,
                "config": cfg.as_dict(), "error": traceback.format_exc()[-2000:]})
            continue

        out_dir = RUNS / f"{name}_seed{args.seed}_tau{cfg.tau:g}_{fp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"snapshots": res.snapshots, "audit": res.audit,
                    "calibration": res.calibration, "config": cfg.as_dict()},
                    out_dir / "trajectory.pt")

        t = res.metrics.get("timing") or {"seconds": {}, "calls": {}}
        secs = t["seconds"]
        wall = float(res.metrics["wall_seconds"])
        total_phases = sum(secs.get(p, 0.0) for p in PHASES)
        row = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fp, "proposer": name, "failed": False,
            "config": cfg.as_dict(), "trajectory_dir": str(out_dir.relative_to(ROOT)),
            "wall_min": wall / 60.0,
            "sum_phases_min": total_phases / 60.0,
            # uninstrumented work plus the cost of phase synchronisation
            "sync_overhead_min": (wall - total_phases) / 60.0,
            "other_min": max(0.0, wall - total_phases) / 60.0,
            **{f"{p}_min": secs.get(p, 0.0) / 60.0 for p in PHASES},
            **{f"{p}_pct": 100.0 * secs.get(p, 0.0) / max(wall, 1e-9) for p in PHASES},
            **{f"{p}_calls": t["calls"].get(p, 0) for p in PHASES},
            "A_final": res.metrics["A_final"],
            "total_steps": res.metrics["total_steps"],
            "peak_memory_mb": res.metrics.get("peak_memory_mb"),
        }
        results.append(LOG, row)
        print(f"[e5] {name:>20s} wall={row['wall_min']:7.1f}min "
              f"grad={row['gradient_min']:6.1f} curv={row['curvature_min']:6.1f} "
              f"solver={row['solver_min']:6.1f} certify={row['certify_min']:5.1f} "
              f"A={row['A_final']:.3f}", flush=True)
        build_table()

    build_table()


if __name__ == "__main__":
    main()
