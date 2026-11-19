"""Tune the curvature bargain over a fixed grid, selecting on calibration accuracy only."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_certify import results  # noqa: E402
from cl_certify.config import Config  # noqa: E402
from cl_certify.ltt import (  # noqa: E402
    calibration_score_fn,
    candidate_pool,
    order_by_calibration,
    weight_average,
)
from cl_certify.runner import run  # noqa: E402

LOG = ROOT / "logs" / "a_tune.jsonl"
RUNS = ROOT / "runs"

TRUST_RADIUS = 0.65
BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAIL_KS = [2, 3, 5, 8, 12, 20]
SCHEMA_VERSION = 1

GRID = [
    {"zeta_floor": 0.01, "tau": 0.4},
    {"zeta_floor": 0.01, "tau": 1.0},
    {"zeta_floor": 0.01, "tau": 4.0},
    {"zeta_floor": 0.10, "tau": 1.0},
    {"zeta_floor": 0.001, "tau": 1.0},
]

COLUMNS = ["proposer", "zeta_floor", "tau", "seed", "calibration_accuracy",
           "calibration_best_candidate", "A_final_theta_final", "A_final_best_candidate",
           "mean_delta_norm", "radius_binding_rate", "wall_min"]
HEADERS = ["proposer", "zeta", "train_tau", "seed", "CAL acc (selects)",
           "cal-best cand", "A_20 theta_final", "A_20 cand", "mean ||D||",
           "radius bind", "wall (min)"]
FMT = dict.fromkeys(
    ("calibration_accuracy", "A_final_theta_final", "A_final_best_candidate",
     "mean_delta_norm", "radius_binding_rate"), "{:.4f}")
FMT["wall_min"] = "{:.1f}"


def build(cand, snapshots, anchors_mean, sd_final):
    from cl_certify.ltt import interpolate

    if cand.family == "anchor_pull":
        return interpolate(anchors_mean, sd_final, cand.param)
    k = int(cand.param)
    keys = sorted(t for t in snapshots if isinstance(t, int))[-k:]
    return weight_average([snapshots[t] for t in keys] + [sd_final])


def measure(model, stream, sd, audit, device, calibration):
    """Return calibration differences, calibration accuracy and per-task test accuracies."""
    from cl_bargain.streams.methods import task_accuracy_corrected
    from cl_certify.runner import _per_example_01

    model.load_state_dict(sd)
    model.eval()
    n_tasks = len(audit["task_range"])
    with torch.no_grad():
        cal = np.concatenate([
            _per_example_01(model, stream, idx.to(device), *audit["task_range"][t])
            - calibration["anchor01"][t]
            for t, idx in sorted(calibration["idx"].items())
        ]) if calibration["idx"] else np.zeros(0)
        cal_acc = float(np.mean(np.concatenate([
            1.0 - _per_example_01(model, stream, idx.to(device), *audit["task_range"][t])
            for t, idx in sorted(calibration["idx"].items())
        ]))) if calibration["idx"] else float("nan")
        accs = {
            t: task_accuracy_corrected(model, stream, stream.tasks[t].test_idx,
                                       *audit["task_range"][t])
            for t in range(n_tasks)
        }
    return cal, cal_acc, accs


def build_table() -> None:
    rows = [r for r in results.read(LOG) if not r.get("failed")]
    rows.sort(key=lambda r: (r["seed"], r["tau"], r["zeta_floor"]))
    if rows:
        print(results.to_markdown(rows, ROOT / "tables" / "a_tune.md",
                                  COLUMNS, HEADERS, FMT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-best", action="store_true",
                    help="run only the grid point chosen on seed 0's calibration")
    ap.add_argument("--points", nargs="+", type=int, default=None,
                    help="1-based indices into GRID, to split the grid across GPUs")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        build_table()
        return

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.cifar import SplitCIFAR100
    from cl_certify.runner import _data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [r for r in results.read(LOG) if not r.get("failed")]
    done = {r.get("fingerprint") for r in results.read(LOG)}

    grid = [GRID[i - 1] for i in args.points] if args.points else GRID
    if args.only_best:
        seed0 = [r for r in rows if r["seed"] == 0]
        if len(seed0) < len(GRID):
            raise SystemExit(f"seed 0 grid incomplete ({len(seed0)}/{len(GRID)})")
        best = max(seed0, key=lambda r: r["calibration_accuracy"])
        grid = [{"zeta_floor": best["zeta_floor"], "tau": best["tau"]}]
        print(f"[a] chosen on calibration: zeta={best['zeta_floor']:g} "
              f"tau={best['tau']:g} cal_acc={best['calibration_accuracy']:.4f}",
              flush=True)

    streams: dict[int, object] = {}
    for seed in args.seeds:
        for point in grid:
            cfg = Config(
                proposer="bargain_curvature", seed=seed, admission=False,
                trust_radius=TRUST_RADIUS, replay_capacity=2000, tag="atune",
                extra={"save_snapshots": True, "branch": "ltt"},
                **point,
            )
            fp = cfg.fingerprint()
            if fp in done:
                print(f"[a] skip zeta={point['zeta_floor']:g} tau={point['tau']:g} "
                      f"seed={seed} (done)", flush=True)
                continue
            stream = streams.get(seed)
            if stream is None:
                stream = streams[seed] = SplitCIFAR100(
                    str(_data_root()), "split20", device=device,
                    class_order_seed=seed, download=False, task_masked=True)
            try:
                res = run(cfg, device, stream=stream, log_steps=True)
            except Exception:
                traceback.print_exc()
                results.append(LOG, {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": fp, "failed": True, "config": cfg.as_dict(),
                    "error": traceback.format_exc()[-2000:]})
                continue

            out_dir = RUNS / f"bargain_curvature_seed{seed}_tau{cfg.tau:g}_{fp}"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"snapshots": res.snapshots, "audit": res.audit,
                        "calibration": res.calibration, "config": cfg.as_dict()},
                        out_dir / "trajectory.pt")

            sd_final = res.snapshots["final"]
            anchors_mean = weight_average([
                res.snapshots[t] for t in sorted(
                    t for t in res.snapshots if isinstance(t, int))])
            probe = make_backbone(stream.n_classes, device, nf=cfg.backbone_width,
                                  norm=cfg.norm)
            cache: dict[str, tuple] = {}

            def evaluate(c, _s=stream, _r=res, _p=probe, _c=cache,
                         _am=anchors_mean, _sf=sd_final):
                if c.name not in _c:
                    sd = {k: v.to(device)
                          for k, v in build(c, _r.snapshots, _am, _sf).items()}
                    _c[c.name] = measure(_p, _s, sd, _r.audit, device, _r.calibration)
                return _c[c.name]

            pool = candidate_pool(BETAS, TAIL_KS)
            ladder = order_by_calibration(pool, calibration_score_fn(evaluate, cfg.delta))
            # selection reads calibration only; audit and test are recorded, not used
            top = ladder[0]
            _, cal_acc_top, accs_top = evaluate(top)
            theta_final = next(c for c in ladder if c.name == "anchor_pull@1.00")
            _, cal_acc_final, accs_final = evaluate(theta_final)

            norms = [r["delta_norm"] for r in res.step_rows
                     if r.get("proposed") and r.get("delta_norm") is not None]
            results.append(LOG, {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": fp, "proposer": "bargain_curvature", "failed": False,
                "zeta_floor": point["zeta_floor"], "tau": point["tau"], "seed": seed,
                "config": cfg.as_dict(), "trajectory_dir": str(out_dir.relative_to(ROOT)),
                "calibration_accuracy": cal_acc_top,
                "calibration_best_candidate": top.name,
                "calibration_accuracy_theta_final": cal_acc_final,
                "A_final_best_candidate": float(np.mean(list(accs_top.values()))),
                "A_final_theta_final": float(np.mean(list(accs_final.values()))),
                "mean_delta_norm": float(np.mean(norms)) if norms else None,
                "radius_binding_rate": None,
                "wall_min": res.metrics["wall_seconds"] / 60.0,
                "selection_statistic": "calibration_accuracy",
            })
            print(f"[a] zeta={point['zeta_floor']:<6g} tau={point['tau']:<5g} "
                  f"seed={seed} CALacc={cal_acc_top:.4f} ({top.name}) "
                  f"A_20={np.mean(list(accs_top.values())):.4f}", flush=True)
            build_table()

    build_table()


if __name__ == "__main__":
    main()
