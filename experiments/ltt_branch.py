"""Train without admission, then certify by fixed-sequence testing over a candidate pool."""

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
from cl_certify.bernstein import empirical_bernstein_upper  # noqa: E402
from cl_certify.config import Config  # noqa: E402
from cl_certify.ltt import (  # noqa: E402
    calibration_score_fn,
    candidate_pool,
    fixed_sequence_select,
    interpolate,
    order_by_calibration,
    weight_average,
)
from cl_certify.runner import _per_example_01, run  # noqa: E402

LOG = ROOT / "logs" / "ltt.jsonl"
RUNS = ROOT / "runs"
TAUS = [0.05, 0.1, 0.4]

# both candidate families are weight averages that include theta_final
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAIL_KS = [2, 3, 5, 8, 12, 20]

# schema 1 rows lack per-candidate accuracy fields
SCHEMA_VERSION = 2

TAU_IN_PROPOSER = frozenset({"constrained_qp", "bargain_first_order", "bargain_curvature"})

COLUMNS = [
    "proposer", "tau", "seed", "abstained", "selected_name", "n_tested",
    "selected_bound", "selected_A_final", "selected_last_task_accuracy",
    "selected_mean_old_task_accuracy", "A_final_uncertified",
    "price_of_certification", "min_bound_diagnostic", "wall_seconds",
]
HEADERS = [
    "proposer", "tau", "seed", "abstain", "selected", "tested",
    "bound", "A_20", "last-task", "old-task", "A_20 uncert",
    "price", "min bound (diag)", "wall (s)",
]
FMT = dict.fromkeys(
    ("selected_bound", "selected_A_final", "selected_last_task_accuracy",
     "selected_mean_old_task_accuracy", "A_final_uncertified",
     "price_of_certification", "min_bound_diagnostic"), "{:.4f}")
FMT["wall_seconds"] = "{:.0f}"


def build(cand, snapshots, anchors_mean, sd_final):
    if cand.family == "anchor_pull":
        return interpolate(anchors_mean, sd_final, cand.param)
    if cand.family == "tail_average":
        k = int(cand.param)
        keys = sorted(t for t in snapshots if isinstance(t, int))[-k:]
        return weight_average([snapshots[t] for t in keys] + [sd_final])
    raise ValueError(f"unknown family {cand.family!r}")


def measure(model, stream, sd, audit, device, calibration):
    """Return audit differences, calibration differences and per-task test accuracies."""
    from cl_bargain.streams.methods import task_accuracy_corrected

    model.load_state_dict(sd)
    model.eval()
    n_tasks = len(audit["task_range"])
    with torch.no_grad():
        aud = np.concatenate([
            _per_example_01(model, stream, idx.to(device), *audit["task_range"][t])
            - audit["anchor01"][t]
            for t, idx in sorted(audit["idx"].items())
        ]) if audit["idx"] else np.zeros(0)
        cal = np.concatenate([
            _per_example_01(model, stream, idx.to(device), *audit["task_range"][t])
            - calibration["anchor01"][t]
            for t, idx in sorted(calibration["idx"].items())
        ]) if calibration["idx"] else np.zeros(0)
        accs = {
            t: task_accuracy_corrected(model, stream, stream.tasks[t].test_idx,
                                       *audit["task_range"][t])
            for t in range(n_tasks)
        }
    return aud, cal, accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", nargs="+", default=["replay"])
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replay-capacity", type=int, default=2000)
    ap.add_argument("--table-only", action="store_true")
    # overrides are Config fields, so they enter the fingerprint and run directory name
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ggn-mu", type=float, default=None)
    ap.add_argument("--derpp-alpha", type=float, default=None)
    ap.add_argument("--derpp-beta", type=float, default=None)
    ap.add_argument("--tag", default="ltt")
    args = ap.parse_args()

    if not args.table_only:
        from cl_bargain.streams.backbone import make_backbone
        from cl_bargain.streams.cifar import SplitCIFAR100
        from cl_certify.runner import _data_root

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        done = {r.get("fingerprint") for r in results.read(LOG)}
        streams: dict[int, object] = {}

        jobs = []
        for seed in args.seeds:
            for name in args.proposers:
                # tau does not affect replay-family training, so one run is certified at every tau
                if name in TAU_IN_PROPOSER:
                    jobs.extend((name, seed, [t]) for t in args.taus)
                else:
                    jobs.append((name, seed, list(args.taus)))

        for name, seed, taus in jobs:
            over = {k: v for k, v in (("lr", args.lr), ("ggn_mu", args.ggn_mu),
                                       ("derpp_alpha", args.derpp_alpha),
                                       ("derpp_beta", args.derpp_beta))
                    if v is not None}
            cfg = Config(
                proposer=name, tau=(taus[0] if name in TAU_IN_PROPOSER else 0.0),
                seed=seed, smoke=args.smoke, admission=False, tag=args.tag,
                replay_capacity=args.replay_capacity,
                extra={"save_snapshots": True, "branch": "ltt"},
                **over,
            )
            fp = cfg.fingerprint()
            if all(f"{fp}-tau{t}" in done for t in taus):
                print(f"[ltt] skip {name} seed={seed} taus={taus} (done)", flush=True)
                continue
            stream = streams.get(seed)
            if stream is None:
                stream = streams[seed] = SplitCIFAR100(
                    str(_data_root()), "split20", device=device,
                    class_order_seed=seed, download=False, task_masked=True,
                )
            try:
                res = run(cfg, device, stream=stream)
            except Exception:
                traceback.print_exc()
                results.append(LOG, {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": fp, "proposer": name, "seed": seed,
                    "failed": True, "config": cfg.as_dict(),
                    "error": traceback.format_exc()[-2000:]})
                continue

            # saved trajectory allows offline re-selection without retraining
            out_dir = RUNS / f"{name}_seed{seed}_tau{cfg.tau:g}_{fp}"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"snapshots": res.snapshots, "audit": res.audit,
                        "calibration": res.calibration, "config": cfg.as_dict()},
                        out_dir / "trajectory.pt")

            n_tasks = len(res.audit["task_range"])
            sd_final = res.snapshots["final"]
            anchors = [res.snapshots[t] for t in sorted(
                t for t in res.snapshots if isinstance(t, int))]
            anchors_mean = weight_average(anchors)
            probe = make_backbone(stream.n_classes, device,
                                  nf=cfg.backbone_width, norm=cfg.norm)

            pool = candidate_pool(BETAS, TAIL_KS)
            cache: dict[str, tuple] = {}

            def evaluate(c, _s=stream, _r=res, _p=probe, _c=cache,
                         _am=anchors_mean, _sf=sd_final):
                if c.name not in _c:
                    sd = build(c, _r.snapshots, _am, _sf)
                    sd = {k: v.to(device) for k, v in sd.items()}
                    _c[c.name] = measure(_p, _s, sd, _r.audit, device, _r.calibration)
                return _c[c.name]

            # ordering reads calibration only, never the audit slice
            ladder = order_by_calibration(
                pool, calibration_score_fn(evaluate, cfg.delta))

            def accrow(c, _e=evaluate, _n=n_tasks):
                _, _, accs = _e(c)
                return {
                    "A_final": float(np.mean(list(accs.values()))),
                    "last_task_accuracy": float(accs[_n - 1]),
                    "mean_old_task_accuracy": float(
                        np.mean([accs[t] for t in range(_n - 1)])),
                }

            for tau in taus:
                def bound_fn(c, _delta=cfg.delta, _e=evaluate):
                    # fixed-sequence testing spends no multiplicity, hence n_checkpoints=1
                    b = empirical_bernstein_upper(_e(c)[0], _delta, 0.0, 1.0,
                                                  n_checkpoints=1)
                    return b.bound, b.mean

                out = fixed_sequence_select(ladder, bound_fn, tau, evaluate_all=True)
                cands = [
                    {"name": c.name, "family": c.family, "param": c.param,
                     "rank_score_calibration": c.rank_score,
                     "audit_bound": out.bounds[i], "audit_mean": out.means[i],
                     **accrow(c)}
                    for i, c in enumerate(ladder) if i < len(out.bounds)
                ]
                sel = accrow(out.selected) if out.selected else {}
                unc = accrow(next(c for c in ladder if c.name == "anchor_pull@1.00"))

                results.append(LOG, {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": f"{fp}-tau{tau}", "proposer": name, "tau": tau,
                    "seed": seed, "failed": False, "config": cfg.as_dict(),
                    "trajectory_dir": str(out_dir.relative_to(ROOT)),
                    "abstained": out.selected is None,
                    "selected_name": out.selected.name if out.selected else None,
                    "n_tested": out.n_tested, "monotone": out.monotone,
                    "candidates": cands,
                    "selected_bound": (out.bounds[out.selected_index]
                                       if out.selected_index >= 0 else None),
                    "selected_A_final": sel.get("A_final"),
                    "selected_last_task_accuracy": sel.get("last_task_accuracy"),
                    "selected_mean_old_task_accuracy": sel.get("mean_old_task_accuracy"),
                    "A_final_uncertified": unc["A_final"],
                    "price_of_certification": (
                        unc["A_final"] - sel["A_final"] if sel else None),
                    # diagnostic only: min over candidates chosen post hoc, not certified
                    "min_bound_diagnostic": float(min(out.bounds)),
                    "wall_seconds": res.metrics["wall_seconds"],
                })
                sel_b = (out.bounds[out.selected_index]
                         if out.selected_index >= 0 else float("nan"))
                sel_name = out.selected.name if out.selected else "ABSTAIN"
                print(f"[ltt] {name:>20s} tau={tau:<5g} seed={seed} "
                      f"sel={sel_name:>16s} bound={sel_b:.4f} "
                      f"A={sel.get('A_final', float('nan')):.3f} "
                      f"minb(diag)={min(out.bounds):.4f}", flush=True)

    rows = [r for r in results.read(LOG) if not r.get("failed")]
    rows.sort(key=lambda r: (r["tau"], r["proposer"], r["seed"]))
    n_abs = sum(bool(r.get("abstained")) for r in rows)
    if rows:
        print(f"[ltt] abstain rate: {n_abs}/{len(rows)} = {n_abs / len(rows):.3f}")
    results.to_csv(rows, ROOT / "tables" / "ltt.csv", COLUMNS)
    print(results.to_markdown(rows, ROOT / "tables" / "ltt.md", COLUMNS, HEADERS, FMT))


if __name__ == "__main__":
    main()
