"""Certified model selection offline from saved trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_certify import results  # noqa: E402
from cl_certify.bernstein import empirical_bernstein_upper  # noqa: E402
from cl_certify.ltt import (  # noqa: E402
    calibration_score_fn,
    candidate_pool,
    fixed_sequence_select,
    interpolate,
    order_by_calibration,
    weight_average,
)
from cl_certify.runner import _per_example_01  # noqa: E402

LOG = ROOT / "logs" / "ltt_select.jsonl"
# beta = 0 is excluded: the plain anchor mean gives the last task only 1/T of the weight
BETA_FLOOR = 0.1
BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAIL_KS = [2, 3, 5, 8, 12, 20]
TAUS = [0.05, 0.1, 0.4]
SCHEMA_VERSION = 4


def build(cand, snapshots, anchors_mean, sd_final):
    if cand.family == "anchor_pull":
        return interpolate(anchors_mean, sd_final, cand.param)
    if cand.family == "tail_average":
        k = int(cand.param)
        keys = sorted(t for t in snapshots if isinstance(t, int))[-k:]
        return weight_average([snapshots[t] for t in keys] + [sd_final])
    raise ValueError(f"unknown family {cand.family!r}")


def measure(model, stream, sd, audit, calibration, device):
    """Return audit differences, calibration differences, calibration accuracy and test accuracy."""
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
        cal_err = {
            t: _per_example_01(model, stream, idx.to(device), *audit["task_range"][t])
            for t, idx in sorted(calibration["idx"].items())
        }
        cal = np.concatenate([
            cal_err[t] - calibration["anchor01"][t] for t in sorted(cal_err)
        ]) if cal_err else np.zeros(0)
        test = {
            t: task_accuracy_corrected(model, stream, stream.tasks[t].test_idx,
                                       *audit["task_range"][t])
            for t in range(n_tasks)
        }
    cal_acc = {t: 1.0 - float(v.mean()) for t, v in cal_err.items()}
    return aud, cal, cal_acc, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.cifar import SplitCIFAR100
    from cl_certify.runner import _data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dirs = ([ROOT / d for d in args.runs] if args.runs
            else sorted((ROOT / "runs").glob("*/")))
    streams: dict[int, object] = {}
    done = {r.get("fingerprint") for r in results.read(LOG)}

    for d in dirs:
        traj = d / "trajectory.pt"
        if not traj.exists():
            continue
        blob = torch.load(traj, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        key = d.name
        if all(f"{key}-{rule}-tau{t}" in done
               for t in args.taus for rule in ("fixed_sequence", "best_calibration")):
            print(f"[select] skip {key} (done)", flush=True)
            continue
        seed = cfg["seed"]
        stream = streams.get(seed)
        if stream is None:
            stream = streams[seed] = SplitCIFAR100(
                str(_data_root()), cfg["stream"], device=device,
                class_order_seed=seed, download=False, task_masked=cfg["task_masked"],
            )
        snapshots, audit, calibration = blob["snapshots"], blob["audit"], blob["calibration"]
        n_tasks = len(audit["task_range"])
        sd_final = snapshots["final"]
        anchors_mean = weight_average(
            [snapshots[t] for t in sorted(t for t in snapshots if isinstance(t, int))])
        probe = make_backbone(stream.n_classes, device,
                              nf=cfg["backbone_width"], norm=cfg["norm"])
        pool = candidate_pool(BETAS, TAIL_KS)
        cache: dict[str, tuple] = {}

        def evaluate(c, _s=stream, _p=probe, _c=cache, _sn=snapshots,
                     _am=anchors_mean, _sf=sd_final, _a=audit, _cal=calibration):
            if c.name not in _c:
                sd = {k: v.to(device) for k, v in build(c, _sn, _am, _sf).items()}
                _c[c.name] = measure(_p, _s, sd, _a, _cal, device)
            return _c[c.name]

        ladder = order_by_calibration(pool, calibration_score_fn(evaluate, cfg["delta"]))

        def row_for(c, _e=evaluate, _n=n_tasks):
            _aud, _cal, cal_acc, test = _e(c)
            return {
                "name": c.name, "family": c.family, "param": c.param,
                "rank_score_calibration": c.rank_score,
                "calibration_accuracy": float(np.mean(list(cal_acc.values()))),
                "calibration_accuracy_old": float(
                    np.mean([cal_acc[t] for t in sorted(cal_acc) if t < _n - 1])),
                "A_final": float(np.mean(list(test.values()))),
                "last_task_accuracy": float(test[_n - 1]),
                "mean_old_task_accuracy": float(
                    np.mean([test[t] for t in range(_n - 1)])),
            }

        for tau in args.taus:
            def bound_fn(c, _d=cfg["delta"], _e=evaluate):
                b = empirical_bernstein_upper(_e(c)[0], _d, 0.0, 1.0, n_checkpoints=1)
                return b.bound, b.mean

            out = fixed_sequence_select(ladder, bound_fn, tau, evaluate_all=True)
            # simultaneous comparison bound: union bound at delta/K over the whole ladder
            k_simul = len(ladder)
            cands = []
            for i, c in enumerate(ladder):
                if i >= len(out.bounds):
                    break
                b1 = empirical_bernstein_upper(evaluate(c)[0], cfg["delta"], 0.0, 1.0,
                                               n_checkpoints=1)
                bk = empirical_bernstein_upper(evaluate(c)[0], cfg["delta"], 0.0, 1.0,
                                               n_checkpoints=k_simul)
                cands.append({**row_for(c), "audit_bound": out.bounds[i],
                              "audit_mean": out.means[i],
                              "audit_var": b1.var, "audit_m": b1.n,
                              "audit_bound_simultaneous": bk.bound,
                              "k_simultaneous": k_simul,
                              "certified_simultaneous": bool(bk.bound <= tau),
                              "certified": bool(i <= out.selected_index)})
            certified = [c for c in cands if c["certified"]]

            picks = {"fixed_sequence": cands[out.selected_index] if certified else None}
            picks["best_calibration"] = (
                max(certified, key=lambda r: (r["calibration_accuracy"], r["name"]))
                if certified else None)

            for rule, pick in picks.items():
                results.append(LOG, {
                    "schema_version": SCHEMA_VERSION,
                    "fingerprint": f"{key}-{rule}-tau{tau}", "run_key": key,
                    "rule": rule, "proposer": cfg["proposer"], "tau": tau,
                    "seed": seed, "config_tau": cfg["tau"],
                    "abstained": pick is None, "n_certified": len(certified),
                    "n_tested": out.n_tested, "n_candidates": len(cands),
                    "selected": pick, "candidates": cands if rule == "fixed_sequence" else None,
                    "uncertified_A_final": next(
                        (c["A_final"] for c in cands if c["name"] == "anchor_pull@1.00"), None),
                    "min_bound_diagnostic": float(min(out.bounds)),
                })
            fs, bc = picks["fixed_sequence"], picks["best_calibration"]
            print(
                f"[select] {cfg['proposer']:>20s} tau={tau:<5g} seed={seed} "
                f"certified={len(certified):2d}/{len(cands)} "
                f"fs={fs['name'] if fs else 'ABSTAIN':>16s} "
                f"A={fs['A_final'] if fs else float('nan'):.3f} | "
                f"bestcal={bc['name'] if bc else 'ABSTAIN':>16s} "
                f"A={bc['A_final'] if bc else float('nan'):.3f}",
                flush=True)


if __name__ == "__main__":
    main()
