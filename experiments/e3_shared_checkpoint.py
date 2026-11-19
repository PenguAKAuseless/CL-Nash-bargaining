"""Compare bargain and constrained-QP steps on one shared proxy at a frozen checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.bargain.solver import solve  # noqa: E402
from cl_certify import results  # noqa: E402
from cl_certify.proposers import ProposalState  # noqa: E402
from cl_certify.reduce import reduced_problem  # noqa: E402

LOG = ROOT / "logs" / "e3_shared.jsonl"


class ProxyBuffer:
    """Fixed per-task sample of training indices standing in for the replay buffer."""

    def __init__(self, idx_by_task: dict[int, torch.Tensor]):
        self._idx = idx_by_task
        self.task_sizes = {t: int(v.numel()) for t, v in idx_by_task.items()}

    def __len__(self) -> int:
        return sum(self.task_sizes.values())

    def held(self, t: int) -> torch.Tensor:
        return self._idx[t]


def solve_qp(problem, trust_radius: float):
    from scipy.optimize import NonlinearConstraint, minimize

    g, h, tau = problem.g, problem.H, problem.tau
    cons = [NonlinearConstraint((lambda z, i=i: float(tau[i] - g[i] @ z - 0.5 * z @ h[i] @ z)),
                                0.0, np.inf, jac=(lambda z, i=i: -(g[i] + h[i] @ z)))
            for i in range(1, problem.n)]
    cons.append(NonlinearConstraint(lambda z: float(trust_radius**2 - z @ z), 0.0, np.inf,
                                    jac=lambda z: -2.0 * z))
    res = minimize(lambda z: float(g[0] @ z + 0.5 * z @ h[0] @ z), np.zeros(problem.p),
                   jac=lambda z: g[0] + h[0] @ z, constraints=cons, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-10})
    return (res.x if res.success and np.all(np.isfinite(res.x)) else None), res.message


def predicted(problem, z):
    return np.array([float(problem.g[i] @ z + 0.5 * z @ problem.H[i] @ z)
                     for i in range(problem.n)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", nargs="+", type=int, default=[5, 10, 15])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--buffer-per-task", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    # optional QP variant: explicit step radius and per-step allowance tau / steps_remaining
    ap.add_argument("--qp-trust-radius", type=float, default=None)
    ap.add_argument("--qp-steps-remaining", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.cifar import SplitCIFAR100
    from cl_bargain.streams.methods import task_accuracy_corrected
    from cl_bargain.streams.paramvec import add_flat_
    from cl_certify.runner import _data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    blob = torch.load(ROOT / "runs" / args.run / "trajectory.pt",
                      map_location="cpu", weights_only=False)
    cfg, snaps, audit, calib = (blob["config"], blob["snapshots"], blob["audit"],
                                blob["calibration"])
    stream = SplitCIFAR100(str(_data_root()), cfg["stream"], device=device,
                           class_order_seed=cfg["seed"], download=False,
                           task_masked=cfg["task_masked"])
    model = make_backbone(stream.n_classes, device, nf=cfg["backbone_width"],
                          norm=cfg["norm"])
    tau = float(cfg["tau"])
    rows = []

    for t_snap in args.tasks:
        live = t_snap + 1
        if live >= len(audit["task_range"]):
            continue
        reserved = torch.cat([v.reshape(-1) for d in (audit["idx"], calib["idx"])
                              for v in d.values()]).to("cpu")
        gen = torch.Generator(device=device).manual_seed(1000 + t_snap)
        idx_by_task = {}
        for t in range(live):
            tr = stream.tasks[t].train_idx.to("cpu")
            keep = tr[~torch.isin(tr, reserved)]
            perm = torch.randperm(keep.numel(), generator=torch.Generator().manual_seed(t))
            idx_by_task[t] = keep[perm[: args.buffer_per_task]].to(device)
        buf = ProxyBuffer(idx_by_task)

        for rep in range(args.repeats):
            model.load_state_dict({k: v.to(device) for k, v in snaps[t_snap].items()})
            model.eval()
            with torch.no_grad():
                base = {t: float(task_accuracy_corrected(
                    model, stream, stream.tasks[t].test_idx, *audit["task_range"][t]))
                    for t in range(live + 1)}

            live_idx = stream.tasks[live].train_idx
            g2 = torch.Generator(device=device).manual_seed(7000 + 97 * t_snap + rep)
            sel = live_idx[torch.randperm(live_idx.numel(), generator=g2,
                                          device=device)[: cfg["minibatch_size"]]]
            state = ProposalState(model=model, stream=stream, idx=sel, task_index=live,
                                  buffer=buf, generator=gen,
                                  minibatch_size=cfg["minibatch_size"], device=device,
                                  step=0)
            red = reduced_problem(state, tau, cfg["chunk_size"],
                                  cfg["max_active_players"], cfg["jvp_batched"],
                                  with_curvature=True, zeta_floor=cfg["zeta_floor"])
            if red is None or red.get("degenerate"):
                print(f"[e3] task {t_snap} rep {rep}: degenerate proxy, skipped")
                continue
            problem = red["problem"]

            out = solve(problem)
            z_b = np.asarray(out.Delta, dtype=float) if out.Delta is not None else None
            z_q, qp_msg = solve_qp(problem, tau)

            solves = [("bargain_curvature", z_b, problem),
                      ("constrained_qp", z_q, problem)]
            if args.qp_trust_radius is not None or args.qp_steps_remaining is not None:
                import copy

                fixed = copy.copy(problem)
                div = max(1, args.qp_steps_remaining or 1)
                fixed.tau = problem.tau / div
                radius = (args.qp_trust_radius if args.qp_trust_radius is not None
                          else tau)
                z_f, fixed_msg = solve_qp(fixed, radius)
                if z_f is None:
                    qp_msg = f"{qp_msg} | fixed: {fixed_msg}"
                solves.append(("constrained_qp_fixed", z_f, fixed))

            for method, z, prob in solves:
                if z is None:
                    rows.append({"task": t_snap, "rep": rep, "method": method,
                                 "solver_failed": True, "message": qp_msg})
                    print(f"[e3] task {t_snap} rep {rep} {method}: SOLVER FAILED")
                    continue
                # saturation is relative to the budget this solver was given
                pred = predicted(prob, z)
                sat = pred[1:] / np.maximum(prob.tau[1:], 1e-12)
                delta = red["to_full"](z)
                add_flat_(model, delta)
                model.eval()
                with torch.no_grad():
                    after = {t: float(task_accuracy_corrected(
                        model, stream, stream.tasks[t].test_idx, *audit["task_range"][t]))
                        for t in range(live + 1)}
                add_flat_(model, -delta)
                dmg = {t: base[t] - after[t] for t in range(live)}
                row = {
                    "task": t_snap, "live_task": live, "rep": rep, "method": method,
                    "solver_failed": False, "tau": tau, "n_players": problem.n - 1,
                    "tag": args.tag,
                    "step_tau_scalar": float(np.mean(prob.tau[1:])),
                    "qp_trust_radius": args.qp_trust_radius,
                    "qp_steps_remaining": args.qp_steps_remaining,
                    "delta_norm_reduced": float(np.linalg.norm(z)),
                    "delta_norm_full": float(delta.norm()),
                    "predicted_damage_max": float(pred[1:].max()),
                    "predicted_damage_mean": float(pred[1:].mean()),
                    "saturation_mean": float(sat.mean()),
                    "saturation_max": float(sat.max()),
                    "n_tight": int((sat > 0.99).sum()),
                    "true_damage_mean": float(np.mean(list(dmg.values()))),
                    "true_damage_max": float(np.max(list(dmg.values()))),
                    "progress_live": after[live] - base[live],
                    "per_task_damage": dmg,
                }
                rows.append(row)
                print(f"[e3] task {t_snap:2d} rep {rep} {method:18s} "
                      f"|z|={row['delta_norm_reduced']:.4f} "
                      f"|D|={row['delta_norm_full']:.4f} "
                      f"sat(mean/max)={row['saturation_mean']:.3f}/{row['saturation_max']:.3f} "
                      f"tight={row['n_tight']}/{problem.n - 1} "
                      f"true dmg mean={row['true_damage_mean']:+.4f} "
                      f"progress={row['progress_live']:+.4f}", flush=True)

    for r in rows:
        results.append(LOG, {"run": args.run, **r})

    ok = [r for r in rows if not r["solver_failed"]]
    lines = ["| task | method | n | |Delta| | saturation mean | saturation max | tight | "
             "true damage mean | true damage max | progress (live) |",
             "|" + "|".join("---" for _ in range(10)) + "|"]
    for t_snap in sorted({r["task"] for r in ok}):
        for method in ("bargain_curvature", "constrained_qp"):
            rs = [r for r in ok if r["task"] == t_snap and r["method"] == method]
            if not rs:
                continue
            m = lambda k, _rs=rs: np.mean([r[k] for r in _rs])  # noqa: E731
            lines.append(
                f"| {t_snap} | {method} | {len(rs)} | {m('delta_norm_full'):.4f} | "
                f"{m('saturation_mean'):.3f} | {m('saturation_max'):.3f} | "
                f"{m('n_tight'):.1f} | {m('true_damage_mean'):+.4f} | "
                f"{m('true_damage_max'):+.4f} | {m('progress_live'):+.4f} |")
    out_text = "\n".join(lines)
    (ROOT / "tables" / "e3_shared_checkpoint.md").write_text(out_text + "\n")
    print("\n" + out_text)


if __name__ == "__main__":
    main()
