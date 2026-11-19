"""Repeat one configuration at a fixed seed and report the run-to-run spread of A_20."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_certify import results  # noqa: E402
from cl_certify.config import Config  # noqa: E402
from cl_certify.runner import run  # noqa: E402

LOG = ROOT / "logs" / "determinism.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", default="replay")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--admission", default="on", choices=["on", "off"])
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.deterministic:
        # force deterministic cuDNN/cuBLAS reduction order; TF32 off for comparability
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config(
        proposer=args.proposer, tau=0.1, seed=args.seed, admission=args.admission == "on",
        tag="determinism", replay_capacity=2000, audit_heldout_n=250,
        admission_budget=0.2,
    )
    out = []
    for rep in range(args.repeats):
        t0 = time.time()
        res = run(cfg, device, log_steps=False)
        m, a = res.metrics, res.admission
        row = {
            "proposer": args.proposer, "repeat": rep,
            "device": args.device, "deterministic": args.deterministic,
            "admission": args.admission, "tag": args.tag,
            "A_final": m["A_final"], "current_task_accuracy": m["current_task_accuracy"],
            "forgetting": m["forgetting"], "n_skipped": a.get("n_skipped"),
            "accept_rate": a.get("accept_rate"), "mean_beta": a.get("mean_beta"),
            "acc_final": m["acc_final"], "wall": time.time() - t0,
            "config": cfg.as_dict(),
        }
        results.append(LOG, row)
        out.append(row)
        print(f"[det] {args.proposer} rep={rep} det={args.deterministic} "
              f"adm={args.admission} A={row['A_final']:.4f} "
              f"skipped={row['n_skipped']} accept={row['accept_rate']}", flush=True)

    a = np.array([r["A_final"] for r in out])
    print(f"[det] A_20 over {len(a)} identical repeats: "
          f"mean={a.mean():.4f} sd={a.std(ddof=1) if len(a) > 1 else 0:.4f} "
          f"range={a.max() - a.min():.4f}", flush=True)


if __name__ == "__main__":
    main()
