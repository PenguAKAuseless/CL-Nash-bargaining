"""A GPU-pinned job pool for the benchmark sweep.

The method's cost is dominated by the curvature block, which is a sequence of
small forward-mode passes over a 1.1M-parameter backbone. That workload is
launch-latency bound rather than throughput bound: a single run was measured
holding one GPU at roughly a quarter of its utilisation and under half a
gigabyte of memory. Two consequences follow, and this file exists to exploit
both:

  * several runs fit on one device at once, on memory by a wide margin and on
    compute because each leaves the device idle most of the time;
  * runs are independent, so the only coordination needed is to hand each
    worker a disjoint slice of the sweep and let the experiment scripts' own
    resume logic skip anything already finished.

Each job is a full command line for an existing experiment script, so nothing
about the experiments themselves changes; they simply see CUDA_VISIBLE_DEVICES
restricted to one device. Jobs that fail are reported and do not stop the pool,
and a killed pool can be restarted: every experiment script keys its resume on
the records already in its own log.

Examples
--------
Gate ablation, five seeds, split over two GPUs with two workers each:

    uv run python experiments/run_sweep.py --sweep gate --gpus 0,1 --procs-per-gpu 2

Dry run (print the plan and the cost estimate, execute nothing):

    uv run python experiments/run_sweep.py --sweep full --gpus 0,1 --procs-per-gpu 3 --dry-run
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Measured single-run wall clock on the development GPU (an 8GB laptop part),
# used only to print an estimate before a long sweep starts. Sources:
# logs/e2_frontier.jsonl for the uncapped numbers, logs/e6_benchmark.jsonl for
# the capped v2 and nashmtl numbers.
MEASURED_SECONDS = {
    "er": 40.0,
    "agem": 74.0,
    "ewc": 270.0,
    "v1": 274.0,
    "nashmtl": 418.0,
    "v2": 5953.0,  # capped at four active players
    "v2_uncapped": 13990.0,
}
# The forward-only curvature path is exact and measured at just over 2x on the
# same device (tables/e7c_jvp_speedup.md).
JVP_ONLY_SPEEDUP = 2.1

SWEEPS = {
    # Feasibility check for the pool itself: two tiny runs, so that GPU
    # pinning, resume and failure reporting are exercised before a sweep that
    # takes hours is launched.
    "smoke": {
        "methods": ["nashmtl", "v2"],
        "seeds": [0],
        "extra": ["--gate-only", "--smoke", "--jvp-batched", "--log-suffix", "_smoke"],
    },
    # The benchmark of Section "Continual-Learning Benchmarks": 20 tasks of 5
    # classes, task-incremental (each example competes only with its own
    # task's classes), 2000 buffered examples for every method, 5 epochs per
    # task, three seeds, one knob value per method. v2 first, since it is the
    # longest run.
    "bench": {
        "methods": ["v2", "ewc", "v1", "nashmtl", "agem", "er"],
        "seeds": [0, 1, 2],
        "extra": ["--jvp-batched", "--stream", "split20", "--buffer", "2000",
                  "--epochs", "5", "--log-suffix", "_split20"],
    },
}


def build_jobs(sweep: str) -> list[list[str]]:
    spec = SWEEPS[sweep]
    jobs = []
    # Method-major, in the listed order, so the longest runs start first.
    for method in spec["methods"]:
        for seed in spec["seeds"]:
            jobs.append(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "e6_benchmark.py"),
                    "--methods",
                    method,
                    "--seeds",
                    str(seed),
                    *spec["extra"],
                ]
            )
    return jobs


def estimate_seconds(sweep: str, n_workers: int, jvp_only: bool) -> tuple[float, float]:
    """(sequential seconds on one device, wall-clock seconds with n_workers).

    The parallel figure assumes workers share devices without perfect scaling;
    the discount below is deliberately conservative rather than optimistic.
    """
    spec = SWEEPS[sweep]
    if "--smoke" in spec["extra"] or "--jvp-batched" in spec["extra"]:
        return 0.0, 0.0  # no measured single-run time for this configuration
    total = 0.0
    for method in spec["methods"]:
        per_run = MEASURED_SECONDS[method]
        if method == "v2" and jvp_only:
            # only the curvature block speeds up, and it is essentially all of
            # v2's cost (tables/e7c_jvp_speedup.md, and the n=10 block timing
            # times the step count reproduces the measured run length)
            per_run /= JVP_ONLY_SPEEDUP
        total += per_run * len(spec["seeds"])
    efficiency = 0.75  # measured contention allowance for co-resident workers
    return total, total / max(n_workers * efficiency, 1e-9)


def run_pool(jobs: list[list[str]], gpus: list[str], procs_per_gpu: int, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    slots = [(gpu, k) for gpu in gpus for k in range(procs_per_gpu)]
    queue = list(enumerate(jobs))
    lock = threading.Lock()
    failures: list[tuple[int, int]] = []

    def worker(gpu: str, slot: int):
        while True:
            with lock:
                if not queue:
                    return
                idx, cmd = queue.pop(0)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            out_path = log_dir / f"job{idx:03d}_gpu{gpu}_slot{slot}.log"
            started = time.time()
            print(f"[gpu {gpu} slot {slot}] job {idx}: {' '.join(cmd[-6:])}", flush=True)
            with out_path.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            elapsed = time.time() - started
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(
                f"[gpu {gpu} slot {slot}] job {idx} {status} in {elapsed / 60:.1f} min",
                flush=True,
            )
            if proc.returncode != 0:
                with lock:
                    failures.append((idx, proc.returncode))

    threads = [threading.Thread(target=worker, args=(g, k), daemon=False) for g, k in slots]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\nsweep finished in {(time.time() - started) / 3600:.2f} h")
    if failures:
        print(f"{len(failures)} job(s) failed: {failures}")
        print("Re-run the same command: completed runs are skipped by each script's resume.")
    return len(failures)


if __name__ == "__main__":
    sweep = "gate"
    if "--sweep" in sys.argv:
        sweep = sys.argv[sys.argv.index("--sweep") + 1]
    gpus = ["0"]
    if "--gpus" in sys.argv:
        gpus = sys.argv[sys.argv.index("--gpus") + 1].split(",")
    procs = 1
    if "--procs-per-gpu" in sys.argv:
        procs = int(sys.argv[sys.argv.index("--procs-per-gpu") + 1])

    if sweep not in SWEEPS:
        raise SystemExit(f"unknown sweep {sweep!r}; known: {list(SWEEPS)}")
    jobs = build_jobs(sweep)
    workers = len(gpus) * procs
    jvp_only = "--jvp-only" in SWEEPS[sweep]["extra"]
    seq, par = estimate_seconds(sweep, workers, jvp_only)
    print(f"sweep={sweep} jobs={len(jobs)} gpus={gpus} procs/gpu={procs} workers={workers}")
    print(
        f"estimate on the development GPU: {seq / 3600:.1f} h sequential, "
        f"{par / 3600:.1f} h at this worker count "
        f"(before any per-device speed difference)"
    )
    if "--dry-run" in sys.argv:
        for i, cmd in enumerate(jobs):
            print(f"  job {i}: {' '.join(cmd)}")
        raise SystemExit(0)

    n_failed = run_pool(jobs, gpus, procs, ROOT / "logs" / f"sweep_{sweep}")
    raise SystemExit(1 if n_failed else 0)
