"""Real-stream benchmark on task-incremental Split-CIFAR100.

The benchmark includes naive Nash-MTL as a first-order comparison.

Methods: er, agem, ewc, v1, v2, nashmtl (naive Nash-MTL ablation --
first-order utilities, d_i=0 for every player, no curvature, no robustness;
see streams/methods.py's bargain_step). `nashmtl` sometimes has no feasible
common-descent direction within the trust region (observed on 25-60% of
random low-dimensional instances in isolated testing, not just a solver
artefact -- see bargain/baselines.py's `_feasible_start_v1`); those steps
are skipped (no update applied) and counted, not silently absorbed.

The curvature method also has an optional BatchNorm ablation arm.

NOT implemented in this pass: the oracle arm (true g_i,
true damage, realised r_i/mu/lambda_i measured per step against retained
full data) and the equal 10x10-split control stream. Both are natural
follow-ups, deferred given this pass's time/compute budget.

The `--gate-only` option restricts METHODS to {nashmtl, v2} for a cheaper
comparison.

v2 now also receives a real `zeta_floor` (Assumption "Curvature-uncertainty
floor", zeta_i >= zeta_min > 0) rather than the implicit 0.0 default in
streams/methods.py's bargain_step -- previously omitted here, which would
have left the theorem's uniqueness hypothesis unmet on every real-stream run.

Run: uv run python experiments/e6_benchmark.py [--smoke] [--batchnorm-arm] [--gate-only]
Writes: logs/e6_benchmark.jsonl, tables/e6_benchmark.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.logging_utils import log_run  # noqa: E402

LOG = ROOT / "logs" / "e6_benchmark.jsonl"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
LOG_SUFFIX = ""

# One representative knob per method, taken from the middle of E2's
# full-scale sweep range -- E6 is a benchmark table, not another knob
# This driver uses one fixed operating point per method.
METHOD_KNOB = {
    "er": 0.1,
    "agem": 0.1,
    "ewc": 3.0,
    "v1": 0.05,
    "v2": 0.05,
    "nashmtl": 0.05,
}
METHODS = list(METHOD_KNOB)
GATE_METHODS = ["nashmtl", "v2"]  # the decisive ablation quoted in the module docstring

# Assumption "Curvature-uncertainty floor" (zeta_i >= zeta_min > 0), applied
# to v2 only (the other methods have no curvature term).
ZETA_FLOOR = 0.01


def build_stream(seed: int, device, stream: str = "split20", task_masked: bool = True):
    from cl_bargain.streams.cifar import SplitCIFAR100

    return SplitCIFAR100(
        str(ROOT / "data"), stream, device=device, class_order_seed=seed,
        download=False, task_masked=task_masked,
    )


def run_one(
    method: str,
    seed: int,
    smoke: bool,
    stream=None,
    norm: str = "group",
    max_active_players: int | None = None,
    jvp_only: bool = False,
    jvp_batched: bool = False,
    knob: float | None = None,
    epochs: int | None = None,
    buffer_capacity: int | None = None,
    zeta_floor: float = ZETA_FLOOR,
) -> dict:
    """`knob` overrides METHOD_KNOB[method]; experiments/e2b_frontier_sweep.py
    uses it to sweep each method's magnitude knob on this same loop."""
    import torch

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.buffer import ReservoirBuffer
    from cl_bargain.streams.methods import (
        RunState,
        agem_step,
        bargain_step,
        er_step,
        ewc_snapshot,
        ewc_step,
        task_accuracy_corrected,
        task_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    if knob is None:
        knob = METHOD_KNOB[method]

    if epochs is None:
        epochs = 1 if smoke else 5
    nf = 8 if smoke else 16
    if buffer_capacity is None:
        buffer_capacity = 60 if smoke else 2000
    minibatch_size = 32
    chunk_size = 64
    fisher_chunk_size = 8 if smoke else 16

    if stream is None:
        stream = build_stream(seed, device)
    n_tasks = 2 if smoke else len(stream.tasks)
    model = make_backbone(stream.n_classes, device, nf=nf, norm=norm)
    buffer = ReservoirBuffer(buffer_capacity, device)
    state = RunState()

    opt = None
    if method in ("er", "agem"):
        opt = torch.optim.SGD(model.parameters(), lr=knob)
    elif method == "ewc":
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    skipped_steps = 0
    total_steps = 0
    acc_at_end: dict[int, float] = {}
    loss_at_end: dict[int, float] = {}

    for t in range(n_tasks):
        task = stream.tasks[t]
        n = task.n_examples
        for _ in range(epochs):
            perm = torch.randperm(n, device=task.train_idx.device, generator=generator)
            for start in range(0, n, minibatch_size):
                sub = task.train_idx[perm[start : start + minibatch_size]]
                total_steps += 1
                if method == "er":
                    er_step(model, opt, stream, sub, buffer, generator, minibatch_size, device)
                elif method == "agem":
                    agem_step(model, opt, stream, sub, buffer, generator, minibatch_size, device)
                elif method == "ewc":
                    ewc_step(
                        model, opt, stream, sub, buffer, generator, minibatch_size, device,
                        state, lam=knob,
                    )
                else:
                    zeta = zeta_floor if method == "v2" else 0.0
                    applied, _diag = bargain_step(
                        model, stream, sub, task.index, buffer, generator, minibatch_size, device,
                        method, knob, chunk_size, zeta_floor=zeta,
                        max_active_players=max_active_players, jvp_only=jvp_only,
                        jvp_batched=jvp_batched,
                    )
                    if not applied:
                        skipped_steps += 1

        buffer.add(task.train_idx, t, generator)
        if method == "ewc":
            state.ewc_snapshots[t] = ewc_snapshot(model, stream, task, fisher_chunk_size, generator)

        model.eval()
        lo, hi = stream.task_range(t)
        with torch.no_grad():
            acc_at_end[t] = task_accuracy_corrected(model, stream, task.test_idx, lo, hi)
            loss_at_end[t] = float(task_loss(model, stream, task.test_idx, lo, hi))
        model.train()

    model.eval()
    acc_final = {}
    loss_final = {}
    with torch.no_grad():
        for t in range(n_tasks):
            task = stream.tasks[t]
            lo, hi = stream.task_range(t)
            acc_final[t] = task_accuracy_corrected(model, stream, task.test_idx, lo, hi)
            loss_final[t] = float(task_loss(model, stream, task.test_idx, lo, hi))
    model.train()

    wall_seconds = time.time() - started
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    )

    # Backward transfer (Lopez-Paz & Ranzato): mean, over tasks except the
    # last, of (final accuracy - accuracy right when the task finished).
    bwt_terms = [acc_final[t] - acc_at_end[t] for t in range(n_tasks - 1)]
    corrected_avg_accuracy = float(np.mean(list(acc_final.values())))
    worst_task_accuracy = float(np.min(list(acc_final.values())))
    forgetting = [acc_at_end[t] - acc_final[t] for t in range(n_tasks - 1)]  # positive = forgetting
    # The frontier of Section "Continual-Learning Benchmarks" is in loss, not
    # accuracy: final live-task loss against the test-loss increase on each
    # earlier task between the end of its own training and the end of the run.
    loss_forgetting = [loss_final[t] - loss_at_end[t] for t in range(n_tasks - 1)]

    return {
        "method": method,
        "knob": knob,
        "zeta_floor": zeta_floor if method == "v2" else 0.0,
        "epochs": epochs,
        "seed": seed,
        "norm": norm,
        # Part of the resume key: a --smoke run uses a different model width,
        # buffer and task count, so its record must not satisfy a later
        # full-scale request for the same (method, seed, norm).
        "smoke": bool(smoke),
        "corrected_avg_accuracy": corrected_avg_accuracy,
        "backward_transfer": float(np.mean(bwt_terms)) if bwt_terms else 0.0,
        "worst_task_accuracy": worst_task_accuracy,
        "forgetting_variance": float(np.var(forgetting)) if forgetting else 0.0,
        "wall_seconds": wall_seconds,
        "peak_memory_mb": peak_memory_mb,
        "skipped_steps": skipped_steps,
        "total_steps": total_steps,
        "plasticity_final_task_loss": loss_at_end[n_tasks - 1],
        "mean_loss_forgetting": float(np.mean(loss_forgetting)) if loss_forgetting else 0.0,
        "worst_task_loss_forgetting": (
            float(np.max(loss_forgetting)) if loss_forgetting else 0.0
        ),
        "acc_at_end": acc_at_end,
        "acc_final": acc_final,
        "loss_at_end": loss_at_end,
        "loss_final": loss_final,
    }


def _run_with_retries(
    method, seed, smoke, stream, norm="group", max_active_players=None,
    jvp_only=False, jvp_batched=False, epochs=None, buffer_capacity=None,
    zeta_floor=ZETA_FLOOR, max_attempts: int = 3
):
    import torch

    for attempt in range(1, max_attempts + 1):
        try:
            return run_one(
                method, seed, smoke, stream=stream, norm=norm,
                max_active_players=max_active_players,
                jvp_only=jvp_only, jvp_batched=jvp_batched, epochs=epochs,
                buffer_capacity=buffer_capacity, zeta_floor=zeta_floor,
            )
        except RuntimeError as e:
            print(f"attempt {attempt}/{max_attempts} for {method} (norm={norm}) seed={seed}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2 * attempt)
    return None


def _load_completed(log_path: Path) -> tuple[list[dict], set[tuple]]:
    """Records and (method, seed, norm, smoke) keys already logged, mirroring
    e2_frontier.py's `_load_completed` -- resuming a killed/restarted run
    without redoing finished work. Ignores summary/crashed-runs lines.

    `smoke` is part of the key because a feasibility run and a full run differ
    in model width, buffer size and task count while sharing the other three
    fields: without it, running the pool's smoke sweep once would mark the real
    sweep as already finished. Records written before this field existed are
    treated as full runs, which is what they were."""
    records: list[dict] = []
    done: set[tuple] = set()
    if not log_path.exists():
        return records, done
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "method" not in rec or "seed" not in rec or "norm" not in rec:
                continue
            records.append(rec)
            done.add(
                (
                    rec["method"], rec["seed"], rec["norm"], bool(rec.get("smoke", False)),
                    rec.get("zeta_floor"), rec.get("epochs"),
                )
            )
    return records, done


def main(
    smoke: bool,
    batchnorm_arm: bool,
    gate_only: bool = False,
    max_active_players: int | None = None,
    max_seeds: int | None = None,
    only_seeds: list[int] | None = None,
    only_methods: list[str] | None = None,
    jvp_only: bool = False,
    jvp_batched: bool = False,
    epochs: int | None = None,
    buffer_capacity: int | None = None,
    stream_name: str = "split20",
    zeta_floor: float = ZETA_FLOOR,
) -> None:
    import torch

    seeds = [0] if smoke else [0, 1, 2]
    if max_seeds is not None:
        seeds = seeds[:max_seeds]  # declared reduced-scope deviation
    methods = GATE_METHODS if gate_only else METHODS
    # Selection flags exist so experiments/run_sweep.py can shard one sweep
    # across several GPU workers: each worker runs a disjoint slice and the
    # resume logic below keeps them from repeating each other's completed runs.
    if only_seeds is not None:
        seeds = [s for s in seeds if s in only_seeds]
    if only_methods is not None:
        methods = [m for m in methods if m in only_methods]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, done = _load_completed(LOG)
    crashed = []
    for seed in seeds:
        stream = None
        for method in methods:
            key_zeta = zeta_floor if method == "v2" else 0.0
            if (method, seed, "group", smoke, key_zeta, epochs) in done:
                continue
            if stream is None:
                stream = build_stream(seed, device, stream_name)
            rec = _run_with_retries(
                method, seed, smoke, stream, max_active_players=max_active_players,
                jvp_only=jvp_only, jvp_batched=jvp_batched, epochs=epochs,
                buffer_capacity=buffer_capacity, zeta_floor=zeta_floor,
            )
            if rec is None:
                crashed.append({"method": method, "seed": seed, "norm": "group"})
                print(f"CRASHED: {method} seed={seed}")
                continue
            summary = {k: v for k, v in rec.items() if not isinstance(v, dict)}
            print(summary)
            log_run(LOG, rec)
            records.append(rec)

        if batchnorm_arm and ("v2", seed, "batch", smoke, zeta_floor, epochs) not in done:
            if stream is None:
                stream = build_stream(seed, device, stream_name)
            rec = _run_with_retries(
                "v2", seed, smoke, stream, norm="batch",
                max_active_players=max_active_players, jvp_only=jvp_only,
                jvp_batched=jvp_batched, epochs=epochs,
            )
            if rec is None:
                crashed.append({"method": "v2", "seed": seed, "norm": "batch"})
                print(f"CRASHED: v2 (batchnorm) seed={seed}")
            else:
                summary = {k: v for k, v in rec.items() if k not in ("acc_at_end", "acc_final")}
                print(summary)
                log_run(LOG, rec)
                records.append(rec)

    if crashed:
        log_run(LOG, {"deliverable": "e6_benchmark_crashed_runs", "crashed": crashed})
    write_table(records, crashed)


def write_table(records: list[dict], crashed: list[dict]) -> None:
    by_method_norm: dict[tuple, list[dict]] = {}
    for r in records:
        by_method_norm.setdefault((r["method"], r["norm"]), []).append(r)

    lines = ["# E6 -- real-stream benchmark (imbalanced Split-CIFAR100, all 10 tasks)\n"]
    lines.append(
        "| method | norm | n seeds | corrected avg acc | backward transfer | "
        "worst-task acc | forgetting variance | mean wall (s) | mean peak mem (MB) | "
        "mean skipped/total steps |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (method, norm), recs in sorted(by_method_norm.items()):
        acc = [r["corrected_avg_accuracy"] for r in recs]
        bwt = [r["backward_transfer"] for r in recs]
        worst = [r["worst_task_accuracy"] for r in recs]
        fvar = [r["forgetting_variance"] for r in recs]
        wall = [r["wall_seconds"] for r in recs]
        mem = [r["peak_memory_mb"] for r in recs if r["peak_memory_mb"] is not None]
        skip_frac = [
            r["skipped_steps"] / r["total_steps"] if r["total_steps"] else 0.0 for r in recs
        ]
        mem_str = f"{np.mean(mem):.0f}" if mem else "n/a"
        lines.append(
            f"| {method} | {norm} | {len(recs)} | {np.mean(acc):.4f} +/- {np.std(acc):.4f} | "
            f"{np.mean(bwt):+.4f} | {np.mean(worst):.4f} | {np.mean(fvar):.5f} | "
            f"{np.mean(wall):.1f} | {mem_str} | {np.mean(skip_frac):.1%} |"
        )

    if crashed:
        lines.append(f"\n**{len(crashed)} run(s) crashed (all retries exhausted):**\n")
        for c in crashed:
            lines.append(f"- {c['method']} norm={c['norm']} seed={c['seed']}")

    lines.append(
        "\nNot implemented this pass: the oracle arm (true "
        "g_i/damage/r_i/mu/lambda_i measured per step against retained full "
        "data) and the equal 10x10-split control stream.\n"
    )
    (TABLES / f"e6_benchmark{LOG_SUFFIX}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    # A configuration that differs from the one already logged (cap, curvature
    # path) gets its own log and table: the resume key is (method, seed, norm,
    # smoke), so sharing one log would let the old configuration's records
    # satisfy the new request.
    if "--log-suffix" in sys.argv:
        LOG_SUFFIX = sys.argv[sys.argv.index("--log-suffix") + 1]
        LOG = ROOT / "logs" / f"e6_benchmark{LOG_SUFFIX}.jsonl"
    _max_active = None
    if "--max-active" in sys.argv:
        _max_active = int(sys.argv[sys.argv.index("--max-active") + 1])
    _max_seeds = None
    if "--max-seeds" in sys.argv:
        _max_seeds = int(sys.argv[sys.argv.index("--max-seeds") + 1])
    _only_seeds = None
    if "--seeds" in sys.argv:
        _only_seeds = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    _only_methods = None
    if "--methods" in sys.argv:
        _only_methods = sys.argv[sys.argv.index("--methods") + 1].split(",")
    main(
        smoke="--smoke" in sys.argv,
        batchnorm_arm="--batchnorm-arm" in sys.argv,
        gate_only="--gate-only" in sys.argv,
        max_active_players=_max_active,
        max_seeds=_max_seeds,
        only_seeds=_only_seeds,
        only_methods=_only_methods,
        jvp_only="--jvp-only" in sys.argv,
        jvp_batched="--jvp-batched" in sys.argv,
        epochs=int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else None,
        buffer_capacity=(
            int(sys.argv[sys.argv.index("--buffer") + 1]) if "--buffer" in sys.argv else None
        ),
        stream_name=(
            sys.argv[sys.argv.index("--stream") + 1] if "--stream" in sys.argv else "split20"
        ),
        zeta_floor=(
            float(sys.argv[sys.argv.index("--zeta") + 1]) if "--zeta" in sys.argv else ZETA_FLOOR
        ),
    )
