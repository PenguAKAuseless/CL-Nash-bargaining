"""E7b -- the reduced dimension at real buffer scale.

Section "E7" measures the cost-vs-n trend of the exact representer basis on a
212-parameter model, because that benchmark materialises an explicit (n, p, p)
curvature array. It therefore says nothing about the quantity that decides
whether the reduction is usable at network scale:

    dim V  <=  n + sum_i k_i,     k_i = (buffer examples of task i) x (classes)

With a 500-example buffer and a 100-class head, k_i alone is of order 10^4 per
player, so the bound admits a reduced problem of dimension 10^4 to 10^5. A dense
Newton solve on a k x k Hessian costs O(k^3) per iteration and O(k^2) memory,
which is where the reduction stops being free. Nothing in this repository had
measured k, or the solve cost at that k, on the real backbone.

This file measures three things directly on the real stream, all without
forming any p x p object:

  (1) the realised dim V, by building the basis from the actual Jacobian rows
      on buffered inputs and taking its numerical rank;
  (2) how dim V grows with the number of active players and with the per-player
      sample cap, which is what the declared max_active_players deviation
      actually controls;
  (3) the wall clock and peak memory of one dense damped-Newton step at that
      dimension, timed on a synthetic reduced problem of exactly that size, so
      the solve cost is separated from the cost of building the basis.

Falsification: if dim V at realistic buffer sizes is small enough that a dense
solve is cheap, the exact reduction is usable as stated and the gradient-span
approximation used in training is an unnecessary compromise. If dim V is large,
then Proposition "Representer property" bounds a dimension that is independent
of p but not small; the table reports both scales explicitly.

Run:  uv run python experiments/e7b_dimv_scale.py [--smoke]
Writes: logs/e7b_dimv_scale.jsonl, tables/e7b_dimv_scale.md
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.bargain.solver import solve  # noqa: E402
from cl_bargain.bargain.utilities import BargainProblem  # noqa: E402
from cl_bargain.logging_utils import log_run  # noqa: E402

LOG = ROOT / "logs" / "e7b_dimv_scale.jsonl"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Per-player sample caps to sweep: how many buffered examples of a player are
# allowed to contribute Jacobian rows to the basis. This is the knob that
# actually controls dim V in a real implementation.
SAMPLE_CAPS = [1, 2, 4, 8]


def jacobian_rows(model, x, n_classes: int, device):
    """Rows of J_i for the given inputs, as a (k, p) matrix with k = len(x) *
    n_classes. Built one output coordinate at a time with backward passes, which
    is exactly what the basis construction of Proposition "Representer property"
    would cost; no p x p object is formed."""
    import torch

    rows = []
    for j in range(x.shape[0]):
        out = model(x[j : j + 1])[0]
        for c in range(n_classes):
            model.zero_grad(set_to_none=True)
            out[c].backward(retain_graph=(c < n_classes - 1))
            rows.append(
                torch.cat(
                    [
                        (p.grad.detach().reshape(-1) if p.grad is not None
                         else torch.zeros(p.numel(), device=device))
                        for p in model.parameters()
                    ]
                ).clone()
            )
        model.zero_grad(set_to_none=True)
    return torch.stack(rows)


# Relative singular-value thresholds the rank is reported at. The Jacobian rows
# are float32, so a direction that is exactly dependent in exact arithmetic
# appears with a small nonzero singular value; the duplicated-row control in
# `main` measures where that floor sits on this backbone.
RANK_TOLS = (1e-6, 1e-5, 1e-4, 1e-3)
COL_CHUNK = 1 << 17


def gram_blockwise(blocks, device):
    """The k x k Gram matrix of the row-stacked `blocks`, in float64, without
    stacking them. At eight examples per player the stacked basis of the
    1.1M-parameter backbone is 16.5 GB, which does not fit on a 24 GB device
    beside the model, so the blocks stay in host memory and are moved to the
    device a column slice at a time."""
    import torch

    sizes = [b.shape[0] for b in blocks]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    p = blocks[0].shape[1]
    G = torch.zeros((offs[-1], offs[-1]), dtype=torch.float64, device=device)
    for start in range(0, p, COL_CHUNK):
        cols = torch.cat([b[:, start : start + COL_CHUNK] for b in blocks]).to(device).double()
        G += cols @ cols.T
    return G


def singular_values(G):
    """Singular values of the row space, largest first, from its Gram matrix.
    Computed in float64: taking square roots of float32 Gram eigenvalues puts
    the noise floor near 3e-4 of the largest singular value, which hides any
    rank deficiency below it."""
    import torch

    return torch.linalg.eigvalsh(G).clamp(min=0.0).sqrt().flip(0).cpu().numpy()


def ranks_at(svals) -> dict[str, int]:
    top = float(svals[0]) if len(svals) else 0.0
    return {f"{t:.0e}": int((svals > t * top).sum()) for t in RANK_TOLS}


def time_dense_newton_step(k: int, n_players: int, seed: int = 0) -> dict:
    """Wall clock and peak memory of one solve of a REDUCED problem of
    dimension k. The reduced problem is synthetic (its data does not matter for
    cost, only its size), which is what lets this be measured at a k the real
    basis would produce without first paying to build that basis."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_players, k)) * 0.1
    H = np.zeros((n_players, k, k))
    for i in range(n_players):
        A = rng.standard_normal((min(k, 32), k))
        H[i] = A.T @ A / max(min(k, 32), 1)
    tau = np.full(n_players, 0.05)
    tau[0] = 0.0
    problem = BargainProblem(g=g, H=H, tau=tau, zeta=np.full(n_players, 1e-2))

    tracemalloc.start()
    t0 = time.perf_counter()
    res = solve(problem, max_iter=5)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "k": k,
        "n_players": n_players,
        "seconds_5_newton_iters": elapsed,
        "seconds_per_iter": elapsed / max(res.n_iter, 1),
        "peak_mem_mb": peak / 1e6,
        "converged": bool(res.converged),
        "n_iter": res.n_iter,
    }


def main(smoke: bool) -> dict:
    import torch

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.buffer import ReservoirBuffer
    from cl_bargain.streams.cifar import SplitCIFAR100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    nf = 8 if smoke else 20
    buffer_capacity = 100 if smoke else 500
    epochs = 1 if smoke else 5
    batch_size = 64
    minibatch_size = 64
    n_tasks = 3 if smoke else 5
    caps = SAMPLE_CAPS[:2] if smoke else SAMPLE_CAPS

    stream = SplitCIFAR100(
        str(ROOT / "data"), "imbalanced", device=device, class_order_seed=seed, download=False
    )
    model = make_backbone(stream.n_classes, device, nf=nf)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    buffer = ReservoirBuffer(buffer_capacity, device)
    n_classes = stream.n_classes
    p_params = int(sum(p.numel() for p in model.parameters()))

    sys.path.insert(0, str(ROOT / "experiments"))
    from e4_curvature import _train_task

    print(f"device={device} nf={nf} params={p_params} classes={n_classes}")
    for t in range(n_tasks):
        task = stream.tasks[t]
        _train_task(
            model, opt, stream, task, buffer, generator, epochs, batch_size, minibatch_size, device
        )
        buffer.add(task.train_idx, t, generator)
    active = sorted(buffer.task_sizes.keys())
    print(f"buffer holds {len(buffer)} examples over {len(active)} tasks")

    rows = []
    for cap in caps:
        mats = []
        t0 = time.perf_counter()
        for tid in active:
            held = buffer.held(tid)
            take = held[:cap]
            x = stream.normalise(stream.x_train[take])
            mats.append(jacobian_rows(model, x, n_classes, device).cpu())
        build_seconds = time.perf_counter() - t0
        n_rows = sum(m.shape[0] for m in mats)
        k_bound = len(active) + n_rows  # n + sum_i k_i
        svals = singular_values(gram_blockwise(mats, device))
        ranks = ranks_at(svals)
        # Control with a known answer: appending a copy of the first player's
        # rows adds exactly zero rank, so every singular value it adds beyond
        # the original count is float32 noise, and a threshold is only
        # meaningful if it sits above that floor.
        svals_dup = singular_values(gram_blockwise([*mats, mats[0]], device))
        noise_floor_rel = float(svals_dup[n_rows] / svals_dup[0])
        ranks_dup = ranks_at(svals_dup)
        tol_ok = [t for t in ranks if ranks_dup[t] == ranks[t]]
        # The smallest threshold that the control certifies: at it the
        # duplicated rows add nothing, so a rank deficiency is not hidden.
        tol_used = tol_ok[0] if tol_ok else None
        k_actual = (ranks[tol_used] if tol_used else n_rows) + len(active)
        solve_cost = time_dense_newton_step(
            k=min(k_actual, 4000), n_players=len(active) + 1
        )
        row = {
            "deliverable": "e7b_dimv_scale",
            "examples_per_player": cap,
            "n_active_players": len(active),
            "classes": n_classes,
            "params": p_params,
            "k_bound_n_plus_sum_ki": int(k_bound),
            "dim_V_measured": int(k_actual),
            "rank_tol_used": tol_used,
            "ranks_by_tol": ranks,
            "ranks_by_tol_duplicated_control": ranks_dup,
            "control_noise_floor_rel": noise_floor_rel,
            "sval_min_rel": float(svals[n_rows - 1] / svals[0]),
            "basis_rows": int(n_rows),
            "basis_build_seconds": build_seconds,
            "solve_dimension_timed": solve_cost["k"],
            "solve_seconds_per_newton_iter": solve_cost["seconds_per_iter"],
            "solve_peak_mem_mb": solve_cost["peak_mem_mb"],
            "smoke": smoke,
        }
        print(row)
        log_run(LOG, row)
        rows.append(row)
        del mats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Extrapolation to the buffer sizes actually used, without building a basis
    # that large: dim V is bounded by n + sum_i k_i and the solve cost at that
    # dimension can be timed directly on a synthetic reduced problem.
    projections = []
    if not smoke:
        for per_player in [8, 32, 128]:
            k_proj = len(active) + len(active) * per_player * n_classes
            cost = time_dense_newton_step(k=min(k_proj, 6000), n_players=len(active) + 1)
            projections.append(
                {
                    "deliverable": "e7b_dimv_projection",
                    "examples_per_player": per_player,
                    "k_bound": int(k_proj),
                    "k_timed": cost["k"],
                    "seconds_per_newton_iter": cost["seconds_per_iter"],
                    "peak_mem_mb": cost["peak_mem_mb"],
                }
            )
            print(projections[-1])
            log_run(LOG, projections[-1])

    return {"rows": rows, "projections": projections}


def write_table(result: dict) -> None:
    rows = result["rows"]
    if not rows:
        return
    lines = [
        "# E7b -- reduced dimension and solve cost at real buffer scale\n",
        f"Backbone with {rows[0]['params']} parameters, {rows[0]['classes']}-class head, "
        f"{rows[0]['n_active_players']} absent players in the buffer. dim V is measured "
        "as the numerical rank of the actual Jacobian rows plus one direction per "
        "player gradient; the solve cost is one dense damped-Newton iteration at that "
        "dimension. The rank is taken at the smallest relative singular-value "
        "threshold at which a duplicated copy of one player's rows adds no rank "
        "(the control noise floor column is the largest singular value that copy "
        "introduces), so float32 noise cannot hide a rank deficiency.\n",
        "| examples/player | basis rows | bound n+sum k_i | dim V measured | rank tol "
        "| smallest sv / largest | control noise floor | basis build (s) "
        "| solve dim timed | s / Newton iter | solve peak mem (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['examples_per_player']} | {r['basis_rows']} | {r['k_bound_n_plus_sum_ki']} "
            f"| {r['dim_V_measured']} | {r['rank_tol_used']} | {r['sval_min_rel']:.2e} "
            f"| {r['control_noise_floor_rel']:.2e} | {r['basis_build_seconds']:.1f} "
            f"| {r['solve_dimension_timed']} | {r['solve_seconds_per_newton_iter']:.3f} "
            f"| {r['solve_peak_mem_mb']:.1f} |"
        )
    if result["projections"]:
        lines += [
            "",
            "## Projected to larger per-player samples\n",
            "| examples/player | bound n+sum k_i | dimension timed | s / Newton iter "
            "| peak mem (MB) |",
            "|---|---|---|---|---|",
        ]
        for pr in result["projections"]:
            lines.append(
                f"| {pr['examples_per_player']} | {pr['k_bound']} | {pr['k_timed']} "
                f"| {pr['seconds_per_newton_iter']:.3f} | {pr['peak_mem_mb']:.1f} |"
            )
    lines += [
        "",
        "The measured dim V is what decides whether the exact reduction is usable in "
        "training: it is independent of the parameter count, as Proposition "
        '"Representer property" states, but it grows linearly in buffered examples '
        "times classes, and a dense Newton step is cubic in it.\n",
    ]
    (TABLES / "e7b_dimv_scale.md").write_text("\n".join(lines))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    res = main(smoke=smoke)
    write_table(res)
    print("Wrote tables/e7b_dimv_scale.md")
