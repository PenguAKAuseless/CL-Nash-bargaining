"""E4b -- the curvature proxy against the TRUE Hessian, not against a second
Gauss-Newton estimate.

Section "E4" compares three buffer-based curvature estimators against a
Gauss-Newton matrix computed on each task's full retained data. That comparison
is informative about estimator agreement, but it cannot check the hypothesis the
theory actually uses: Assumption "Derivative approximation errors" bounds
||hat_H_i - H_i||_op with H_i = grad^2 L_i, the true Hessian, and Theorem
"Computable damage certificate" is conditional on that bound holding with the
zeta_i the solver was given. Two quantities decide it, and neither is measurable
from Gauss-Newton comparisons alone:

  (1) lambda_min(H_i). The true Hessian of a cross-entropy loss at a
      non-stationary point is indefinite. A PSD curvature model hat_H_i can
      only satisfy ||hat_H_i - H_i||_op <= zeta_i if zeta_i >= -lambda_min(H_i),
      so the measured negative spectrum is a LOWER BOUND on the admissible
      curvature-uncertainty floor of Assumption "Curvature-uncertainty floor".
  (2) ||hat_H_i - H_i||_op itself, for the estimator actually used at network
      scale.

Both are computed matrix free. The true Hessian acts by double backward; the
buffer estimator acts through the same Gauss-Newton-vector product the training
loop uses; extreme eigenvalues come from Lanczos on those actions
(scipy.sparse.linalg.eigsh over a LinearOperator), so no p x p object is ever
formed.

Falsification, stated in advance: if the measured ||hat_H_i - H_i||_op is of the
same order as the zeta_i actually used on real streams, the certificate's
hypothesis is met and its measured under-coverage must be explained elsewhere.
If it is orders of magnitude larger, then the real-stream certificate was never
operating inside its own assumptions, and that, rather than the bound, is what
needs fixing.

Run:  uv run python experiments/e4b_true_hessian.py [--smoke] [--tasks N]
Writes: logs/e4b_true_hessian.jsonl, tables/e4b_true_hessian.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_bargain.logging_utils import log_run  # noqa: E402

LOG = ROOT / "logs" / "e4b_true_hessian.jsonl"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# The zeta floor E-cert and E6 actually ran with, quoted here so the table can
# report the measured error against the value the solver was told to assume.
ZETA_USED = 0.01


def _flat_params(model):
    import torch

    return torch.cat([p.reshape(-1) for p in model.parameters()])


def _unflatten_like(vec, model):
    out, i = [], 0
    for p in model.parameters():
        k = p.numel()
        out.append(vec[i : i + k].view_as(p))
        i += k
    return out


def true_hessian_vector_product(model, x, y, lo: int, hi: int, v_flat, chunk_size: int):
    """(grad^2 of the task-restricted cross-entropy) @ v, by double backward.

    Chunked over x and accumulated with example-count weights, so the result is
    the Hessian of the MEAN loss regardless of chunking. Unlike the Gauss-Newton
    action in bargain/curvature.py this keeps the term that makes the true
    Hessian indefinite, which is the whole point of this file.
    """
    import torch
    import torch.nn.functional as f

    params = list(model.parameters())
    v_parts = _unflatten_like(v_flat, model)
    total = torch.zeros_like(v_flat)
    n_total = 0
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        n = xb.shape[0]
        model.zero_grad(set_to_none=True)
        out = model(xb)[:, lo:hi]
        # lo=0, hi=n_classes throughout this repo: the backbone carries one
        # head over every class and the loss is taken over all logits, so
        # labels need no offset (matches e4_curvature.py's own convention).
        loss = f.cross_entropy(out, yb) * n  # un-average, re-average at the end
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(grads, v_parts, strict=True))
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        total = total + torch.cat([h.reshape(-1) for h in hv]).detach()
        n_total += n
    model.zero_grad(set_to_none=True)
    return total / max(n_total, 1)


def buffer_curvature_vector_product(
    model, x_buf, lo: int, hi: int, v_flat, chunk_size: int, weighted: bool = True
):
    """hat_H_i v on buffer inputs. weighted=True is the Gauss-Newton operator
    J^T (diag(p) - p p^T) J that streams/methods.py's bargain_step builds its
    curvature block from, and therefore the estimator the certificate is
    evaluated with. weighted=False is the unweighted function-space operator
    J^T J of E4, which the first version of this file measured in its place;
    it is kept as a second column because E4 already reports that its scale
    grows with model confidence."""
    import torch

    from cl_bargain.bargain.curvature import ggn_vector_product
    from cl_bargain.streams.paramvec import flat_to_dict, param_shapes

    shapes = param_shapes(model)
    names = list(shapes.keys())
    v_dict = flat_to_dict(v_flat, names, shapes)
    hv = ggn_vector_product(
        model, x_buf, lo, hi, v_dict, chunk_size=chunk_size, weighted=weighted
    )
    return torch.cat([hv[k].reshape(-1) for k in names]).detach()


def _linear_operator(matvec_flat, dim, device, counter):
    """Wrap a torch matvec as a scipy LinearOperator so Lanczos can drive it."""
    import torch
    from scipy.sparse.linalg import LinearOperator

    def mv(v_np):
        counter["n"] += 1
        v = torch.as_tensor(np.asarray(v_np, dtype=np.float32).ravel(), device=device)
        out = matvec_flat(v)
        return out.detach().cpu().numpy().astype(np.float64)

    return LinearOperator((dim, dim), matvec=mv, dtype=np.float64)


def extreme_eigenvalues(matvec_flat, dim, device, k_tol: float, maxiter: int):
    """(lambda_min, lambda_max) by Lanczos on the operator's action."""
    from scipy.sparse.linalg import eigsh

    counter = {"n": 0}
    op = _linear_operator(matvec_flat, dim, device, counter)
    lo = float(eigsh(op, k=1, which="SA", tol=k_tol, maxiter=maxiter, return_eigenvectors=False)[0])
    hi = float(eigsh(op, k=1, which="LA", tol=k_tol, maxiter=maxiter, return_eigenvectors=False)[0])
    return lo, hi, counter["n"]


def operator_norm(matvec_flat, dim, device, k_tol: float, maxiter: int):
    """||A||_op for a symmetric A, as the largest-magnitude eigenvalue."""
    from scipy.sparse.linalg import eigsh

    counter = {"n": 0}
    op = _linear_operator(matvec_flat, dim, device, counter)
    val = float(
        eigsh(op, k=1, which="LM", tol=k_tol, maxiter=maxiter, return_eigenvectors=False)[0]
    )
    return abs(val), counter["n"]


def main(
    smoke: bool,
    n_tasks_limit: int | None,
    hessian_examples_override: int | None = None,
    maxiter_override: int | None = None,
) -> list[dict]:
    import torch

    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.buffer import ReservoirBuffer
    from cl_bargain.streams.cifar import SplitCIFAR100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    seed = 0
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    nf = 8 if smoke else 20
    buffer_capacity = 100 if smoke else 500
    epochs = 1 if smoke else 5
    batch_size = 64
    minibatch_size = 64
    lr = 0.1
    chunk_size = 64 if smoke else 128
    # The true-Hessian action is a double backward over the task's retained
    # data; capping how many examples enter it keeps one Lanczos matvec at a
    # few seconds rather than a few minutes. Declared, and reported per row.
    # Cost per boundary scales as (examples x params x matvecs). The defaults
    # below put a full nine-boundary run in the low hours on one GPU; both
    # knobs are exposed so the measurement can be traded against wall clock
    # without editing the file.
    hessian_examples = 128 if smoke else 512
    lanczos_tol = 1e-2 if smoke else 3e-3
    lanczos_maxiter = 30 if smoke else 150
    if hessian_examples_override is not None:
        hessian_examples = hessian_examples_override
    if maxiter_override is not None:
        lanczos_maxiter = maxiter_override

    stream = SplitCIFAR100(
        str(ROOT / "data"), "imbalanced", device=device, class_order_seed=seed, download=False
    )
    model = make_backbone(stream.n_classes, device, nf=nf)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    buffer = ReservoirBuffer(buffer_capacity, device)

    # Reuse E4's own training loop so this runs on the same trajectory rather
    # than on a separately trained model.
    sys.path.insert(0, str(ROOT / "experiments"))
    from e4_curvature import _train_task

    dim = int(sum(p.numel() for p in model.parameters()))
    n_tasks = 2 if smoke else len(stream.tasks)
    if n_tasks_limit is not None:
        n_tasks = min(n_tasks, n_tasks_limit)
    print(f"device={device} nf={nf} params={dim} tasks={n_tasks} smoke={smoke}")

    results = []
    for t in range(n_tasks):
        task = stream.tasks[t]
        started = time.time()
        _train_task(
            model, opt, stream, task, buffer, generator, epochs, batch_size, minibatch_size, device
        )
        buffer.add(task.train_idx, t, generator)
        print(f"task {t} trained in {time.time() - started:.1f}s")

        if t == 0:
            continue
        prev = t - 1
        prev_task = stream.tasks[prev]
        lo, hi = 0, stream.n_classes

        idx_full = prev_task.train_idx[:hessian_examples]
        x_full = stream.normalise(stream.x_train[idx_full])
        y_full = stream.y_train[idx_full]
        held = buffer.held(prev)
        if len(held) == 0:
            continue
        x_buf = stream.normalise(stream.x_train[held])

        t0 = time.time()

        def h_true(v, x_full=x_full, y_full=y_full, lo=lo, hi=hi):
            return true_hessian_vector_product(model, x_full, y_full, lo, hi, v, chunk_size)

        def h_hat(v, x_buf=x_buf, lo=lo, hi=hi):
            return buffer_curvature_vector_product(model, x_buf, lo, hi, v, chunk_size)

        def h_jtj(v, x_buf=x_buf, lo=lo, hi=hi):
            return buffer_curvature_vector_product(
                model, x_buf, lo, hi, v, chunk_size, weighted=False
            )

        lam_min, lam_max, n_mv_true = extreme_eigenvalues(
            h_true, dim, device, lanczos_tol, lanczos_maxiter
        )
        gap_norm, n_mv_gap = operator_norm(
            lambda v: h_hat(v) - h_true(v), dim, device, lanczos_tol, lanczos_maxiter
        )
        hat_norm, _ = operator_norm(h_hat, dim, device, lanczos_tol, lanczos_maxiter)
        jtj_gap_norm, _ = operator_norm(
            lambda v: h_jtj(v) - h_true(v), dim, device, lanczos_tol, lanczos_maxiter
        )
        jtj_norm, _ = operator_norm(h_jtj, dim, device, lanczos_tol, lanczos_maxiter)
        seconds = time.time() - t0

        record = {
            "deliverable": "e4b_true_hessian",
            "task": prev,
            "measured_after_task": t,
            "n_hessian_examples": len(idx_full),
            "n_buffer_examples": len(held),
            "params": dim,
            "lambda_min_true_hessian": lam_min,
            "lambda_max_true_hessian": lam_max,
            "op_norm_hat_H": hat_norm,
            "op_norm_gap": gap_norm,
            "estimator": "ggn_weighted",
            "op_norm_hat_H_unweighted_jtj": jtj_norm,
            "op_norm_gap_unweighted_jtj": jtj_gap_norm,
            "zeta_used_on_real_streams": ZETA_USED,
            "gap_over_zeta_used": gap_norm / ZETA_USED,
            # zeta_i must be at least -lambda_min for a PSD hat_H to sit inside
            # the uncertainty set at all (Assumption "Curvature-uncertainty floor").
            "zeta_floor_implied_by_negative_spectrum": max(0.0, -lam_min),
            "matvecs_true": n_mv_true,
            "matvecs_gap": n_mv_gap,
            "seconds": seconds,
            "smoke": smoke,
        }
        print(record)
        log_run(LOG, record)
        results.append(record)

    return results


def write_table(records: list[dict]) -> None:
    if not records:
        return
    lines = [
        "# E4b -- buffer curvature against the TRUE Hessian\n",
        "All quantities matrix free: the true Hessian acts by double backward, the "
        "buffer estimator by the same weighted Gauss-Newton product the training loop "
        "builds its curvature block from, and extreme eigenvalues come from Lanczos on "
        "those actions. The last two columns are the unweighted function-space "
        "operator J^T J of E4, for comparison.\n",
        "| task | Hess. examples | buffer | lambda_min(H) | lambda_max(H) | ||hat_H||_op "
        "| ||hat_H - H||_op | vs zeta used | zeta floor implied | s "
        "| ||J^T J||_op | ||J^T J - H||_op |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['task']} | {r['n_hessian_examples']} | {r['n_buffer_examples']} "
            f"| {r['lambda_min_true_hessian']:.3e} | {r['lambda_max_true_hessian']:.3e} "
            f"| {r['op_norm_hat_H']:.3e} | {r['op_norm_gap']:.3e} "
            f"| {r['gap_over_zeta_used']:.1f}x | "
            f"{r['zeta_floor_implied_by_negative_spectrum']:.3e} | {r['seconds']:.0f} "
            f"| {r['op_norm_hat_H_unweighted_jtj']:.3e} | {r['op_norm_gap_unweighted_jtj']:.3e} |"
        )
    worst = max(r["gap_over_zeta_used"] for r in records)
    neg = [r for r in records if r["lambda_min_true_hessian"] < 0]
    lines += [
        "",
        f"Largest measured ||hat_H - H||_op relative to the zeta = {ZETA_USED} used on "
        f"real streams: **{worst:.1f}x**.\n",
        f"Boundaries with an indefinite true Hessian: **{len(neg)}/{len(records)}**; the "
        "most negative eigenvalue seen sets a hard lower bound on any admissible "
        "curvature-uncertainty floor.\n",
        "Claim (the zeta actually used bounds the real curvature error) holds: "
        f"**{bool(worst <= 1.0)}**\n",
    ]
    (TABLES / "e4b_true_hessian.md").write_text("\n".join(lines))


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    limit = None
    if "--tasks" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--tasks") + 1])
    hess_ex = None
    if "--hessian-examples" in sys.argv:
        hess_ex = int(sys.argv[sys.argv.index("--hessian-examples") + 1])
    maxit = None
    if "--maxiter" in sys.argv:
        maxit = int(sys.argv[sys.argv.index("--maxiter") + 1])
    recs = main(
        smoke=smoke,
        n_tasks_limit=limit,
        hessian_examples_override=hess_ex,
        maxiter_override=maxit,
    )
    write_table(recs)
    print(f"Wrote tables/e4b_true_hessian.md ({len(recs)} boundaries)")
