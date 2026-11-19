"""Continual-learning methods compared in E2's frontier experiment.

ER and A-GEM are standard baselines. EWC reuses bargain/curvature.py's
diagonal_empirical_fisher. v1/v2 solve the bargain restricted to the
subspace spanned by the active players' own gradients, Delta = G^T beta
(beta in R^n) -- the standard simplification used across the gradient-
combination literature (Nash-MTL, CAGrad) to make an n-player program
tractable at network scale without E7's matrix-free machinery.

This is EXACT for v1 (first-order): the KKT stationarity condition of the
trust-region-active first-order NBS literally is a linear combination of the
g_i's, so restricting to span(g_i) loses nothing for first-order updates.

It is an APPROXIMATION for v2: once curvature enters, H_i Delta need not
stay in span(g_i), so the true unconstrained maximizer over the full
parameter space can leave this subspace. Reported as a deliberate,
documented subspace simplification, not the literal full-dimensional
solve the reduced curvature problem at network scale.

In this reduced subspace, <g_j, Delta> = (K beta)_j and Delta^T H_j Delta =
beta^T M_j beta with K = G G^T (n x n Gram) and M_j = G H_j G^T (n x n) --
so the reduced bargain is exactly a `bargain.utilities.BargainProblem` with
g=K, H=stack(M_j), p_reduced=n, solved by the existing (unmodified) damped-
Newton solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cl_bargain.bargain.solver import solve
from cl_bargain.bargain.utilities import BargainProblem


def sample_batch(idx, batch_size: int, generator):
    import torch

    n = int(idx.numel())
    picks = torch.randint(0, n, (min(batch_size, n),), device=idx.device, generator=generator)
    return idx[picks]


def stream_loss(stream, out, y):
    """Training cross-entropy. On a task-masked stream every example competes
    only with the classes of its own task (task-incremental training); on an
    unmasked stream it competes with the whole head."""
    import torch.nn.functional as f

    mask = getattr(stream, "class_mask", None)
    if mask is None:
        return f.cross_entropy(out, y)
    return f.cross_entropy(out.masked_fill(~mask[y], -1e9), y)


def task_loss(model, stream, test_idx, lo: int, hi: int):
    """Mean cross-entropy on the TEST set, task-incrementally restricted to
    [lo, hi). `test_idx` must index stream.x_test/y_test (e.g. task.test_idx)
    -- this is an evaluation-only helper, never used on training indices."""
    import torch.nn.functional as f

    x = stream.normalise(stream.x_test[test_idx])
    y = stream.y_test[test_idx] - lo
    out = model(x)[:, lo:hi]
    return f.cross_entropy(out, y)


def probe_loss(model, stream, probe_idx, lo: int, hi: int):
    """Mean cross-entropy on a task's PROBE slice.
    "Probe slice"), task-incrementally restricted to [lo, hi). `probe_idx`
    indexes stream.x_train/y_train (e.g. `ProbeReplayBuffer.probe_held(t)`)
    -- the probe slice is never replayed, so evaluating it here does not
    reintroduce the circularity `task_loss` avoids by using the held-out
    TEST set; the probe slice plays the analogous role for a quantity
    (per-step loss CHANGE during training) the test set cannot measure
    without re-running the whole stream."""
    import torch.nn.functional as f

    x = stream.normalise(stream.x_train[probe_idx])
    y = stream.y_train[probe_idx] - lo
    out = model(x)[:, lo:hi]
    return f.cross_entropy(out, y)


def task_accuracy_corrected(
    model, stream, test_idx, lo: int, hi: int, chunk_size: int = 512
) -> float:
    """Task-incremental accuracy on the TEST set, chance-corrected:
    (acc - 1/k)/(1 - 1/k) where k = hi-lo is the task's class count
    Sec 9): on an imbalanced stream, an uncorrected average measures the
    class split as much as the method, since a k=2 task's raw accuracy
    floor (50%) is far higher than a k=20 task's (5%)."""
    import torch

    k = hi - lo
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, test_idx.numel(), chunk_size):
            chunk = test_idx[start : start + chunk_size]
            x = stream.normalise(stream.x_test[chunk])
            y = stream.y_test[chunk] - lo
            out = model(x)[:, lo:hi]
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(chunk.numel())
    acc = correct / total
    chance = 1.0 / k
    return (acc - chance) / (1.0 - chance)


@dataclass
class RunState:
    """Persistent state a method's training step needs across the stream."""

    # task -> (theta_star, fisher), both param dicts
    ewc_snapshots: dict = field(default_factory=dict)


def er_step(model, opt, stream, idx, buffer, generator, minibatch_size: int, device) -> None:
    import torch

    x = stream.augment(stream.x_train[idx], generator)
    y = stream.y_train[idx]

    opt.zero_grad(set_to_none=True)
    if len(buffer) > 0:
        b_idx = sample_batch(buffer.idx[: buffer.filled], minibatch_size, generator)
        bx = stream.augment(stream.x_train[b_idx], generator)
        by = stream.y_train[b_idx]
        out = model(torch.cat([x, bx]))
        loss = stream_loss(stream, out, torch.cat([y, by]))
    else:
        loss = stream_loss(stream, model(x), y)
    loss.backward()
    opt.step()


def agem_step(model, opt, stream, idx, buffer, generator, minibatch_size: int, device) -> None:
    """A-GEM: project the current gradient onto the half-space of non-negative
    inner product with a reference gradient from a buffer batch, only when
    the two conflict (docs: Lopez-Paz & Ranzato 2017)."""
    import torch

    from cl_bargain.streams.paramvec import flatten_grad

    x = stream.augment(stream.x_train[idx], generator)
    y = stream.y_train[idx]

    if len(buffer) == 0:
        opt.zero_grad(set_to_none=True)
        stream_loss(stream, model(x), y).backward()
        opt.step()
        return

    b_idx = sample_batch(buffer.idx[: buffer.filled], minibatch_size, generator)
    bx = stream.augment(stream.x_train[b_idx], generator)
    by = stream.y_train[b_idx]

    opt.zero_grad(set_to_none=True)
    stream_loss(stream, model(bx), by).backward()
    g_ref = flatten_grad(model).clone()

    opt.zero_grad(set_to_none=True)
    stream_loss(stream, model(x), y).backward()
    g = flatten_grad(model)

    dot = torch.dot(g, g_ref)
    if dot < 0:
        g = g - (dot / torch.dot(g_ref, g_ref).clamp_min(1e-12)) * g_ref

    off = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = g[off : off + n].reshape(p.shape).clone()
        off += n
    opt.step()


def ewc_snapshot(
    model, stream, task, fisher_chunk_size: int, generator, max_examples: int = 300
) -> tuple[dict, dict]:
    """Diagonal Fisher estimated from a bounded random subsample of the
    task's data, not the whole thing -- common practice for empirical
    Fisher, and keeps the per-example-loop cost (see
    bargain/curvature.py's diagonal_empirical_fisher) bounded regardless of
    task size."""
    from cl_bargain.bargain.curvature import diagonal_empirical_fisher

    model.eval()
    idx = sample_batch(task.train_idx, max_examples, generator)
    x = stream.normalise(stream.x_train[idx])
    y = stream.y_train[idx]
    lo, hi = (
        stream.task_range(task.index)
        if getattr(stream, "class_mask", None) is not None
        else (0, stream.n_classes)
    )
    fisher = diagonal_empirical_fisher(model, x, y, lo=lo, hi=hi, chunk_size=fisher_chunk_size)
    theta_star = {k: v.detach().clone() for k, v in model.named_parameters()}
    model.train()
    return theta_star, {k: v.detach().clone() for k, v in fisher.items()}


def ewc_step(
    model, opt, stream, idx, buffer, generator, minibatch_size: int, device,
    state: RunState, lam: float,
) -> None:

    x = stream.augment(stream.x_train[idx], generator)
    y = stream.y_train[idx]

    opt.zero_grad(set_to_none=True)
    loss = stream_loss(stream, model(x), y)
    if state.ewc_snapshots:
        penalty = 0.0
        for theta_star, fisher in state.ewc_snapshots.values():
            for name, p in model.named_parameters():
                penalty = penalty + (fisher[name] * (p - theta_star[name]) ** 2).sum()
        # Mean over the retained tasks, not sum: with one penalty per past
        # task the summed version multiplies lam by the number of tasks seen,
        # so the same lam freezes the network on a 20-task stream while
        # barely constraining it on a 5-task one.
        loss = loss + 0.5 * lam * penalty / len(state.ewc_snapshots)
    loss.backward()
    opt.step()


def _gram_curvature_jvp_only(model, g_list, x_list, n_classes: int, chunk_size: int):
    """G H_j G^T for every player j, using forward-mode products ONLY.

    The default path in `_flat_gram_and_curvature` calls ggn_vector_product
    once per (j, a) pair, and each of those calls is one JVP followed by one
    VJP: n^2 forward passes plus n^2 backward passes per step.

    Nothing in the reduced problem needs H_j v in parameter space, though.
    Only the n x n numbers g_b^T H_j g_a are used, and with H_j the
    Gauss-Newton operator (1/N) sum J^T W J, W = diag(p) - p p^T, those are

        g_b^T H_j g_a = (1/N) sum_examples (J g_b)^T W (J g_a),

    which needs the directional derivatives J g_a and no backward pass at
    all. This computes them with n JVPs per player, contracts them into the
    n x n block, and never touches a VJP. The result is EXACT, not an
    approximation: it is the same quantity, computed through the other side
    of the same bilinear form (checked against the default path in
    tests/test_methods_torch.py).
    """
    import torch
    from torch.func import functional_call, jvp

    from cl_bargain.streams.paramvec import flat_to_dict, param_shapes

    shapes = param_shapes(model)
    names = list(shapes.keys())
    n = len(g_list)
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    tangents = [flat_to_dict(g_list[a], names, shapes) for a in range(n)]

    m_stack = np.zeros((n, n, n), dtype=np.float64)
    for j in range(n):
        x = x_list[j]
        acc = torch.zeros((n, n), dtype=torch.float64, device=g_list[0].device)
        n_total = 0
        for start in range(0, x.shape[0], chunk_size):
            xb = x[start : start + chunk_size]

            def f(p, xb=xb):
                return functional_call(model, (p, buffers), (xb,))[:, 0:n_classes]

            jvs = []
            out = None
            for a in range(n):
                out, jv = jvp(f, (params,), (tangents[a],))
                jvs.append(jv)
            probs = out.softmax(-1)
            # W jv_a = p * jv_a - p * <p, jv_a>, the Gauss-Newton middle factor
            # applied in output space, where it costs nothing.
            w_jvs = [probs * jv - probs * (probs * jv).sum(-1, keepdim=True) for jv in jvs]
            for a in range(n):
                for b in range(a, n):
                    val = (jvs[b].double() * w_jvs[a].double()).sum()
                    acc[b, a] += val
                    if b != a:
                        acc[a, b] += val
            n_total += xb.shape[0]
        m_stack[j] = (acc / max(n_total, 1)).detach().cpu().numpy()

    m_stack = 0.5 * (m_stack + np.transpose(m_stack, (0, 2, 1)))
    m_stack += 1e-6 * np.eye(n)[None]
    return m_stack


def _has_batch_coupled_norm(model) -> bool:
    import torch

    # In eval mode BatchNorm uses its running statistics and acts per example.
    return any(
        isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
        for m in model.modules()
    )


def _gram_curvature_jvp_batched(
    model, g_list, x_list, n_classes: int, max_batch: int = 192, class_masks=None
):
    """The same n x n x n block as `_gram_curvature_jvp_only`, with every
    player's examples in one batch.

    `_gram_curvature_jvp_only` runs n JVPs on each player's own batch: n^2
    forward passes of a few dozen examples each, which on a GPU costs launch
    latency rather than arithmetic. When every normalisation layer is
    per-example (GroupNorm, the default backbone), the output on one example
    does not depend on which other examples share its batch, so the rows of
    J g_a are the same whether computed per player or on the concatenation of
    all players' examples. n JVPs over the concatenated batch then replace the
    n^2 per-player ones, and player j's block is the per-example contraction
    summed over j's own rows only. With a BatchNorm layer the concatenation
    would couple players' examples and change the answer, so the caller falls
    back to the per-player path in that case.

    TF32 convolutions are disabled for the duration of the call. With them on
    (the PyTorch default on Ampere), the block differs from its FP32 value by
    about 7e-4 relative, and by an amount that changes with the batch shape,
    so the two paths would disagree for a reason unrelated to the algebra.
    With them off the batched block matches the FP32 per-player block to
    about 1e-8 relative on the reduced ResNet-18.

    class_masks: one boolean vector over the head per player, for a
    task-incremental stream on which player j's loss sees only its own task's
    logits. The softmax is then taken over those logits, which zeroes the
    Gauss-Newton middle factor W outside them, so the block is player j's
    Gauss-Newton operator on its own task head.
    """
    import torch

    from cl_bargain.streams.paramvec import flat_to_dict, param_shapes

    shapes = param_shapes(model)
    names = list(shapes.keys())
    n = len(g_list)
    device = g_list[0].device
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    tangents = [flat_to_dict(g_list[a], names, shapes) for a in range(n)]

    x_all = torch.cat(list(x_list))
    owner = torch.cat(
        [
            torch.full((x.shape[0],), j, dtype=torch.long, device=device)
            for j, x in enumerate(x_list)
        ]
    )
    counts = torch.tensor([x.shape[0] for x in x_list], dtype=torch.float64, device=device)
    row_mask = None
    if class_masks is not None:
        row_mask = torch.stack([m[:n_classes] for m in class_masks])[owner]

    acc = torch.zeros((n, n, n), dtype=torch.float64, device=device)
    tf32_before = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        _accumulate_batched_blocks(
            model, params, buffers, tangents, x_all, owner, acc, n_classes, max_batch, row_mask
        )
    finally:
        torch.backends.cudnn.allow_tf32 = tf32_before

    m_stack = (acc / counts.clamp_min(1.0)[:, None, None]).cpu().numpy()
    m_stack = 0.5 * (m_stack + np.transpose(m_stack, (0, 2, 1)))
    m_stack += 1e-6 * np.eye(n)[None]
    return m_stack


def _accumulate_batched_blocks(
    model, params, buffers, tangents, x_all, owner, acc, n_classes: int, max_batch: int,
    row_mask=None,
) -> None:
    import torch
    from torch.func import functional_call, jvp

    n = len(tangents)
    for start in range(0, x_all.shape[0], max_batch):
        xb = x_all[start : start + max_batch]

        def f(p, xb=xb):
            return functional_call(model, (p, buffers), (xb,))[:, 0:n_classes]

        jvs = []
        out = None
        for a in range(n):
            out, jv = jvp(f, (params,), (tangents[a],))
            jvs.append(jv)
        if row_mask is not None:
            out = out.masked_fill(~row_mask[start : start + max_batch], -1e9)
        probs = out.softmax(-1)
        j_stack = torch.stack(jvs)  # (n, B, C)
        w_stack = probs * j_stack - probs * (probs * j_stack).sum(-1, keepdim=True)
        # per_example[e, b, a] = (J g_b)_e^T W_e (J g_a)_e
        per_example = torch.einsum("bec,aec->eba", j_stack.double(), w_stack.double())
        # Each player's rows are contiguous in x_all, so a slice sum per player
        # gives the segment sum in a fixed order; index_add_ on CUDA
        # accumulates with atomics and made runs non-reproducible.
        own = owner[start : start + max_batch]
        for j in torch.unique_consecutive(own).tolist():
            acc[j] += per_example[own == j].sum(0)


def _flat_gram_and_curvature(
    model,
    stream,
    g_list,
    x_list,
    n_classes: int,
    chunk_size: int,
    need_curvature: bool,
    jvp_only: bool = False,
    jvp_batched: bool = False,
    class_masks=None,
):
    import torch

    from cl_bargain.bargain.curvature import ggn_vector_product
    from cl_bargain.streams.paramvec import flat_to_dict, param_shapes

    if torch.cuda.is_available():
        # jvp/vjp occasionally trip a spurious CUDA OOM on a fragmented
        # allocator cache even at trivial total memory use (see
        # bargain/curvature.py's diagonal_empirical_fisher for the same
        # symptom); called once here rather than inside the O(n^2)
        # ggn_vector_product loop below, since this is a training-time hot
        # path and empty_cache() is not free.
        torch.cuda.empty_cache()

    n = len(g_list)
    shapes = param_shapes(model)
    # Must match model.named_parameters()' own insertion order, NOT sorted:
    # torch.func.jvp requires the tangent dict's key order to match the
    # primal dict's exactly (caught during E4's toy-model validation, see
    # tests/test_curvature_torch.py and bargain/curvature.py's flatten_dict,
    # which sorts only for its OWN internal self-consistent use, never for
    # feeding into jvp/vjp directly).
    names = list(shapes.keys())
    stacked = torch.stack(g_list)  # (n, P)
    k_mat = (stacked @ stacked.T).detach().cpu().numpy().astype(np.float64)

    if not need_curvature:
        return k_mat, None

    if jvp_batched and not _has_batch_coupled_norm(model):
        return k_mat, _gram_curvature_jvp_batched(
            model, g_list, x_list, n_classes, class_masks=class_masks
        )
    if class_masks is not None:
        raise ValueError("per-player class masks are implemented on the batched path only")

    if jvp_only or jvp_batched:
        return k_mat, _gram_curvature_jvp_only(model, g_list, x_list, n_classes, chunk_size)

    m_stack = np.zeros((n, n, n), dtype=np.float64)
    for j in range(n):
        for a in range(n):
            v_dict = flat_to_dict(g_list[a], names, shapes)
            hv = ggn_vector_product(
                model, x_list[j], 0, n_classes, v_dict, chunk_size=chunk_size, weighted=True
            )
            hv_flat = torch.cat([hv[k].reshape(-1) for k in names])
            m_stack[j, :, a] = (stacked @ hv_flat).detach().cpu().numpy().astype(np.float64)
    # symmetrise (numerical noise from the two-sided jvp/vjp construction) and
    # add a small jitter -- G H_j G^T is PSD by congruence since H_j (a GGN)
    # is PSD, but floating-point noise can leave a tiny negative eigenvalue.
    m_stack = 0.5 * (m_stack + np.transpose(m_stack, (0, 2, 1)))
    m_stack += 1e-6 * np.eye(n)[None]
    return k_mat, m_stack


def bargain_step(
    model,
    stream,
    idx0,
    task_index: int,
    buffer,
    generator,
    minibatch_size: int,
    device,
    method: str,
    knob: float,
    chunk_size: int,
    zeta_floor: float = 0.0,
    max_active_players: int | None = None,
    jvp_only: bool = False,
    jvp_batched: bool = False,
) -> tuple[bool, dict | None]:
    """One v1/v2/nashmtl bargain-based update, reduced to span(active
    players' gradients). Returns (applied, diagnostics): applied is False if
    the bargain had no feasible/converged solution (see the nashmtl branch
    below) or n==1 diagnostics is None; diagnostics carries the solved
    problem, its Delta (in the normalised-gradient basis) and the per-player
    rescale factors `c`, which bargain/certificate.py's certified_damage
    needs to report damage in the ORIGINAL units
    "Numerical preconditioning without breaking the units", final sentence).

    zeta_floor: uniform curvature-uncertainty floor zeta_i
    Assumption "Curvature-uncertainty floor", zeta_i >= zeta_min > 0), applied
    to the "v2" method only (v1/nashmtl have no curvature term). Default 0.0
    preserves the exact behaviour of every existing caller (E2, E6) that does
    not pass it; E-cert (bargain/certificate.py's consumers) passes a real
    floor.

    max_active_players: optional cap on the number of absent players bargained
    with at each step. Players are subsampled uniformly through `generator` when
    more are available. This is an approximation, not a reduction of the exact
    method: excluded players are omitted from that update.
    before being relied on (see tests/test_methods_torch.py and E-cert's own
    report of the induced deviation). Default None preserves exact behaviour
    for every existing caller (E2, E6)."""
    import torch

    from cl_bargain.streams.paramvec import add_flat_, flatten_grad

    active = sorted(buffer.task_sizes.keys()) if len(buffer) > 0 else []
    active = [t for t in active if t != task_index]
    if max_active_players is not None and len(active) > max_active_players:
        perm = torch.randperm(len(active), device=device, generator=generator).tolist()
        active = sorted(active[i] for i in perm[:max_active_players])

    x0 = stream.augment(stream.x_train[idx0], generator)
    y0 = stream.y_train[idx0]
    model.zero_grad(set_to_none=True)
    stream_loss(stream, model(x0), y0).backward()
    g_list = [flatten_grad(model).clone()]
    x_list = [stream.normalise(stream.x_train[idx0])]

    for t in active:
        held = buffer.held(t)
        idx_i = sample_batch(held, minibatch_size, generator)
        xi = stream.augment(stream.x_train[idx_i], generator)
        yi = stream.y_train[idx_i]
        model.zero_grad(set_to_none=True)
        stream_loss(stream, model(xi), yi).backward()
        g_list.append(flatten_grad(model).clone())
        x_list.append(stream.normalise(stream.x_train[idx_i]))
    model.zero_grad(set_to_none=True)

    n = len(g_list)
    if n == 1:
        # No absent tasks yet: this reduces to plain gradient descent with
        # the knob playing the role of a step size (tau/epsilon both bound
        # how far the update may move against the live task's own loss).
        add_flat_(model, -knob * g_list[0] / (g_list[0].norm() + 1e-12))
        return True, None

    # Rescale each player's basis vector to unit norm -- Lemma "Per-player
            # scale invariance: c_i = 1/||g_i|| applied to
    # (g_i, tau_i) TOGETHER leaves Delta* exactly unchanged while making K's
    # diagonal O(1) regardless of each player's raw gradient scale (K's
    # diagonal was ~65 pre-fix, swamping a tau_base of 0.01 and diverging to
    # NaN in E2's calibration pass). Scaling g_i ALONE and leaving tau fixed
    # at `knob` for every player -- the previous behaviour here -- is exactly
    # what Remark "Numerical preconditioning without breaking the units"
    # warns against: it silently redenominates tau_i by ||g_i||, so the same
    # `knob` means a different real budget for every player depending on
    # their incidental raw gradient norm, and any certificate computed
    # downstream would be in meaningless units.
    raw_norms = torch.stack([g.norm() for g in g_list])
    c = 1.0 / (raw_norms + 1e-12)
    c_np = c.detach().cpu().numpy().astype(np.float64)
    g_list = [g * c[i] for i, g in enumerate(g_list)]

    k_mat, m_stack = _flat_gram_and_curvature(
        model,
        stream,
        g_list,
        x_list,
        stream.n_classes,
        chunk_size,
        need_curvature=(method == "v2"),
        jvp_only=jvp_only,
        jvp_batched=jvp_batched,
        class_masks=(
            None
            if getattr(stream, "class_mask", None) is None
            else [
                stream.class_mask[stream.task_range(t)[0]]
                for t in [task_index, *active]
            ]
        ),
    )
    if m_stack is not None:
        # _flat_gram_and_curvature evaluates player j's curvature via
        # ggn_vector_product against the (already basis-rescaled) vectors
        # g_list, which correctly picks up a c_i*c_a factor on each entry
        # m_stack[j][i,a] from the TWO basis-vector indices -- but it uses
        # the TRUE (unscaled) hat_H_j itself, so it is still missing the
        # THIRD factor c_j that Lemma "Per-player scale invariance" requires
        # for player j's OWN curvature (hat_H_j -> c_j hat_H_j), exactly
        # parallel to tau_j -> c_j tau_j above. Caught by
        # tests/test_methods_torch.py's raw-vs-rescaled equivalence check.
        m_stack = m_stack * c_np[:, None, None]
    alpha = np.full(n, 1.0 / n)

    # tau_i = knob for absent players in the ORIGINAL (unscaled) problem;
    # rescale by the SAME c_i applied to g_i, per Lemma "Per-player scale
    # invariance", so Delta* is recovered exactly rather than redefined.
    tau_original = np.zeros(n)
    tau_original[1:] = knob
    tau = tau_original * c_np

    # Orthonormal coordinates for span(g_i). In beta, Delta = G^T beta has
    # ||Delta||^2 = beta^T K beta, not ||beta||^2, because the unit-norm basis
    # vectors are not orthogonal; solving in beta therefore measured the
    # program's norm terms (zeta_i/2 ||Delta||^2, the v1 trust region) in the
    # wrong metric. With K = V diag(lam) V^T and beta = V lam^{-1/2} gamma,
    # ||Delta|| = ||gamma||, <g_i, Delta> = (V lam^{1/2} gamma)_i and
    # Delta^T H_j Delta = gamma^T (T^T M_j T) gamma with T = V lam^{-1/2}.
    lam, eigvec = np.linalg.eigh(k_mat)
    keep = lam > 1e-10 * lam.max()
    to_beta = eigvec[:, keep] / np.sqrt(lam[keep])
    g_red = eigvec[:, keep] * np.sqrt(lam[keep])
    k_dim = int(keep.sum())

    if method in ("v1", "nashmtl"):
        from cl_bargain.bargain.baselines import solve_v1_trust_region

        if method == "nashmtl":
            # naive Nash-MTL ablation: d_i=0 for EVERY
            # player, no live-task protection -- tau stays all-zero
            # regardless of c (0*c_i == 0).
            tau = np.zeros(n)
        result = solve_v1_trust_region(g_red, tau, alpha, epsilon=knob)
        gamma = result.Delta
        if not (result.converged and result.feasible):
            # No common improving direction found within the trust region
            # (observed on ~25-60% of random instances for the fully
            # symmetric nashmtl case in isolated testing -- a real
            # structural property of some gradient configurations, not
            # purely a solver-robustness gap). Skip this step's update
            # rather than applying a meaningless Delta.
            return False, None
        # Equivalent BargainProblem (H=0, first-order utilities) so
        # bargain/certificate.py's certified_damage/admission_scale work
        # identically for v1/nashmtl diagnostics as for v2 -- used by
        # E-cert's "first-order utilities, d_i=0" ablation
        # (experiments/e8_certificate.py's `run_one(method="nashmtl")`).
        problem = BargainProblem(
            g=g_red, H=np.zeros((n, k_dim, k_dim)), tau=tau, alpha=alpha
        )
    else:
        # zeta_floor > 0 satisfies Assumption "Curvature-uncertainty floor"
        # (zeta_i >= zeta_min > 0): without it, lambda_min(H_bar) can be
        # exactly zero whenever p exceeds the players' total curvature rank
        # (the generic case at network scale), voiding both the uniqueness
        # clause of Theorem "Convexity, attainment, and uniqueness" and the
        # representer property's rho > 0 requirement. Rescaled by the same
        # c_i as tau, per Lemma "Per-player scale invariance".
        zeta = np.full(n, zeta_floor) * c_np
        m_red = np.einsum("ab,jbc,cd->jad", to_beta.T, m_stack, to_beta)
        m_red = 0.5 * (m_red + np.transpose(m_red, (0, 2, 1)))
        problem = BargainProblem(g=g_red, H=m_red, tau=tau, alpha=alpha, zeta=zeta)
        result = solve(problem)
        gamma = result.Delta
        if not result.converged:
            return False, None

    beta = to_beta @ gamma
    beta_t = torch.as_tensor(beta, dtype=g_list[0].dtype, device=device)
    stacked = torch.stack(g_list)
    delta_flat = beta_t @ stacked
    add_flat_(model, delta_flat)

    # For bargain/certificate.py's certified_damage: `c` recovers
    # original-unit damage (divide by c_i), `active` maps result rows
    # (index 0 = live task) to buffer task ids (index 1.. = active[0..]).
    # `delta_flat` is the applied update in FULL parameter space;
    # ||delta_flat|| = ||gamma|| up to float error, and it is q_t in
    # prop:calibration's eq:calibration-lp. "beta" is gamma, the coordinates
    # `problem` is posed in; "beta_gram" is the same update in the basis of
    # rescaled gradients.
    # V1Result (v1/nashmtl) has no .shadow_prices attribute -- recompute it
    # the same way BargainProblem.shadow_prices does (alpha/slack), so
    # diagnostics has an identical shape regardless of method.
    slacks = result.slacks
    shadow_prices = getattr(result, "shadow_prices", None)
    if shadow_prices is None:
        shadow_prices = alpha / np.maximum(slacks, 1e-300)
    diagnostics = {
        "problem": problem,
        "beta": gamma,
        "beta_gram": beta,
        "c": c_np,
        "active": active,
        "slacks": slacks,
        "shadow_prices": shadow_prices,
        "delta_flat": delta_flat.detach(),
        "delta_norm": float(delta_flat.norm()),
    }
    return True, diagnostics


__all__ = [
    "RunState",
    "agem_step",
    "bargain_step",
    "er_step",
    "ewc_snapshot",
    "ewc_step",
    "sample_batch",
    "task_loss",
]
