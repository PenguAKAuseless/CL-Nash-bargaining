"""Continual-learning update rules; bargain methods solve in the span of the players' gradients."""

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
    """Training cross-entropy, restricted to the example's own task logits if task-masked."""
    import torch.nn.functional as f

    mask = getattr(stream, "class_mask", None)
    if mask is None:
        return f.cross_entropy(out, y)
    return f.cross_entropy(out.masked_fill(~mask[y], -1e9), y)


def task_loss(model, stream, test_idx, lo: int, hi: int):
    """Mean test-set cross-entropy, task-incrementally restricted to [lo, hi)."""
    import torch.nn.functional as f

    x = stream.normalise(stream.x_test[test_idx])
    y = stream.y_test[test_idx] - lo
    out = model(x)[:, lo:hi]
    return f.cross_entropy(out, y)


def probe_loss(model, stream, probe_idx, lo: int, hi: int):
    """Mean cross-entropy on a task's probe slice, restricted to [lo, hi)."""
    import torch.nn.functional as f

    x = stream.normalise(stream.x_train[probe_idx])
    y = stream.y_train[probe_idx] - lo
    out = model(x)[:, lo:hi]
    return f.cross_entropy(out, y)


def task_accuracy_corrected(
    model, stream, test_idx, lo: int, hi: int, chunk_size: int = 512
) -> float:
    """Chance-corrected task-incremental test accuracy, (acc - 1/k) / (1 - 1/k)."""
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
    """A-GEM: project out the gradient component that conflicts with a buffer reference gradient."""
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
    """Parameter snapshot and diagonal Fisher from a bounded random subsample of the task."""
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
        # mean over retained tasks, so lam does not scale with the number of tasks
        loss = loss + 0.5 * lam * penalty / len(state.ewc_snapshots)
    loss.backward()
    opt.step()


def _gram_curvature_jvp_only(model, g_list, x_list, n_classes: int, chunk_size: int):
    """G H_j G^T for every player j via forward-mode JVPs only (exact, no VJP)."""
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
    """Same block as `_gram_curvature_jvp_only` from one concatenated batch."""
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
    # optional per-player task-head mask for task-incremental streams
    row_mask = None
    if class_masks is not None:
        row_mask = torch.stack([m[:n_classes] for m in class_masks])[owner]

    acc = torch.zeros((n, n, n), dtype=torch.float64, device=device)
    # TF32 convolutions make the block depend on batch shape
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
        # fixed-order per-player slice sums; index_add_ on CUDA is non-deterministic
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
        # avoid spurious OOMs from a fragmented allocator cache
        torch.cuda.empty_cache()

    n = len(g_list)
    shapes = param_shapes(model)
    # must match named_parameters() order: jvp requires tangent keys in primal order
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
    # symmetrise and jitter: G H_j G^T is PSD but float noise can break that slightly
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
    trust_radius: float | None = None,
) -> tuple[bool, dict | None]:
    """One bargain update in the span of the players' gradients; returns (applied, diagnostics)."""
    import torch

    from cl_bargain.streams.paramvec import add_flat_, flatten_grad
    from cl_certify.timing import section

    active = sorted(buffer.task_sizes.keys()) if len(buffer) > 0 else []
    active = [t for t in active if t != task_index]
    if max_active_players is not None and len(active) > max_active_players:
        perm = torch.randperm(len(active), device=device, generator=generator).tolist()
        active = sorted(active[i] for i in perm[:max_active_players])

    with section("gradient"):
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
        # no absent tasks: normalised gradient step of length knob
        add_flat_(model, -knob * g_list[0] / (g_list[0].norm() + 1e-12))
        return True, None

    # rescale g_i (and tau_i, H_i below) by c_i = 1/||g_i||; this leaves Delta* unchanged
    raw_norms = torch.stack([g.norm() for g in g_list])
    c = 1.0 / (raw_norms + 1e-12)
    c_np = c.detach().cpu().numpy().astype(np.float64)
    g_list = [g * c[i] for i, g in enumerate(g_list)]

    with section("curvature"):
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
        # player j's own curvature also needs its factor c_j, on top of the basis factors
        m_stack = m_stack * c_np[:, None, None]
    alpha = np.full(n, 1.0 / n)

    # tau_i = knob for absent players, rescaled by the same c_i as g_i
    tau_original = np.zeros(n)
    tau_original[1:] = knob
    tau = tau_original * c_np

    # orthonormal coordinates gamma for span(g_i): beta = V lam^{-1/2} gamma, ||Delta|| = ||gamma||
    lam, eigvec = np.linalg.eigh(k_mat)
    keep = lam > 1e-10 * lam.max()
    to_beta = eigvec[:, keep] / np.sqrt(lam[keep])
    g_red = eigvec[:, keep] * np.sqrt(lam[keep])
    k_dim = int(keep.sum())

    if method in ("v1", "nashmtl"):
        from cl_bargain.bargain.baselines import solve_v1_trust_region

        if method == "nashmtl":
            # naive Nash-MTL ablation: d_i = 0 for every player
            tau = np.zeros(n)
        # epsilon is a step radius, knob a damage budget; trust_radius decouples them
        eps = knob if trust_radius is None else trust_radius
        with section("solver"):
            result = solve_v1_trust_region(g_red, tau, alpha, epsilon=eps)
        gamma = result.Delta
        if not (result.converged and result.feasible):
            # no common improving direction within the trust region: skip the update
            return False, None
        # equivalent first-order BargainProblem (H = 0) for downstream diagnostics
        problem = BargainProblem(
            g=g_red, H=np.zeros((n, k_dim, k_dim)), tau=tau, alpha=alpha
        )
    else:
        # zeta_floor > 0 keeps the program strongly concave when H_bar is rank-deficient
        zeta = np.full(n, zeta_floor) * c_np
        m_red = np.einsum("ab,jbc,cd->jad", to_beta.T, m_stack, to_beta)
        m_red = 0.5 * (m_red + np.transpose(m_red, (0, 2, 1)))
        problem = BargainProblem(g=g_red, H=m_red, tau=tau, alpha=alpha, zeta=zeta)
        with section("solver"):
            result = solve(problem)
        gamma = result.Delta
        if not result.converged:
            return False, None

    # ||Delta|| = ||gamma||, so the step radius is enforced here for v2 too
    gamma_norm = float(np.linalg.norm(gamma))
    radius_binding = False
    if trust_radius is not None and gamma_norm > trust_radius:
        gamma = gamma * (trust_radius / gamma_norm)
        radius_binding = True

    beta = to_beta @ gamma
    beta_t = torch.as_tensor(beta, dtype=g_list[0].dtype, device=device)
    stacked = torch.stack(g_list)
    delta_flat = beta_t @ stacked
    add_flat_(model, delta_flat)

    # c converts damage back to original units (divide by c_i); active maps rows 1.. to task ids
    # V1Result has no shadow_prices, so recompute alpha / slack
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
        "gamma_norm_presolve": gamma_norm,
        "radius_binding": radius_binding,
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
