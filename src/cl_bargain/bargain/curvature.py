"""Curvature construction, PSD helpers, and torch-backed curvature estimators."""

from __future__ import annotations

import numpy as np


def random_psd(rng: np.random.Generator, p: int, rank: int) -> np.ndarray:
    """H = A^T A / rank, PSD by construction, effective rank ~ rank."""
    A = rng.standard_normal((rank, p))
    return A.T @ A / rank


def damp(H: np.ndarray, eps: float) -> np.ndarray:
    """H + eps*I -- keeps a PSD operator strictly positive definite."""
    p = H.shape[-1]
    return H + eps * np.eye(p)


def min_eigval(H: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(H).min())



def ggn_vector_product(
    model, x, lo: int, hi: int, v_dict, chunk_size: int = 128, weighted: bool = True
):
    """(GGN of the task-restricted logits) @ v; weighted=False gives (J^T J) @ v instead."""
    import torch
    from torch.func import functional_call, jvp, vjp

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    total = {k: torch.zeros_like(v) for k, v in v_dict.items()}
    n_total = 0
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        n = xb.shape[0]

        def f(p, xb=xb):
            return functional_call(model, (p, buffers), (xb,))[:, lo:hi]

        out, jv = jvp(f, (params,), (v_dict,))
        if weighted:
            probs = out.softmax(-1)
            u = probs * jv - probs * (probs * jv).sum(-1, keepdim=True)
        else:
            u = jv
        _, vjp_fn = vjp(f, params)
        (hv_chunk,) = vjp_fn(u)
        # vjp already sums over the chunk; normalise by the total count at the end
        for k in total:
            total[k] = total[k] + hv_chunk[k]
        n_total += n
    return {k: v_ / n_total for k, v_ in total.items()}


def jacobian_outer_vector_product(model, x, lo: int, hi: int, v_dict, chunk_size: int = 128):
    """Function-space Jacobian estimator: (J^T J) @ v. See `ggn_vector_product`."""
    return ggn_vector_product(model, x, lo, hi, v_dict, chunk_size=chunk_size, weighted=False)


def diagonal_empirical_fisher(model, x, y, lo: int, hi: int, chunk_size: int = 64):
    """Diagonal empirical Fisher of the task-restricted cross-entropy (per-example loop)."""
    import torch
    import torch.nn.functional as f

    params = list(model.parameters())
    names = [name for name, _ in model.named_parameters()]
    total = [torch.zeros_like(p) for p in params]
    n = x.shape[0]
    was_training = model.training
    model.eval()
    for i in range(n):
        model.zero_grad(set_to_none=True)
        out = model(x[i : i + 1])[:, lo:hi]
        loss = f.cross_entropy(out, y[i : i + 1] - lo)
        grads = torch.autograd.grad(loss, params)
        for t, g in zip(total, grads, strict=True):
            t += g.detach() ** 2
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return {name: t / n for name, t in zip(names, total, strict=True)}


def random_probe(param_shapes: dict, rng, device):
    """Random parameter-shaped direction with unit L2 norm."""
    import torch

    v = {
        k: torch.as_tensor(rng.standard_normal(shape), dtype=torch.float32, device=device)
        for k, shape in param_shapes.items()
    }
    norm = flatten_dict(v).norm()
    return {k: t / norm for k, t in v.items()}


def flatten_dict(d):
    import torch

    return torch.cat([d[k].reshape(-1) for k in sorted(d)])


def quadratic_form(hv_dict, v_dict) -> float:
    """v^T H v given Hv and v as matching parameter-shaped dicts."""
    return float(flatten_dict(hv_dict).double() @ flatten_dict(v_dict).double())


def diag_quadratic_form(diag_dict, v_dict) -> float:
    """v^T diag(F) v given a diagonal Fisher dict and v as a parameter-shaped dict."""
    total = 0.0
    for k in diag_dict:
        total += float((diag_dict[k].double() * v_dict[k].double() ** 2).sum())
    return total


def trace_estimate(hv_fn, param_shapes, rng, device, n_probes: int = 8) -> float:
    """Hutchinson trace estimate: E[v^T H v] = tr(H) for v ~ N(0, I)."""
    import torch

    total = 0.0
    for _ in range(n_probes):
        v = {
        k: torch.as_tensor(rng.standard_normal(shape), dtype=torch.float32, device=device)
        for k, shape in param_shapes.items()
    }
        hv = hv_fn(v)
        total += float(flatten_dict(hv).double() @ flatten_dict(v).double())
    return total / n_probes
