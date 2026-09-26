"""Curvature construction and PSD utilities.

The pure-numpy pieces (E0/E1, closed-form sandbox curvature) are at the top
of this file and keep the module importable with no torch installed. The
torch-backed estimators below import torch locally inside
each function body -- this file and streams/cifar.py are the only two
places in this repo allowed to import torch (see pyproject.toml).

Candidates, in order of preference:
  1. function-space Jacobian J^T J on stored inputs (`jacobian_outer_vector_product`)
  2. GGN on the buffer (`ggn_vector_product`) -- expected to collapse, included
    to demonstrate it: under cross-entropy, GGN's middle factor
     diag(p)-pp^T -> 0 as the model overfits the exact examples it is replayed on)
  3. diagonal empirical Fisher snapshotted at the task boundary (`diagonal_empirical_fisher`)
  4. K-FAC -- not implemented (lowest preference; out of scope for this pass)
"""

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


# ---- torch-backed estimators (E4) -----------------------------------------


def ggn_vector_product(
    model, x, lo: int, hi: int, v_dict, chunk_size: int = 128, weighted: bool = True
):
    """(GGN of the task-restricted logits) @ v, batched over x.

    weighted=True: full GGN, J^T (diag(p)-pp^T) J -- the "GGN on buffer" /
    "ground truth GGN" estimator (candidate 2), computed via one JVP + one
    VJP per chunk (no Jacobian ever formed explicitly). Validated against an
    explicit brute-force Jacobian construction on a toy model (see
    tests/test_curvature_torch.py).

    weighted=False: J^T J with no softmax reweighting -- the function-space
    Jacobian estimator (candidate 1), a property of the network's outputs
    alone, not of how well it fits these particular inputs.
    """
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
        # vjp already sums its cotangent's contribution over the chunk's
        # batch dimension -- accumulate that sum across chunks directly and
        # divide by the total example count once at the end. Multiplying by
        # `n` here would double-count (caught by
        # tests/test_curvature_torch.py against a brute-force Jacobian).
        for k in total:
            total[k] = total[k] + hv_chunk[k]
        n_total += n
    return {k: v_ / n_total for k, v_ in total.items()}


def jacobian_outer_vector_product(model, x, lo: int, hi: int, v_dict, chunk_size: int = 128):
    """Function-space Jacobian estimator: (J^T J) @ v. See `ggn_vector_product`."""
    return ggn_vector_product(model, x, lo, hi, v_dict, chunk_size=chunk_size, weighted=False)


def diagonal_empirical_fisher(model, x, y, lo: int, hi: int, chunk_size: int = 64):
    """Diagonal empirical Fisher: per-parameter mean squared per-example
    gradient of the task-restricted cross-entropy loss (EWC-style).

    Deliberately NOT `torch.func.vmap` + `grad`: during E2's calibration pass
    that combination triggered a fatal CUDA error on this machine's GPU/
    driver that then poisoned the CUDA context for the rest of the process
    (every subsequent, otherwise-unrelated CUDA call failed too, including
    ones that never touch vmap). A plain per-example loop over ordinary
    autograd is slower but avoids that code path entirely.
    """
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
    """A random direction v (dict matching param shapes), unit L2 norm over
    the flattened parameter vector."""
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
