"""Flatten/unflatten a model's parameters to/from a single vector (torch).

Needed by the bargain-based training methods (E2): the numpy bargain solver
(bargain/solver.py) operates on flat vectors, and the update Delta it
produces has to be written back into the model's actual parameter tensors.
"""

from __future__ import annotations


def param_shapes(model) -> dict:
    return {k: tuple(v.shape) for k, v in model.named_parameters()}


def flatten_grad(model):
    import torch

    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
    return torch.cat([g.reshape(-1) for g in grads])


def flatten_params(model):
    import torch

    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def add_flat_(model, delta) -> None:
    """In-place: model's parameters += delta (a flat vector matching flatten_params)."""
    import torch

    with torch.no_grad():
        off = 0
        for p in model.parameters():
            n = p.numel()
            p.add_(delta[off : off + n].reshape(p.shape))
            off += n


def dict_to_flat(d: dict, names: list[str]):
    import torch

    return torch.cat([d[k].reshape(-1) for k in names])


def flat_to_dict(flat, names: list[str], shapes: dict) -> dict:
    out = {}
    off = 0
    for k in names:
        n = 1
        for s in shapes[k]:
            n *= s
        out[k] = flat[off : off + n].reshape(shapes[k])
        off += n
    return out


__all__ = [
    "add_flat_",
    "dict_to_flat",
    "flat_to_dict",
    "flatten_grad",
    "flatten_params",
    "param_shapes",
]
