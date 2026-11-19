# Continual Learning as a Bargaining Game with Absent Players

This repository contains the implementation and experiment drivers for bargaining
over continual-learning updates when past tasks are represented by stored data.
The core package provides quadratic utilities, a feasible log-slack solver,
curvature operators, subspace reduction, certificate calibration, and admission
control.

## Structure

- `src/cl_bargain/bargain/`: utilities, solver, curvature, reduction, certificates,
  and first-order baselines.
- `src/cl_bargain/streams/`: CIFAR streams, backbone, replay buffer, and training
  methods.
- `experiments/`: retained structural, benchmark, curvature, certificate, and
  plotting scripts.
- `tables/`: summaries for retained experiment runs.

## Setup

```bash
uv sync
uv sync --extra torch
```

CIFAR data is intentionally not included. GPU experiments expect torchvision
CIFAR files under `data/`; download them with the standard torchvision loader
before running a real-data experiment.

## Current Protocol

The real-stream protocol uses CIFAR-100 split into 20 tasks of 5 classes,
task-masked training and evaluation, a reduced GroupNorm ResNet, five epochs per
task, a 2000-example reservoir, and fixed seeds 0, 1, and 2.

## Experiment Drivers

```bash
uv run python experiments/e0_solver.py
uv run python experiments/e1_sensitivity.py
uv run python experiments/e4b_true_hessian.py
uv run python experiments/e6_benchmark.py
uv run python experiments/e7b_dimv_scale.py
uv run python experiments/e7c_jvp_speedup.py
uv run python experiments/e7d_batched_curvature.py
uv run python experiments/e8_certificate.py --jvp-batched --stream split20 --buffer 2000 --epochs 5 --seeds 0
```

Each run writes structured records and can regenerate its summary table. Generated
logs, downloaded data, virtual environments, caches, and plots are local artifacts
and are not required for installing the package.

## Quality Checks

```bash
uv run ruff check .
```
