# Certified forgetting for continual learning

Code for training continual-learning proposers on Split-CIFAR100 and certifying their forgetting on a held-out audit slice.

## Setup

```bash
uv sync --extra torch --group dev
```

Python 3.12, PyTorch 2.6.0 + torchvision 0.21.0 built for CUDA 12.4. Experiments were run on a single NVIDIA GPU with 24 GB of memory.
CIFAR-100 is expected in torchvision format under `data/` (`torchvision.datasets.CIFAR100("data", download=True)`).
Trained trajectories are written to `runs/`, result rows to `logs/`, tables to `tables/`.

Tests: `uv run pytest`. Lint: `uv run ruff check .`

## Experiments

All commands are prefixed with `uv run python`. Times are GPU wall time for one job running alone.

| Paper | Command | Time |
|---|---|---|
| Fig. 2; Tables 5–7 (sandbox) | not included in this release | -- |
| Table 8 (logistic) | not included in this release | -- |
| Table H.1 (admission) | `experiments/determinism_check.py --admission off --repeats 2`<br>`experiments/determinism_check.py --admission on --repeats 3`<br>`CUBLAS_WORKSPACE_CONFIG=:4096:8 experiments/determinism_check.py --admission on --deterministic --repeats 3`<br>`experiments/ltt_branch.py --proposers replay --lr 0.045` | ~2 min per run |
| Table H.2 (shared checkpoint) | `experiments/e3_shared_checkpoint.py --run bargain_curvature_seed0_tau0.4_2086db4f1015 --qp-trust-radius 0.65 --qp-steps-remaining 5325 --tag fixedqp` (needs the seed-0 `bargain_curvature` run below) | ~20 min |
| Tables H.3–H.4 (main comparison, certification) | `CL_TIMING=1 experiments/e5_walltime.py --seed 0`<br>`experiments/e5_walltime.py --no-timing --seed 1` and `--seed 2`<br>`experiments/e5_walltime.py --proposers replay_ggn --ggn-mu {0.01,0.03} --no-timing`<br>`experiments/a_tune_bargain.py`<br>`experiments/ltt_select.py`<br>`experiments/report_main.py --tau 0.4` and `--tau 0.1` | ~2.5 h per seed; ~6 h for `a_tune_bargain.py`; selection and tables a few minutes |
| Table H.5 (cost) | `CL_TIMING=1 experiments/e5_walltime.py --seed 0`, then `experiments/e5_walltime.py --table-only` | ~2.5 h |
| H.6 (selection rule) | `experiments/report_main.py --tau 0.4` (last section) | < 1 min |

## Results

Bonferroni δ/K over the 16-candidate pool, then best calibration accuracy inside the certified set; δ = 0.05, cert τ = 0.4.

| Proposer | Seed | A_20 after selection | UCB at selection | Tightest certified τ |
|---|---|---|---|---|
| replay | 0 / 1 / 2 | 0.5950 / 0.6269 / 0.6560 | 0.1185 / 0.1327 / 0.1084 | 0.4 / 0.4 / 0.4 |
| DER++ (α = β = 1) | 0 / 1 / 2 | 0.6509 / 0.6681 / 0.6784 | 0.1041 / 0.0851 / 0.1148 | 0.4 / 0.1 / 0.4 |
| replay + GGN (μ = 0.1) | 0 / 1 / 2 | 0.4954 / 0.4593 / 0.4372 | 0.1419 / 0.1357 / 0.1455 | 0.4 / 0.4 / 0.4 |
| constrained QP | 0 / 1 / 2 | 0.6610 / 0.6692 / 0.6860 | 0.1204 / 0.1084 / 0.1091 | 0.4 / 0.4 / 0.4 |
| first-order bargain | 0 / 1 / 2 | 0.6561 / 0.6610 / 0.6770 | 0.1095 / 0.1033 / 0.1074 | 0.4 / 0.4 / 0.4 |
| curvature bargain | 0 / 1 / 2 | 0.6250 / 0.6375 / 0.6794 | 0.1262 / 0.1245 / 0.1142 | 0.4 / 0.4 / 0.4 |

Tables 2–3 of the original submission are not reproduced by this code; see the correction note in the paper.
