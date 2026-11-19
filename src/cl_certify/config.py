"""Run configuration; every hyper-parameter is logged verbatim with the result."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    dataset: str = "split_cifar100"
    stream: str = "split20"
    task_masked: bool = True
    epochs: int = 5
    minibatch_size: int = 32
    buffer_capacity: int = 2000
    # replay slots, with calibration/audit added on top (None: split buffer_capacity)
    replay_capacity: int | None = None
    backbone_width: int = 16
    norm: str = "group"
    lr: float = 0.1
    seed: int = 0

    split_seed: int = 1234
    # replay / calibration / audit fractions of the buffer
    split_fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)

    proposer: str = "bargain_curvature"
    tau: float = 0.05
    zeta_floor: float = 0.01
    # step radius / trust region for proposers that have one; None means use tau
    trust_radius: float | None = None
    # constrained_qp uses tau / steps_remaining per step (only with budget_mode="per_step")
    qp_cumulative_budget: bool = False
    chunk_size: int = 64
    max_active_players: int | None = None
    # required when task_masked=True (per-player head masks are batched-path only)
    jvp_batched: bool = True
    bargain_every_k: int = 1
    ewc_lambda: float = 3.0
    derpp_alpha: float = 0.5
    derpp_beta: float = 0.5
    ggn_mu: float = 1.0

    admission: bool = True
    max_backtrack: int = 6
    budget_schedule: str = "constant"
    # "cumulative": drift since task anchor vs tau; "per_step": step damage vs tau/steps_left
    budget_mode: str = "cumulative"
    # admission budget on calibration cross-entropy; None means use tau
    admission_budget: float | None = None

    # "heldout_train": per-task slice withheld from training/buffer; "buffer": carved from buffer
    audit_source: str = "heldout_train"
    audit_heldout_n: int = 250
    delta: float = 0.05
    loss_max: float = 10.0
    certify_every_epoch: bool = True
    certify_at_task_end: bool = True
    certify_on_test_set: bool = True

    smoke: bool = False
    tag: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        import hashlib

        payload = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


__all__ = ["Config"]
