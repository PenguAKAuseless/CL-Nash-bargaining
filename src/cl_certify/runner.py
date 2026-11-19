"""Shared training loop: propose, admit on calibration slice, apply, certify on audit slice."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cl_bargain.streams.backbone import make_backbone
from cl_bargain.streams.methods import task_accuracy_corrected, task_loss
from cl_certify import timing
from cl_certify.admission import BacktrackingAdmission, ramp_budget
from cl_certify.certifier import PairedBernsteinCertifier
from cl_certify.config import Config
from cl_certify.coverage import clopper_pearson
from cl_certify.proposers import (
    AGEMProposer,
    BargainProposer,
    ConstrainedQPProposer,
    DERPPProposer,
    EWCProposer,
    ProposalState,
    ReplayGaussNewtonProposer,
    ReplayProposer,
)
from cl_certify.splits import ThreeWayBuffer

PROPOSERS = (
    "replay", "agem", "ewc", "derpp", "replay_ggn",
    "constrained_qp", "bargain_first_order", "bargain_curvature",
)


def build_proposer(name: str, model, cfg: Config):
    if name == "replay":
        return ReplayProposer(model, cfg.lr)
    if name == "agem":
        return AGEMProposer(model, cfg.lr)
    if name == "ewc":
        return EWCProposer(model, cfg.ewc_lambda, cfg.lr)
    if name == "derpp":
        return DERPPProposer(model, cfg.lr, cfg.derpp_alpha, cfg.derpp_beta)
    if name == "replay_ggn":
        return ReplayGaussNewtonProposer(model, cfg.lr, cfg.ggn_mu)
    if name == "constrained_qp":
        return ConstrainedQPProposer(
            cfg.tau, cfg.chunk_size, cfg.max_active_players, cfg.jvp_batched,
            trust_radius=cfg.trust_radius,
            cumulative_budget=(cfg.qp_cumulative_budget and cfg.budget_mode == "per_step"),
        )
    if name in ("bargain_first_order", "bargain_curvature"):
        method = "v1" if name == "bargain_first_order" else "v2"
        p = BargainProposer(
            method, cfg.tau, cfg.zeta_floor, cfg.chunk_size,
            cfg.max_active_players, cfg.jvp_batched, cfg.bargain_every_k,
            trust_radius=cfg.trust_radius,
        )
        if cfg.bargain_every_k > 1:
            p.attach_fallback(model, cfg.lr)
        return p
    raise ValueError(f"unknown proposer {name!r}; known: {PROPOSERS}")


@dataclass
class RunResult:
    config: dict
    metrics: dict
    admission: dict
    certificate_rows: list = field(default_factory=list)
    step_rows: list = field(default_factory=list)
    slice_sizes: list = field(default_factory=list)
    # populated only when cfg.extra["save_snapshots"] is set
    snapshots: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    # frozen calibration slice with per-example 0-1 anchors
    calibration: dict = field(default_factory=dict)


def _per_example_01(model, stream, idx, lo: int, hi: int) -> np.ndarray:
    if idx.numel() == 0:
        return np.zeros(0)
    with torch.no_grad():
        out = model(stream.normalise(stream.x_train[idx]))[:, lo:hi]
        wrong = (out.argmax(1) != (stream.y_train[idx] - lo)).float()
    return wrong.detach().cpu().numpy().astype(float)


def _per_example_loss(model, stream, idx, lo: int, hi: int, loss_max: float) -> np.ndarray:
    # clip must match the certifier's loss_max (finite Bernstein range)
    if idx.numel() == 0:
        return np.zeros(0)
    with torch.no_grad():
        out = model(stream.normalise(stream.x_train[idx]))[:, lo:hi]
        ce = F.cross_entropy(out, stream.y_train[idx] - lo, reduction="none")
    return np.clip(ce.detach().cpu().numpy().astype(float), 0.0, loss_max)


def _batched_slice_loss(model, stream, idx_list, ranges, loss_max: float) -> np.ndarray:
    """Mean clipped loss per (idx, task-range) pair, in one forward pass."""
    if not idx_list:
        return np.zeros(0)
    counts = [int(i.numel()) for i in idx_list]
    if sum(counts) == 0:
        return np.zeros(len(idx_list))
    keep = [k for k, c in enumerate(counts) if c > 0]
    cat = torch.cat([idx_list[k] for k in keep])
    with torch.no_grad():
        logits = model(stream.normalise(stream.x_train[cat]))
    out = np.zeros(len(idx_list))
    off = 0
    for k in keep:
        c = counts[k]
        lo, hi = ranges[k]
        chunk = logits[off : off + c, lo:hi]
        y = stream.y_train[idx_list[k]] - lo
        out[k] = float(
            F.cross_entropy(chunk, y, reduction="none").clamp(0.0, loss_max).mean()
        )
        off += c
    return out


def _per_example_test_loss(model, stream, idx, lo: int, hi: int, loss_max: float) -> np.ndarray:
    if idx.numel() == 0:
        return np.zeros(0)
    with torch.no_grad():
        out = model(stream.normalise(stream.x_test[idx]))[:, lo:hi]
        ce = F.cross_entropy(out, stream.y_test[idx] - lo, reduction="none")
    return np.clip(ce.detach().cpu().numpy().astype(float), 0.0, loss_max)


def run(cfg: Config, device: torch.device, stream=None, log_steps: bool = False) -> RunResult:
    from cl_bargain.streams.cifar import SplitCIFAR100

    if stream is None:
        stream = SplitCIFAR100(
            str(_data_root()), cfg.stream, device=device,
            class_order_seed=cfg.seed, download=False, task_masked=cfg.task_masked,
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_tasks = 2 if cfg.smoke else len(stream.tasks)
    epochs = 1 if cfg.smoke else cfg.epochs
    if cfg.smoke:
        capacity = 120
    elif cfg.replay_capacity is not None:
        # grow the total so the replay slice equals replay_capacity
        capacity = round(cfg.replay_capacity / cfg.split_fractions[0])
    else:
        capacity = cfg.buffer_capacity

    model = make_backbone(stream.n_classes, device, nf=cfg.backbone_width, norm=cfg.norm)
    buffer = ThreeWayBuffer(capacity, device, cfg.split_seed, cfg.split_fractions)
    proposer = build_proposer(cfg.proposer, model, cfg)

    # number of checkpoints K is fixed before the run, not the number reached
    per_task_checkpoints = (epochs if cfg.certify_every_epoch else 0) + (
        1 if cfg.certify_at_task_end else 0
    )
    n_checkpoints = max(1, n_tasks * per_task_checkpoints)

    certifier = PairedBernsteinCertifier(
        delta=cfg.delta, loss_max=cfg.loss_max,
        n_checkpoints=n_checkpoints, n_tasks_max=max(1, n_tasks - 1),
        source="audit",
    )
    # 0-1 loss certifier: same audit slice, same anchors, range [0, 1].
    zo_certifier = PairedBernsteinCertifier(
        delta=cfg.delta, loss_max=1.0,
        n_checkpoints=n_checkpoints, n_tasks_max=max(1, n_tasks - 1),
        source="audit01",
    )
    # test-set certifier is a measurement used to score audit-bound coverage, not a guarantee
    test_certifier = PairedBernsteinCertifier(
        delta=cfg.delta, loss_max=cfg.loss_max,
        n_checkpoints=n_checkpoints, n_tasks_max=max(1, n_tasks - 1),
        source="test",
    )

    steps_per_epoch = {
        t: int(np.ceil(stream.tasks[t].n_examples / cfg.minibatch_size)) for t in range(n_tasks)
    }
    total_steps_after = {
        t: sum(steps_per_epoch[u] * epochs for u in range(t + 1, n_tasks)) for t in range(n_tasks)
    }
    # earliest task has the most remaining steps, giving the tightest per-step budget
    admission = BacktrackingAdmission(
        budget_per_step=cfg.tau / max(1, total_steps_after[0]),
        max_backtrack=cfg.max_backtrack,
        enabled=cfg.admission,
        schedule=cfg.budget_schedule,
        mode=cfg.budget_mode,
    )
    # calibration-slice loss per task, taken at its task boundary
    cal_anchor: dict[int, float] = {}
    snapshots: dict[int, dict] = {}
    task_elapsed: list[float] = []
    cal_frozen: dict[int, torch.Tensor] = {}
    cal01_anchor: dict[int, np.ndarray] = {}

    # audit pools fixed before training; "heldout_train" slices never enter training or the buffer
    audit_pool: dict[int, torch.Tensor] = {}
    train_idx_for: dict[int, torch.Tensor] = {}
    split_rng = torch.Generator(device="cpu")
    split_rng.manual_seed(cfg.split_seed)
    for t in range(n_tasks):
        full = stream.tasks[t].train_idx
        if cfg.audit_source == "heldout_train":
            k = min(cfg.audit_heldout_n, max(0, int(full.numel()) - cfg.minibatch_size))
            perm = torch.randperm(int(full.numel()), generator=split_rng)
            audit_pool[t] = full[perm[:k].to(full.device)].clone()
            train_idx_for[t] = full[perm[k:].to(full.device)].clone()
        elif cfg.audit_source == "buffer":
            train_idx_for[t] = full
        else:
            raise ValueError(
                f"unknown audit_source {cfg.audit_source!r}; "
                "known: 'heldout_train', 'buffer'"
            )

    acc_at_end: dict[int, float] = {}
    loss_at_end: dict[int, float] = {}
    cur_task_acc: dict[int, float] = {}
    step_rows: list[dict] = []
    slice_sizes: list[dict] = []
    proposer_failures = 0
    total_steps = 0
    checkpoint = 0
    audit_frozen: dict[int, torch.Tensor] = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing.reset()
    started = time.time()

    for t in range(n_tasks):
        task = stream.tasks[t]
        train_idx = train_idx_for[t]
        n = int(train_idx.numel())
        active_tasks = [a for a in buffer.tasks() if a != t]

        def cal_loss(tasks: list[int]) -> np.ndarray:
            return _batched_slice_loss(
                model, stream,
                [buffer.calibration_held(a) for a in tasks],
                [stream.task_range(a) for a in tasks],
                cfg.loss_max,
            )

        adm_tau = cfg.tau if cfg.admission_budget is None else cfg.admission_budget
        if cfg.budget_mode == "cumulative":
            budget = adm_tau
        elif cfg.budget_mode == "cumulative_ramp":
            # grows from 0 at the earliest absent task's boundary to the full budget at stream end
            budget = (
                ramp_budget(adm_tau, t, min(active_tasks), n_tasks)
                if active_tasks and n_tasks > 1 else adm_tau
            )
        elif active_tasks:
            budget = adm_tau / max(1, total_steps_after[min(active_tasks)])
        else:
            budget = adm_tau
        # proposer allowance uses cfg.tau (certificate units), not adm_tau (cross-entropy units)
        step_budget = (
            cfg.tau / max(1, total_steps_after[min(active_tasks)])
            if active_tasks else cfg.tau
        )
        reference = (
            np.array([cal_anchor[a] for a in active_tasks], dtype=float)
            if cfg.budget_mode == "cumulative" and active_tasks
            and all(a in cal_anchor for a in active_tasks)
            else None
        )

        for epoch in range(epochs):
            perm = torch.randperm(n, device=train_idx.device, generator=generator)
            for start in range(0, n, cfg.minibatch_size):
                sub = train_idx[perm[start : start + cfg.minibatch_size]]
                total_steps += 1

                state = ProposalState(
                    model=model, stream=stream, idx=sub, task_index=t, buffer=buffer,
                    generator=generator, minibatch_size=cfg.minibatch_size,
                    device=device, step=total_steps, step_budget=step_budget,
                )
                proposal = proposer.propose(state)
                if proposal.delta is None:
                    proposer_failures += 1
                    if log_steps:
                        step_rows.append({
                            "step": total_steps, "task": t, "epoch": epoch,
                            "proposed": False, "beta": 0.0,
                        })
                    continue

                with timing.section("admission"):
                    outcome = admission.admit(
                        model, proposal.delta, active_tasks, cal_loss,
                        budget_override=budget, reference=reference, task=t,
                    )
                if log_steps:
                    step_rows.append({
                        "step": total_steps, "task": t, "epoch": epoch, "proposed": True,
                        "beta": outcome.beta, "applied": outcome.applied,
                        "n_backtracks": outcome.n_backtracks,
                        "worst_cal_damage": outcome.worst_damage,
                        "budget": outcome.budget,
                        "delta_norm": float(proposal.delta.norm()),
                        "n_active": proposal.diagnostics.get("n_active"),
                    })

            if cfg.certify_every_epoch:
                checkpoint += 1
                with timing.section("certify"):
                    _certify(
                        certifier, zo_certifier, test_certifier, model, stream, buffer,
                        audit_frozen, active_tasks, checkpoint, total_steps, cfg, t,
                        epoch,
                    )

        # task boundary: ingest, anchor, snapshot
        buffer.add(train_idx, t)
        buffer.assert_disjoint()
        lo, hi = stream.task_range(t)
        if cfg.audit_source == "buffer":
            audit_idx = buffer.audit_held(t, for_certification=True).clone()
        else:
            audit_idx = audit_pool[t]
        audit_frozen[t] = audit_idx
        certifier.set_anchor(t, _per_example_loss(model, stream, audit_idx, lo, hi, cfg.loss_max))
        zo_certifier.set_anchor(t, _per_example_01(model, stream, audit_idx, lo, hi))
        if cfg.certify_on_test_set:
            test_certifier.set_anchor(
                t, _per_example_test_loss(model, stream, task.test_idx, lo, hi, cfg.loss_max)
            )
        cal_anchor[t] = float(cal_loss([t])[0])
        if cfg.extra.get("save_snapshots"):
            # frozen at the boundary so later measurements pair with the anchor
            cal_idx = buffer.calibration_held(t).clone()
            cal_frozen[t] = cal_idx
            cal01_anchor[t] = _per_example_01(model, stream, cal_idx, lo, hi)
        if isinstance(proposer, DERPPProposer):
            proposer.store_logits(model, stream, buffer.held(t))

        slice_sizes.append({"after_task": t, **{
            k: dict(v) for k, v in buffer.all_slice_sizes().items()
        }})

        model.eval()
        with torch.no_grad():
            acc_at_end[t] = task_accuracy_corrected(model, stream, task.test_idx, lo, hi)
            loss_at_end[t] = float(task_loss(model, stream, task.test_idx, lo, hi))
            cur_task_acc[t] = acc_at_end[t]
        model.train()

        if cfg.extra.get("save_snapshots"):
            snapshots[t] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # env var rather than cfg so the switch does not change the config fingerprint
        if os.environ.get("CL_PROGRESS"):
            elapsed = time.time() - started
            task_elapsed.append(elapsed)
            # cumulative time ~ t^p; fit p from the last two boundaries
            if len(task_elapsed) >= 2 and task_elapsed[-2] > 0:
                ratio = task_elapsed[-1] / task_elapsed[-2]
                power = (math.log(ratio) / math.log((t + 1) / t)) if ratio > 1 else 1.0
                total = elapsed * (n_tasks / (t + 1)) ** max(1.0, power)
            else:
                total = elapsed * n_tasks / (t + 1)
            print(f"[progress] {cfg.proposer} tau={cfg.tau:g} seed={cfg.seed} "
                  f"task {t + 1}/{n_tasks} elapsed={elapsed / 60:.1f}min "
                  f"eta={(total - elapsed) / 60:.1f}min "
                  f"(total~{total / 60:.1f}min)", flush=True)

        if cfg.certify_at_task_end:
            checkpoint += 1
            with timing.section("certify"):
                _certify(
                    certifier, zo_certifier, test_certifier, model, stream, buffer,
                    audit_frozen, [a for a in buffer.tasks() if a != t],
                    checkpoint, total_steps, cfg, t, -1,
                )

    model.eval()
    acc_final, loss_final = {}, {}
    with torch.no_grad():
        for t in range(n_tasks):
            lo, hi = stream.task_range(t)
            acc_final[t] = task_accuracy_corrected(model, stream, stream.tasks[t].test_idx, lo, hi)
            loss_final[t] = float(task_loss(model, stream, stream.tasks[t].test_idx, lo, hi))
    model.train()

    wall = time.time() - started
    forgetting = [acc_at_end[t] - acc_final[t] for t in range(n_tasks - 1)]
    adm = admission.summary()

    metrics = {
        "A_final": float(np.mean(list(acc_final.values()))),
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "worst_task_accuracy": float(np.min(list(acc_final.values()))),
        "current_task_accuracy": float(np.mean(list(cur_task_acc.values()))),
        "plasticity_final_task_loss": loss_at_end[n_tasks - 1],
        "step_accept_rate": adm.get("accept_rate", 1.0),
        "mean_beta": adm.get("mean_beta", 1.0),
        "proposer_failure_rate": proposer_failures / max(1, total_steps),
        "total_steps": total_steps,
        "n_checkpoints_planned": n_checkpoints,
        "n_checkpoints_used": checkpoint,
        "wall_seconds": wall,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
        ),
        "acc_at_end": acc_at_end,
        "acc_final": acc_final,
        "timing": timing.snapshot() if timing.enabled() else None,
        **_certificate_metrics(certifier, test_certifier, cfg),
        **{f"zo_{k}": v for k, v in _certificate_metrics(zo_certifier, None, cfg).items()},
    }

    audit_export: dict = {}
    calibration_export: dict = {}
    if cfg.extra.get("save_snapshots"):
        audit_export = {
            "idx": {t: v.detach().cpu().clone() for t, v in audit_frozen.items()},
            "anchor01": {t: np.asarray(a) for t, a in zo_certifier.anchors.items()},
            "anchor_ce": {t: np.asarray(a) for t, a in certifier.anchors.items()},
            "task_range": {t: stream.task_range(t) for t in range(n_tasks)},
            "acc_at_end": dict(acc_at_end),
        }
        snapshots["final"] = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        calibration_export = {
            "idx": {t: v.detach().cpu().clone() for t, v in cal_frozen.items()},
            "anchor01": {t: np.asarray(a) for t, a in cal01_anchor.items()},
        }

    return RunResult(
        config=cfg.as_dict(), metrics=metrics, admission=adm,
        snapshots=snapshots, audit=audit_export, calibration=calibration_export,
        certificate_rows=(
            certifier.as_rows() + zo_certifier.as_rows() + test_certifier.as_rows()
        ),
        step_rows=step_rows, slice_sizes=slice_sizes,
    )


def _certify(certifier, zo_certifier, test_certifier, model, stream, buffer,
             audit_frozen, active_tasks, checkpoint, step, cfg, live_task, epoch):
    del buffer, live_task, epoch
    model.eval()
    per_task, per_task01 = {}, {}
    for a in active_tasks:
        if a not in audit_frozen or not certifier.has_anchor(a):
            continue
        lo, hi = stream.task_range(a)
        v = _per_example_loss(model, stream, audit_frozen[a], lo, hi, cfg.loss_max)
        if v.size < 2:
            continue
        per_task[a] = v
        per_task01[a] = _per_example_01(model, stream, audit_frozen[a], lo, hi)
        certifier.certify_task(a, v, checkpoint, step, budget=cfg.tau)
        zo_certifier.certify_task(a, per_task01[a], checkpoint, step, budget=cfg.tau)
    if per_task:
        # pooling by example equals the per-task average only when every task has the same m
        equal_m = len({v.size for v in per_task.values()}) == 1
        if equal_m:
            certifier.certify_pooled_examples(per_task, checkpoint, step, budget=cfg.tau)
            certifier.certify_betting(per_task, checkpoint, step, budget=cfg.tau)
            zo_certifier.certify_pooled_examples(per_task01, checkpoint, step, budget=cfg.tau)
            zo_certifier.certify_betting(per_task01, checkpoint, step, budget=cfg.tau)
        # per-task intervals at delta/T, averaged; does not need equal slice sizes
        certifier.certify_average_of_per_task(per_task, checkpoint, step, budget=cfg.tau)
        zo_certifier.certify_average_of_per_task(per_task01, checkpoint, step, budget=cfg.tau)
    if len(per_task) >= 2:
        certifier.certify_pooled(per_task, checkpoint, step, budget=cfg.tau)

    if cfg.certify_on_test_set:
        for a in active_tasks:
            if not test_certifier.has_anchor(a):
                continue
            lo, hi = stream.task_range(a)
            v = _per_example_test_loss(model, stream, stream.tasks[a].test_idx, lo, hi,
                                       cfg.loss_max)
            test_certifier.certify_task(a, v, checkpoint, step, budget=cfg.tau)
    model.train()


def _certificate_metrics(certifier, test_certifier, cfg) -> dict:
    # budget violation: bound > tau; coverage violation: audit bound < test-set mean
    nan = float("nan")
    rows = [r for r in certifier.records if r.mode == "per_task"]  # audit only
    test_by_key = (
        {}
        if test_certifier is None
        else {
            (r.task, r.checkpoint): r.mean
            for r in test_certifier.records
            if r.mode == "per_task"
        }
    )
    pooled_ex = [r for r in certifier.records if r.mode == "pooled_examples"]
    betting = [r for r in certifier.records if r.mode == "betting_pooled"]
    budget_viol = [r.bound > cfg.tau for r in rows]
    cov = [
        (r.bound < test_by_key[(r.task, r.checkpoint)])
        for r in rows
        if (r.task, r.checkpoint) in test_by_key
    ]
    widths = [r.bound - r.mean for r in rows]
    pooled = [r for r in certifier.records if r.mode == "pooled"]
    avg_pt = [r for r in certifier.records if r.mode == "avg_per_task_bernstein"]
    # pooled estimand's test counterpart: mean of per-task test means at the checkpoint
    test_by_ckpt: dict[int, list[float]] = {}
    for (t_, c_), mu in test_by_key.items():
        del t_
        test_by_ckpt.setdefault(c_, []).append(mu)
    pooled_cov = [
        r.bound < float(np.mean(test_by_ckpt[r.checkpoint]))
        for r in pooled_ex
        if r.checkpoint in test_by_ckpt
    ]
    cov_lo, cov_hi = clopper_pearson(int(np.sum(cov)), len(cov)) if cov else (nan, nan)
    pcov_lo, pcov_hi = (
        clopper_pearson(int(np.sum(pooled_cov)), len(pooled_cov)) if pooled_cov else (nan, nan)
    )
    return {
        "cert_coverage_violation_cp_lo": cov_lo,
        "cert_coverage_violation_cp_hi": cov_hi,
        "cert_pooledex_coverage_violation_rate": (
            float(np.mean(pooled_cov)) if pooled_cov else nan
        ),
        "cert_pooledex_coverage_cp_lo": pcov_lo,
        "cert_pooledex_coverage_cp_hi": pcov_hi,
        "cert_pooledex_n_coverage_scored": len(pooled_cov),
        "cert_avg_per_task_mean_bound": (
            float(np.mean([r.bound for r in avg_pt])) if avg_pt else nan
        ),
        "cert_avg_per_task_mean_width": (
            float(np.mean([r.bound - r.mean for r in avg_pt])) if avg_pt else nan
        ),
        "cert_avg_per_task_budget_violation_rate": (
            float(np.mean([r.bound > cfg.tau for r in avg_pt])) if avg_pt else nan
        ),
        "cert_budget_violation_rate": float(np.mean(budget_viol)) if budget_viol else float("nan"),
        "cert_coverage_violation_rate": float(np.mean(cov)) if cov else float("nan"),
        "cert_n_scored": len(rows),
        "cert_n_coverage_scored": len(cov),
        "cert_mean_width": float(np.mean(widths)) if widths else float("nan"),
        "cert_median_width": float(np.median(widths)) if widths else float("nan"),
        "cert_mean_bound": float(np.mean([r.bound for r in rows])) if rows else float("nan"),
        "cert_mean_audit_m": float(np.mean([r.n_audit for r in rows])) if rows else float("nan"),
        "cert_min_audit_m": int(np.min([r.n_audit for r in rows])) if rows else 0,
        "cert_pooled_mean_bound": (
            float(np.mean([r.bound for r in pooled])) if pooled else float("nan")
        ),
        "cert_pooled_budget_violation_rate": (
            float(np.mean([r.bound > cfg.tau for r in pooled])) if pooled else float("nan")
        ),
        "cert_pooledex_mean_bound": (
            float(np.mean([r.bound for r in pooled_ex])) if pooled_ex else float("nan")
        ),
        "cert_pooledex_mean_width": (
            float(np.mean([r.bound - r.mean for r in pooled_ex])) if pooled_ex else float("nan")
        ),
        "cert_pooledex_budget_violation_rate": (
            float(np.mean([r.bound > cfg.tau for r in pooled_ex])) if pooled_ex else float("nan")
        ),
        "cert_pooledex_mean_m": (
            float(np.mean([r.n_audit for r in pooled_ex])) if pooled_ex else float("nan")
        ),
        "cert_betting_mean_bound": (
            float(np.mean([r.bound for r in betting])) if betting else float("nan")
        ),
        "cert_betting_mean_width": (
            float(np.mean([r.bound - r.mean for r in betting])) if betting else float("nan")
        ),
        "cert_betting_budget_violation_rate": (
            float(np.mean([r.bound > cfg.tau for r in betting])) if betting else float("nan")
        ),
    }


def _data_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "data"


__all__ = ["PROPOSERS", "RunResult", "build_proposer", "run"]
