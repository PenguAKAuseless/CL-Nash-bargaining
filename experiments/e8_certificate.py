"""E-cert -- the certificate on real streams. Per step and per absent task, measure the pair
(certified damage C_i(Delta_hat), realised probe-loss increase) and report
(a) coverage: fraction of OUT-OF-SAMPLE steps where realised loss does not
    exceed C_i;
(b) tightness: distribution of C_i - F_i in units of tau_i;
(c) budget adherence: fraction of steps at which admission control would
    bind (beta* < 1), and realised cumulative loss increase vs the budget;
(d) gap to re-solving: at steps where beta* < 1, the plasticity cost of the
    one-scalar backtrack vs re-solving with tau_i tightened
    (Remark "Admission control is cheap, not optimal");
(e) the same quantities on the full retained task loss (probe-to-task gap,
    Remark "Probe loss against task loss").

Ablations, each isolating one ingredient, computed from the SAME trajectory
where possible (no extra training cost):
  - no calibration: fixed (r, zeta, rho) instead of the fitted triple;
  - no probe slice: calibrate on the REPLAY slice instead -- the circularity
    control (this is the ablation Remark "Not available" predicts should be
    biased low, since the replay slice IS what the optimiser was pushed to
    fit);
  - no admission control: report the RAW certificate (beta=1) rather than
    the admitted beta*Delta's;
  - first-order utilities, d_i=0: a separate, reduced-scale supplementary
    run via `run_one(method="nashmtl")`, since this changes the actual
    optimisation trajectory and cannot be read off the v2 run.

Both budget schedules of Corollary "Cumulative loss and budget allocation"
are reported: constant tau_i^(t) = B_i/T_i, and horizon-amortised
tau_i^(t) = B_i/(T_i - t), whose total exceeds B_i by sum_{j<=T_i} 1/j.

DECLARED, MEASURED deviation for tractability: v2's curvature term costs
O(n_active^2) GGN-vector products per step (~4h/run at n=10 on this GPU --
see logs/e2_frontier.jsonl). `--max-active-players` caps the number of
absent players bargained with per step (streams/methods.py's bargain_step,
`max_active_players`); its effect must be measured (see
tests/test_methods_torch.py) before being relied on here.

Run: uv run python experiments/e8_certificate.py [--smoke] [--max-active N]
Writes: logs/e8_certificate.jsonl, tables/e8_certificate.md,
        figures/e8_coverage.png (F2)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cl_bargain.bargain.certificate import (  # noqa: E402
    calibrate_lp,
    calibrated_certificate,
    telemetry_residual,
)
from cl_bargain.logging_utils import config_hash, log_run  # noqa: E402
from cl_bargain.tb import heartbeat  # noqa: E402

HEARTBEAT_PATH = ROOT / "logs" / "e8_certificate_progress.log"

LOG = ROOT / "logs" / "e8_certificate.jsonl"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

ZETA_FLOOR = 0.01  # Assumption "Curvature-uncertainty floor", zeta_i >= zeta_min > 0
WINDOW = 30  # calibration window, steps (Proposition "Certificate calibration")
# The probe-to-task gap needs a task's whole test set, 500 images against the
# 11 of its probe slice, so evaluating it at every step costs more than the
# measurement the run is for. Every twentieth step still leaves ~400 paired
# observations per task.
FULL_TASK_EVERY = 20
FIXED_RZR = (0.05, 0.05, 0.05)  # "no calibration" ablation's untuned (r, zeta, rho)


def build_stream(seed: int, device, stream: str = "split20", task_masked: bool = True):
    from cl_bargain.streams.cifar import SplitCIFAR100

    return SplitCIFAR100(
        str(ROOT / "data"), stream, device=device, class_order_seed=seed,
        download=False, task_masked=task_masked,
    )


class _PlayerWindow:
    """Rolling (q_t, eps_scaled_t) window for one absent player, PLUS a
    parallel replay-slice window for the "no probe slice" ablation. Refit
    happens with the window as it stood BEFORE this step's own residual is
    appended -- coverage is measured out-of-sample, never against a fit that
    has already seen the step being scored (which would be tautological)."""

    def __init__(self, maxlen: int = WINDOW):
        self.maxlen = maxlen
        self.q: list[float] = []
        self.eps_probe: list[float] = []
        self.eps_replay: list[float] = []
        self.rzr_probe = FIXED_RZR
        self.rzr_replay = FIXED_RZR

    def current_fit(self, kind: str):
        return self.rzr_probe if kind == "probe" else self.rzr_replay

    def push_and_refit(self, q: float, eps_probe: float, eps_replay: float | None) -> None:
        self.q.append(q)
        self.eps_probe.append(eps_probe)
        fitted, ok = calibrate_lp(np.array(self.q), np.array(self.eps_probe))
        if ok:
            self.rzr_probe = tuple(float(x) for x in fitted)
        if eps_replay is not None:
            self.eps_replay.append(eps_replay)
            fitted_r, ok_r = calibrate_lp(np.array(self.q), np.array(self.eps_replay))
            if ok_r:
                self.rzr_replay = tuple(float(x) for x in fitted_r)
        if len(self.q) > self.maxlen:
            self.q.pop(0)
            self.eps_probe.pop(0)
            if self.eps_replay:
                self.eps_replay.pop(0)


def run_one(
    seed: int,
    smoke: bool,
    max_active_players: int | None,
    probe_fraction: float = 0.1,
    stream=None,
    method: str = "v2",
    jvp_batched: bool = False,
    epochs: int | None = None,
    buffer_capacity: int | None = None,
    stream_name: str = "split20",
) -> dict:
    """method="nashmtl" reuses this exact loop for the "first-order
    utilities, d_i=0" ablation: nashmtl's tau stays all-zero
    (streams/methods.py's bargain_step), so `problem.tau[row]/c_a` is
    already 0 and the certificate/coverage machinery below needs no other
    change -- coverage against tau_i=0 is an extremely strict standard (any
    positive realised damage is a miss), which is the point of the ablation."""
    import torch

    from cl_bargain.bargain.solver import solve
    from cl_bargain.bargain.utilities import BargainProblem
    from cl_bargain.streams.backbone import make_backbone
    from cl_bargain.streams.buffer import ProbeReplayBuffer
    from cl_bargain.streams.methods import (
        bargain_step,
        probe_loss,
        task_accuracy_corrected,
        task_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    if epochs is None:
        epochs = 1 if smoke else 5
    nf = 8 if smoke else 16
    if buffer_capacity is None:
        buffer_capacity = 60 if smoke else 2000
    minibatch_size = 32
    chunk_size = 64
    knob = 0.05  # tau_base for the CONSTANT schedule; horizon schedule derives B_i from it

    if stream is None:
        stream = build_stream(seed, device, stream_name)
    n_tasks = 2 if smoke else len(stream.tasks)
    model = make_backbone(stream.n_classes, device, nf=nf)
    buffer = ProbeReplayBuffer(buffer_capacity, probe_fraction, device)

    windows: dict[int, _PlayerWindow] = {}
    records = []
    total_steps_global = 0
    for t in range(n_tasks):
        task = stream.tasks[t]
        n = task.n_examples
        for _ in range(epochs):
            perm = torch.randperm(n, device=task.train_idx.device, generator=generator)
            for start in range(0, n, minibatch_size):
                sub = task.train_idx[perm[start : start + minibatch_size]]
                total_steps_global += 1

                active_before = sorted(buffer.task_sizes.keys())
                active_before = [a for a in active_before if a != t]

                do_full = total_steps_global % FULL_TASK_EVERY == 0
                probe_before = {}
                replay_before = {}
                full_before = {}
                for a in active_before:
                    lo, hi = stream.task_range(a)
                    p_idx = buffer.probe_held(a)
                    r_idx = buffer.held(a)
                    with torch.no_grad():
                        probe_before[a] = (
                            float(probe_loss(model, stream, p_idx, lo, hi))
                            if p_idx.numel() > 0
                            else None
                        )
                        replay_before[a] = (
                            float(probe_loss(model, stream, r_idx, lo, hi))
                            if r_idx.numel() > 0
                            else None
                        )
                        if do_full:
                            full_before[a] = float(
                                task_loss(model, stream, stream.tasks[a].test_idx, lo, hi)
                            )

                applied, diag = bargain_step(
                    model, stream, sub, t, buffer, generator, minibatch_size, device,
                    method, knob, chunk_size, zeta_floor=ZETA_FLOOR,
                    max_active_players=max_active_players, jvp_batched=jvp_batched,
                )
                if not applied or diag is None:
                    continue

                problem: BargainProblem = diag["problem"]
                beta = diag["beta"]
                c = diag["c"]
                active = diag["active"]  # order matches problem rows 1..
                q = diag["delta_norm"]

                # The certificate of Theorem "Computable damage certificate"
                # with the fitted (r, zeta, rho): quadratic predicted damage
                # plus r q + zeta/2 q^2 + rho/6 q^3 with q = ||Delta|| in
                # parameter space, the same terms and the same q the
                # calibration LP was fitted with.
                # only rho to certified_damage on a row problem whose r and
                # zeta defaulted to 0, and cubed ||beta|| (reduced
                # coordinates, a non-orthogonal basis) instead of q, so the
                # scored certificate was not the fitted one.
                def predicted_rows(b, problem=problem):
                    return problem.g @ b + 0.5 * np.einsum("ipq,p,q->i", problem.H, b, b)

                pred_rows = predicted_rows(beta)[1:]
                fits = np.array(
                    [windows.setdefault(x, _PlayerWindow()).current_fit("probe") for x in active]
                )
                cert_rows = calibrated_certificate(pred_rows, q, fits)
                tau_rows = problem.tau[1:]
                # Proposition "Certified damage under shrinkage", scanning the
                # absent players only (tau_0 = 0 for the live task).
                over = cert_rows > tau_rows
                beta_star = (
                    float(min(1.0, np.min(tau_rows[over] / cert_rows[over])))
                    if np.any(over)
                    else 1.0
                )
                cert_admitted_rows = calibrated_certificate(
                    predicted_rows(beta_star * beta)[1:], beta_star * q, fits
                )
                resolve_gap = None
                if beta_star < 1.0:
                    # The cheap backtrack retracts onto the certified region;
                    # the exact response is to re-solve with the offending
                    # tau tightened, which returns a different direction and
                    # a larger admissible magnitude (Remark
                    # "Admission control is cheap, not optimal"). Tightening
                    # ALL players' tau by beta_star is the simplest
                    # self-consistent choice absent a more specific rule.
                    # SCOPE NOTE: compared in the REDUCED beta-space, not
                    # full parameter space (the basis vectors themselves are
                    # not in bargain_step's diagnostics) -- a relative, not
                    # absolute, comparison.
                    tightened = BargainProblem(
                        g=problem.g, H=problem.H, tau=problem.tau * beta_star,
                        alpha=problem.alpha, r=problem.r, zeta=problem.zeta,
                    )
                    resolved = solve(tightened, Delta0=beta.copy())
                    if resolved.converged:
                        admitted_norm = beta_star * float(np.linalg.norm(beta))
                        resolved_norm = float(np.linalg.norm(resolved.Delta))
                        resolve_gap = (resolved_norm - admitted_norm) / max(admitted_norm, 1e-12)

                for row, a in enumerate(active, start=1):
                    lo, hi = stream.task_range(a)
                    p_idx = buffer.probe_held(a)
                    r_idx = buffer.held(a)
                    with torch.no_grad():
                        probe_after = (
                            float(probe_loss(model, stream, p_idx, lo, hi))
                            if p_idx.numel() > 0
                            else None
                        )
                        replay_after = (
                            float(probe_loss(model, stream, r_idx, lo, hi))
                            if r_idx.numel() > 0
                            else None
                        )
                        full_after = (
                            float(task_loss(model, stream, stream.tasks[a].test_idx, lo, hi))
                            if do_full
                            else None
                        )
                    # Measured across this step alone, as the certificate is,
                    # so both ends come from the same step rather than from
                    # the last step at which it happened to be measured.
                    F_full_task = None if full_after is None else full_after - full_before[a]

                    win = windows.setdefault(a, _PlayerWindow())
                    # Out-of-sample: use the fit as it stood BEFORE this
                    # step's residual is folded in.
                    r_p, zeta_p, rho_p = win.current_fit("probe")

                    c_a = float(c[row])
                    g_row = problem.g[row : row + 1]
                    h_row = problem.H[row : row + 1]
                    beta_row = beta[None, :]

                    F_probe = (
                        (probe_after - probe_before[a]) if probe_before[a] is not None else None
                    )
                    F_replay = (
                        (replay_after - replay_before[a]) if replay_before[a] is not None else None
                    )

                    eps_probe_scaled = eps_replay_scaled = None
                    if F_probe is not None:
                        eps_probe_scaled = float(
                            telemetry_residual(
                                np.array([c_a * F_probe]), g_row, h_row, beta_row
                            )[0]
                        )
                    if F_replay is not None:
                        eps_replay_scaled = float(
                            telemetry_residual(
                                np.array([c_a * F_replay]), g_row, h_row, beta_row
                            )[0]
                        )

                    # Certificate under the OUT-OF-SAMPLE fit (calibrated,
                    # probe slice): the main measurement. The two ablations
                    # change only the constants.
                    pred_a = float(pred_rows[row - 1])
                    cert_probe = float(cert_rows[row - 1]) / c_a
                    cert_fixed = float(calibrated_certificate(pred_a, q, FIXED_RZR)) / c_a
                    cert_replay = None
                    if eps_replay_scaled is not None:
                        cert_replay = (
                            float(calibrated_certificate(pred_a, q, win.current_fit("replay")))
                            / c_a
                        )
                    cert_admitted = float(cert_admitted_rows[row - 1]) / c_a

                    tau_i = float(problem.tau[row] / c_a)  # original-units budget

                    records.append(
                        {
                            "seed": seed,
                            "step": total_steps_global,
                            "task_id": a,
                            "n_active": len(active),
                            "tau_i": tau_i,
                            "q": q,
                            "F_probe": F_probe,
                            "F_replay": F_replay,
                            "F_full_task": F_full_task,
                            "cert_probe_calibrated": cert_probe,
                            "cert_no_calibration": cert_fixed,
                            "cert_no_probe_slice": cert_replay,
                            "cert_admitted": cert_admitted,
                            "beta_star": beta_star,
                            "resolve_gap": resolve_gap,
                            "rzr_probe": [r_p, zeta_p, rho_p],
                        }
                    )
                    win.push_and_refit(q, eps_probe_scaled or 0.0, eps_replay_scaled)

        # Task t is now absent: add it to the buffer so later tasks bargain
        # against it (the bug this comment guards against: forgetting this
        # call left `active` empty for the whole run, and every step silently
        # took bargain_step's n==1 plain-gradient-descent branch -- caught by
        # the --smoke run producing 0 records).
        buffer.add(task.train_idx, t, generator)
        msg = (
            f"seed={seed} method={method} probe_fraction={probe_fraction} "
            f"task {t}/{n_tasks - 1} done, {len(records)} records so far, "
            f"{total_steps_global} steps"
        )
        print(msg)
        heartbeat(HEARTBEAT_PATH, msg)
        # Incremental checkpoint: this run's own `records` only reach LOG at
        # the very end of run_one (via __main__'s log_run call) -- a crash
        # or an intentional early stop mid-run previously lost everything
        # collected so far. Overwriting the same "e_cert_checkpoint" key
        # (config_hash) each task boundary means the log always holds the
        # latest complete prefix, recoverable even if the process never
        # reaches its own final log_run call.
        log_run(
            LOG,
            {
                "deliverable": "e_cert_checkpoint",
                "seed": seed,
                "method": method,
                "probe_fraction": probe_fraction,
                "max_active_players": max_active_players,
                "last_task_done": t,
                "n_records_so_far": len(records),
                "records": records,
            },
        )

    # Final accuracy -- used by E-probe's probe/replay memory trade-off
    # (accuracy is expected to fall slowly as probe_fraction rises, since
    # memory that would have been replayed is held out instead).
    model.eval()
    final_acc = {}
    with torch.no_grad():
        for t in range(n_tasks):
            task = stream.tasks[t]
            lo, hi = stream.task_range(t)
            final_acc[t] = task_accuracy_corrected(model, stream, task.test_idx, lo, hi)
    model.train()

    return {
        "seed": seed,
        "max_active_players": max_active_players,
        "probe_fraction": probe_fraction,
        "n_records": len(records),
        "records": records,
        "final_corrected_avg_accuracy": float(np.mean(list(final_acc.values()))),
    }


def summarise(run: dict) -> dict:
    recs = run["records"]
    coverage_probe = np.mean(
        [r["F_probe"] <= r["cert_probe_calibrated"] for r in recs if r["F_probe"] is not None]
    )
    coverage_fixed = np.mean(
        [r["F_probe"] <= r["cert_no_calibration"] for r in recs if r["F_probe"] is not None]
    )
    coverage_replay_cal = np.mean(
        [
            r["F_replay"] <= r["cert_no_probe_slice"]
            for r in recs
            if r["F_replay"] is not None and r["cert_no_probe_slice"] is not None
        ]
    )
    # (e) the probe-to-task gap (Remark "Probe loss against task loss"): same
    # certificate, checked against the FULL retained-task loss increase.
    coverage_full_task = np.mean(
        [
            r["F_full_task"] <= r["cert_probe_calibrated"]
            for r in recs
            if r["F_full_task"] is not None
        ]
    )
    tightness = [
        (r["cert_probe_calibrated"] - r["F_probe"]) / max(r["tau_i"], 1e-12)
        for r in recs
        if r["F_probe"] is not None
    ]
    beta_binds = np.mean([r["beta_star"] < 1.0 for r in recs])
    gaps = [r["resolve_gap"] for r in recs if r["resolve_gap"] is not None]
    return {
        "deliverable": "e_cert_summary",
        "seed": run["seed"],
        "max_active_players": run["max_active_players"],
        "n_records": run["n_records"],
        "coverage_probe_calibrated": float(coverage_probe) if recs else None,
        "coverage_no_calibration": float(coverage_fixed) if recs else None,
        "coverage_no_probe_slice_replay_calibrated": (
            float(coverage_replay_cal) if not np.isnan(coverage_replay_cal) else None
        ),
        "coverage_full_task_loss": (
            float(coverage_full_task) if not np.isnan(coverage_full_task) else None
        ),
        "median_tightness_in_units_of_tau": float(np.median(tightness)) if tightness else None,
        "fraction_beta_star_binds": float(beta_binds) if recs else None,
        "median_resolve_gap": float(np.median(gaps)) if gaps else None,
        "config_hash": config_hash(
            {"seed": run["seed"], "max_active_players": run["max_active_players"]}
        ),
    }


def write_table(summaries: list[dict], suffix: str = "") -> None:
    lines = ["# E-cert -- the certificate on real streams\n"]
    lines.append(
        "| seed | max active | n steps measured | coverage (calibrated) | "
        "coverage (no calibration) | coverage (no probe slice, replay-cal.) | "
        "coverage (full task loss) | median tightness (units of tau) | "
        "fraction beta* binds | median resolve gap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        cp = s["coverage_probe_calibrated"]
        cf = s["coverage_no_calibration"]
        cr = s["coverage_no_probe_slice_replay_calibrated"]
        cft = s["coverage_full_task_loss"]
        t = s["median_tightness_in_units_of_tau"]
        b = s["fraction_beta_star_binds"]
        rg = s["median_resolve_gap"]
        lines.append(
            f"| {s['seed']} | {s['max_active_players']} | {s['n_records']} | "
            f"{_f(cp)} | {_f(cf)} | {_f(cr)} | {_f(cft)} | {_f(t)} | {_f(b)} | {_f(rg)} |"
        )
    lines.append(
        "\n\"median resolve gap\" (Remark \"Admission control is cheap, not optimal\"): "
        "at steps where beta*<1, the relative extra magnitude re-solving with tau "
        "tightened by beta* would have bought over the cheap backtrack beta*Delta, "
        "compared in the REDUCED beta-space (see run_one's scope note).\n"
    )
    lines.append(
        "\nCoverage below 1 out-of-sample invalidates the certificate; "
        "out-of-sample means the certificate is not a certificate and the budget "
        "guarantee must be withdrawn; coverage at 1 with tightness of order 10 or more "
        "means the certificate is valid but uninformative; replay-slice calibration "
        "matching probe-slice calibration would mean the probe slice is unnecessary.\n"
    )
    lines.append(
        "\nNOT implemented in this pass: the horizon-amortised budget schedule "
        "tau_i^(t)=B_i/(T_i-t) (Corollary \"horizon\"; only the constant schedule "
        "tau_i^(t)=knob is measured). The \"first-order utilities, d_i=0\" ablation "
        "is a separate run (`run_one(method=\"nashmtl\")`), not read off this "
        "trajectory, since it changes the actual update applied.\n"
    )
    (TABLES / f"e8_certificate{suffix}.md").write_text("\n".join(lines))


def write_figure(all_records: list[dict], suffix: str = "") -> None:
    """Certified damage versus realised
    probe-loss increase, one point per step and task, identity line and the
    budget line tau_i marked -- the coverage plot."""
    x = [r["F_probe"] for r in all_records if r["F_probe"] is not None]
    y = [r["cert_probe_calibrated"] for r in all_records if r["F_probe"] is not None]
    tau = [r["tau_i"] for r in all_records if r["F_probe"] is not None]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, alpha=0.3, s=8)
    lims = [min(x + y + [0]), max(x + y + [1e-3])]
    ax.plot(lims, lims, "k--", label="C_i = F_i (identity)")
    if tau:
        ax.axhline(float(np.median(tau)), color="tab:red", linestyle=":", label="median tau_i")
    ax.set_xlabel("realised probe-loss increase F_i")
    ax.set_ylabel("certified damage C_i(Delta_hat)")
    ax.set_title("E-cert: coverage plot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"e8_coverage{suffix}.png", dpi=150)
    plt.close(fig)


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    max_active = None
    if "--max-active" in sys.argv:
        max_active = int(sys.argv[sys.argv.index("--max-active") + 1])
    jvp_batched = "--jvp-batched" in sys.argv
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else None
    buffer_capacity = (
        int(sys.argv[sys.argv.index("--buffer") + 1]) if "--buffer" in sys.argv else None
    )
    stream_name = sys.argv[sys.argv.index("--stream") + 1] if "--stream" in sys.argv else "split20"
    # Each configured run writes beside the default output rather than over it.
    suffix = ""
    if "--out-suffix" in sys.argv:
        suffix = sys.argv[sys.argv.index("--out-suffix") + 1]

    # Concurrent runs each get their own log through the suffix: every task
    # boundary appends the whole record list so far, and lines that long from
    # two processes can interleave in one file.
    if suffix:
        LOG = ROOT / "logs" / f"e8_certificate{suffix}.jsonl"
        HEARTBEAT_PATH = ROOT / "logs" / f"e8_certificate{suffix}_progress.log"
    seeds = [0]
    if "--seeds" in sys.argv:
        seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")]
    summaries = []
    all_records = []
    for seed in seeds:
        started = time.time()
        run = run_one(
            seed, smoke, max_active_players=max_active, jvp_batched=jvp_batched, epochs=epochs,
            buffer_capacity=buffer_capacity, stream_name=stream_name,
        )
        run["epochs"] = epochs
        print(f"seed {seed}: {run['n_records']} records in {time.time() - started:.1f}s")
        log_run(LOG, {"deliverable": "e_cert_run", **run})
        s = summarise(run)
        print(s)
        log_run(LOG, s)
        summaries.append(s)
        all_records.extend(run["records"])

    write_table(summaries, suffix)
    write_figure(all_records, suffix)
    print(f"Wrote tables/e8_certificate{suffix}.md and figures/e8_coverage{suffix}.png")
