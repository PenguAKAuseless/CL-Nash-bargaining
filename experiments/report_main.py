"""Build the main, rank and appendix tables from the offline selection log."""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = ROOT / "logs" / "ltt_select.jsonl"
TAUS = [0.05, 0.1, 0.4]
# only runs listed in these logs enter the tables
TAG_DIRS = ("logs/e5_walltime.jsonl", "logs/a_tune.jsonl")


def post_fix_keys() -> dict[str, dict]:
    keys = {}
    for f in TAG_DIRS:
        p = ROOT / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("failed") or not r.get("trajectory_dir"):
                continue
            keys[Path(r["trajectory_dir"]).name] = r
    return keys


def variant(rec: dict) -> tuple[str, bool]:
    """Return a label for the configuration and whether it is the canonical arm for its proposer."""
    c = rec["config"]
    p = c["proposer"]
    if rec.get("zeta_floor") is not None or c.get("tag") == "atune":
        return f"zeta={c['zeta_floor']:g} train_tau={c['tau']:g}", False
    if p == "derpp":
        a, b = c["derpp_alpha"], c["derpp_beta"]
        return f"alpha=beta={a:g}", (a, b) == (1.0, 1.0)
    if p == "replay_ggn":
        mu = c["ggn_mu"]
        return f"mu={mu:g}", mu == 0.1
    return "--", True


def bonferroni_pick(cands: list[dict]) -> dict | None:
    """Return the best-calibration candidate within the simultaneously certified set."""
    ok = [c for c in cands if c.get("certified_simultaneous")]
    if not ok:
        return None
    return max(ok, key=lambda c: (c["calibration_accuracy"], c["name"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.1,
                    help="the cert_tau the main table reports at")
    args = ap.parse_args()

    keys = post_fix_keys()
    rows = [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]
    # candidates are stored once per (run, tau), on the fixed_sequence row.
    by = {}
    fs_pick = {}
    for r in rows:
        if r["run_key"] not in keys:
            continue
        if r["rule"] == "fixed_sequence":
            if r.get("candidates"):
                by[(r["run_key"], r["tau"])] = r
            fs_pick[(r["run_key"], r["tau"])] = r.get("selected")

    cells = []
    for (key, tau), r in sorted(by.items()):
        pick = bonferroni_pick(r["candidates"])
        label, canonical = variant(keys[key])
        cells.append({
            "key": key, "tau": tau, "proposer": r["proposer"], "seed": r["seed"],
            "variant": label, "canonical": canonical,
            "n_simul": sum(1 for c in r["candidates"] if c.get("certified_simultaneous")),
            "n_cand": len(r["candidates"]),
            "pick": pick,
            "theta_final": r.get("uncertified_A_final"),
            "fs": fs_pick.get((key, tau)),
        })

    # tightest cert_tau: smallest tau at which the Bonferroni set is non-empty
    tightest = {}
    for c in cells:
        if c["pick"] is not None:
            k = c["key"]
            tightest[k] = min(tightest.get(k, 1e9), c["tau"])

    print(f"\n## Main table (cert_tau = {args.tau:g}, Bonferroni d/K + best_calibration)\n")
    hdr = ("proposer", "variant", "seed", "A_20 theta_final", "A_20 after selection",
           "UCB at selection", "certified", "tightest cert_tau", "candidate")
    all_at_tau = [c for c in cells if abs(c["tau"] - args.tau) < 1e-12]
    at_tau = [c for c in all_at_tau if c["canonical"]]
    others = [c for c in all_at_tau if not c["canonical"]]

    def emit(cs):
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for c in sorted(cs, key=lambda c: (c["proposer"], c["variant"], c["seed"])):
            p = c["pick"]
            tt = tightest.get(c["key"])
            print("| {} | {} | {} | {:.4f} | {} | {} | {} | {} | {} |".format(
                c["proposer"], c["variant"], c["seed"], c["theta_final"],
                f"{p['A_final']:.4f}" if p else "ABSTAIN",
                f"{p['audit_bound_simultaneous']:.4f}" if p else "--",
                f"{c['n_simul']}/{c['n_cand']}",
                f"{tt:g}" if tt else "none",
                p["name"] if p else "--"))

    emit(at_tau)
    if others:
        print("\n### Secondary arms (not in the rank table)\n")
        emit(others)

    # abstaining cells are ranked by theta_final accuracy and have no UCB rank
    print(f"\n## Rank by accuracy vs rank by UCB (cert_tau = {args.tau:g})\n")
    per_seed = defaultdict(list)
    for c in at_tau:
        per_seed[c["seed"]].append(c)
    acc_rank = defaultdict(dict)
    ucb_rank = defaultdict(dict)
    for seed, cs in sorted(per_seed.items()):
        order = sorted(cs, key=lambda c: -(c["pick"]["A_final"] if c["pick"]
                                           else c["theta_final"]))
        for i, c in enumerate(order, 1):
            acc_rank[c["proposer"]][seed] = i
        cert = [c for c in cs if c["pick"]]
        for i, c in enumerate(sorted(cert,
                                     key=lambda c: c["pick"]["audit_bound_simultaneous"]), 1):
            ucb_rank[c["proposer"]][seed] = i
    props = sorted(acc_rank, key=lambda p: st.mean(acc_rank[p].values()))
    seeds = sorted(per_seed)
    print("| proposer | " + " | ".join(f"acc rank s{s}" for s in seeds)
          + " | mean acc rank | " + " | ".join(f"UCB rank s{s}" for s in seeds)
          + " | mean UCB rank |")
    print("|" + "---|" * (2 * len(seeds) + 3))
    for p in props:
        a = [acc_rank[p].get(s) for s in seeds]
        u = [ucb_rank[p].get(s) for s in seeds]
        uu = [x for x in u if x]
        print("| {} | {} | {:.1f} | {} | {} |".format(
            p, " | ".join(str(x) if x else "--" for x in a),
            st.mean([x for x in a if x]),
            " | ".join(str(x) if x else "--" for x in u),
            f"{st.mean(uu):.1f}" if uu else "--"))

    # compared only on cells where both rules select a candidate
    print("\n## Appendix: fixed_sequence vs best_calibration\n")
    gaps = []
    for c in cells:
        p, f = c["pick"], c["fs"]
        if p and f:
            gaps.append((c["proposer"], c["seed"], c["tau"],
                         f["A_final"], p["A_final"], p["A_final"] - f["A_final"]))
    if gaps:
        worst = max(gaps, key=lambda g: g[5])
        print(f"over {len(gaps)} certifying cells the fixed sequence picks a candidate "
              f"{st.mean(g[5] for g in gaps) * 100:.1f} accuracy points worse on average "
              f"and up to {worst[5] * 100:.1f} points worse "
              f"({worst[0]}, seed {worst[1]}, cert_tau {worst[2]:g}: "
              f"{worst[3]:.4f} vs {worst[4]:.4f}).")


if __name__ == "__main__":
    main()
