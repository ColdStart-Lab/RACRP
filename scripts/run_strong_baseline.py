"""Give the baseline every advantage the derived policy has, then compare.

The derived window optimises a **pair** — when to bring the container back and
how long to hold it. The ATC'20 policy as published fixes the first at the 5th
percentile of the idle-gap histogram and the second at the 99th, and our sweeps
so far have varied only the second. That is a real asymmetry, and it is the first
thing a reviewer should attack: *of course* a policy that tunes two degrees of
freedom beats one that tunes one.

So this script removes the asymmetry. It sweeps the baseline over **both**
percentiles, 6 x 6, and keeps the lower envelope of the resulting cloud — the
best cost/latency frontier reachable by any fixed percentile pair. That envelope
is a stronger baseline than anything in the ATC'20 paper, because the paper does
not tune the head at all and the envelope is chosen with the test period in view.

If the derived window still wins against that, the gain is not extra degrees of
freedom. It is that a percentile is the wrong functional form: the cost-optimal
window depends on the *shape* of the distribution between the percentiles, which
no fixed pair of quantiles can see.
"""
from __future__ import annotations

import argparse, json, statistics as st
import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import conditional_policy
from racrp.baselines.hazard import hazard_policy
from racrp.sim.simulator import simulate_app, SimResult

COHORT_FILE = PROCESSED / "cohort.npy"
HEADS = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0)
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
RATIOS = (0.30, 0.12, 0.05, 0.02, 0.008, 0.003)


def run(counts, rows, mem, policy):
    tot = SimResult()
    for i, r in enumerate(rows):
        inv = counts[r].astype(np.int64)
        w = np.asarray(policy(inv))
        tot.merge(simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                               w[TEST_SLICE[0]:TEST_SLICE[1]], float(mem[i])))
    return tot.idle_gb_minutes, tot.cold_starts


def lower_envelope(points):
    """Pareto frontier: keep a point only if nothing else is cheaper AND colder."""
    pts = sorted(points)
    out, best = [], np.inf
    for gb, cold in pts:
        if cold < best - 1e-9:
            out.append((gb, cold))
            best = cold
    return out


def interp(curve, budget):
    pts = sorted(curve)
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=600)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows_all = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all),
                                       min(args.apps, len(rows_all)), replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[strong] {len(rows)} applications, 6x6 baseline sweep")

    cloud = []
    for h in HEADS:
        row = []
        for t in TAILS:
            if h >= t:
                continue
            gb, cold = run(counts, rows, mem,
                           lambda inv, h=h, t=t: conditional_policy(
                               inv, eval_slice=TEST_SLICE, key="none",
                               head_margin=h, tail_margin=t))
            cloud.append({"head_pct": h, "tail_pct": t,
                          "idle_gb_minutes": gb, "cold_starts": cold})
            row.append(f"{cold/1000:.0f}k@{gb/1000:.0f}k")
        print(f"  head p{h:<5g} " + " ".join(row), flush=True)

    env = lower_envelope([(c["idle_gb_minutes"], c["cold_starts"]) for c in cloud])
    print("\n  ATC'20 best envelope over 6x6 " +
          " ".join(f"{c/1000:.0f}k@{g/1000:.0f}k" for g, c in env), flush=True)

    der = []
    for r in RATIOS:
        gb, cold = run(counts, rows, mem,
                       lambda inv, r=r: hazard_policy(
                           inv, eval_slice=TEST_SLICE, key="none",
                           price_ratio=r, joint=True))
        der.append((gb, cold))
    print("  DERIVED window, marginal      " +
          " ".join(f"{c/1000:.0f}k@{g/1000:.0f}k" for g, c in der), flush=True)

    rows_out, vals = [], []
    for gb, cold in env:
        v = interp(der, gb)
        if v is None or not cold:
            continue
        pct = round(100 * (1 - v / cold), 2)
        vals.append(pct)
        rows_out.append({"budget_idle_gb_min": round(gb, 1),
                         "best_baseline_cold_starts": cold,
                         "derived_cold_starts": round(v, 1),
                         "reduction_pct": pct})

    summary = {"n_apps": int(len(rows)), "baseline_configs": len(cloud),
               "median_reduction_pct": st.median(vals) if vals else None,
               "min": min(vals) if vals else None,
               "max": max(vals) if vals else None, "n_points": len(vals)}
    (TABLES / "strong_baseline_cloud.json").write_text(json.dumps(cloud, indent=2))
    (TABLES / "strong_baseline_matched.json").write_text(json.dumps(rows_out, indent=2))
    (TABLES / "strong_baseline_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print("\nvs the BEST of 6x6 tuned ATC'20 percentile pairs, at matched cost")
    print(pd.DataFrame(rows_out).to_string(index=False))
    print(f"\nmedian {summary['median_reduction_pct']}%  "
          f"(range {summary['min']} .. {summary['max']}, n={summary['n_points']})")


if __name__ == "__main__":
    main()
