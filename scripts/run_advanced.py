"""Three refinements, each measured against the same baseline at matched cost.

  A. **Derived window** — the keep-alive is the cost-optimal stopping time rather
     than a tail percentile. Removes the percentile knob; what is swept instead
     is the price of a cold start in GB-minutes, which a platform can quote.
  B. **Per-application selection** — condition only where a held-out likelihood
     test on the application's *own training data* says conditioning predicts
     better. Follows directly from the mechanism check, which found the gain is
     negative in one quartile.
  C. **Regime weighting** — down-weight history that a two-sample test says no
     longer describes the application's behaviour.

Arms are compared by interpolating each cost/benefit curve at the ATC'20
baseline's own operating points, as everywhere else in this project. Each
refinement is also run alone, so a combined gain can be attributed.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import conditional_policy
from racrp.baselines.hazard import hazard_policy
from racrp.baselines.selective import selective_policy
from racrp.sim.simulator import simulate_app, SimResult

COHORT_FILE = PROCESSED / "cohort.npy"
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
# price of memory per minute relative to one cold start; smaller = cold starts
# are more expensive, so the optimal policy holds on longer
RATIOS = (0.30, 0.12, 0.05, 0.02, 0.008, 0.003)


def score(counts, rows, mem, policy, knobs):
    curve = []
    for k in knobs:
        tot = SimResult()
        for i, r in enumerate(rows):
            inv = counts[r].astype(np.int64)
            w = np.asarray(policy(inv, k))
            tot.merge(simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                                   w[TEST_SLICE[0]:TEST_SLICE[1]], float(mem[i])))
        curve.append((tot.idle_gb_minutes, tot.cold_starts))
    return curve


def interp(curve, budget):
    pts = sorted(curve)
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=500)
    ap.add_argument("--kappa", type=float, default=8.0)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows_all = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all),
                                       min(args.apps, len(rows_all)), replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[advanced] {len(rows)} applications")

    K = args.kappa
    arms = {
        "ATC'20 marginal (baseline)":
            (lambda inv, t: conditional_policy(inv, eval_slice=TEST_SLICE,
                                               key="none", tail_margin=t), TAILS),
        "conditional, percentile window":
            (lambda inv, t: conditional_policy(inv, eval_slice=TEST_SLICE,
                                               key="prev_gap+tod", kappa=K,
                                               tail_margin=t), TAILS),
        # The derived rule minimises the cost of holding CONTINUOUSLY from the
        # gap's start, so it is applied without a pre-warm offset. Composing it
        # with one shifts the window the T was optimised for and destroys the
        # result -- measured, and recorded in the experiment log.
        "A: marginal, DERIVED window":
            (lambda inv, r: hazard_policy(inv, eval_slice=TEST_SLICE, key="none",
                                          price_ratio=r, kappa=K,
                                          head_margin=0.0), RATIOS),
        "A: conditional + DERIVED window":
            (lambda inv, r: hazard_policy(inv, eval_slice=TEST_SLICE,
                                          key="prev_gap+tod", price_ratio=r,
                                          kappa=K, head_margin=0.0), RATIOS),
        "B: + per-app selection":
            (lambda inv, r: selective_policy(inv, eval_slice=TEST_SLICE,
                                             price_ratio=r, kappa=K,
                                             use_selection=True,
                                             use_regime=False), RATIOS),
        "C: + regime weighting":
            (lambda inv, r: selective_policy(inv, eval_slice=TEST_SLICE,
                                             price_ratio=r, kappa=K,
                                             use_selection=False,
                                             use_regime=True), RATIOS),
        "B+C: selection and regime":
            (lambda inv, r: selective_policy(inv, eval_slice=TEST_SLICE,
                                             price_ratio=r, kappa=K,
                                             use_selection=True,
                                             use_regime=True), RATIOS),
    }

    curves, records = {}, []
    for name, (pol, knobs) in arms.items():
        c = score(counts, rows, mem, pol, knobs)
        curves[name] = c
        for (gb, cold), k in zip(c, knobs):
            records.append({"arm": name, "knob": float(k),
                            "idle_gb_minutes": gb, "cold_starts": cold})
        print(f"  {name:34s} " +
              " ".join(f"{cold/1000:.0f}k@{gb/1000:.0f}k" for gb, cold in c),
              flush=True)

    base = curves["ATC'20 marginal (baseline)"]
    matched, summary = [], []
    for gb, cold in sorted(base):
        row = {"budget_idle_gb_min": round(gb, 1), "baseline_cold_starts": cold}
        for name, c in curves.items():
            if name.startswith("ATC"):
                continue
            v = interp(c, gb)
            row[name] = None if v is None else round(v, 1)
            if v is not None and cold:
                row[name + " %"] = round(100 * (1 - v / cold), 2)
        matched.append(row)

    for name in curves:
        if name.startswith("ATC"):
            continue
        vals = [r[name + " %"] for r in matched if r.get(name + " %") is not None]
        if vals:
            summary.append({"arm": name, "median_reduction_pct": st.median(vals),
                            "min": min(vals), "max": max(vals), "n": len(vals)})

    (TABLES / "advanced_sweep.json").write_text(json.dumps(records, indent=2))
    (TABLES / "advanced_matched_cost.json").write_text(json.dumps(matched, indent=2))
    (TABLES / "advanced_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 250)
    print("\nmedian cold-start reduction vs ATC'20, at matched memory cost")
    df = pd.DataFrame(summary).sort_values("median_reduction_pct", ascending=False)
    print(df.to_string(index=False))
    print(f"\nwrote advanced_summary.json")


if __name__ == "__main__":
    main()
