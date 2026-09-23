"""The headline experiment: derive the keep-alive window from the cost model.

The deployed ATC'20 policy reads its pre-warm offset and keep-alive length off
fixed percentiles (5th and 99th) of an application's idle-gap histogram. Those
percentiles are a heuristic. We derive the pair instead, by minimising the
expected cost of the warm window:

    C(a, L) = m · E[ overlap of [a, a+L) with the gap ]  +  K · P(gap outside it)

over a small grid of offsets, each in O(range). Nothing is tuned: what is swept
is `K/m`, the price of a cold start in GB-minutes, which a platform can quote.

Arms, all through the same simulator and compared at matched memory cost:

  ATC'20 marginal                the deployed policy
  DERIVED, marginal              same distribution, window derived not tuned
  DERIVED + conditioning         also conditioned on the application's state
  conditional, percentile        the project's previous best
  CONTROL: donor app             the same optimisation run on another
                                 application's gap distribution, to test whether the
                                 gain needs the application's OWN history
"""
from __future__ import annotations

import argparse, json, statistics as st
import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.hazard import hazard_policy
from racrp.baselines.conditional import conditional_policy
from racrp.sim.simulator import simulate_app, SimResult

COHORT_FILE = PROCESSED / "cohort.npy"
HIST_RANGE = 240
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
RATIOS = (0.30, 0.12, 0.05, 0.02, 0.008, 0.003)


def build_donors(counts, rows):
    """Each application gets another application's training-period gap histogram.

    A derangement, so nobody is handed their own. Built from the training period
    only, exactly like the real estimator, so the control differs from the treated
    arm in one respect and one only: whose history it is.
    """
    rng = np.random.default_rng(SEED)
    hists = []
    for r in rows:
        inv = counts[r].astype(np.int64)
        a = np.flatnonzero(inv[:TEST_SLICE[0]] > 0)
        h = np.zeros(HIST_RANGE + 1)
        if len(a) > 1:
            np.add.at(h, np.minimum(np.diff(a), HIST_RANGE), 1.0)
        hists.append(h)
    n = len(rows)
    perm = rng.permutation(n)
    fixed = np.flatnonzero(perm == np.arange(n))          # break any fixed points
    for i in fixed:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return [hists[j] if hists[j].sum() > 0 else hists[i]
            for i, j in enumerate(perm)]


def curve(counts, rows, mem, fn, knobs, label):
    out = []
    for k in knobs:
        t = SimResult()
        for i, r in enumerate(rows):
            inv = counts[r].astype(np.int64)
            w = np.asarray(fn(inv, k, i))
            t.merge(simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                                 w[TEST_SLICE[0]:TEST_SLICE[1]], float(mem[i])))
        out.append((t.idle_gb_minutes, t.cold_starts))
    print(f"  {label:38s} " +
          " ".join(f"{c/1000:.0f}k@{g/1000:.0f}k" for g, c in out), flush=True)
    return out


def interp(c, x):
    p = sorted(c)
    if not p or x < p[0][0] or x > p[-1][0]:
        return None
    return float(np.interp(x, [q[0] for q in p], [q[1] for q in p]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=0)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    if args.apps:
        rows = np.sort(rows[np.random.default_rng(SEED)
                            .choice(len(rows), args.apps, replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[derived] {len(rows)} applications, test days 11-14")

    donors = build_donors(counts, rows)

    arms = {
        "ATC'20 marginal (baseline)":
            (lambda inv, t, i: conditional_policy(inv, eval_slice=TEST_SLICE,
                                                  key="none", tail_margin=t), TAILS),
        "DERIVED window, marginal":
            (lambda inv, r, i: hazard_policy(inv, eval_slice=TEST_SLICE, key="none",
                                             price_ratio=r, joint=True), RATIOS),
        "DERIVED window + conditioning":
            (lambda inv, r, i: hazard_policy(inv, eval_slice=TEST_SLICE,
                                             key="prev_gap+tod", price_ratio=r,
                                             joint=True), RATIOS),
        "conditional, percentile window":
            (lambda inv, t, i: conditional_policy(inv, eval_slice=TEST_SLICE,
                                                  key="prev_gap+tod",
                                                  tail_margin=t), TAILS),
        "CONTROL: derived from a donor app":
            (lambda inv, r, i: hazard_policy(inv, eval_slice=TEST_SLICE, key="none",
                                             price_ratio=r, joint=True,
                                             donor_hist=donors[i]), RATIOS),
    }

    curves, records = {}, []
    for name, (fn, knobs) in arms.items():
        c = curve(counts, rows, mem, fn, knobs, name)
        curves[name] = c
        for (gb, cold), k in zip(c, knobs):
            records.append({"arm": name, "knob": float(k),
                            "idle_gb_minutes": gb, "cold_starts": cold})

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

    (TABLES / "derived_sweep.json").write_text(json.dumps(records, indent=2))
    (TABLES / "derived_matched_cost.json").write_text(json.dumps(matched, indent=2))
    (TABLES / "derived_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 240)
    print("\nmedian cold-start reduction vs ATC'20, at matched memory cost")
    print(pd.DataFrame(summary).sort_values("median_reduction_pct",
                                            ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
