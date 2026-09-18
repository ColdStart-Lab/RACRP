"""Is the result an artefact of the two hyper-parameters we chose by hand?

The conditional estimator has exactly two: ``kappa``, the strength of the pull
towards the marginal measured in equivalent observed gaps, and the number of
buckets the previous gap is discretised into. Both were set by judgement rather
than by search -- kappa = 8 because a handful of gaps should not yet outvote an
application's whole history, and six log2 buckets because that spans one minute
to a day at a resolution the data can support.

Neither was tuned on the test split, and this script exists so that claim can be
checked rather than asserted: it sweeps both and reports the matched-cost
reduction at every setting. A result that only appears at one corner of that grid
would be a tuning artefact; a result that is flat across it is a property of the
method.

Everything is compared against the ATC'20 baseline at matched memory cost, as
everywhere else in this project.
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
from racrp.sim.simulator import simulate_app, SimResult

COHORT_FILE = PROCESSED / "cohort.npy"
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
KAPPAS = (2.0, 4.0, 8.0, 16.0, 32.0)
BUCKETS = (4, 6, 8)


def curve(counts, rows, mem, **kw):
    out = []
    for tail in TAILS:
        total = SimResult()
        for i, r in enumerate(rows):
            inv = counts[r].astype(np.int64)
            w = conditional_policy(inv, eval_slice=TEST_SLICE, tail_margin=tail, **kw)
            total.merge(simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                                     w[TEST_SLICE[0]:TEST_SLICE[1]], float(mem[i])))
        out.append((total.idle_gb_minutes, total.cold_starts))
    return out


def interp(c, budget):
    pts = sorted(c)
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=400)
    ap.add_argument("--key", default="prev_gap+tod")
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows_all = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all),
                                       min(args.apps, len(rows_all)), replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[sensitivity] {len(rows)} applications, key={args.key}")

    base = curve(counts, rows, mem, key="none")
    print("  baseline done", flush=True)

    recs = []
    for nb in BUCKETS:
        for kap in KAPPAS:
            c = curve(counts, rows, mem, key=args.key, kappa=kap, n_gap_buckets=nb)
            reds = [100 * (1 - v / cold)
                    for budget, cold in base
                    if cold and (v := interp(c, budget)) is not None]
            med = st.median(reds) if reds else None
            recs.append({"buckets": nb, "kappa": kap,
                         "median_reduction_pct": med,
                         "n_matched_points": len(reds)})
            print(f"  buckets={nb} kappa={kap:5.1f}: "
                  f"{med:+.1f}%" if med is not None else "  (no overlap)",
                  flush=True)

    df = pd.DataFrame(recs)
    piv = df.pivot(index="buckets", columns="kappa", values="median_reduction_pct")
    pd.set_option("display.width", 200)
    print("\nmedian cold-start reduction vs ATC'20 at matched cost, by setting")
    print(piv.round(1).to_string())

    vals = [r["median_reduction_pct"] for r in recs
            if r["median_reduction_pct"] is not None]
    out = {"key": args.key, "n_apps": int(len(rows)), "grid": recs,
           "min_pct": min(vals) if vals else None,
           "max_pct": max(vals) if vals else None,
           "median_pct": st.median(vals) if vals else None,
           "all_positive": bool(vals) and min(vals) > 0}
    (TABLES / "sensitivity.json").write_text(json.dumps(out, indent=2))

    print("\nREAD-OUT")
    if out["all_positive"]:
        print(f"  Positive at every one of the {len(vals)} settings tried "
              f"({out['min_pct']:+.1f}% to {out['max_pct']:+.1f}%). The result is "
              f"a property of the method, not of the two values we picked.")
    else:
        print("  The result changes sign somewhere in the grid. The chosen "
              "setting matters and that has to be reported.")
    print(f"\nwrote {TABLES / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
