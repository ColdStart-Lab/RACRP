"""Does the gain come from where the mechanism says it should?

A falsification condition stated before the method was run: if conditioning on the
previous gap helps because an application's gaps genuinely carry information about
each other, then the improvement must concentrate in the applications where that
dependence exists, and must be small where gaps are effectively independent.

Per application we measure the mutual information between consecutive idle gaps on
the **training** period only, split the applications into quartiles by it, and
compare the two policies **within each quartile**.

**Matched cost within each stratum.** A first version of this check compared the
two arms at a single fixed tail percentile inside each quartile, and appeared to
show the method failing catastrophically in the most-autocorrelated quartile
(-149% cold starts). It was spending 20% less memory there. At a fixed percentile
the two arms simply sit at different points on their own cost curves, so the
comparison says nothing -- the same trap that made an earlier "+38% improvement"
evaporate once cost was matched. Each stratum is therefore swept over the cost
knob and the arms compared by interpolating at a common budget, as everywhere
else in this project.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import conditional_policy
from racrp.sim.simulator import simulate_app

COHORT_FILE = PROCESSED / "cohort.npy"
N_BINS = 10
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)


def gap_autocorrelation_mi(inv: np.ndarray, upto: int) -> tuple[float, int]:
    """MI between consecutive idle gaps, on the training period."""
    a = np.flatnonzero(inv[:upto] > 0)
    if len(a) < 40:
        return np.nan, 0
    gaps = np.diff(a)
    g = np.clip(np.log2(np.maximum(gaps, 1)).astype(int), 0, N_BINS - 1)
    prev, cur = g[:-1], g[1:]
    if len(cur) < 30:
        return np.nan, len(gaps)
    J = np.zeros((N_BINS, N_BINS))
    np.add.at(J, (prev, cur), 1.0)
    J /= J.sum()
    px, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = J * np.log2(J / (px * py))
    return float(np.nansum(t)), len(gaps)


def interp(curve, budget):
    pts = sorted(curve)
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="prev_gap+tod")
    ap.add_argument("--apps", type=int, default=500)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows_all = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all),
                                       min(args.apps, len(rows_all)), replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[mechanism] {len(rows)} applications, key={args.key}")

    idx, mis = [], []
    for i, r in enumerate(rows):
        mi, _ = gap_autocorrelation_mi(counts[r].astype(np.int64), TEST_SLICE[0])
        if np.isfinite(mi):
            idx.append(i)
            mis.append(mi)
    idx = np.array(idx)
    mis = np.array(mis)
    q = pd.qcut(pd.Series(mis), 4,
                labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    qa = q.to_numpy()
    print(f"[mechanism] {len(idx)} applications with a usable gap sequence")

    curves: dict[tuple, list] = {}
    for arm, key in (("baseline", "none"), ("conditional", args.key)):
        for tail in TAILS:
            per = {}
            for i in idx:
                inv = counts[rows[i]].astype(np.int64)
                w = conditional_policy(inv, eval_slice=TEST_SLICE, key=key,
                                       tail_margin=tail)
                s = simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                                 w[TEST_SLICE[0]:TEST_SLICE[1]], float(mem[i]))
                per[i] = (s.idle_gb_minutes, s.cold_starts)
            for lab in q.cat.categories:
                sel = idx[qa == lab]
                curves.setdefault((arm, lab), []).append(
                    (sum(per[i][0] for i in sel), sum(per[i][1] for i in sel)))
            print(f"  {arm:12s} tail={tail:5.1f} done", flush=True)

    out_rows = []
    for lab in q.cat.categories:
        base = sorted(curves[("baseline", lab)])
        cond = curves[("conditional", lab)]
        reds = []
        for budget, cold in base:
            v = interp(cond, budget)
            if v is not None and cold:
                reds.append(100 * (1 - v / cold))
        out_rows.append({
            "stratum": str(lab),
            "apps": int((qa == lab).sum()),
            "median_mi_bits": float(np.median(mis[qa == lab])),
            "matched_cost_reduction_pct": float(np.median(reds)) if reds else None,
            "n_matched_points": len(reds),
        })

    g = pd.DataFrame(out_rows)
    pd.set_option("display.width", 200)
    print("\ncold-start reduction at MATCHED memory cost, within each stratum")
    print(g.to_string(index=False))

    valid = [r for r in out_rows if r["matched_cost_reduction_pct"] is not None]
    corr = (float(np.corrcoef([r["median_mi_bits"] for r in valid],
                              [r["matched_cost_reduction_pct"] for r in valid])[0, 1])
            if len(valid) > 2 else float("nan"))
    lo = valid[0]["matched_cost_reduction_pct"] if valid else float("nan")
    hi = valid[-1]["matched_cost_reduction_pct"] if valid else float("nan")
    print(f"\ncorrelation across strata (median MI vs matched-cost reduction): "
          f"{corr:+.3f}")

    out = {"key": args.key, "n_apps": int(len(idx)),
           "correlation_mi_vs_reduction": corr,
           "reduction_lowest_quartile_pct": lo,
           "reduction_highest_quartile_pct": hi,
           "by_quartile": out_rows}
    (TABLES / "mechanism_check_summary.json").write_text(json.dumps(out, indent=2))

    print("\nREAD-OUT")
    if np.isfinite(hi) and np.isfinite(lo) and hi > lo + 3:
        print("  The gain concentrates in applications whose gaps genuinely depend "
              "on each other, and is smallest where they do not. The mechanism is "
              "behaving as claimed.")
    else:
        print("  The gain does NOT concentrate where the mechanism predicts. The "
              "aggregate improvement must be attributed to something else, and "
              "reported that way.")
    print(f"\nwrote {TABLES / 'mechanism_check_summary.json'}")


if __name__ == "__main__":
    main()
