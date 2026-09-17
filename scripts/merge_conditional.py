"""Merge conditional-sweep runs and recompute the matched-cost comparison.

The full ablation was run in two passes (the cell-count-matched control for the
joint key was added after the first pass), so this stitches the per-arm curves
back together and recomputes the matched-cost table over all of them at once.
Merging is by ``(key, tail_margin)`` and later files win, so re-running a single
arm updates it in place rather than duplicating it.
"""
from __future__ import annotations

import json
import statistics as st

import numpy as np
import pandas as pd

from racrp.config import TABLES

SOURCES = ["conditional_sweep_5arm.json", "conditional_sweep_joint.json",
           "conditional_sweep.json"]
PRETTY = {
    "none": "ATC'20 marginal histogram (baseline)",
    "prev_gap": "Conditioned on previous gap",
    "tod": "Conditioned on time of day",
    "prev_gap+tod": "Conditioned on previous gap x time of day (ours)",
    "shuffled": "RANDOM key, 6 cells (control for prev_gap)",
    "shuffled_joint": "RANDOM key, 24 cells (control for the joint key)",
}
# which control is the right one for each arm; a control must match its arm in
# cell count or it does not control for the variance of splitting the data
CONTROL_FOR = {"prev_gap": "shuffled", "tod": "shuffled",
               "prev_gap+tod": "shuffled_joint"}


def interp_at(curve, budget, xk="idle_gb_minutes", yk="cold_starts"):
    pts = sorted(((r[xk], r[yk]) for r in curve), key=lambda t: t[0])
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main() -> None:
    merged: dict[tuple, dict] = {}
    for name in SOURCES:
        p = TABLES / name
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            merged[(r["key"], r["tail_margin"])] = r
    records = list(merged.values())
    keys = sorted({r["key"] for r in records})
    print(f"[merge] {len(records)} records over {len(keys)} arms: {keys}")

    by: dict[str, list] = {}
    for r in records:
        by.setdefault(r["key"], []).append(r)

    matched = []
    for b in sorted(by.get("none", []), key=lambda r: r["idle_gb_minutes"]):
        budget = b["idle_gb_minutes"]
        row = {"budget_idle_gb_min": round(budget, 1),
               "baseline_cold_starts": b["cold_starts"],
               "baseline_tail": b["tail_margin"]}
        for key in keys:
            if key == "none":
                continue
            v = interp_at(by[key], budget)
            row[key] = None if v is None else round(v, 1)
            if v is not None and b["cold_starts"]:
                row[key + "_reduction_pct"] = round(
                    100.0 * (1 - v / b["cold_starts"]), 2)
        matched.append(row)

    (TABLES / "conditional_sweep.json").write_text(json.dumps(records, indent=2))
    (TABLES / "conditional_matched_cost.json").write_text(json.dumps(matched, indent=2))

    pd.set_option("display.width", 250)
    print("\ncold starts at MATCHED memory budget")
    print(pd.DataFrame(matched).to_string(index=False))

    print("\nsummary (median reduction vs the ATC'20 baseline, at matched cost)")
    print(f"{'arm':52s} {'median':>9} {'range':>18} {'its control':>13}")
    print("-" * 98)
    best = None
    for key in keys:
        if key == "none":
            continue
        red = [r.get(key + "_reduction_pct") for r in matched
               if r.get(key + "_reduction_pct") is not None]
        if not red:
            continue
        ctrl_key = CONTROL_FOR.get(key)
        ctrl = [r.get(ctrl_key + "_reduction_pct") for r in matched
                if ctrl_key and r.get(ctrl_key + "_reduction_pct") is not None]
        med = st.median(red)
        ctrl_txt = f"{st.median(ctrl):+.1f}%" if ctrl else "-"
        print(f"{PRETTY.get(key, key):52s} {med:+8.1f}% "
              f"{min(red):+7.1f}% .. {max(red):+6.1f}% {ctrl_txt:>13}")
        if not key.startswith("shuffled") and (best is None or med > best[1]):
            best = (key, med, min(red), max(red), ctrl_txt)

    if best:
        summary = {"best_arm": best[0], "best_label": PRETTY.get(best[0], best[0]),
                   "median_reduction_pct": best[1],
                   "min_reduction_pct": best[2], "max_reduction_pct": best[3],
                   "its_control_median_pct": best[4]}
        (TABLES / "conditional_headline.json").write_text(json.dumps(summary, indent=2))
        print(f"\nHEADLINE: {summary['best_label']} -> "
              f"{best[1]:.1f}% median cold-start reduction at matched memory cost "
              f"(range {best[2]:.1f}%..{best[3]:.1f}%), "
              f"its cell-count-matched control at {best[4]}.")


if __name__ == "__main__":
    main()
