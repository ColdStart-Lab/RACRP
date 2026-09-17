"""The headline experiment: state-conditioned episodic memory vs. the ATC'20 policy.

Every arm runs through the identical policy implementation with only the
conditioning key changed, so the comparison is an ablation of the conditioning
and not a comparison of two codebases. Every arm is swept over the same
tail-percentile cost knob, and arms are compared by interpolating each curve at a
common memory budget -- comparing single points would be meaningless, since a
policy that holds more containers warm always shows fewer cold starts.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import conditional_policy
from racrp.sim import policies as P
from racrp.eval.harness import evaluate_policy
from racrp.eval.strata import macro_summary, app_sparsity, stratify

COHORT_FILE = PROCESSED / "cohort.npy"
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
KEYS = ("none", "prev_gap", "tod", "prev_gap+tod", "shuffled", "shuffled_joint")
PRETTY = {
    "none": "ATC'20 marginal histogram (baseline)",
    "prev_gap": "Conditioned on previous gap (ours)",
    "tod": "Conditioned on time of day",
    "prev_gap+tod": "Conditioned on previous gap x time of day",
    "shuffled": "RANDOM key, 6 cells (control for prev_gap)",
    "shuffled_joint": "RANDOM key, 24 cells (control for the joint key)",
}


def interp_at(curve, budget, xk="idle_gb_minutes", yk="cold_starts"):
    pts = sorted(((r[xk], r[yk]) for r in curve), key=lambda t: t[0])
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--apps", type=int, default=0, help="0 = whole cohort")
    ap.add_argument("--keys", nargs="*", default=list(KEYS))
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    if args.apps:
        rows = rows[np.random.default_rng(SEED).choice(len(rows), args.apps, False)]
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    sparsity = app_sparsity(counts, rows)
    print(f"[cond] {len(rows)} applications, test days 11-14, kappa={args.kappa}")

    records, strata_rows = [], []
    for key in args.keys:
        for tail in TAILS:
            r = evaluate_policy(
                counts, rows, mem,
                lambda inv, i, k=key, t=tail: conditional_policy(
                    inv, eval_slice=TEST_SLICE, key=k, kappa=args.kappa,
                    tail_margin=t),
                TEST_SLICE, name=f"{key}-t{tail:g}", per_app=True)
            pa = r.pop("_per_app")
            r.update(macro_summary(pa))
            r["key"] = key
            r["tail_margin"] = tail
            records.append(r)
            if key in ("none", "prev_gap") and tail == 92.0:
                s = stratify(pa, sparsity, rows)
                s.insert(0, "key", key)
                strata_rows.append(s)
            print(f"  {PRETTY[key][:38]:40s} tail={tail:5.1f} "
                  f"cold={r['cold_starts']:8,}  gb-min={r['idle_gb_minutes']:11,.0f}")

    (TABLES / "conditional_sweep.json").write_text(json.dumps(records, indent=2))
    if strata_rows:
        pd.concat(strata_rows, ignore_index=True).to_json(
            TABLES / "conditional_strata.json", orient="records")

    # ---------------------------------------------------- matched-cost comparison
    by = {}
    for r in records:
        by.setdefault(r["key"], []).append(r)
    base = by.get("none", [])
    matched = []
    if base:
        for b in sorted(base, key=lambda r: r["idle_gb_minutes"]):
            budget = b["idle_gb_minutes"]
            row = {"budget_idle_gb_min": round(budget, 1),
                   "baseline_cold_starts": b["cold_starts"],
                   "baseline_tail": b["tail_margin"]}
            for key in args.keys:
                if key == "none":
                    continue
                v = interp_at(by.get(key, []), budget)
                row[key] = None if v is None else round(v, 1)
                if v is not None and b["cold_starts"]:
                    row[key + "_reduction_pct"] = round(
                        100.0 * (1 - v / b["cold_starts"]), 2)
            matched.append(row)
    (TABLES / "conditional_matched_cost.json").write_text(json.dumps(matched, indent=2))

    df = pd.DataFrame(matched)
    pd.set_option("display.width", 220)
    print("\ncold starts at MATCHED memory budget (baseline's own operating points)")
    print(df.to_string(index=False))

    red = [r.get("prev_gap_reduction_pct") for r in matched
           if r.get("prev_gap_reduction_pct") is not None]
    if red:
        print(f"\nconditioning on the previous gap: median reduction "
              f"{np.median(red):+.1f}%  (range {min(red):+.1f}% .. {max(red):+.1f}%)")
    ctrl = [r.get("shuffled_reduction_pct") for r in matched
            if r.get("shuffled_reduction_pct") is not None]
    if ctrl:
        print(f"random-key control:                median reduction "
              f"{np.median(ctrl):+.1f}%   <- must be ~0 for the result to mean anything")


if __name__ == "__main__":
    main()
