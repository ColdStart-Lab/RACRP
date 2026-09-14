"""The central experiment: does experience transfer across applications?

Evaluated on **unseen** applications -- ones whose behaviour was never available
when the fleet bank was built -- while sweeping how much of their *own* history
the scheduler is allowed to have observed.

Policies, all scored through the same simulator on the same applications:

  fixed keep-alive K        needs no history at all (the platform default)
  own-histogram (ATC'20)    the deployed policy, restricted to H minutes of the
                            application's own past -- this is the real baseline
  fleet-prior only          retrieved similar applications' pooled gap
                            distribution, ignoring the target's own history
  RACRP-shrinkage           blend of the two, weighted by how much own evidence
                            exists (kappa equivalent gaps)
  oracle                    perfect foresight, as an upper bound

Three things this is designed to show, and to be capable of *failing* to show:

  1. at small H, borrowing strength beats the per-application histogram;
  2. as H grows the two converge, because the shrinkage estimator provably
     reduces to the histogram -- a method that won everywhere would be suspect;
  3. the **equivalence point**: how many hours of an application's own history a
     shared memory is worth.

Two things this script is careful about, because without them the comparison is
not valid:

  * **Matched cost.** A policy that keeps more containers warm will always show
    fewer cold starts. Every variant is therefore swept over its tail-percentile
    knob to trace a cost/benefit curve, and variants are compared by interpolating
    each curve at a common idle-GB-minute budget. A single point comparison
    between policies that spend different amounts of memory says nothing.
  * **Where scarcity actually binds.** Policies keep learning from gaps revealed
    during evaluation, so after four days even an application that started with no
    history has accumulated some. Results are reported both over the whole test
    slice and over the **first six hours**, which is the regime the argument is
    about: a function that has just been deployed.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from racrp.config import (
    TABLES, PROCESSED, TEST_SLICE, KEEPALIVE_SWEEP, SEED,
)

from racrp.data.trace import build_app_matrix
from racrp.data.splits import (
    split_apps, observable_slice, app_profile, HISTORY_SWEEP, HISTORY_LABEL,
)
from racrp.baselines.fleet import FleetBank, fleet_policy
from racrp.sim import policies as P
from racrp.eval.harness import evaluate_policy
from racrp.eval.strata import macro_summary
from racrp.sim.simulator import SimResult, simulate_app

COHORT_FILE = PROCESSED / "cohort.npy"

# Tail percentile of the (blended) gap distribution -> how long to hold warm.
# This is the cost knob every variant is swept over, so that curves can be
# compared at equal spend rather than at whatever point each happens to land on.
TAIL_SWEEP = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
EARLY_HOURS = 6


def interp_at(curve, budget, xkey, ykey):
    """Cold starts of a swept policy at a given memory budget.

    Linear interpolation along the policy's own cost/benefit curve, so two
    policies that cannot be run at exactly the same spend are still compared at
    the same spend. Returns None when the budget lies outside the curve, because
    extrapolating a Pareto curve would manufacture a result.
    """
    pts = sorted(((r[xkey], r[ykey]) for r in curve), key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if not xs or budget < xs[0] or budget > xs[-1]:
        return None
    return float(np.interp(budget, xs, ys))


def matched_cost_table(records):
    """Compare each variant against the ATC'20 histogram at the histogram's spend."""
    out = []
    by = {}
    for r in records:
        if r["history_minutes"] < 0:
            continue
        by.setdefault((r["history_minutes"], r["method"]), []).append(r)

    for H in sorted({h for h, _ in by}):
        base = by.get((H, "own-histogram-ATC20"), [])
        if not base:
            continue
        for scope, xk, yk in (("full", "idle_gb_minutes", "cold_starts"),
                              ("early", "early_idle_gb_minutes", "early_cold_starts")):
            budgets = sorted(r[xk] for r in base)
            budget = budgets[len(budgets) // 2]
            row = {"history_minutes": H, "history_label": HISTORY_LABEL[H],
                   "scope": scope, "budget_idle_gb_min": round(budget, 1)}
            ref = interp_at(base, budget, xk, yk)
            row["own_histogram"] = ref
            for method in ("fleet-prior-only", "RACRP-shrinkage"):
                v = interp_at(by.get((H, method), []), budget, xk, yk)
                row[method] = None if v is None else round(v, 1)
                if v is not None and ref:
                    row[method + "_red_pct"] = round(100.0 * (1 - v / ref), 2)
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16, help="applications retrieved")
    ap.add_argument("--kappa", type=float, default=8.0,
                    help="fleet prior strength, in equivalent observed gaps")
    ap.add_argument("--unseen-frac", type=float, default=0.30)
    ap.add_argument("--keepalive-sweep", type=int, nargs="*",
                    default=[3, 5, 10, 20, 30, 60, 120])
    ap.add_argument("--early-hours", type=int, default=EARLY_HOURS)
    args = ap.parse_args()
    early_slice = (TEST_SLICE[0], TEST_SLICE[0] + args.early_hours * 60)

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    seen, unseen = split_apps(meta, rows, unseen_frac=args.unseen_frac, seed=SEED)
    print(f"[scarcity] cohort={len(rows)}  seen={len(seen)}  unseen={len(unseen)}")

    urows = rows[unseen]
    umem = meta.loc[urows, "memory_mb"].to_numpy()
    inv_total = counts[urows, TEST_SLICE[0]:TEST_SLICE[1]].sum()
    print(f"[scarcity] unseen apps contribute {inv_total:,} invocations "
          f"over the test slice")

    records = []

    # ---- history-free reference: the platform default, swept
    def score(pol, name, H, method, **extra):
        """Score one policy over the full test slice and the early window.

        The policy's decisions are computed **once** per application and both
        windows are scored from the same array. Calling the evaluation harness
        twice would recompute every policy, which turns a minutes-long sweep into
        an hours-long one for no change in the result.
        """
        full, early = SimResult(), SimResult()
        per_app = []
        for i, r_ in enumerate(urows):
            inv = counts[r_].astype(np.int64)
            warm = np.asarray(pol(inv, i))
            m = float(umem[i])
            rf = simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                              warm[TEST_SLICE[0]:TEST_SLICE[1]], m)
            re_ = simulate_app(inv[early_slice[0]:early_slice[1]],
                               warm[early_slice[0]:early_slice[1]], m)
            full.merge(rf)
            early.merge(re_)
            d = rf.as_dict(); d["app"] = int(r_)
            per_app.append(d)

        r = full.as_dict()
        r["policy"] = name
        r.update(macro_summary(per_app))
        r["early_cold_starts"] = early.cold_starts
        r["early_idle_gb_minutes"] = early.idle_gb_minutes
        r["early_invocations"] = early.invocations
        r["history_minutes"] = H
        r["method"] = method
        r.update(extra)
        records.append(r)
        return r

    for K in args.keepalive_sweep:
        r = score(lambda inv, i, K=K: P.fixed_keepalive(inv, K),
                  f"fixed-keepalive-{K}", -1, "fixed-keepalive", keepalive=K)
        print(f"  fixed keep-alive {K:3d}m: cold={r['cold_starts']:7,} "
              f"gbmin={r['idle_gb_minutes']:10,.0f} "
              f"early_cold={r['early_cold_starts']:6,}")

    score(lambda inv, i: P.oracle(inv, 0), "oracle", -1, "oracle")

    # ---- the sweep
    for H in HISTORY_SWEEP:
        hw = observable_slice(H)
        # keys are built from the SAME budget the target has (so they are
        # comparable), but the distributions being borrowed come from the seen
        # applications' full observed history -- they are the ones we know well.
        bank = FleetBank(counts, rows, meta, seen, hw,
                         dist_window=(0, TEST_SLICE[0]))
        # correct the pooled prior's systematic scale bias, fitted leave-one-out
        # on the SEEN applications only
        scale = bank.calibrate()
        profiles = np.stack([
            app_profile(counts, int(rows[p]), hw, meta.loc[int(rows[p])])
            for p in unseen])
        print(f"\n[scarcity] H={H:6d} min ({HISTORY_LABEL[H]:>13s})  "
              f"observable window={hw}")

        variants = {
            "own-histogram-ATC20": dict(use_own=True, use_prior=False),
            "fleet-prior-only":    dict(use_own=False, use_prior=True),
            "RACRP-shrinkage":     dict(use_own=True, use_prior=True),
        }
        for name, kw in variants.items():
            for tail in TAIL_SWEEP:
                def pol(inv, i, kw=kw, tail=tail):
                    return fleet_policy(inv, profile=profiles[i], bank=bank,
                                        history_window=hw, eval_slice=TEST_SLICE,
                                        k=args.k, kappa=args.kappa,
                                        tail_margin=tail, **kw)

                r = score(pol, f"{name}-H{H}-t{tail:g}", H, name,
                          history_label=HISTORY_LABEL[H], tail_margin=tail,
                          k=args.k, kappa=args.kappa)
                print(f"    {name:22s} tail={tail:5.1f} cold={r['cold_starts']:7,} "
                      f"gbmin={r['idle_gb_minutes']:10,.0f} "
                      f"early_cold={r['early_cold_starts']:6,}")

    out = TABLES / "scarcity_sweep.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\n[scarcity] wrote {out}")

    matched = matched_cost_table(records)
    (TABLES / "scarcity_matched_cost.json").write_text(json.dumps(matched, indent=2))
    print("\ncold starts at MATCHED memory budget, unseen applications")
    print(pd.DataFrame(matched).to_string(index=False))


if __name__ == "__main__":
    main()
