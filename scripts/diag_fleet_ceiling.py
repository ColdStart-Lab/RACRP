"""Diagnostic: is borrowing strength across applications possible at all here?

The measured result is that the fleet prior loses to an application's own
histogram even when that application has no history. Two explanations fit:

  (H1) **Retrieval is the bottleneck.** At a history budget of zero the profile is
       little more than trigger type, memory and declared duration, which may
       simply not determine how an application idles. A better key would fix it.
  (H2) **There is nothing to borrow.** Applications' idle-gap distributions are so
       individual that no other application's distribution is a useful stand-in,
       however well chosen.

This separates them with an oracle retriever that picks the ``k`` seen
applications whose *actual* idle-gap distribution is closest to the target's
actual distribution, measured in quantile (Wasserstein) space. It uses the
target's future and is therefore never a method -- only a bound on what any
retrieval rule could achieve with this fleet.

Reference points, all scored through the same simulator on the same held-out
applications and swept over the same cost knob:

  own-histogram      the ATC'20 policy on whatever history the target has
  fixed keep-alive   the platform default, no history at all
  fleet (profile)    what we actually built
  fleet (ORACLE)     the ceiling: best possible neighbours

If the oracle also loses, H2 holds and the honest conclusion is a negative
result, reported as such.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.data.splits import split_apps, observable_slice, app_profile, HISTORY_LABEL
from racrp.baselines.fleet import (
    FleetBank, fleet_policy, gap_histogram, hist_to_quantiles,
    quantiles_to_windows, QGRID,
)
from racrp.sim import policies as P
from racrp.sim.simulator import simulate_app, SimResult

COHORT_FILE = PROCESSED / "cohort.npy"
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)


class OracleBank(FleetBank):
    """Same fleet, but neighbours are chosen by true distributional closeness."""

    def set_oracle_target(self, q_true: np.ndarray) -> None:
        self._q_true = q_true

    def retrieve(self, profile, k=16):
        # distance in quantile space between the target's TRUE gap distribution
        # and each seen application's; log scale so a factor-of-two error at 5
        # minutes counts like a factor-of-two error at 500
        d = np.abs(np.log1p(self.quant) - np.log1p(self._q_true)[None, :]).mean(axis=1)
        d = np.where(self.has_gaps, d, np.inf)
        k = min(k, int(np.isfinite(d).sum()) or 1)
        idx = np.argsort(d)[:k]
        return idx, 1.0 / (1.0 + d[idx])


def score(counts, urows, umem, pol, eval_slice, early_slice):
    full, early = SimResult(), SimResult()
    for i, r in enumerate(urows):
        inv = counts[r].astype(np.int64)
        warm = np.asarray(pol(inv, i))
        m = float(umem[i])
        full.merge(simulate_app(inv[eval_slice[0]:eval_slice[1]],
                                warm[eval_slice[0]:eval_slice[1]], m))
        early.merge(simulate_app(inv[early_slice[0]:early_slice[1]],
                                 warm[early_slice[0]:early_slice[1]], m))
    return full, early


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--early-hours", type=int, default=6)
    ap.add_argument("--max-apps", type=int, default=600)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    seen, unseen = split_apps(meta, rows, seed=SEED)
    if args.max_apps and len(unseen) > args.max_apps:
        unseen = unseen[:args.max_apps]
    urows = rows[unseen]
    umem = meta.loc[urows, "memory_mb"].to_numpy()
    hw = observable_slice(args.history)
    early = (TEST_SLICE[0], TEST_SLICE[0] + args.early_hours * 60)

    bank = FleetBank(counts, rows, meta, seen, hw, dist_window=(0, TEST_SLICE[0]))
    print(f"[diag] prior scale calibrated on seen apps: {bank.calibrate():.3f}")
    obank = OracleBank(counts, rows, meta, seen, hw, dist_window=(0, TEST_SLICE[0]))
    obank.calibrate()

    profiles = np.stack([app_profile(counts, int(rows[p]), hw, meta.loc[int(rows[p])])
                         for p in unseen])
    # each target's TRUE gap distribution over the evaluation period
    q_true = []
    for r in urows:
        h, _ = gap_histogram(counts[r].astype(np.int64), TEST_SLICE)
        q_true.append(hist_to_quantiles(h))
    q_true = np.stack(q_true)

    records = []

    def add(method, tail, full, ear):
        records.append({
            "method": method, "tail_margin": tail,
            "cold_starts": full.cold_starts,
            "idle_gb_minutes": full.idle_gb_minutes,
            "early_cold_starts": ear.cold_starts,
            "early_idle_gb_minutes": ear.idle_gb_minutes,
        })
        print(f"  {method:24s} tail={tail:5.1f}  cold={full.cold_starts:7,} "
              f"gbmin={full.idle_gb_minutes:10,.0f} early={ear.cold_starts:6,}")

    print(f"[diag] {len(unseen)} held-out apps, history budget "
          f"{HISTORY_LABEL.get(args.history, args.history)}")

    for K in (5, 10, 20, 30, 60):
        f, e = score(counts, urows, umem,
                     lambda inv, i, K=K: P.fixed_keepalive(inv, K), TEST_SLICE, early)
        add("fixed-keepalive", float(K), f, e)

    for tail in TAILS:
        f, e = score(counts, urows, umem,
                     lambda inv, i, t=tail: fleet_policy(
                         inv, profile=profiles[i], bank=bank, history_window=hw,
                         eval_slice=TEST_SLICE, k=args.k, kappa=args.kappa,
                         tail_margin=t, use_own=True, use_prior=False),
                     TEST_SLICE, early)
        add("own-histogram-ATC20", tail, f, e)

    for tail in TAILS:
        f, e = score(counts, urows, umem,
                     lambda inv, i, t=tail: fleet_policy(
                         inv, profile=profiles[i], bank=bank, history_window=hw,
                         eval_slice=TEST_SLICE, k=args.k, kappa=args.kappa,
                         tail_margin=t, use_own=True, use_prior=True),
                     TEST_SLICE, early)
        add("fleet-profile-retrieval", tail, f, e)

    def oracle_pol(inv, i, t):
        obank.set_oracle_target(q_true[i])
        return fleet_policy(inv, profile=profiles[i], bank=obank, history_window=hw,
                            eval_slice=TEST_SLICE, k=args.k, kappa=args.kappa,
                            tail_margin=t, use_own=True, use_prior=True)

    for tail in TAILS:
        f, e = score(counts, urows, umem,
                     lambda inv, i, t=tail: oracle_pol(inv, i, t), TEST_SLICE, early)
        add("fleet-ORACLE-retrieval", tail, f, e)

    out = TABLES / f"diag_fleet_ceiling_H{args.history}.json"
    out.write_text(json.dumps(records, indent=2))

    df = pd.DataFrame(records)
    print("\n--- cost/benefit curves (first 6 h after deployment) ---")
    for m, g in df.groupby("method"):
        g = g.sort_values("early_idle_gb_minutes")
        pts = ", ".join(f"({r.early_idle_gb_minutes:,.0f} GB-min -> {r.early_cold_starts:,})"
                        for r in g.itertuples())
        print(f"{m:24s} {pts}")

    print("\nREAD-OUT")
    own = df[df.method == "own-histogram-ATC20"]
    orc = df[df.method == "fleet-ORACLE-retrieval"]
    prof = df[df.method == "fleet-profile-retrieval"]

    def at_budget(g, budget):
        g = g.sort_values("early_idle_gb_minutes")
        xs, ys = g.early_idle_gb_minutes.to_numpy(), g.early_cold_starts.to_numpy()
        if budget < xs.min() or budget > xs.max():
            return None
        return float(np.interp(budget, xs, ys))

    budget = float(np.median(own.early_idle_gb_minutes))
    b_own, b_orc, b_prof = (at_budget(own, budget), at_budget(orc, budget),
                            at_budget(prof, budget))
    print(f"  at {budget:,.0f} idle GB-min: own={b_own}, profile={b_prof}, oracle={b_orc}")
    if b_orc is not None and b_own is not None and b_orc < b_own * 0.9:
        print("  Oracle beats the own-histogram -> there IS transferable structure; "
              "retrieval is the bottleneck (H1).")
    else:
        print("  Even oracle retrieval does not beat the own-histogram -> "
              "another application's gap distribution is not a useful stand-in "
              "for this one (H2). Report the negative result.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
