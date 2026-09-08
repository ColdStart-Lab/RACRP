"""Phase 1.3 - establish the beat-bar.

Runs every non-learned policy over the test slice and publishes their
cost/benefit curves. These curves are the standard RACRP has to beat: a result
only counts as an improvement if it sits left-and-below them, i.e. fewer cold
starts for the same wasted memory-time (or the same cold starts for less).

Policies
--------
  scale-to-zero            no warming at all (worst latency, zero waste)
  fixed keep-alive K       the platform default, swept over K
  hybrid histogram         Shahrad et al., ATC'20 -- the deployed Azure policy
  oracle                   perfect foresight (unreachable lower bound)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from racrp.config import (
    TEST_SLICE, TRAIN_SLICE, KEEPALIVE_SWEEP, TABLES, PROCESSED, COHORT_SIZE, SEED,
)
from racrp.data.trace import build_app_matrix, select_cohort
from racrp.sim import policies as P
from racrp.baselines.histogram import hybrid_histogram_policy
from racrp.eval.harness import evaluate_policy, save_records, summarise

COHORT_FILE = PROCESSED / "cohort.npy"


def load_cohort(counts, meta, size=COHORT_SIZE, force=False):
    if COHORT_FILE.exists() and not force:
        rows = np.load(COHORT_FILE)
        if len(rows) and (size is None or abs(len(rows) - size) < size):
            return rows
    rows = select_cohort(counts, meta, size=size, seed=SEED)
    np.save(COHORT_FILE, rows)
    print(f"[cohort] selected {len(rows)} apps -> {COHORT_FILE}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-size", type=int, default=COHORT_SIZE)
    ap.add_argument("--force-cohort", action="store_true")
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = load_cohort(counts, meta, args.cohort_size, args.force_cohort)
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[baselines] cohort={len(rows)} apps, "
          f"test slice={TEST_SLICE} ({(TEST_SLICE[1]-TEST_SLICE[0])//1440} days)")
    print(f"[baselines] cohort invocations over test slice: "
          f"{counts[rows, TEST_SLICE[0]:TEST_SLICE[1]].sum():,}")

    records = []

    records.append(evaluate_policy(
        counts, rows, mem, lambda inv, i: P.never_warm(inv), TEST_SLICE,
        name="scale-to-zero"))
    print(f"  scale-to-zero: {records[-1]['cold_starts']:,} cold starts")

    for k in KEEPALIVE_SWEEP:
        r = evaluate_policy(counts, rows, mem,
                            lambda inv, i, k=k: P.fixed_keepalive(inv, k),
                            TEST_SLICE, name=f"fixed-keepalive-{k}")
        r["keepalive"] = k
        records.append(r)
        print(f"  keep-alive {k:3d} min: {r['cold_starts']:8,} cold starts, "
              f"{r['idle_gb_minutes']:12,.0f} idle GB-min")

    def hist_policy(inv, i):
        return hybrid_histogram_policy(inv, warmup_slice=TRAIN_SLICE)

    r = evaluate_policy(counts, rows, mem, hist_policy, TEST_SLICE,
                        name="hybrid-histogram-ATC20")
    records.append(r)
    print(f"  hybrid histogram (ATC'20): {r['cold_starts']:,} cold starts, "
          f"{r['idle_gb_minutes']:,.0f} idle GB-min")

    r = evaluate_policy(counts, rows, mem,
                        lambda inv, i: P.oracle(inv, 0), TEST_SLICE, name="oracle")
    records.append(r)
    print(f"  oracle: {r['cold_starts']:,} cold starts, "
          f"{r['idle_gb_minutes']:,.0f} idle GB-min")

    save_records(records, "baselines_test.json")
    print()
    print(summarise(records).to_string(index=False))


if __name__ == "__main__":
    main()
