"""How much of an application's next idle gap is predictable from anything at all?

This is the measurement that explains every negative result in the project and
that selected the method which finally worked, so it is worth stating plainly.

For each application we take the sequence of idle gaps observed during training,
discretise both the gap and each candidate context, and compute the mutual
information between them. Mutual information is the right instrument here because
it is model-free: it bounds what *any* predictor could extract from that context,
so a small value is evidence about the problem rather than about our modelling.

Contexts tested, all computable by a scheduler at the moment it must decide:

  time of day            8 three-hour blocks
  day of week            7 values
  recent 5-min activity  how busy the last five minutes were
  recent 60-min activity the same over an hour
  burst level            activity over the preceding day
  previous gap           the length of the gap that just closed

Applications whose gap distribution is nearly degenerate (entropy below 0.2 bits
- perfectly periodic functions) are excluded, since there is nothing to predict
and including them would flatter every context equally.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from racrp.config import TABLES, PROCESSED, TEST_SLICE, SEED, MIN_PER_DAY
from racrp.data.trace import build_app_matrix

COHORT_FILE = PROCESSED / "cohort.npy"
N_GAP_BINS = 10          # log2 buckets of the gap
MIN_GAPS = 60
MIN_ENTROPY = 0.2


def mutual_information(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> float:
    J = np.zeros((nx, ny))
    np.add.at(J, (x, y), 1.0)
    if J.sum() < 30:
        return np.nan
    J /= J.sum()
    px, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = J * np.log2(J / (px * py))
    return float(np.nansum(t))


def entropy(y: np.ndarray, ny: int) -> float:
    p = np.bincount(y, minlength=ny).astype(float)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=600)
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(rows), min(args.apps, len(rows)), replace=False)

    keys = ("tod", "dow", "recent5", "recent60", "burst", "prevgap")
    acc = {k: [] for k in keys}
    H = []

    for r in rows[sel]:
        inv = counts[r].astype(np.int64)[:TEST_SLICE[0]]      # training period only
        a = np.flatnonzero(inv > 0)
        if len(a) < MIN_GAPS:
            continue
        starts, gaps = a[:-1], np.diff(a)
        g = np.clip(np.log2(gaps).astype(int), 0, N_GAP_BINS - 1)
        h = entropy(g, N_GAP_BINS)
        if h < MIN_ENTROPY:            # nearly periodic: nothing to predict
            continue
        H.append(h)

        acc["tod"].append(mutual_information(
            ((starts % MIN_PER_DAY) // 180).astype(int), g, 8, N_GAP_BINS))
        acc["dow"].append(mutual_information(
            ((starts // MIN_PER_DAY) % 7).astype(int), g, 7, N_GAP_BINS))

        active = (inv > 0).astype(float)
        cs = np.concatenate([[0.0], np.cumsum(active)])
        for w, key in ((5, "recent5"), (60, "recent60")):
            lo = np.maximum(starts - w + 1, 0)
            cnt = cs[starts + 1] - cs[lo]
            acc[key].append(mutual_information(
                np.clip((cnt / w * 8).astype(int), 0, 7), g, 8, N_GAP_BINS))

        burst = np.clip(
            (cs[starts + 1] - cs[np.maximum(starts - MIN_PER_DAY + 1, 0)]).astype(int) // 180,
            0, 7)
        acc["burst"].append(mutual_information(burst, g, 8, N_GAP_BINS))

        prev = np.concatenate([[g[0]], g[:-1]])
        acc["prevgap"].append(mutual_information(prev, g, N_GAP_BINS, N_GAP_BINS))

    Hm = float(np.median(H))
    out = {"n_apps": len(H), "gap_entropy_median_bits": Hm, "contexts": {}}
    pretty = {"tod": "time of day", "dow": "day of week",
              "recent5": "recent 5-min activity", "recent60": "recent 60-min activity",
              "burst": "burst level (24 h)", "prevgap": "previous gap"}

    print(f"applications analysed: {len(H)}")
    print(f"median entropy of the idle gap itself: {Hm:.3f} bits\n")
    print(f"{'context':>24} {'median MI':>10} {'p90':>8} {'% of gap entropy':>18}")
    print("-" * 64)
    order = sorted(keys, key=lambda k: np.nanmedian(acc[k]))
    for k in order:
        v = np.array(acc[k], float)
        v = v[np.isfinite(v)]
        med = float(np.median(v))
        out["contexts"][k] = {
            "label": pretty[k], "median_mi_bits": med,
            "mean_mi_bits": float(v.mean()),
            "p90_mi_bits": float(np.percentile(v, 90)),
            "share_of_gap_entropy": med / Hm,
        }
        print(f"{pretty[k]:>24} {med:10.4f} {np.percentile(v, 90):8.4f} "
              f"{100 * med / Hm:17.1f}%")

    best = max(out["contexts"].values(), key=lambda d: d["share_of_gap_entropy"])
    out["irreducible_share"] = 1 - best["share_of_gap_entropy"]
    print(f"\nBest single context: {best['label']} "
          f"({100*best['share_of_gap_entropy']:.1f}% of gap entropy).")
    print(f"So at least {100*out['irreducible_share']:.0f}% of the uncertainty in "
          f"an application's next idle gap is irreducible from any of these.")
    print("Serverless idle gaps on this trace are close to memoryless, which is "
          "why rich retrieval keys lose: they pay a large variance cost for "
          "signal that is mostly not there.")

    p = TABLES / "context_information.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
