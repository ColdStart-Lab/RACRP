"""Choose the estimator's prior strength on the validation split, not the test one.

The derived window integrates over the whole idle-gap distribution, so it is
sensitive to how the sparse tail is smoothed in a way a percentile rule is not.
That makes the prior strength a real hyper-parameter, and it has to be chosen
somewhere.

It matters which somewhere. Measured on 250 applications, a *heavy* uniform prior
(241 pseudo-observations spread over the 241-minute support -- the accidental
behaviour of adding one count per bin) scored **better on the test period** than
the weak prior that replaced it: +74.3% against +69.3%. Picking it on that basis
would be choosing a hyper-parameter by its test-set score, which is the one thing
every other result in this project has been careful not to do.

So the sweep runs on the validation days (9-10) and the winner is carried to the
test days unchanged. Whatever it picks is what gets reported.
"""
from __future__ import annotations

import json
import statistics as st

import numpy as np
import pandas as pd

from racrp.config import TABLES, PROCESSED, VAL_SLICE, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import conditional_policy
from racrp.baselines.hazard import hazard_policy
from racrp.sim.simulator import simulate_app, SimResult

TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)
RATIOS = (0.30, 0.12, 0.05, 0.02, 0.008, 0.003)
# total pseudo-count mass spread over the support
SMOOTH_GRID = (0.0, 1.0, 5.0, 25.0, 60.0, 120.0, 241.0)
N_APPS = 400


def curve(counts, rows, mem, fn, knobs, sl):
    out = []
    for k in knobs:
        t = SimResult()
        for i, r in enumerate(rows):
            inv = counts[r].astype(np.int64)
            w = np.asarray(fn(inv, k))
            t.merge(simulate_app(inv[sl[0]:sl[1]], w[sl[0]:sl[1]], float(mem[i])))
        out.append((t.idle_gb_minutes, t.cold_starts))
    return out


def median_gain(base, arm):
    pts = sorted(arm)
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    vals = [100 * (1 - float(np.interp(g, xs, ys)) / c)
            for g, c in base if xs and xs[0] <= g <= xs[-1] and c]
    return (st.median(vals) if vals else None), len(vals)


def main():
    counts, meta = build_app_matrix()
    rows_all = np.load(PROCESSED / "cohort.npy")
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all), N_APPS, replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[tune] {len(rows)} applications, VALIDATION days 9-10")

    base = curve(counts, rows, mem,
                 lambda inv, t: conditional_policy(inv, eval_slice=VAL_SLICE,
                                                   key="none", tail_margin=t),
                 TAILS, VAL_SLICE)
    print("  ATC'20 baseline " +
          " ".join(f"{c/1000:.1f}k@{g/1000:.0f}k" for g, c in base), flush=True)

    rowsout = []
    for sm in SMOOTH_GRID:
        arm = curve(counts, rows, mem,
                    lambda inv, r, sm=sm: hazard_policy(
                        inv, eval_slice=VAL_SLICE, key="none",
                        price_ratio=r, joint=True, smooth=sm),
                    RATIOS, VAL_SLICE)
        med, n = median_gain(base, arm)
        rowsout.append({"smooth_total": sm, "val_median_gain_pct": med,
                        "n_points": n})
        print(f"  smooth={sm:<6g} val median {med:+.1f}%  (n={n})", flush=True)

    ok = [r for r in rowsout if r["val_median_gain_pct"] is not None]
    best = max(ok, key=lambda r: r["val_median_gain_pct"])
    out = {"grid": rowsout, "chosen_smooth_total": best["smooth_total"],
           "chosen_on": "validation days 9-10", "n_apps": int(len(rows))}
    (TABLES / "smoothing_tuning.json").write_text(json.dumps(out, indent=2))

    pd.set_option("display.width", 160)
    print("\n" + pd.DataFrame(rowsout).to_string(index=False))
    print(f"\nchosen on validation: smooth (total pseudo-mass) = "
          f"{best['smooth_total']}")


if __name__ == "__main__":
    main()
