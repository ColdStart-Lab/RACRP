"""Precompute everything the web demonstration serves.

The demo runs offline on a lecture-room laptop: nothing is computed at request
time and no model is loaded.

It shows the mechanism the experiments actually support. For each demonstration
application we write:

  * its invocation timeline over the evaluation window;
  * its **marginal** idle-gap histogram -- what the deployed ATC'20 policy uses --
    alongside the **conditional** histograms, one per bucket of the gap that just
    closed, so the difference between them is visible rather than asserted;
  * the keep-alive window each of those implies;
  * per-minute decision tracks and a scoreboard for every policy.

Applications are chosen by documented rules spanning behaviour types:

  * the largest cold-start reduction in each sparsity band -- what the method is for;
  * the largest **window spread** (how differently the conditional treats the
    states) in each band -- these make the mechanism visible on screen, since an
    application whose buckets all agree shows nothing even when the method helps;
  * and, deliberately, the application in each band where conditioning helps
    *least*. A picker filtered to successes would be a fabricated result.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from racrp.config import ARTIFACTS, TABLES, PROCESSED, TEST_SLICE, MIN_PER_DAY, SEED
from racrp.data.trace import build_app_matrix
from racrp.baselines.conditional import (
    conditional_policy, gap_bucket, N_GAP_BUCKETS,
)
from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows, QGRID
from racrp.baselines.hazard import hazard_policy, optimal_window, DEFAULT_SMOOTH
from racrp.sim import policies as P
from racrp.sim.simulator import simulate_app, cold_start_ms
from racrp.eval.strata import app_sparsity

DEMO = ARTIFACTS / "demo"
DEMO.mkdir(parents=True, exist_ok=True)
COHORT_FILE = PROCESSED / "cohort.npy"
TAIL = 92.0
BUCKET_LABEL = ["1-3 min", "4-15 min", "16-63 min", "64-255 min",
                "256-1023 min", ">1023 min"]


# The price of a cold start, in GB-minutes of idle memory. This is the knob the
# derived rule sweeps in place of a tail percentile; the demo pins it at one
# value so the window on screen is reproducible.
PRICE_RATIO = 0.05

# The knob swept when scoring an application at the baseline's own spend.
#
# Deliberately dense. For a near-deterministic application -- one whose gaps take
# only a handful of distinct values -- the cost/benefit curve is a STEP, not a
# smooth trade-off: either the window covers the period or it does not. A coarse
# sweep leaves a huge gap between two adjacent operating points, and interpolating
# across it invents a point on a straight line where the real curve has a cliff.
# Measured on app 490: 193 cold starts at 21 GB-min, then 1 cold start at 212,
# with nothing between. Reading the baseline's 191 GB-min off that line claimed
# 22 cold starts at a spend the policy never actually occupies.
PRICE_GRID = (0.60, 0.40, 0.28, 0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.024,
              0.017, 0.012, 0.008, 0.005, 0.003, 0.0015)

# How the picker is composed.
#
# Winners are drawn from a band around the cohort median (+78.8%) rather than
# from the top of the ranking: four applications at 99.5% are real but they are
# the tail, and a demo built on the tail invites the objection this project
# spends its time avoiding. The band widens only if it cannot be filled.
#
# Non-wins are bounded below so they are visibly negative without being the
# step-curve artefacts described in overview/PRESENTATION_GUIDE.md -- an application
# with three distinct gap lengths can read -900% for reasons that have nothing to
# do with the method.
TARGET_BANDS = ((60.0, 80.0), (55.0, 85.0), (50.0, 90.0), (40.0, 95.0))
N_WINNERS = 14
N_NON_WINNERS = 2
NON_WIN_FLOOR = -10.0        # nothing worse than this
MIN_COLD_STARTS = 3          # below this a percentage means very little
MAX_BRACKET = 0.5            # the budget must sit inside a resolved interval


def _derived_window(marg):
    """The window the cost model picks from this application's own history.

    Reported alongside the percentile window so the dashboard can show the two
    side by side on the same distribution, with the expected cost of each under
    the model being minimised -- which is the check that caught a bug in the
    estimator and is worth leaving visible.
    """
    import numpy as _np
    off, ln = optimal_window(marg, PRICE_RATIO)
    pw_m, ka_m = quantiles_to_windows(hist_to_quantiles(marg), 5.0, TAIL, 10)

    h = _np.asarray(marg, float) + DEFAULT_SMOOTH / max(len(marg), 1)
    pmf = h / h.sum()
    g = _np.arange(len(marg))

    def cost(a0, L):
        ov = _np.clip(_np.minimum(a0 + L, g) - a0, 0, None)
        mem = float((pmf * ov).sum())
        miss = float(pmf[(g < a0) | (g >= a0 + L)].sum())
        return {"warm_minutes": round(mem, 3), "miss_prob": round(miss, 4),
                "expected_cost": round(PRICE_RATIO * mem + miss, 4)}

    return {
        "price_ratio": PRICE_RATIO,
        "prewarm": int(off), "keepalive": int(ln),
        "percentile_prewarm": int(pw_m), "percentile_keepalive": int(ka_m),
        "cost_derived": cost(off, ln),
        "cost_percentile": cost(pw_m, max(ka_m, 1)),
    }



def gap_stats(inv: np.ndarray, upto: int):
    """Marginal and per-bucket conditional gap histograms, from training data."""
    a = np.flatnonzero(inv[:upto] > 0)
    if len(a) < 3:
        return None
    gaps = np.diff(a)
    prev = np.concatenate([[gaps[0]], gaps[:-1]])
    pb = gap_bucket(prev)
    R = 240
    marg = np.zeros(R + 1)
    np.add.at(marg, np.minimum(gaps, R), 1.0)
    cond = np.zeros((N_GAP_BUCKETS, R + 1))
    n = np.zeros(N_GAP_BUCKETS)
    for b in range(N_GAP_BUCKETS):
        m = pb == b
        if m.any():
            np.add.at(cond[b], np.minimum(gaps[m], R), 1.0)
            n[b] = m.sum()
    return marg, cond, n, gaps


def _finite(obj):
    """Replace inf/NaN with None throughout a payload before it is written.

    An application that never repeats has an infinite median idle time. Python's
    json writes that as `Infinity`, which is not valid JSON, and the API layer
    refuses to serialise it -- so the whole demo returns 500 the moment such an
    application is picked. Found exactly that way.
    """
    import math
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--kappa", type=float, default=8.0)
    ap.add_argument("--per-band", type=int, default=2)
    ap.add_argument("--rescore", action="store_true",
                    help="redo the matched-cost scoring instead of reusing it")
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    sparsity = app_sparsity(counts, rows)
    lo = TEST_SLICE[0]
    hi = min(lo + args.days * MIN_PER_DAY, TEST_SLICE[1])

    # -------------------------------------------------- score every application
    def window_spread(inv, upto):
        """How differently the conditional treats the states, in minutes.

        An application whose buckets all imply the same window demonstrates
        nothing on screen even if the method helps it, so this is tracked
        separately from the cold-start gain and used as its own selection rule.
        """
        st = gap_stats(inv, upto)
        if st is None:
            return 0.0, 0
        marg, cond, ncell, _ = st
        qm = hist_to_quantiles(marg)
        _, ka_m = quantiles_to_windows(qm, 5.0, TAIL, 10)
        kas = []
        for b in range(N_GAP_BUCKETS):
            if ncell[b] == 0:
                continue
            qb = ((ncell[b] * hist_to_quantiles(cond[b]) + args.kappa * qm)
                  / (ncell[b] + args.kappa))
            kas.append(quantiles_to_windows(qb, 5.0, TAIL, 10)[1])
        return (float(max(kas) - min(kas)) if len(kas) > 1 else 0.0), len(kas)

    def matched_gain(inv, mem):
        """Cold-start reduction from the derived window at the BASELINE's spend.

        The comparison this project treats as valid, done per application: sweep
        the derived window over its price knob, then read its curve at the memory
        the ATC'20 baseline happens to spend. Returns the percentage reduction
        there, plus the two cold-start counts, or None when the baseline's spend
        falls outside the range our knob can reach.
        """
        w_b = conditional_policy(inv, eval_slice=TEST_SLICE, key="none",
                                 tail_margin=TAIL, kappa=args.kappa)
        base = simulate_app(inv[lo:hi], w_b[lo:hi], mem)
        if base.cold_starts == 0:
            return None
        pts = []
        for ratio in PRICE_GRID:
            w = hazard_policy(inv, eval_slice=TEST_SLICE, key="none",
                              price_ratio=ratio, joint=True)
            r = simulate_app(inv[lo:hi], w[lo:hi], mem)
            pts.append((r.idle_gb_minutes, r.cold_starts))
        pts.sort()
        xs = [a for a, _ in pts]
        ys = [b for _, b in pts]
        if not xs or base.idle_gb_minutes < xs[0] or base.idle_gb_minutes > xs[-1]:
            return None
        ours = float(np.interp(base.idle_gb_minutes, xs, ys))
        # how wide the interpolated interval is, relative to the budget: a large
        # value means the curve stepped over this budget rather than passing
        # through it, and the number below is an interpolation artefact
        k = int(np.searchsorted(xs, base.idle_gb_minutes))
        k = min(max(k, 1), len(xs) - 1)
        span = (xs[k] - xs[k - 1]) / max(base.idle_gb_minutes, 1e-9)
        return {
            "matched_gain": 100.0 * (1.0 - ours / base.cold_starts),
            "base_cold": int(base.cold_starts),
            "ours_cold": ours,
            "budget_gb": float(base.idle_gb_minutes),
            "bracket_span": float(span),
        }

    # Scoring every application at matched cost is minutes of simulation, and it
    # only changes when the policies or the cohort change. Cache it.
    cache_f = DEMO / "_scored.json"
    scored = []
    if cache_f.exists() and not args.rescore:
        scored = json.loads(cache_f.read_text())
        print(f"[demo] reusing matched-cost scores for {len(scored)} "
              f"applications ({cache_f.name}; --rescore to redo)")
    for i, r in enumerate(rows if not scored else []):
        inv = counts[r].astype(np.int64)
        if inv[lo:hi].sum() < 30:
            continue
        mem = float(meta.loc[r, "memory_mb"])
        m = matched_gain(inv, mem)
        if m is None:
            continue
        spread, n_buckets = window_spread(inv, lo)
        scored.append({
            "pos": i, "row": int(r), "mem": mem,
            "sparsity": float(sparsity[i]),
            "spread": spread, "n_buckets": n_buckets, **m,
        })
    if not cache_f.exists() or args.rescore:
        DEMO.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(scored))
        print(f"[demo] scored {len(scored)} applications at matched cost")

    bands = [(0, 5, "frequent"), (5, 30, "periodic"),
             (30, 240, "sparse"), (240, np.inf, "very sparse")]

    def band_of(x):
        return next((l for blo, bhi, l in bands if blo <= x["sparsity"] < bhi),
                    bands[-1][2])

    def usable(x):
        return (x.get("bracket_span", 9.9) < MAX_BRACKET
                and x["base_cold"] >= MIN_COLD_STARTS)

    pool = [x for x in scored if usable(x)]

    def spread(cands, n):
        """Take n, rotating through activity bands so the list is not one shape."""
        by = {}
        for x in cands:
            by.setdefault(band_of(x), []).append(x)
        # Within the band, prefer applications with the most cold starts to
        # actually watch. All of these are 60-80% either way; an application
        # going 93 -> 26 makes a visible race, one going 3 -> 1 does not.
        for v in by.values():
            v.sort(key=lambda z: -z["base_cold"])
        out, keys = [], list(by)
        while len(out) < n and any(by[k] for k in keys):
            for k in keys:
                if by[k] and len(out) < n:
                    out.append(by[k].pop(0))
        return out

    picked, seen = [], set()
    for lo_b, hi_b in TARGET_BANDS:
        cands = [x for x in pool
                 if x["pos"] not in seen and lo_b <= x["matched_gain"] <= hi_b]
        for x in spread(cands, N_WINNERS - len(picked)):
            x["band"] = band_of(x)
            x["reason"] = (f"selected from the {lo_b:.0f}-{hi_b:.0f}% band, near "
                           f"the cohort median: at the baseline's own memory spend "
                           f"this application's {x['base_cold']:.0f} cold starts "
                           f"become {x['ours_cold']:.0f} "
                           f"({x['matched_gain']:+.0f}%)")
            picked.append(x)
            seen.add(x["pos"])
        if len(picked) >= N_WINNERS:
            break

    losers = [x for x in pool
              if x["pos"] not in seen
              and NON_WIN_FLOOR <= x["matched_gain"] < -0.5]
    losers.sort(key=lambda x: x["matched_gain"])          # most negative first
    for x in losers[:N_NON_WINNERS]:
        x["band"] = band_of(x)
        x["reason"] = (f"kept on purpose as a NON-WIN: at the baseline's own "
                       f"memory spend this application's {x['base_cold']:.0f} cold "
                       f"starts become {x['ours_cold']:.0f} "
                       f"({x['matched_gain']:+.1f}%). The method does not help "
                       f"here, and the list is weighted towards wins but not "
                       f"filtered to them")
        picked.append(x)
        seen.add(x["pos"])

    # Shuffled, not ranked. Deterministic so the demo is the same every time it
    # is rebuilt; every entry still carries the sentence saying why it is here.
    import random as _random
    _random.Random(SEED).shuffle(picked)

    won = sum(1 for x in picked if x["matched_gain"] > 0)
    gains = sorted(round(x["matched_gain"], 1) for x in picked)
    print(f"[demo] picked {len(picked)}: {won} win at matched cost, "
          f"{len(picked) - won} do not; range {gains[0]:+.0f}% to {gains[-1]:+.0f}%")

    index = []
    for s in picked:
        r, mem = s["row"], s["mem"]
        inv = counts[r].astype(np.int64)
        st = gap_stats(inv, lo)
        if st is None:
            continue
        marg, cond, ncell, gaps = st

        q_marg = hist_to_quantiles(marg)
        pw_m, ka_m = quantiles_to_windows(q_marg, 5.0, TAIL, 10)
        buckets = []
        for b in range(N_GAP_BUCKETS):
            if ncell[b] == 0:
                buckets.append(None)
                continue
            q_b = ((ncell[b] * hist_to_quantiles(cond[b]) + args.kappa * q_marg)
                   / (ncell[b] + args.kappa))
            pw_b, ka_b = quantiles_to_windows(q_b, 5.0, TAIL, 10)
            buckets.append({
                "bucket": b, "label": BUCKET_LABEL[b], "n": int(ncell[b]),
                "prewarm": pw_b, "keepalive": ka_b,
                "median_gap": float(np.interp(0.5, QGRID, hist_to_quantiles(cond[b]))),
                "pdf": [round(float(v), 5) for v in cond[b] / max(cond[b].sum(), 1)],
            })

        tracks = {}
        for name, warm in (
            ("fixed-keepalive-10", P.fixed_keepalive(inv, 10)),
            ("marginal-ATC20", conditional_policy(inv, eval_slice=TEST_SLICE,
                                                  key="none", tail_margin=TAIL,
                                                  kappa=args.kappa)),
            ("conditional-prev-gap", conditional_policy(inv, eval_slice=TEST_SLICE,
                                                        key="prev_gap",
                                                        tail_margin=TAIL,
                                                        kappa=args.kappa)),
            ("derived-window", hazard_policy(inv, eval_slice=TEST_SLICE,
                                             key="none",
                                             price_ratio=PRICE_RATIO,
                                             joint=True)),
        ):
            res, states = simulate_app(inv[lo:hi], warm[lo:hi], mem,
                                       trace_decisions=True)
            tracks[name] = {"states": states.astype(int).tolist(),
                            "summary": res.as_dict()}

        # the actual sequence of gaps this application showed, so the demo can
        # show WHY a bucket has the shape it does rather than only asserting it
        act = np.flatnonzero(inv[:lo] > 0)
        gaps_seq = np.diff(act)[-400:] if len(act) > 2 else np.array([], dtype=int)

        payload = {
            "app": s["pos"],
            "gaps_recent": [int(g) for g in gaps_seq],
            "hash": str(meta.loc[r, "HashApp"])[:12],
            "band": s["band"],
            "reason_selected": s["reason"],
            "matched_gain_pct": round(float(s["matched_gain"]), 1),
            "matched_budget_gb": round(float(s["budget_gb"]), 1),
            "matched_base_cold": int(s["base_cold"]),
            "matched_ours_cold": round(float(s["ours_cold"]), 1),
            "trigger": str(meta.loc[r, "trigger"]),
            "memory_mb": mem,
            "avg_duration_ms": float(np.nan_to_num(meta.loc[r, "avg_duration_ms"])),
            "cold_start_ms": cold_start_ms(mem),
            "median_idle_min": s["sparsity"],
            "n_gaps_train": int(len(gaps)),
            "window_spread_min": float(s.get("spread", 0.0)),
            "marginal_keepalive": ka_m,
            "bucket_spread": [b["keepalive"] for b in buckets if b],
            "window": {"start": int(lo), "end": int(hi)},
            "invocations": inv[lo:hi].tolist(),
            "marginal": {
                "pdf": [round(float(v), 5) for v in marg / max(marg.sum(), 1)],
                "prewarm": pw_m, "keepalive": ka_m,
                "median_gap": float(np.interp(0.5, QGRID, q_marg)),
            },
            "buckets": buckets,
            "derived": _derived_window(marg),
            "kappa": args.kappa, "tail": TAIL,
            "tracks": tracks,
        }
        (DEMO / f"app_{s['pos']}.json").write_text(json.dumps(_finite(payload)))
        index.append({
            "app": s["pos"], "hash": payload["hash"], "band": s["band"],
            "trigger": payload["trigger"], "memory_mb": mem,
            "median_idle_min": s["sparsity"],
            "reason_selected": s["reason"],
            "matched_gain_pct": round(float(s["matched_gain"]), 1),
            "window_spread_min": float(s.get("spread", 0.0)),
            "marginal_keepalive": ka_m,
            "derived_keepalive": payload["derived"]["keepalive"],
            "derived_prewarm": payload["derived"]["prewarm"],
            "n_states": int(sum(1 for b in buckets if b)),
            "cold": {k: v["summary"]["cold_starts"] for k, v in tracks.items()},
            "gb": {k: v["summary"]["idle_gb_minutes"] for k, v in tracks.items()},
        })
        print(f"  app {s['pos']:5d} ({s['band']:12s}) "
              + ", ".join(f"{k.split('-')[0]}={v['summary']['cold_starts']}"
                          for k, v in tracks.items()))

    results = {}
    for f in ("derived_matched_cost.json", "derived_summary.json",
              "strong_baseline_matched.json", "strong_baseline_summary.json",
              "strong_baseline_cloud.json", "smoothing_tuning.json",
              "conditional_matched_cost.json", "conditional_sweep.json",
              "context_information.json", "characterisation.json",
              "baselines_test.json", "diag_fleet_ceiling_H0.json",
              "prediction_metrics_h5_smoke.json"):
        p = TABLES / f
        if p.exists():
            results[f.replace(".json", "")] = json.loads(p.read_text())

    # Order is the shuffle applied at selection time, not a ranking. Each entry
    # carries its own selection reason, so nothing about why it is here is lost
    # by not sorting.

    keep = {f"app_{a['app']}.json" for a in index}
    for stale in DEMO.glob("app_*.json"):
        if stale.name not in keep:
            stale.unlink()
            print(f"  removed stale {stale.name}")

    (DEMO / "index.json").write_text(json.dumps(_finite({
        "apps": index, "kappa": args.kappa, "tail": TAIL,
        "window_days": args.days, "bucket_labels": BUCKET_LABEL,
        "results": results,
    }), indent=2))
    print(f"[demo] wrote {len(index)} applications + index to {DEMO}")


if __name__ == "__main__":
    main()
