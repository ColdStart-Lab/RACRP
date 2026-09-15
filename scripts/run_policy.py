"""Phase 3.5 - turn predicted risk into scheduling decisions, and score it.

The prediction stage produced a calibrated ``P(cold start within dt)`` for every
(app, idle minute) in the test split. This stage converts those probabilities
into keep-warm decisions and runs them through the same simulator that scored the
heuristic baselines, so the comparison is like for like.

Decision rule
-------------
Keeping an app warm for one more minute costs ``mem_gb`` GB-minutes. Dropping it
risks a cold start worth ``cold_start_ms`` of user-visible latency. So keep warm
iff

    risk_t  >  lambda * mem_gb / cold_start_penalty

Sweeping lambda traces the whole cost/benefit curve, which is the honest way to
compare against a keep-alive window that has its own tunable knob. A single
operating point proves nothing; the curve is the claim.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from racrp.config import (
    TEST_SLICE, TRAIN_SLICE, LAMBDA_SWEEP, KEEPALIVE_SWEEP, TABLES, PROCESSED,
    PRIMARY_HORIZON, COLD_START_BASE_MS, COLD_START_PER_GB_MS,
)
from racrp.data.trace import build_app_matrix
from racrp.sim import policies as P
from racrp.sim.simulator import cold_start_ms
from racrp.eval.harness import evaluate_policy, save_records, summarise, pareto_front

COHORT_FILE = PROCESSED / "cohort.npy"


def risk_matrix(pred_file, model_key: str, n_apps: int, n_minutes: int) -> np.ndarray:
    """Scatter the per-situation predictions back onto an (apps x minutes) grid.

    Minutes with no prediction (the app was active, so the instance is warm
    anyway, or the minute is outside the evaluated slice) are left at zero; the
    policy's unconditional one-minute grace window covers exactly those.
    """
    z = np.load(pred_file)
    key = model_key.replace("-", "_")
    if key not in z:
        raise KeyError(f"{model_key} not in {pred_file}; have {list(z.keys())}")
    p = z[key]
    grid = np.zeros((n_apps, n_minutes), dtype=np.float32)
    grid[z["app_idx"], z["t_idx"]] = p
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-keepalive", type=int, default=1,
                    help="unconditional grace minutes after activity")
    args = ap.parse_args()

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    tag = f"_{args.tag}" if args.tag else ""
    pred_file = TABLES / f"test_predictions_h{args.horizon}{tag}.npz"

    z = np.load(pred_file)
    model_keys = [k for k in z.files
                  if k not in ("y", "app_idx", "t_idx", "rows")]
    print(f"[policy] models in {pred_file.name}: {model_keys}")

    records = []
    for key in model_keys:
        grid = risk_matrix(pred_file, key, len(rows), counts.shape[1])
        # per-app cost ratio: bigger apps are more expensive to hold but also
        # more painful to cold start, so the threshold is app-specific
        gb = mem / 1024.0
        penalty = np.array([cold_start_ms(m) for m in mem])
        base = gb / (penalty / 1000.0)          # GB-min per second of latency
        for lam in LAMBDA_SWEEP:
            thr = np.clip(lam * base, 1e-6, 0.999)

            def pol(inv, i, grid=grid, thr=thr):
                return P.risk_threshold(inv, grid[i], float(thr[i]),
                                        horizon=args.horizon,
                                        min_keepalive=args.min_keepalive)

            r = evaluate_policy(counts, rows, mem, pol, TEST_SLICE,
                                name=f"{key}-lam{lam}")
            r["model"] = key
            r["lambda"] = lam
            records.append(r)
            print(f"  {key:34s} lam={lam:5.2f}: {r['cold_starts']:8,} cold starts, "
                  f"{r['idle_gb_minutes']:12,.0f} idle GB-min")

    save_records(records, f"policy_sweep_h{args.horizon}{tag}.json")
    front = pareto_front(records)
    print("\nPareto front (risk-aware policies):")
    print(summarise(front).to_string(index=False))


if __name__ == "__main__":
    main()
