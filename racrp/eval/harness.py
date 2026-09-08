"""Policy evaluation harness.

Runs a ``keep_warm`` policy across the whole app cohort over a time slice and
aggregates the simulator's accounting into a single record. Every policy in the
project -- heuristic, learned, or oracle -- is scored through this one function,
so the comparison is apples to apples by construction.

Policies are given the *full* timeline to compute their decisions (so a policy
may use the training days as history) but are only **scored** on the evaluation
slice. This mirrors a production deployment evaluated on a later week.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd

from racrp.config import TABLES
from racrp.sim.simulator import simulate_app, SimResult

# a policy maps (invocations_full, app_row_index) -> keep_warm array (full length)
PolicyFn = Callable[[np.ndarray, int], np.ndarray]


def evaluate_policy(
    counts: np.ndarray,
    rows: np.ndarray,
    memory_mb: np.ndarray,
    policy: PolicyFn,
    eval_slice: tuple[int, int],
    *,
    name: str = "policy",
    per_app: bool = False,
) -> dict:
    """Score ``policy`` over ``rows`` on ``eval_slice``. Returns a flat record."""
    lo, hi = eval_slice
    total = SimResult()
    per_app_rows = []
    t0 = time.time()

    for i, r in enumerate(rows):
        inv_full = counts[r].astype(np.int64)
        warm_full = policy(inv_full, i)
        inv = inv_full[lo:hi]
        warm = np.asarray(warm_full)[lo:hi]
        res = simulate_app(inv, warm, float(memory_mb[i]))
        total.merge(res)
        if per_app:
            d = res.as_dict()
            d["app"] = int(r)
            per_app_rows.append(d)

    rec = total.as_dict()
    rec["policy"] = name
    rec["n_apps"] = len(rows)
    rec["eval_minutes"] = hi - lo
    rec["runtime_s"] = round(time.time() - t0, 2)
    if per_app:
        rec["_per_app"] = per_app_rows
    return rec


def save_records(records: list[dict], filename: str) -> None:
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    (TABLES / filename).write_text(json.dumps(clean, indent=2))
    print(f"[eval] wrote {TABLES / filename}  ({len(clean)} records)")


def pareto_front(records: list[dict],
                 x: str = "idle_gb_minutes",
                 y: str = "cold_starts") -> list[dict]:
    """Records not dominated on (minimise x, minimise y)."""
    pts = sorted(records, key=lambda r: (r[x], r[y]))
    front, best_y = [], float("inf")
    for r in pts:
        if r[y] < best_y:
            front.append(r)
            best_y = r[y]
    return front


def summarise(records: list[dict]) -> pd.DataFrame:
    cols = ["policy", "cold_starts", "cold_start_rate", "mean_added_latency_ms",
            "idle_gb_minutes", "idle_warm_minutes", "wasted_usd", "invocations"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in records])
    return df.sort_values("cold_starts")
