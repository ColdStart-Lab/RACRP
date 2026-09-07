"""Per-app and stratified reporting.

Aggregate cold-start counts are dominated by a handful of very dense
applications: in our cohort the busiest 1% of apps issue most of the
invocations, so a policy can look excellent overall while failing every sparse
app in the trace. Since sparse, bursty applications are exactly the regime this
project claims to improve, results are reported three ways:

  * **micro**  -- totals over the whole cohort (what the platform's bill sees)
  * **macro**  -- mean of per-app cold-start rates (what a typical app sees)
  * **strata** -- broken down by how sparse the app is

A claim that only survives in one of these is reported as such, not generalised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racrp.config import TRAIN_SLICE
from racrp.data.trace import idle_times

SPARSITY_BINS = [0, 1, 5, 20, 60, np.inf]
SPARSITY_LABELS = ["dense (<1 min gap)", "1-5 min", "5-20 min", "20-60 min", ">60 min"]


def app_sparsity(counts: np.ndarray, rows: np.ndarray,
                 slice_: tuple[int, int] = TRAIN_SLICE) -> np.ndarray:
    """Median idle time per app, measured on the TRAIN slice only.

    Using the training slice keeps the stratification itself out of the test
    data -- otherwise the choice of which apps count as 'sparse' would be made
    using the very period we evaluate on.
    """
    lo, hi = slice_
    out = np.zeros(len(rows))
    for i, r in enumerate(rows):
        its = idle_times(counts[r, lo:hi])
        out[i] = np.median(its) if len(its) else np.inf
    return out


def stratify(per_app: list[dict], sparsity: np.ndarray,
             rows: np.ndarray) -> pd.DataFrame:
    """Aggregate per-app simulator records within sparsity bands."""
    df = pd.DataFrame(per_app)
    pos = {int(r): i for i, r in enumerate(rows)}
    df["sparsity"] = [sparsity[pos[int(a)]] for a in df["app"]]
    df["band"] = pd.cut(df["sparsity"], bins=SPARSITY_BINS,
                        labels=SPARSITY_LABELS, right=False)
    g = df.groupby("band", observed=False).agg(
        n_apps=("app", "count"),
        cold_starts=("cold_starts", "sum"),
        invocations=("invocations", "sum"),
        idle_gb_minutes=("idle_gb_minutes", "sum"),
        mean_added_latency_ms=("mean_added_latency_ms", "mean"),
    )
    g["cold_start_rate_micro"] = g["cold_starts"] / g["invocations"].replace(0, np.nan)
    macro = df.groupby("band", observed=False)["cold_start_rate"].mean()
    g["cold_start_rate_macro"] = macro
    return g.reset_index()


def macro_summary(per_app: list[dict]) -> dict:
    """Cohort summary that weights every application equally."""
    df = pd.DataFrame(per_app)
    return {
        "n_apps": int(len(df)),
        "macro_cold_start_rate": float(df["cold_start_rate"].mean()),
        "median_app_cold_start_rate": float(df["cold_start_rate"].median()),
        "apps_with_zero_cold_starts": int((df["cold_starts"] == 0).sum()),
        "macro_added_latency_ms": float(df["mean_added_latency_ms"].mean()),
        "p95_app_cold_start_rate": float(df["cold_start_rate"].quantile(0.95)),
    }
