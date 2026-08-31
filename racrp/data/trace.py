"""Azure Functions Trace 2019 -> app-level minute matrix.

The trace ships one CSV per day with 1440 per-minute invocation-count columns
per *function*. Azure allocates resources per **application**, and warm-up
decisions are made at the application level (see the dataset notes), so we
aggregate functions into apps and simulate apps.

Everything expensive is cached under ``data/processed`` as compressed ``.npz``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from racrp.config import (
    RAW, PROCESSED, N_DAYS, MIN_PER_DAY, N_MINUTES,
    DEFAULT_APP_MEMORY_MB, MIN_INVOCATIONS_14D, COHORT_SIZE, SEED,
)

TRIGGERS = ["http", "timer", "event", "queue", "storage", "orchestration", "others"]

INV_FILE = "invocations_per_function_md.anon.d{:02d}.csv"
DUR_FILE = "function_durations_percentiles.anon.d{:02d}.csv"
MEM_FILE = "app_memory_percentiles.anon.d{:02d}.csv"

CACHE_FILE = PROCESSED / "app_minute_matrix.npz"
META_FILE = PROCESSED / "app_meta.json"


def _find_raw_dir() -> Path:
    """The tarball may extract into RAW directly or into a subdirectory."""
    if (RAW / INV_FILE.format(1)).exists():
        return RAW
    for cand in sorted(RAW.iterdir()):
        if cand.is_dir() and (cand / INV_FILE.format(1)).exists():
            return cand
    raise FileNotFoundError(
        "Could not find " + INV_FILE.format(1) + f" under {RAW}. Extract the trace first."
    )


def build_app_matrix(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ``(counts, meta)``.

    counts : uint16 array (n_apps, 20160) of invocations per minute
    meta   : DataFrame indexed by row with HashApp, HashOwner, trigger mix,
             duration stats, memory (MB), and aggregate activity statistics.
    """
    if CACHE_FILE.exists() and META_FILE.exists() and not force:
        z = np.load(CACHE_FILE, allow_pickle=False)
        counts = z["counts"]
        meta = pd.read_json(META_FILE, orient="records")
        return counts, meta

    src = _find_raw_dir()
    print(f"[trace] reading from {src}")

    minute_cols = [str(i) for i in range(1, MIN_PER_DAY + 1)]
    per_day: dict[int, pd.DataFrame] = {}
    app_trigger_counts: dict[str, dict[str, int]] = {}
    app_owner: dict[str, str] = {}

    for d in range(1, N_DAYS + 1):
        df = pd.read_csv(src / INV_FILE.format(d))
        for app, trig, owner in zip(df["HashApp"], df["Trigger"], df["HashOwner"]):
            tc = app_trigger_counts.setdefault(app, {})
            tc[trig] = tc.get(trig, 0) + 1
            app_owner[app] = owner
        agg = df.groupby("HashApp", sort=False)[minute_cols].sum().astype(np.uint32)
        per_day[d] = agg
        print(f"[trace] day {d:2d}: {len(df):7d} function rows -> {len(agg):6d} apps")

    apps = sorted(set().union(*(set(v.index) for v in per_day.values())))
    app_index = {a: i for i, a in enumerate(apps)}
    counts = np.zeros((len(apps), N_MINUTES), dtype=np.uint32)

    for d, agg in per_day.items():
        rows = np.array([app_index[a] for a in agg.index], dtype=np.int64)
        lo = (d - 1) * MIN_PER_DAY
        counts[rows, lo:lo + MIN_PER_DAY] = agg.to_numpy()
    del per_day

    # ---- durations: function level -> app level, invocation-weighted
    dur_w = np.zeros(len(apps))
    dur_s = np.zeros(len(apps))
    dur_p99 = np.zeros(len(apps))
    for d in range(1, N_DAYS + 1):
        f = src / DUR_FILE.format(d)
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["HashApp"].isin(app_index)]
        df = df.assign(
            _s=df["Average"] * df["Count"],
            _p=df["percentile_Average_99"] * df["Count"],
        )
        g = df.groupby("HashApp", sort=False).agg(w=("Count", "sum"),
                                                  s=("_s", "sum"),
                                                  p=("_p", "sum"))
        rows = np.array([app_index[a] for a in g.index], dtype=np.int64)
        dur_w[rows] += g["w"].to_numpy()
        dur_s[rows] += g["s"].to_numpy()
        dur_p99[rows] += g["p"].to_numpy()
    safe_w = np.maximum(dur_w, 1.0)
    avg_dur = np.where(dur_w > 0, dur_s / safe_w, np.nan)
    p99_dur = np.where(dur_w > 0, dur_p99 / safe_w, np.nan)

    # ---- memory: app level, averaged over the 12 available days
    mem_s = np.zeros(len(apps))
    mem_n = np.zeros(len(apps))
    for d in range(1, N_DAYS + 1):
        f = src / MEM_FILE.format(d)
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df["HashApp"].isin(app_index)]
        g = df.groupby("HashApp", sort=False)["AverageAllocatedMb"].mean()
        rows = np.array([app_index[a] for a in g.index], dtype=np.int64)
        vals = g.to_numpy()
        ok = np.isfinite(vals)
        mem_s[rows[ok]] += vals[ok]
        mem_n[rows[ok]] += 1
    memory_mb = np.where(mem_n > 0, mem_s / np.maximum(mem_n, 1), DEFAULT_APP_MEMORY_MB)

    total_inv = counts.sum(axis=1)
    active_min = (counts > 0).sum(axis=1)

    def dominant(a: str) -> str:
        tc = app_trigger_counts.get(a, {})
        return max(tc, key=tc.get) if tc else "others"

    meta = pd.DataFrame({
        "HashApp": apps,
        "HashOwner": [app_owner.get(a, "") for a in apps],
        "trigger": [dominant(a) for a in apps],
        "n_functions": [sum(app_trigger_counts.get(a, {}).values()) for a in apps],
        "total_invocations": total_inv,
        "active_minutes": active_min,
        "avg_duration_ms": avg_dur,
        "p99_duration_ms": p99_dur,
        "memory_mb": memory_mb,
    })
    meta["invocations_per_minute"] = meta["total_invocations"] / N_MINUTES
    meta["duty_cycle"] = meta["active_minutes"] / N_MINUTES

    counts = np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)
    np.savez_compressed(CACHE_FILE, counts=counts)
    meta.to_json(META_FILE, orient="records")
    print(f"[trace] cached {counts.shape} -> {CACHE_FILE}")
    return counts, meta


def idle_times(row: np.ndarray) -> np.ndarray:
    """Gaps (minutes) between consecutive active minutes of one app.

    This is the 'IT' distribution the ATC'20 hybrid histogram policy is built on.
    """
    idx = np.flatnonzero(row > 0)
    if len(idx) < 2:
        return np.empty(0, dtype=np.int64)
    return np.diff(idx)


def select_cohort(counts: np.ndarray, meta: pd.DataFrame,
                  size: int = COHORT_SIZE, seed: int = SEED) -> np.ndarray:
    """Stratified sample of app row indices: activity decile x dominant trigger.

    The full trace has tens of thousands of apps; a stratified cohort keeps the
    experiment tractable while preserving the heavy-tailed activity mix that
    makes the cold-start problem hard.
    """
    rng = np.random.default_rng(seed)
    eligible = meta.index[meta["total_invocations"] >= MIN_INVOCATIONS_14D].to_numpy()
    sub = meta.loc[eligible]
    decile = pd.qcut(sub["total_invocations"].rank(method="first"), 10, labels=False)
    strata = sub["trigger"].astype(str) + "|" + decile.astype(str)

    picked: list[int] = []
    total = len(sub)
    for _key, idx in strata.groupby(strata).groups.items():
        idx = np.asarray(idx)
        quota = min(max(1, int(round(size * len(idx) / total))), len(idx))
        picked.extend(rng.choice(idx, size=quota, replace=False).tolist())
    out = np.array(sorted(set(picked)))
    if len(out) > size:
        out = np.sort(rng.choice(out, size=size, replace=False))
    return out
