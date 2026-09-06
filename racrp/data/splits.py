"""Application-level splits and history-limited observability.

The temporal split alone is not enough to test whether a *shared* memory helps:
every application in it has its own long history available, which is itself a
per-application memory. To ask whether experience transfers **across**
applications we must evaluate on applications the system has never seen, and
control how much of their own history it is allowed to observe.

  * ``split_apps``      partitions the cohort into ``seen`` / ``unseen``,
                        stratified by activity decile and dominant trigger.
  * ``observable_slice``gives the window of an unseen application's history that
                        the scheduler is permitted to use: the ``H`` minutes
                        immediately preceding the evaluation slice. ``H = 0``
                        models a function on the day it is deployed -- nothing is
                        known about it except its declared metadata.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racrp.config import SEED, TEST_SLICE

# how much of its own history an unseen application is allowed to have observed,
# in minutes. 0 = just deployed; 11520 = the full training period.
HISTORY_SWEEP = (0, 60, 240, 1440, 4320, 11520)

HISTORY_LABEL = {0: "just deployed", 60: "1 hour", 240: "4 hours",
                 1440: "1 day", 4320: "3 days", 11520: "8 days"}


def split_apps(meta: pd.DataFrame, rows: np.ndarray, *, unseen_frac: float = 0.30,
               seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Partition cohort positions into (seen, unseen), stratified.

    Returns positions **into ``rows``**, not raw matrix row indices, because
    every downstream array (features, predictions, cohort) is indexed that way.
    """
    rng = np.random.default_rng(seed)
    sub = meta.loc[rows].reset_index(drop=True)
    decile = pd.qcut(sub["total_invocations"].rank(method="first"), 10, labels=False)
    strata = sub["trigger"].astype(str) + "|" + decile.astype(str)

    unseen: list[int] = []
    for _key, idx in strata.groupby(strata).groups.items():
        idx = np.asarray(idx)
        n = int(round(unseen_frac * len(idx)))
        if n:
            unseen.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    unseen = np.array(sorted(set(unseen)))
    seen = np.setdiff1d(np.arange(len(rows)), unseen)
    return seen, unseen


def observable_slice(history_minutes: int,
                     eval_slice: tuple[int, int] = TEST_SLICE) -> tuple[int, int]:
    """The window of an application's own past the scheduler may use.

    Ends where the evaluation period begins, so nothing after the decision point
    is ever visible.
    """
    end = eval_slice[0]
    return max(end - history_minutes, 0), end


def idle_gaps(row: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """Observed idle times within a window (minutes between active minutes)."""
    lo, hi = window
    idx = np.flatnonzero(row[lo:hi] > 0)
    return np.diff(idx) if len(idx) >= 2 else np.empty(0, dtype=np.int64)


def app_profile(counts: np.ndarray, row: int, window: tuple[int, int],
                meta_row: pd.Series) -> np.ndarray:
    """A behavioural descriptor of one application from its observable history.

    Deliberately includes deployment-time metadata (trigger, memory, duration)
    that is known before the application has ever run, so that the profile is
    still defined when ``H = 0`` -- which is the regime the whole reframing is
    about.
    """
    from racrp.data.trace import TRIGGERS
    lo, hi = window
    seg = counts[row, lo:hi] if hi > lo else np.empty(0, dtype=np.uint16)

    trig = np.zeros(len(TRIGGERS), dtype=np.float32)
    t = str(meta_row["trigger"])
    if t in TRIGGERS:
        trig[TRIGGERS.index(t)] = 1.0

    static = np.array([
        np.log1p(float(meta_row["memory_mb"])),
        np.log1p(float(np.nan_to_num(meta_row["avg_duration_ms"]))),
        np.log1p(float(np.nan_to_num(meta_row["p99_duration_ms"]))),
        np.log1p(float(meta_row["n_functions"])),
    ], dtype=np.float32)

    if len(seg) == 0:
        behav = np.zeros(9, dtype=np.float32)
        diurnal = np.zeros(24, dtype=np.float32)
        observed = np.zeros(1, dtype=np.float32)
    else:
        gaps = idle_gaps(counts[row], window)
        duty = float((seg > 0).mean())
        rate = float(seg.mean())
        if len(gaps):
            q = np.log1p(np.percentile(gaps, [5, 25, 50, 75, 95]))
            g = np.array([np.log1p(gaps.mean()), np.log1p(gaps.std()),
                          np.log1p(len(gaps))], dtype=np.float32)
        else:
            q = np.zeros(5, dtype=np.float32)
            g = np.zeros(3, dtype=np.float32)
        behav = np.concatenate([[duty, np.log1p(rate)], q, g[:2]]).astype(np.float32)
        # diurnal shape: activity by hour of day, normalised
        minutes = np.arange(lo, hi)
        hours = (minutes % 1440) // 60
        d = np.zeros(24, dtype=np.float32)
        np.add.at(d, hours, (seg > 0).astype(np.float32))
        cnt = np.zeros(24, dtype=np.float32)
        np.add.at(cnt, hours, 1.0)
        diurnal = d / np.maximum(cnt, 1.0)
        observed = np.array([1.0], dtype=np.float32)

    return np.concatenate([trig, static, behav, diurnal, observed]).astype(np.float32)


PROFILE_DIM = 7 + 4 + 9 + 24 + 1
