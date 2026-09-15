"""Explicit renewal-state features for the risk predictor.

Why these exist
---------------
The predictor's only view of recent history is a 240-minute invocation window.
The information audit found that the single most informative context for the next
idle gap is **the length of the gap that just closed** (11.0% of the gap's
entropy, ahead of recent activity at 8.6% and time of day at 7.6%). The model can
only see that variable implicitly, through the window, and only while the gap is
shorter than the window. Since 54% of decision points are more than an hour idle,
and any gap beyond 240 minutes leaves the window a vector of zeros, the predictor
is blind to the most informative variable in the problem exactly where the recent
window is least informative and the decision is hardest.

These features make the renewal state explicit and unbounded by the window:

  * elapsed idle age -- how long since the last invocation, uncapped
  * the length of the previous closed gap -- the audited variable
  * that gap's bucket, one-hot in the discretisation the policy uses
  * time since a comparably long gap last occurred -- a crude recurrence signal

All are computable from information strictly before the decision minute, and cost
a handful of scalars at inference.
"""
from __future__ import annotations

import numpy as np

from racrp.baselines.conditional import gap_bucket, N_GAP_BUCKETS

# elapsed age, previous gap, recurrence, plus the one-hot bucket and a validity flag
D_STATE = 3 + N_GAP_BUCKETS + 1
BIG = 1e9


def state_features(counts: np.ndarray, rows: np.ndarray,
                   app_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """Renewal-state features for a batch of situations ``(app_idx, t_idx)``.

    ``rows`` maps a cohort position to a matrix row. Returns float32
    ``(n, D_STATE)``.

    Causality: for a decision at minute ``t`` we use only invocations at minutes
    strictly less than ``t``. The "previous gap" is the one that closed at the
    most recent active minute before ``t``, and it is therefore fully observed.
    """
    n = len(app_idx)
    out = np.zeros((n, D_STATE), dtype=np.float32)
    order = np.argsort(app_idx, kind="stable")

    start = 0
    while start < n:
        a = app_idx[order[start]]
        end = start
        while end < n and app_idx[order[end]] == a:
            end += 1
        sel = order[start:end]
        ts = t_idx[sel].astype(np.int64)

        active = np.flatnonzero(counts[rows[a]] > 0)
        if len(active) == 0:
            out[sel, -1] = 0.0
            start = end
            continue

        # index of the last active minute strictly before t
        pos = np.searchsorted(active, ts, side="left") - 1
        has_prev = pos >= 0
        last_active = np.where(has_prev, active[np.clip(pos, 0, len(active) - 1)], -1)
        age = np.where(has_prev, ts - last_active, BIG)

        # the gap that closed at that last active minute
        has_gap = pos >= 1
        prev_gap = np.where(
            has_gap,
            active[np.clip(pos, 0, len(active) - 1)]
            - active[np.clip(pos - 1, 0, len(active) - 1)],
            0)

        out[sel, 0] = np.log1p(np.minimum(age, BIG)).astype(np.float32)
        out[sel, 1] = np.log1p(prev_gap).astype(np.float32)

        # time since a comparably sized gap (same bucket) last closed
        if len(active) >= 3:
            gaps = np.diff(active)
            gbuck = gap_bucket(gaps)
            closed_at = active[1:]
            recur = np.zeros(len(sel), dtype=np.float64)
            for j in range(len(sel)):
                if not has_gap[j]:
                    recur[j] = BIG
                    continue
                k = pos[j] - 1                      # index into gaps/closed_at
                b = gbuck[k]
                earlier = np.flatnonzero((gbuck[:k] == b))
                recur[j] = (ts[j] - closed_at[earlier[-1]]) if len(earlier) else BIG
            out[sel, 2] = np.log1p(np.minimum(recur, BIG)).astype(np.float32)
        else:
            out[sel, 2] = np.log1p(BIG)

        b = gap_bucket(np.maximum(prev_gap, 1))
        out[sel[has_gap], 3 + b[has_gap]] = 1.0
        out[sel, -1] = has_gap.astype(np.float32)
        start = end

    return out
