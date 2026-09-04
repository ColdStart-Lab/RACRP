"""Hybrid Histogram Policy - Shahrad et al., USENIX ATC 2020.

Reimplementation of the policy deployed in Azure Functions, which is the
strongest non-learned standard this project must beat.

The policy, per application:

  1. Maintain a histogram of **idle times** (IT) between consecutive invocations,
     with 1-minute bins over a bounded range (default 4 hours).
  2. If the app's IT distribution fits inside the range and is not too dispersed
     (coefficient of variation below a cut-off), use the *histogram* mode:
        pre-warm window  = 5th percentile of IT   (drop the instance, recreate
                           it just before the next invocation is expected)
        keep-alive window= 99th percentile of IT - pre-warm window
                           (how long to hold it, covering the tail)
  3. If the app is too sparse for the range (large ITs), fall back to a
     **time-series forecast** of the next IT (the paper uses ARIMA; we use the
     same idea with a light AR model so the baseline runs over thousands of apps
     in reasonable time) and set the windows around the forecast.
  4. If the app is too dense / bursty (CV below the histogram's usefulness, or
     too few samples), fall back to a **standard fixed keep-alive** window.

The histogram is updated online as the simulation proceeds, so the policy only
ever sees the past - the same causality contract as every other policy here.

Reference
---------
M. Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the
Serverless Workload at a Large Cloud Provider," USENIX ATC 2020, Sec. 4.
"""
from __future__ import annotations

import numpy as np

# Paper defaults (Sec. 4.2): 4-hour histogram range, 5th/99th percentile margins.
DEFAULT_RANGE = 240          # minutes covered by the histogram
DEFAULT_HEAD_MARGIN = 5      # pre-warm percentile
DEFAULT_TAIL_MARGIN = 99     # keep-alive percentile
CV_CUTOFF = 2.0              # above this the IT distribution is too dispersed
MIN_SAMPLES = 3              # need at least this many ITs to trust the histogram
OOB_FRACTION = 0.5           # if this share of ITs exceed the range -> ARIMA mode


def _percentile_from_hist(hist: np.ndarray, q: float) -> int:
    """Smallest bin index whose cumulative mass reaches ``q`` percent."""
    total = hist.sum()
    if total <= 0:
        return 0
    target = total * q / 100.0
    c = np.cumsum(hist)
    return int(np.searchsorted(c, target))


def _ar_forecast(its: list[int], k: int = 4) -> float:
    """Light AR(k)-style forecast of the next idle time.

    Stands in for the paper's ARIMA fallback: a least-squares AR fit on the log
    idle times, which is what makes the fallback work on the long-tailed apps
    while staying cheap enough for a whole-cohort sweep.
    """
    if len(its) < k + 2:
        return float(np.mean(its)) if its else 0.0
    x = np.log1p(np.asarray(its, dtype=np.float64))
    X = np.stack([x[i:len(x) - k + i] for i in range(k)], axis=1)
    y = x[k:]
    try:
        coef, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], y, rcond=None)
        pred = float(np.r_[x[-k:], 1.0] @ coef)
    except np.linalg.LinAlgError:
        return float(np.mean(its))
    return float(np.expm1(np.clip(pred, 0.0, 12.0)))


def hybrid_histogram_policy(
    invocations: np.ndarray,
    *,
    hist_range: int = DEFAULT_RANGE,
    head_margin: float = DEFAULT_HEAD_MARGIN,
    tail_margin: float = DEFAULT_TAIL_MARGIN,
    default_keepalive: int = 10,
    cv_cutoff: float = CV_CUTOFF,
    warmup_slice: tuple[int, int] | None = None,
    recompute_every: int = 1,
    ar_history: int = 256,
) -> np.ndarray:
    """Return the per-minute ``keep_warm`` array for one app.

    ``warmup_slice`` pre-loads the histogram from an earlier period (the training
    days) so the policy is not cold-started on the test days, mirroring a
    production deployment that has been running for a while.

    The window parameters are recomputed after **every** idle-time sample by
    default, as in the paper -- measured at ~230 s for the whole cohort, which is
    affordable, and this baseline is given every advantage. Running moments are
    maintained incrementally and the AR fallback is fitted on a bounded tail, so
    the cost is linear rather than quadratic in an app's activity.
    ``recompute_every > 1`` models a deployment that refreshes periodically; it
    measurably degrades the policy on memoryless traces, which is why it is not
    the default.
    """
    T = len(invocations)
    warm = np.zeros(T, dtype=bool)
    active = np.flatnonzero(np.asarray(invocations) > 0)
    if len(active) == 0:
        return warm

    hist = np.zeros(hist_range + 1, dtype=np.int64)
    n = 0                      # number of idle-time samples seen
    s1 = 0.0                   # running sum of ITs
    s2 = 0.0                   # running sum of squares
    n_oob = 0
    recent: list[int] = []     # bounded tail used by the AR fallback

    def observe(gap: int) -> None:
        nonlocal n, s1, s2, n_oob
        n += 1
        s1 += gap
        s2 += gap * gap
        if gap <= hist_range:
            hist[gap] += 1
        else:
            n_oob += 1
        recent.append(gap)
        if len(recent) > ar_history:
            del recent[0]

    if warmup_slice is not None:
        lo, hi = warmup_slice
        pre = np.flatnonzero(np.asarray(invocations[lo:hi]) > 0)
        for gap in np.diff(pre):
            observe(int(gap))

    pre_warm, keep_alive = 0, default_keepalive
    since_refresh = recompute_every        # force a computation on the first sample

    def refresh() -> tuple[int, int]:
        if n < MIN_SAMPLES:
            return 0, default_keepalive
        mean = s1 / n
        var = max(s2 / n - mean * mean, 0.0)
        cv = (var ** 0.5) / max(mean, 1e-9)
        if n_oob / n > OOB_FRACTION:
            nxt = _ar_forecast(recent)
            return int(max(0, nxt * 0.9)), int(max(1, nxt * 0.4))
        if cv > cv_cutoff:
            return 0, default_keepalive
        pw = _percentile_from_hist(hist, head_margin)
        ka = _percentile_from_hist(hist, tail_margin)
        # the window must COVER minute s+ka, so its length is ka-pw+1
        return int(pw), int(max(1, ka - pw + 1))

    prev = None
    for s in active:
        if prev is not None:
            observe(int(s - prev))
            since_refresh += 1
        prev = s

        if since_refresh >= recompute_every:
            pre_warm, keep_alive = refresh()
            since_refresh = 0

        # an idle time of IT means the next invocation lands at minute s+IT, so
        # the pre-warm offset is measured from s itself (warming at s+1+IT would
        # arrive exactly one minute late, every time)
        lo = min(max(s + pre_warm, s + 1), T)
        hi = min(lo + keep_alive, T)
        if lo < hi:
            warm[lo:hi] = True

    return warm
