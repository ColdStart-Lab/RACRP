"""State-conditioned episodic memory over an application's own idle gaps.

This is the method the project arrives at, and the route to it was empirical.

## What the measurement said

An information audit over 264 applications asked how much of the next idle gap's
entropy each available context explains (median gap entropy 1.242 bits):

| context                | share of gap entropy explained |
|------------------------|--------------------------------|
| burst level            | 0.0%                           |
| day of week            | 3.7%                           |
| recent 60-min activity | 5.5%                           |
| time of day            | 7.6%                           |
| recent 5-min activity  | 8.6%                           |
| **previous gap**       | **11.0%**                      |

Two things follow. First, roughly 89% of the uncertainty in an application's next
idle gap is irreducible from anything observable — serverless idle gaps on this
trace are close to memoryless, which is why every richer retrieval scheme we tried
failed: they were paying a large variance cost to chase signal that is mostly not
there. Second, the signal that *does* exist is concentrated in a single, discrete,
low-dimensional variable: the gap that just closed.

## What that implies for the mechanism

When the informative key is one coarse discrete variable, the right non-parametric
conditional estimator is not a learned embedding with k-nearest-neighbour lookup —
it is a **conditional histogram**. Retrieval over a discrete key with many samples
per cell *is* a conditional histogram; the embedding machinery only buys you
anything when the key is high-dimensional and continuous, and here that machinery
was pure variance. So the estimator below keeps one histogram per bucket of the
previous gap:

    h_hat( g | b )  =  ( n_b * h_b  +  kappa * h_marginal ) / ( n_b + kappa )

with the quantile-space blend used elsewhere in this project, and partial pooling
towards the marginal so a thin bucket is not trusted before it has earned it.

## Relationship to the deployed policy

`kappa -> inf`, or a bucket with no observations, gives exactly the ATC'20 hybrid
histogram. This is a strict generalisation along the conditioning axis: the
deployed policy is the special case that conditions on nothing.

## Causality

At a decision following the active minute ``s``, only gaps that had already
*closed* strictly before ``s`` are in the histograms, and the conditioning bucket
is the gap that closed most recently. Nothing about the current gap is used.
"""
from __future__ import annotations

import numpy as np

from racrp.baselines.histogram import (
    DEFAULT_RANGE, DEFAULT_HEAD_MARGIN, DEFAULT_TAIL_MARGIN,
)
from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows

MIN_PER_DAY = 1440
N_GAP_BUCKETS = 6          # log2 buckets of the previous gap, width 2 octaves
N_TOD_BUCKETS = 4          # 6-hour blocks, used only by the joint-key variant


def gap_bucket(g, n: int = N_GAP_BUCKETS) -> np.ndarray:
    """Coarse log-spaced bucket of a gap length, in [0, n)."""
    return np.clip((np.log2(np.maximum(np.asarray(g), 1)) / 2).astype(int), 0, n - 1)


def tod_bucket(minute, n: int = N_TOD_BUCKETS) -> np.ndarray:
    return ((np.asarray(minute) % MIN_PER_DAY) // (MIN_PER_DAY // n)).astype(int) % n


def conditional_policy(
    invocations: np.ndarray,
    *,
    eval_slice: tuple[int, int],
    key: str = "prev_gap",
    kappa: float = 8.0,
    head_margin: float = DEFAULT_HEAD_MARGIN,
    tail_margin: float = DEFAULT_TAIL_MARGIN,
    default_keepalive: int = 10,
    hist_range: int = DEFAULT_RANGE,
    n_gap_buckets: int = N_GAP_BUCKETS,
    recompute_every: int = 4,
) -> np.ndarray:
    """Keep-warm decisions from a conditional idle-gap histogram.

    ``key`` selects the conditioning variable and is the ablation axis:

    * ``"none"``          the ATC'20 marginal histogram (the baseline, same code)
    * ``"prev_gap"``      bucket of the gap that just closed  (the method)
    * ``"tod"``           bucket of time of day
    * ``"prev_gap+tod"``  the joint key
    * ``"shuffled"``      a random bucket with the same number of cells as
                          ``"prev_gap"`` -- the control that separates real
                          conditioning from the variance-reduction artefact of
                          simply splitting the data into more cells
    * ``"shuffled_joint"`` the same control sized for ``"prev_gap+tod"``, which
                          has four times as many cells. A control must match the
                          arm it controls for in cell count, or it is not a
                          control for it.
    """
    inv = np.asarray(invocations)
    T = len(inv)
    warm = np.zeros(T, dtype=bool)
    a = np.flatnonzero(inv > 0)
    if len(a) < 2:
        return warm

    starts, gaps, closed = a[:-1], np.diff(a), a[1:]
    ng = n_gap_buckets

    # bucket for episode i = the context observable when gaps[i] began, i.e. the
    # gap that closed immediately before it
    prev_b = np.concatenate([[0], gap_bucket(gaps[:-1], ng)])
    if key in ("none",):
        cells, cell_of = 1, np.zeros(len(gaps), dtype=int)
    elif key == "prev_gap":
        cells, cell_of = ng, prev_b
    elif key == "tod":
        cells, cell_of = N_TOD_BUCKETS, tod_bucket(starts)
    elif key == "prev_gap+tod":
        cells = ng * N_TOD_BUCKETS
        cell_of = prev_b * N_TOD_BUCKETS + tod_bucket(starts)
    elif key == "shuffled":
        cells = ng
        cell_of = np.random.default_rng(0).integers(0, ng, len(gaps))
    elif key == "shuffled_joint":
        cells = ng * N_TOD_BUCKETS
        cell_of = np.random.default_rng(1).integers(0, cells, len(gaps))
    else:
        raise ValueError(f"unknown key {key!r}")

    marg = np.zeros(hist_range + 1)
    cond = np.zeros((cells, hist_range + 1))
    n_marg = 0
    n_cell = np.zeros(cells)

    lo_e, hi_e = eval_slice
    ev = a[(a >= lo_e) & (a < hi_e)]
    ptr = 0
    pw, ka = 0, default_keepalive
    since = recompute_every

    for s in ev:
        # absorb every gap that has CLOSED before this decision, and no others
        while ptr < len(closed) and closed[ptr] < s:
            g = min(int(gaps[ptr]), hist_range)
            marg[g] += 1.0
            n_marg += 1
            c = int(cell_of[ptr])
            cond[c, g] += 1.0
            n_cell[c] += 1.0
            ptr += 1

        since += 1
        if since >= recompute_every and n_marg:
            q_marg = hist_to_quantiles(marg)
            if key == "none" or ptr == 0:
                q = q_marg
            else:
                # the cell the NEXT gap will fall in, given what just closed
                if key == "prev_gap":
                    c = int(gap_bucket(gaps[ptr - 1], ng))
                elif key == "tod":
                    c = int(tod_bucket(s))
                elif key == "prev_gap+tod":
                    c = int(gap_bucket(gaps[ptr - 1], ng)) * N_TOD_BUCKETS + int(tod_bucket(s))
                else:                                   # shuffled controls
                    c = int(cell_of[min(ptr, len(cell_of) - 1)])
                nb = n_cell[c]
                q = (((nb * hist_to_quantiles(cond[c]) + kappa * q_marg) / (nb + kappa))
                     if nb > 0 else q_marg)
            pw, ka = quantiles_to_windows(q, head_margin, tail_margin,
                                          default_keepalive)
            since = 0

        start = min(max(int(s) + pw, int(s) + 1), T)
        end = min(start + ka, T)
        if start < end:
            warm[start:end] = True
    return warm
