"""Decide *per application* whether to condition, and detect regime changes.

Two refinements that follow directly from what the earlier experiments measured,
rather than from new ideas bolted on.

## 1. Per-application policy selection

The mechanism check found the gain from conditioning tracks how much an
application's consecutive gaps depend on each other: **−0.6%** in the quartile
where they are independent, **+62%** where they are most dependent, correlation
+0.99 across strata. The second quartile was *negative*.

A platform does not have to apply one policy to everything. Whether conditioning
is justified for a given application is testable **on that application's own
training data**, before the evaluation period begins: split its observed gaps,
fit both the marginal and the conditional on one half, and compare their
held-out log-likelihood on the other. Condition only where the conditional
actually predicts better.

This is the "cleaning" step the whole method needed. It cannot do worse than the
baseline except through selection noise, and it removes the strata where
conditioning was actively harmful.

## 2. Regime detection

The information audit measured *stationary* mutual information, averaged over an
application's whole history. It therefore says nothing about applications whose
behaviour **changes** — a deployment, an incident, a traffic shift. In that
regime an application's own past is not merely uninformative, it is misleading,
and the pooled histogram is anchored to behaviour that no longer holds.

We detect a change by comparing the distribution of recent gaps against the
application's longer history (a two-sample test on the log-gap distribution), and
when they diverge we down-weight the stale history rather than discard it.

Both are decided causally, from data already observed at the moment of the
decision.
"""
from __future__ import annotations

import numpy as np

from racrp.baselines.histogram import DEFAULT_RANGE
from racrp.baselines.conditional import gap_bucket, tod_bucket, N_GAP_BUCKETS, N_TOD_BUCKETS
from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows
from racrp.baselines.hazard import optimal_keepalive

MIN_GAPS_TO_SELECT = 40


def conditioning_is_justified(inv: np.ndarray, upto: int, *,
                              key: str = "prev_gap",
                              n_gap_buckets: int = N_GAP_BUCKETS,
                              kappa: float = 8.0,
                              min_gaps: int = MIN_GAPS_TO_SELECT) -> tuple[bool, float]:
    """Does conditioning predict this application's gaps better than the marginal?

    Decided by held-out likelihood **within the training period only**: fit both
    estimators on the first 70% of the application's observed gaps, score the
    remaining 30%. Returns ``(use_conditional, log-likelihood gain per gap)``.

    Nothing from the evaluation period is touched, so this is a legitimate
    per-application model choice rather than a peek at the answer.
    """
    a = np.flatnonzero(inv[:upto] > 0)
    if len(a) < min_gaps + 3:
        return False, 0.0
    gaps = np.diff(a)
    starts = a[:-1]
    if len(gaps) < min_gaps:
        return False, 0.0

    cut = int(0.7 * len(gaps))
    if cut < 10 or len(gaps) - cut < 5:
        return False, 0.0

    ng = n_gap_buckets
    prev_b = np.concatenate([[0], gap_bucket(gaps[:-1], ng)])
    if key == "prev_gap":
        cells, cell_of = ng, prev_b
    else:
        cells = ng * N_TOD_BUCKETS
        cell_of = prev_b * N_TOD_BUCKETS + tod_bucket(starts)

    R = DEFAULT_RANGE
    fit, held = slice(0, cut), slice(cut, len(gaps))
    g_fit = np.minimum(gaps[fit], R)
    g_held = np.minimum(gaps[held], R)

    marg = np.zeros(R + 1)
    np.add.at(marg, g_fit, 1.0)
    cond = np.zeros((cells, R + 1))
    ncell = np.zeros(cells)
    np.add.at(cond, (cell_of[fit], g_fit), 1.0)
    np.add.at(ncell, cell_of[fit], 1.0)

    eps = 1.0                                  # Laplace, so nothing is impossible
    p_marg = (marg + eps) / (marg.sum() + eps * (R + 1))

    ll_m = np.log(p_marg[g_held]).mean()
    cells_held = cell_of[held]
    p_cond = np.empty(len(g_held))
    for i, (c, g) in enumerate(zip(cells_held, g_held)):
        n = ncell[c]
        blend = (n * (cond[c] / max(cond[c].sum(), 1)) + kappa * p_marg) / (n + kappa)
        blend = (blend + 1e-12) / (blend.sum() + 1e-12 * (R + 1))
        p_cond[i] = blend[g]
    ll_c = np.log(p_cond).mean()
    return bool(ll_c > ll_m), float(ll_c - ll_m)


def regime_weights(gaps: np.ndarray, *, recent: int = 20,
                   threshold: float = 0.35) -> float:
    """How much to trust an application's older history, in [0, 1].

    Compares the most recent ``recent`` gaps against everything before them on the
    log-gap scale. A large standardised separation means the application's
    behaviour has shifted, so its older gaps describe a regime that no longer
    holds and should be down-weighted rather than discarded.

    Returns a multiplier applied to the historical counts: 1.0 when nothing has
    changed, falling towards 0.2 when the recent behaviour is clearly different.
    """
    if len(gaps) < recent + 10:
        return 1.0
    old = np.log1p(gaps[:-recent].astype(np.float64))
    new = np.log1p(gaps[-recent:].astype(np.float64))
    sd = np.sqrt(0.5 * (old.var() + new.var())) + 1e-9
    sep = abs(new.mean() - old.mean()) / sd
    if sep <= threshold:
        return 1.0
    return float(np.clip(1.0 / (1.0 + (sep - threshold)), 0.2, 1.0))


def selective_policy(
    invocations: np.ndarray,
    *,
    eval_slice: tuple[int, int],
    price_ratio: float | None = None,
    tail_margin: float = 92.0,
    key: str = "prev_gap+tod",
    kappa: float = 8.0,
    hist_range: int = DEFAULT_RANGE,
    n_gap_buckets: int = N_GAP_BUCKETS,
    recompute_every: int = 4,
    use_selection: bool = True,
    use_regime: bool = True,
) -> np.ndarray:
    """Conditional keep-alive with per-application selection and regime weighting.

    ``price_ratio`` selects the derived optimal-stopping window; leaving it None
    falls back to the tail percentile, so the two window rules can be compared
    with everything else held fixed.
    """
    inv = np.asarray(invocations)
    T = len(inv)
    warm = np.zeros(T, dtype=bool)
    a = np.flatnonzero(inv > 0)
    if len(a) < 2:
        return warm

    lo_e, hi_e = eval_slice
    starts, gaps, closed = a[:-1], np.diff(a), a[1:]
    ng = n_gap_buckets

    # ---- decide, from training data only, whether to condition at all
    condition = True
    if use_selection:
        condition, _ = conditioning_is_justified(inv, lo_e, key="prev_gap",
                                                 n_gap_buckets=ng, kappa=kappa)

    prev_b = np.concatenate([[0], gap_bucket(gaps[:-1], ng)])
    if key == "prev_gap":
        cells, cell_of = ng, prev_b
    else:
        cells = ng * N_TOD_BUCKETS
        cell_of = prev_b * N_TOD_BUCKETS + tod_bucket(starts)

    marg = np.zeros(hist_range + 1)
    cond = np.zeros((cells, hist_range + 1))
    n_marg = 0
    n_cell = np.zeros(cells)
    seen_gaps: list[int] = []

    ev = a[(a >= lo_e) & (a < hi_e)]
    ptr = 0
    ka, pw = 10, 0
    since = recompute_every

    for s in ev:
        while ptr < len(closed) and closed[ptr] < s:
            g = min(int(gaps[ptr]), hist_range)
            marg[g] += 1.0
            n_marg += 1
            c = int(cell_of[ptr])
            cond[c, g] += 1.0
            n_cell[c] += 1.0
            seen_gaps.append(g)
            ptr += 1

        since += 1
        if since >= recompute_every and n_marg:
            w = (regime_weights(np.asarray(seen_gaps[-400:]))
                 if (use_regime and len(seen_gaps) > 30) else 1.0)

            if not condition or ptr == 0:
                dist = marg.copy()
            else:
                if key == "prev_gap":
                    c = int(gap_bucket(gaps[ptr - 1], ng))
                else:
                    c = (int(gap_bucket(gaps[ptr - 1], ng)) * N_TOD_BUCKETS
                         + int(tod_bucket(s)))
                nb = n_cell[c]
                dist = ((nb * (cond[c] / max(cond[c].sum(), 1))
                         + kappa * (marg / max(marg.sum(), 1)))
                        if nb > 0 else marg.copy())

            if w < 1.0 and len(seen_gaps) >= 20:
                # the regime has shifted: re-weight towards what just happened
                recent = np.zeros(hist_range + 1)
                np.add.at(recent, np.array(seen_gaps[-20:]), 1.0)
                dist = w * (dist / max(dist.sum(), 1e-12)) + \
                    (1 - w) * (recent / max(recent.sum(), 1e-12))

            if price_ratio is not None:
                # length from the cost model, offset from the same rule the
                # percentile policy uses, so the two differ only in the length
                # no pre-warm: the derived T is the optimum for holding
                # continuously from the gap's start, and shifting that window by
                # an offset it was not optimised for destroys the result
                ka = optimal_keepalive(dist, price_ratio)
                pw = 0
            else:
                pw, ka = quantiles_to_windows(hist_to_quantiles(dist),
                                              5.0, tail_margin, 10)
            since = 0

        start = min(max(int(s) + pw, int(s) + 1), T)
        end = min(start + ka, T)
        if start < end:
            warm[start:end] = True
    return warm
