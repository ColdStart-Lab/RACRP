"""The keep-alive window, derived rather than tuned.

Every policy in this project so far — including the deployed ATC'20 one — reads
the keep-alive window off a **percentile** of an idle-gap distribution, and sweeps
that percentile to trade memory against latency. The percentile is a heuristic:
nothing says the 92nd is the right one, and it is not a quantity anyone operating
a platform can price.

The decision is really an optimal-stopping problem. Holding an idle container for
`T` minutes after an application goes quiet costs memory for as long as the gap
lasts; dropping it risks a cold start if the gap outlives `T`. In expectation:

    C(T)  =  m · ∫₀ᵀ S(u) du   +   K · S(T)

where `S(u) = P(gap > u)` is the survival function of the idle gap, `m` is the
memory cost of holding one minute, and `K` is what a cold start costs. Then

    dC/dT  =  m·S(T) − K·f(T)  =  0
    ⟹   f(T) / S(T)  =  m / K
    ⟹   **hazard(T) = m / K**

**Drop the container at the first moment its hazard rate falls below the price
ratio.** That is the whole policy, in closed form.

Two things follow that matter for this project:

* The tail-percentile knob disappears. What is swept instead is `K/m` — how many
  GB-minutes of memory a cold start is worth — which is an economic quantity a
  platform actually knows, not a distributional artefact.
* It composes with the conditioning. Using the state-conditional gap distribution
  gives `T*(state)`: a different optimal stopping time per state, each derived
  from the same cost model rather than from a shared percentile.

Serverless idle gaps are heavy-tailed, so their hazard is typically *decreasing*:
the longer an application has already been quiet, the longer it is likely to stay
quiet. That is exactly the regime where a percentile rule and an optimal rule
diverge, and it is why this is worth deriving rather than tuning.
"""
from __future__ import annotations

import numpy as np

from racrp.baselines.histogram import DEFAULT_RANGE
from racrp.baselines.conditional import (
    gap_bucket, tod_bucket, N_GAP_BUCKETS, N_TOD_BUCKETS,
)
from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows

# Total pseudo-count mass spread uniformly over the support, as a weak prior on
# the idle-gap distribution. Chosen on the VALIDATION days (9-10) by
# scripts/tune_smoothing.py, then carried to the test days unchanged.
#
# Worth recording why it is a named constant rather than a default of 1.0. This
# started life as `hist + smooth`, one pseudo-count *per bin*, which on a
# 241-minute support is 241 pseudo-observations. That setting scores BEST on the
# test period (+74.3% against +69.3% for a weak prior, 250 applications) and
# WORST on the validation period (+53.5% against +70.3% for this one). Selecting
# it on the strength of the first number would have been choosing a
# hyper-parameter by its test-set score.
DEFAULT_SMOOTH = 25.0


def optimal_keepalive(hist: np.ndarray, price_ratio: float,
                      *, smooth: float = DEFAULT_SMOOTH,
                      max_t: int | None = None) -> int:
    """The stopping time that minimises expected cost, in closed form.

    ``hist`` is an (unnormalised) histogram of observed gap lengths, bin ``i``
    meaning "a gap of ``i`` minutes". ``price_ratio`` is ``m / K``: the memory
    cost of holding one idle minute divided by the cost of a cold start. A larger
    ratio means memory is relatively expensive, so the container is dropped sooner.

    Expected cost of holding for ``T`` minutes, in units of one cold start:

        C(T) / K  =  ratio · Σ_{u=1..T} S(u-1)   +   S(T)

    where ``S(u) = P(gap > u)``. The first term is memory paid for every minute
    the gap is still running; the second is the chance of paying a cold start.

    **We minimise this directly rather than solving the first-order condition.**
    Setting the derivative to zero gives the elegant identity `hazard(T) = m/K`,
    and that identity is the right way to *understand* the rule -- but it locates
    stationary points, and C(T) is not unimodal for real gap distributions. A
    perfectly periodic application has zero hazard everywhere except at its
    period, so a first-crossing rule stops at minute 1 and cold-starts on every
    single invocation. (Measured: it did exactly that.) The global argmin over the
    grid is exact, costs O(range), and is what the cost model actually asks for.

    ``smooth`` is the **total** pseudo-count mass spread uniformly over the
    support, so that no gap length is assigned probability zero. It is a total
    and not a per-bin count on purpose: the support here is 241 minutes, so
    adding one count per bin would add 241 pseudo-observations against a typical
    application's few hundred real gaps -- a 20-30% uniform prior, which drags
    mass into a tail that is not there and makes the optimiser give up far too
    early. Measured on one application: it chose a 9-minute window whose true
    expected cost was 38% worse than the 21-minute optimum.
    """
    R = len(hist) - 1
    h = np.asarray(hist, dtype=np.float64) + smooth / (len(hist) or 1)
    total = h.sum()
    if total <= 0:
        return 1
    pmf = h / total
    surv = np.maximum(1.0 - np.cumsum(pmf), 0.0)      # S(u) = P(gap > u)
    at_least = np.concatenate([[1.0], surv[:-1]])     # P(gap >= u)

    hi = R if max_t is None else min(max_t, R)
    # memory paid over minutes 1..T, plus the residual cold-start risk
    mem = price_ratio * np.cumsum(at_least[:hi + 1])
    cost = mem + surv[:hi + 1]
    t = int(np.argmin(cost[1:]) + 1) if hi >= 1 else 1
    return max(t, 1)


def optimal_window(hist: np.ndarray, price_ratio: float, *,
                   smooth: float = DEFAULT_SMOOTH,
                   offsets: tuple[int, ...] | None = None) -> tuple[int, int]:
    """Jointly optimal (pre-warm offset, keep-alive length).

    The deployed policy does not merely hold the container; it *releases* it and
    brings it back just before the next invocation is expected. That is a strictly
    richer action, and optimising the length alone while giving up the offset
    concedes the comparison. So we optimise the pair.

    A container warm over ``[a, a+L)`` after the application goes quiet pays
    memory for the overlap of that window with the gap, and pays a cold start
    unless the gap ends inside the window:

        C(a, L) = m · Σ_g P(g)·max(0, min(a+L, g) − a)  +  K · P(g < a or g ≥ a+L)

    Minimised over a small grid of offsets (each an O(range) sweep over ``L``),
    which keeps the whole thing cheap enough to recompute online.

    ``smooth`` is total pseudo-mass spread over the support, not a per-bin count;
    see :func:`optimal_keepalive` for why that distinction mattered.
    """
    R = len(hist) - 1
    h = np.asarray(hist, dtype=np.float64) + smooth / (len(hist) or 1)
    pmf = h / h.sum()
    g = np.arange(R + 1)

    if offsets is None:
        q = np.cumsum(pmf)
        cand = {0}
        for pct in (0.01, 0.05, 0.10, 0.25, 0.50):
            cand.add(int(np.searchsorted(q, pct)))
        offsets = tuple(sorted(x for x in cand if x <= R))

    best = (0, 1, np.inf)
    for a in offsets:
        # Everything below is O(range) per offset rather than O(range^2).
        # With the window starting at a, write u = g - a for gaps beyond a. Then
        #   memory(L) = sum_u pmf[a+u] * min(L, u)
        # which telescopes: memory(L) = memory(L-1) + P(gap - a >= L).
        tail = pmf[a:]
        if tail.size < 2:
            continue
        # P(gap - a >= L) for L = 1..len(tail)-1
        surv_from_a = np.concatenate([tail[::-1].cumsum()[::-1][1:], [0.0]])
        mem = np.concatenate([[0.0], np.cumsum(surv_from_a)])
        covered = np.concatenate([[0.0], np.cumsum(tail)])
        n = min(len(mem), len(covered))
        cost = price_ratio * mem[:n] + (1.0 - covered[:n])
        if n < 2:
            continue
        L = int(np.argmin(cost[1:]) + 1)
        if cost[L] < best[2]:
            best = (a, L, float(cost[L]))
    return best[0], max(best[1], 1)


def hazard_at(hist: np.ndarray, t: int, smooth: float = 1.0) -> float:
    """The hazard rate at minute ``t`` -- reported for interpretation only."""
    h = np.asarray(hist, dtype=np.float64) + smooth / (len(hist) or 1)
    pmf = h / max(h.sum(), 1e-12)
    surv = np.maximum(1.0 - np.cumsum(pmf), 0.0)
    at_least = np.concatenate([[1.0], surv[:-1]])
    t = min(max(t, 0), len(pmf) - 1)
    return float(pmf[t] / max(at_least[t], 1e-12))


def hazard_policy(
    invocations: np.ndarray,
    *,
    eval_slice: tuple[int, int],
    price_ratio: float,
    key: str = "prev_gap+tod",
    kappa: float = 8.0,
    hist_range: int = DEFAULT_RANGE,
    n_gap_buckets: int = N_GAP_BUCKETS,
    recompute_every: int = 4,
    smooth: float = DEFAULT_SMOOTH,
    head_margin: float = 0.0,
    joint: bool = True,
    donor_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Keep-warm decisions from the cost-optimal stopping time.

    Identical bookkeeping to :func:`racrp.baselines.conditional.conditional_policy`
    — the same online, causal, partially-pooled state-conditional histogram — but
    the window is the optimal stopping time for the given price ratio instead of a
    percentile. ``key="none"`` gives the same treatment applied to the marginal,
    which is the fair comparison: it isolates *deriving the window* from
    *conditioning the distribution*.

    ``donor_hist`` is the control. Passing another application's gap histogram
    derives the window from the cost model exactly as before, but from the wrong
    distribution. The optimisation, the price ratio, the simulator and the
    matched-cost accounting are unchanged; only the evidence belongs to someone
    else. If the gain survives that, then the window sizes were simply better on
    average and nothing was learned from the application's own history.
    """
    inv = np.asarray(invocations)
    T = len(inv)
    warm = np.zeros(T, dtype=bool)
    a = np.flatnonzero(inv > 0)
    if len(a) < 2:
        return warm

    starts, gaps, closed = a[:-1], np.diff(a), a[1:]
    ng = n_gap_buckets
    prev_b = np.concatenate([[0], gap_bucket(gaps[:-1], ng)])

    if key == "none":
        cells, cell_of = 1, np.zeros(len(gaps), dtype=int)
    elif key == "prev_gap":
        cells, cell_of = ng, prev_b
    elif key == "prev_gap+tod":
        cells = ng * N_TOD_BUCKETS
        cell_of = prev_b * N_TOD_BUCKETS + tod_bucket(starts)
    else:
        raise ValueError(f"unknown key {key!r}")

    marg = np.zeros(hist_range + 1)
    cond = np.zeros((cells, hist_range + 1))
    n_marg = 0
    n_cell = np.zeros(cells)

    lo_e, hi_e = eval_slice
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
            ptr += 1

        since += 1
        if since >= recompute_every and n_marg:
            if key == "none" or ptr == 0:
                blended = marg
            else:
                if key == "prev_gap":
                    c = int(gap_bucket(gaps[ptr - 1], ng))
                else:
                    c = (int(gap_bucket(gaps[ptr - 1], ng)) * N_TOD_BUCKETS
                         + int(tod_bucket(s)))
                nb = n_cell[c]
                # partial pooling in probability space, then re-weighted to
                # counts: the hazard is a ratio of masses, so the blend has to
                # happen on the distributions, not on raw counts of different sizes
                blended = ((nb * (cond[c] / max(cond[c].sum(), 1))
                            + kappa * (marg / max(marg.sum(), 1)))
                           if nb > 0 else marg)
            if donor_hist is not None:
                blended = donor_hist
            if joint:
                # optimise the offset and the length together: releasing the
                # container and bringing it back is a strictly richer action than
                # holding from the start, and the deployed policy uses it
                pw, ka = optimal_window(blended, price_ratio, smooth=smooth)
            else:
                ka = optimal_keepalive(blended, price_ratio, smooth=smooth)
                # head_margin <= 0 means NO pre-warm. It must be set to zero
                # explicitly, not read as the 0th percentile, which is the
                # shortest gap ever seen and is routinely non-zero.
                pw = (0 if head_margin <= 0 else
                      quantiles_to_windows(hist_to_quantiles(blended),
                                           head_margin, 92.0, 10)[0])
            since = 0

        start = min(max(int(s) + pw, int(s) + 1), T)
        end = min(start + ka, T)
        if start < end:
            warm[start:end] = True
    return warm
