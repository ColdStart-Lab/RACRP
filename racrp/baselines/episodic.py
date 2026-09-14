"""Episodic memory over an application's own past: conditional idle-gap estimation.

The principal weakness of the deployed policy
---------------------------------------------
The ATC'20 hybrid histogram estimates one idle-time distribution per application
by pooling **every** gap that application has ever shown. That is a *marginal*
distribution, and it throws away the context each gap occurred in. An application
that idles two minutes during business hours and six hours overnight is described
by a single blended histogram that is wrong in both regimes: its 99th percentile
is dominated by the overnight gaps, so the daytime keep-alive is far too long,
while its 5th percentile is set by daytime traffic, so the overnight pre-warm
fires far too early.

What the scheduler actually needs is the **conditional** distribution

    P( next idle gap | it is Monday 09:00 and the last hour looked like this )

and that is precisely what an episodic memory can supply. For a decision at
minute ``s`` we retrieve the application's **own** past episodes that occurred in
a similar context -- similar time of day, similar day type, similar recent
activity shape -- and use the gaps that actually followed *those* moments.

This is the mechanism the project set out to test, stated correctly. The earlier
experiments tested transfer *across applications*, which is a different claim and
which failed for reasons documented in the experiment log. Retrieval here is
*within* an application and *across time*, which is what "I have seen this
pattern before" actually means for a workload.

Relationship to the baseline
----------------------------
The estimate is partially pooled towards the application's marginal histogram:

    h_hat( g | c )  =  ( n_c * h_c  +  kappa * h_marginal ) / ( n_c + kappa )

where ``n_c`` is how many past episodes matched the context closely enough. With
no matching context the estimate **is** the ATC'20 histogram, so the deployed
policy is recovered exactly as a special case rather than replaced -- and the
partial pooling is also the "cleaning" that stops a thin, noisy conditional
estimate from being trusted before it has earned it.

Causality
---------
At a decision made after the active minute ``s``, only episodes whose gap had
already *closed* strictly before ``s`` are visible. Nothing about the current gap,
or any later one, can enter the retrieval.
"""
from __future__ import annotations

import numpy as np

from racrp.baselines.histogram import DEFAULT_RANGE, DEFAULT_HEAD_MARGIN, DEFAULT_TAIL_MARGIN
from racrp.baselines.fleet import hist_to_quantiles, quantiles_to_windows

MIN_PER_DAY = 1440

# how far back the recent-activity signature looks, in minutes
SIG_SCALES = (5, 15, 60, 240, 1440)


def context_vectors(inv: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Context of each minute in ``at``: when it is, and what just happened.

    Returns float32 ``(len(at), D)``. Every component is computable from minutes
    strictly at or before ``at``, so the key never contains future information.
    """
    at = np.asarray(at, dtype=np.int64)
    tod = (at % MIN_PER_DAY) / MIN_PER_DAY
    dow = (at // MIN_PER_DAY) % 7

    feats = [
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float64),          # weekend indicator
    ]

    # recent-activity signature: how busy the application has been at several
    # scales ending at this minute. This is what distinguishes "quiet 09:00"
    # from "surging 09:00".
    active = (inv > 0).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(active)])
    rate = np.concatenate([[0.0], np.cumsum(inv.astype(np.float64))])
    for w in SIG_SCALES:
        lo = np.maximum(at - w + 1, 0)
        hi = at + 1
        feats.append(np.log1p((csum[hi] - csum[lo]) / w * 100.0))
        feats.append(np.log1p((rate[hi] - rate[lo]) / w))
    return np.stack(feats, axis=1).astype(np.float32)


CONTEXT_DIM = 5 + 2 * len(SIG_SCALES)


class EpisodicMemory:
    """An application's own (context -> observed gap) episodes, queried by context.

    ``scales`` standardises the context space so no single component dominates the
    neighbourhood; it is computed from the training portion only.
    """

    def __init__(self, inv: np.ndarray, hist_range: int = DEFAULT_RANGE):
        self.inv = np.asarray(inv)
        self.hist_range = hist_range
        self.active = np.flatnonzero(self.inv > 0)
        if len(self.active) >= 2:
            self.starts = self.active[:-1]
            self.gaps = np.diff(self.active).astype(np.int64)
            # an episode only becomes usable once its gap has closed
            self.closed_at = self.active[1:]
        else:
            self.starts = np.empty(0, dtype=np.int64)
            self.gaps = np.empty(0, dtype=np.int64)
            self.closed_at = np.empty(0, dtype=np.int64)
        self.ctx = (context_vectors(self.inv, self.starts)
                    if len(self.starts) else np.zeros((0, CONTEXT_DIM), np.float32))

    def standardise_from(self, upto_minute: int) -> None:
        """Fit the context scaling on episodes closed before ``upto_minute``."""
        m = self.closed_at < upto_minute
        ref = self.ctx[m] if m.any() else self.ctx
        if len(ref) == 0:
            self.mu = np.zeros(CONTEXT_DIM, np.float32)
            self.sd = np.ones(CONTEXT_DIM, np.float32)
        else:
            self.mu = ref.mean(axis=0)
            self.sd = ref.std(axis=0) + 1e-3
        self.z = (self.ctx - self.mu) / self.sd

    def marginal(self, before: int) -> tuple[np.ndarray, int]:
        """The ATC'20 estimate: all gaps closed before ``before``."""
        m = self.closed_at < before
        h = np.zeros(self.hist_range + 1)
        if not m.any():
            return h, 0
        g = np.minimum(self.gaps[m], self.hist_range)
        np.add.at(h, g, 1.0)
        return h, int(m.sum())

    def conditional(self, query_minute: int, before: int, k: int = 24,
                    max_candidates: int = 4000
                    ) -> tuple[np.ndarray, int, float]:
        """Histogram of gaps that followed the ``k`` most similar past contexts.

        Returns ``(histogram, n_used, mean_similarity)``. Only episodes closed
        strictly before ``before`` are eligible.
        """
        h = np.zeros(self.hist_range + 1)
        m = np.flatnonzero(self.closed_at < before)
        if len(m) == 0:
            return h, 0, 0.0
        if len(m) > max_candidates:      # bound the cost on very active apps
            m = m[-max_candidates:]

        q = (context_vectors(self.inv, np.array([query_minute]))[0] - self.mu) / self.sd
        d = np.linalg.norm(self.z[m] - q[None, :], axis=1)
        kk = min(k, len(m))
        sel = np.argpartition(d, kk - 1)[:kk]
        idx = m[sel]
        dd = d[sel]

        # distance -> weight, with a scale set by the neighbourhood itself so the
        # weighting is meaningful for both dense and sparse applications
        scale = np.median(dd) + 1e-6
        w = np.exp(-dd / scale)
        g = np.minimum(self.gaps[idx], self.hist_range)
        np.add.at(h, g, w)
        return h, int(kk), float(np.mean(np.exp(-dd / scale)))


def episodic_policy(
    invocations: np.ndarray,
    *,
    eval_slice: tuple[int, int],
    warmup_end: int,
    k: int = 24,
    kappa: float = 6.0,
    head_margin: float = DEFAULT_HEAD_MARGIN,
    tail_margin: float = DEFAULT_TAIL_MARGIN,
    default_keepalive: int = 10,
    hist_range: int = DEFAULT_RANGE,
    use_context: bool = True,
    recompute_every: int = 1,
    prewarm_from: str = "marginal",
) -> np.ndarray:
    """Keep-warm decisions from a context-conditional idle-gap estimate.

    ``use_context=False`` reduces this to the ATC'20 marginal histogram through
    the identical code path, which is what makes the comparison an ablation of the
    conditioning rather than a comparison of two implementations.

    ``prewarm_from="marginal"`` estimates the pre-warm offset from the marginal
    histogram even when the keep-alive length comes from the conditional. The two
    halves of the window need different amounts of data: the offset is a 5th
    percentile, which a 24-neighbour conditional estimates with essentially its
    minimum sample and therefore very noisily, while the keep-alive length is a
    tail quantity where conditioning pays. Estimating both from the conditional
    reproduced the failure signature seen earlier -- fewer GB-minutes spent and
    MORE cold starts, i.e. windows in the wrong place rather than the wrong size.
    """
    inv = np.asarray(invocations)
    T = len(inv)
    warm = np.zeros(T, dtype=bool)
    mem = EpisodicMemory(inv, hist_range)
    if len(mem.starts) == 0:
        return warm
    mem.standardise_from(warmup_end)

    lo_e, hi_e = eval_slice
    active_eval = mem.active[(mem.active >= lo_e) & (mem.active < hi_e)]
    if len(active_eval) == 0:
        return warm

    pw, ka = 0, default_keepalive
    since = recompute_every
    for s in active_eval:
        since += 1
        if since >= recompute_every:
            h_marg, n_marg = mem.marginal(int(s))
            if use_context:
                h_ctx, n_ctx, _ = mem.conditional(int(s), int(s), k=k)
                if n_ctx > 0 and h_ctx.sum() > 0:
                    # partial pooling: trust the conditional in proportion to how
                    # much matched context supports it
                    q_ctx = hist_to_quantiles(h_ctx)
                    q_marg = (hist_to_quantiles(h_marg) if n_marg
                              else np.zeros_like(q_ctx))
                    n_eff = h_ctx.sum()
                    q = ((n_eff * q_ctx + kappa * q_marg) / (n_eff + kappa)
                         if n_marg else q_ctx)
                    _, ka = quantiles_to_windows(q, head_margin, tail_margin,
                                                 default_keepalive)
                    if prewarm_from == "marginal" and n_marg:
                        pw, _ = quantiles_to_windows(
                            hist_to_quantiles(h_marg), head_margin, tail_margin,
                            default_keepalive)
                    else:
                        pw, _ = quantiles_to_windows(q, head_margin, tail_margin,
                                                     default_keepalive)
                else:
                    pw, ka = quantiles_to_windows(hist_to_quantiles(h_marg),
                                                  head_margin, tail_margin,
                                                  default_keepalive)
            else:
                pw, ka = quantiles_to_windows(hist_to_quantiles(h_marg),
                                              head_margin, tail_margin,
                                              default_keepalive)
            since = 0

        start = min(max(int(s) + pw, int(s) + 1), T)
        end = min(start + ka, T)
        if start < end:
            warm[start:end] = True
    return warm
