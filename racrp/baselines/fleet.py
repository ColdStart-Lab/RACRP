"""Borrowing strength across applications: the retrieval policy, non-parametric.

The deployed Azure policy (Shahrad et al., ATC'20) estimates an application's
idle-time distribution from **that application's own** observed gaps. It is a
per-application episodic memory, and it degrades exactly when the application has
few gaps to learn from -- a newly deployed function, or a rarely-invoked one.

This module implements the generalisation the project argues for. For a target
application we

  1. describe it by a behavioural profile computable from whatever history is
     observable -- including, at ``H = 0``, only its deployment-time metadata;
  2. retrieve the ``k`` most similar applications from a fleet bank built on a
     disjoint set of *seen* applications;
  3. pool the retrieved applications' idle-time histograms into a **prior**;
  4. combine that prior with the target's own histogram by shrinkage:

         h_hat  =  ( n_own * h_own  +  kappa * h_prior ) / ( n_own + kappa )

     where ``n_own`` is the number of gaps the target has actually shown.

The shrinkage is the whole argument in one line. With no own history the estimate
is the fleet prior; with plenty it converges to the per-application histogram, so
**the deployed baseline is recovered as a special case** rather than replaced.
That is what makes this a generalisation and not a competitor, and it predicts
the convergence we then go and measure.

**Pooling happens in quantile space, not density space.** Averaging the
neighbours' densities produces a *mixture*, and a mixture of distributions is
strictly more dispersed than its components: if one analogue typically idles for
5 minutes and another for 200, the mixture has mass at both, so its 92nd
percentile sits near 200 and the policy holds a container warm far longer than
any individual analogue would justify. Measured, this failed badly -- the
density-pooled prior barely responded to its own cost knob at all (see
docs/EXPERIMENT_LOG.md, E4), because every percentile from the 60th to the 99th
landed in the same inflated tail.

Averaging the *quantile functions* instead -- the 1-D Wasserstein barycenter --
produces a distribution with the typical location and spread of the neighbours
rather than the union of their supports. Since the policy reads percentiles off
the estimate, this is the operation that matches how the estimate is used. The
density-pooling path is kept behind ``pool="density"`` so the comparison can be
reported rather than quietly dropped.

Nothing here is learned. It is deliberately a non-parametric estimator, so that
the mechanism can be tested without confounding it with a neural network; the
learned model is the same estimator with a learned similarity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from racrp.baselines.histogram import (
    DEFAULT_RANGE, DEFAULT_HEAD_MARGIN, DEFAULT_TAIL_MARGIN, _percentile_from_hist,
)
from racrp.data.splits import idle_gaps, app_profile, PROFILE_DIM


# probability grid on which quantile functions are represented
QGRID = np.linspace(0.0, 1.0, 501)


def hist_to_quantiles(h: np.ndarray, grid: np.ndarray = QGRID) -> np.ndarray:
    """Quantile function of a histogram, evaluated on ``grid``.

    Returns gap lengths in minutes. An empty histogram yields zeros, which the
    caller treats as "no information" rather than "gaps of length zero".
    """
    total = h.sum()
    if total <= 0:
        return np.zeros(len(grid))
    cdf = np.cumsum(h) / total
    # searchsorted on the CDF gives the inverse-CDF (quantile) function
    idx = np.searchsorted(cdf, np.clip(grid, 0.0, 1.0), side="left")
    return np.minimum(idx, len(h) - 1).astype(np.float64)


def quantiles_to_windows(q: np.ndarray, head_margin: float, tail_margin: float,
                         default_keepalive: int, grid: np.ndarray = QGRID
                         ) -> tuple[int, int]:
    """Pre-warm offset and keep-alive length read off a quantile function."""
    if not np.any(q > 0):
        return 0, default_keepalive
    pw = float(np.interp(head_margin / 100.0, grid, q))
    ka = float(np.interp(tail_margin / 100.0, grid, q))
    pw = int(max(0, round(pw)))
    return pw, int(max(1, round(ka) - pw + 1))


def gap_histogram(row: np.ndarray, window: tuple[int, int],
                  hist_range: int = DEFAULT_RANGE) -> tuple[np.ndarray, int]:
    """Histogram of observed idle times, plus the number of gaps observed."""
    gaps = idle_gaps(row, window)
    h = np.zeros(hist_range + 1, dtype=np.float64)
    if len(gaps) == 0:
        return h, 0
    inside = gaps[gaps <= hist_range]
    np.add.at(h, inside, 1.0)
    # gaps beyond the range still count as evidence, held in the top bin so the
    # tail percentile does not silently pretend the application is dense
    h[hist_range] += float((gaps > hist_range).sum())
    return h, int(len(gaps))


class FleetBank:
    """Profiles and gap histograms of the *seen* applications."""

    def __init__(self, counts: np.ndarray, rows: np.ndarray, meta: pd.DataFrame,
                 seen_pos: np.ndarray, profile_window: tuple[int, int],
                 dist_window: tuple[int, int] | None = None,
                 hist_range: int = DEFAULT_RANGE):
        """Build the fleet memory.

        Two windows, and the distinction matters:

        ``profile_window`` is how much of a seen application's history is used to
        build its **retrieval key**. It is set to the same budget the target has,
        so that keys and queries are computed the same way and are comparable. At
        a budget of zero this reduces to deployment-time metadata on both sides.

        ``dist_window`` is how much is used to estimate its **idle-gap
        distribution** -- the thing actually being borrowed. Seen applications are
        seen: the platform has watched them for the whole training period, so this
        defaults to all history before the evaluation slice.

        Conflating the two was a real bug: with both set to the target's window, a
        just-deployed target retrieved analogues that had also been observed for
        zero minutes, so the prior was empty and silently fell back to the fixed
        default. The fleet is well-observed; only the target is data-poor.
        """
        self.hist_range = hist_range
        self.seen_pos = np.asarray(seen_pos)
        self.profile_window = profile_window
        self.dist_window = dist_window if dist_window is not None else (0, profile_window[1])
        n = len(self.seen_pos)
        self.profiles = np.zeros((n, PROFILE_DIM), dtype=np.float32)
        self.hists = np.zeros((n, hist_range + 1), dtype=np.float64)
        self.n_gaps = np.zeros(n, dtype=np.int64)

        for i, p in enumerate(self.seen_pos):
            r = int(rows[p])
            self.profiles[i] = app_profile(counts, r, profile_window, meta.loc[r])
            self.hists[i], self.n_gaps[i] = gap_histogram(counts[r], self.dist_window,
                                                          hist_range)
        # standardise the profile space so no single feature dominates cosine
        self.mu = self.profiles.mean(axis=0, keepdims=True)
        self.sd = self.profiles.std(axis=0, keepdims=True) + 1e-6
        self.z = self._standardise(self.profiles)
        # normalise each application's histogram to a distribution before pooling,
        # so a very busy application does not swamp the prior
        tot = self.hists.sum(axis=1, keepdims=True)
        self.pdf = self.hists / np.maximum(tot, 1.0)
        # and precompute quantile functions, which is what pooling actually uses
        self.quant = np.stack([hist_to_quantiles(h) for h in self.hists])
        self.has_gaps = self.n_gaps > 0
        self.scale = 1.0

    def calibrate(self, k: int = 16, sample: int = 400, seed: int = 0) -> float:
        """Correct the systematic scale bias of the pooled prior.

        Retrieval turns out to rank applications usefully but to get the LEVEL
        wrong: pooling analogues produced keep-alive windows a median 3.4x longer
        than the applications actually needed. Ranking without level is not
        actionable -- the policy reads an absolute number of minutes off the
        estimate.

        The correction is a single global factor, fitted **leave-one-out on the
        seen applications only** (never on the held-out ones): for a sample of
        seen applications we build the prior from the *other* seen applications,
        compare its median quantile to the application's own, and take the median
        ratio. Storing one scalar keeps the estimator non-parametric in spirit and
        cannot overfit the evaluation set.
        """
        rng = np.random.default_rng(seed)
        idx_pool = np.flatnonzero(self.has_gaps)
        if len(idx_pool) < 10:
            self.scale = 1.0
            return self.scale
        pick = rng.choice(idx_pool, size=min(sample, len(idx_pool)), replace=False)
        ratios = []
        for i in pick:
            held = self.has_gaps.copy()
            held[i] = False                      # leave this application out
            sims = self.z @ self.z[i]
            sims[~held] = -np.inf
            kk = min(k, int(held.sum()))
            nb = np.argpartition(-sims, kk - 1)[:kk]
            w = np.exp(8.0 * (sims[nb] - sims[nb].max()))
            w /= max(w.sum(), 1e-12)
            pred = (w[:, None] * self.quant[nb]).sum(axis=0)
            pm = np.interp(0.5, QGRID, pred)
            tm = np.interp(0.5, QGRID, self.quant[i])
            if tm > 0 and pm > 0:
                ratios.append(pm / tm)
        self.scale = float(np.median(ratios)) if ratios else 1.0
        self.scale = float(np.clip(self.scale, 0.05, 20.0))
        return self.scale

    def _standardise(self, p: np.ndarray) -> np.ndarray:
        z = (p - self.mu) / self.sd
        n = np.linalg.norm(z, axis=1, keepdims=True)
        return z / np.maximum(n, 1e-8)

    def retrieve(self, profile: np.ndarray, k: int = 16
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Indices and similarities of the k most similar seen applications."""
        q = self._standardise(profile.reshape(1, -1))[0]
        sims = self.z @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    def pooled_prior(self, profile: np.ndarray, k: int = 16,
                     temperature: float = 8.0, pool: str = "quantile"
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Similarity-weighted pool of retrieved applications' gap distributions.

        ``pool="quantile"`` averages quantile functions (Wasserstein barycenter),
        which keeps the typical scale of the analogues. ``pool="density"``
        averages the densities, producing a mixture -- kept only so the two can be
        compared, since the difference between them turned out to decide whether
        the method works at all.

        Applications that showed no gaps at all contribute nothing; pooling their
        empty distributions would drag the prior towards zero.
        """
        idx, sims = self.retrieve(profile, min(k * 3, len(self.z)))
        keep = self.has_gaps[idx]
        if keep.any():
            idx, sims = idx[keep][:k], sims[keep][:k]
        else:
            idx, sims = idx[:k], sims[:k]

        w = np.exp(temperature * (sims - sims.max()))
        w /= max(w.sum(), 1e-12)

        if pool == "density":
            prior = (w[:, None] * self.pdf[idx]).sum(axis=0)
            tot = prior.sum()
            prior = (prior / tot if tot > 0
                     else np.full_like(prior, 1.0 / len(prior)))
            return hist_to_quantiles(prior), idx, sims

        q = (w[:, None] * self.quant[idx]).sum(axis=0)
        return q / getattr(self, "scale", 1.0), idx, sims


def windows_from_hist(h: np.ndarray, head_margin: float = DEFAULT_HEAD_MARGIN,
                      tail_margin: float = DEFAULT_TAIL_MARGIN,
                      default_keepalive: int = 10) -> tuple[int, int]:
    """Pre-warm offset and keep-alive length from a (possibly pooled) histogram."""
    if h.sum() <= 0:
        return 0, default_keepalive
    pw = _percentile_from_hist(h, head_margin)
    ka = _percentile_from_hist(h, tail_margin)
    return int(pw), int(max(1, ka - pw + 1))


def fleet_policy(
    invocations: np.ndarray,
    *,
    profile: np.ndarray,
    bank: FleetBank,
    history_window: tuple[int, int],
    eval_slice: tuple[int, int],
    k: int = 16,
    kappa: float = 8.0,
    head_margin: float = DEFAULT_HEAD_MARGIN,
    tail_margin: float = DEFAULT_TAIL_MARGIN,
    hist_range: int = DEFAULT_RANGE,
    default_keepalive: int = 10,
    use_own: bool = True,
    use_prior: bool = True,
    online: bool = True,
    recompute_every: int = 4,
    pool: str = "quantile",
) -> np.ndarray:
    """Keep-warm decisions for one application under the shrinkage estimator.

    ``kappa`` is the strength of the fleet prior measured in "equivalent observed
    gaps": the target's own evidence overtakes the prior once it has seen more
    than ``kappa`` gaps of its own.

    ``use_own`` / ``use_prior`` switch the two halves off, which is how the
    ablation recovers the pure ATC'20 histogram (``use_prior=False``) and the pure
    fleet prior (``use_own=False``) inside the same code path.

    ``online`` lets the target keep learning from gaps it reveals *during* the
    evaluation period -- the same causality contract as every other policy: only
    gaps that have already closed are ever used.

    ``recompute_every`` refreshes the blended windows once per that many newly
    observed gaps rather than after every one. It is applied **identically to
    every arm** (own-histogram, fleet-prior, shrinkage), so it cannot favour one
    of them; it exists because a full percentile recomputation at each of an
    active application's twenty thousand minutes would dominate the sweep.
    """
    T = len(invocations)
    warm = np.zeros(T, dtype=bool)
    lo_e, hi_e = eval_slice

    own_hist, n_own = gap_histogram(np.asarray(invocations), history_window, hist_range)
    prior_q = None
    if use_prior:
        prior_q, _, _ = bank.pooled_prior(profile, k=k, pool=pool)

    def windows(h_own: np.ndarray, n: int) -> tuple[int, int]:
        """Blend own evidence with the fleet prior IN QUANTILE SPACE, then read
        the pre-warm and keep-alive windows off the blended quantile function."""
        if not use_prior:
            return windows_from_hist(h_own, head_margin, tail_margin,
                                     default_keepalive)
        # Confidence in the estimate, in [0, 1]: how much of it comes from this
        # application's own evidence rather than from borrowed analogues.
        conf = n / (n + kappa) if (use_own and n) else 0.0

        if conf == 0.0:
            q = prior_q
        else:
            q = (n * hist_to_quantiles(h_own) + kappa * prior_q) / (n + kappa)

        _, ka = quantiles_to_windows(q, head_margin, tail_margin,
                                     default_keepalive)
        # The pre-warm offset is taken from the application's OWN evidence only.
        pw = 0
        if n > 0:
            pw, _ = quantiles_to_windows(hist_to_quantiles(h_own), head_margin,
                                         tail_margin, default_keepalive)

        # Pre-warming releases the container for the first ``pw`` minutes of an
        # idle gap, betting nothing arrives before then. That bet is well founded
        # on an application's OWN distribution -- which is why the deployed policy
        # makes it -- and reckless on a borrowed one: analogues agree about the
        # typical SCALE of a gap, not about this application's short tail, and a
        # mistimed pre-warm turns a would-be warm hit into a cold start. Measured,
        # letting the prior influence the offset made the policy spend MORE memory
        # and incur MORE cold starts at once (E5 in the experiment log), which is
        # the signature of mis-placed rather than mis-sized windows.
        #
        # So the two halves of the window are estimated from different evidence:
        # WHEN to come back is an own-data question, HOW LONG to stay is where
        # borrowed tail information helps. The offset is additionally scaled by
        # confidence so it ramps in as own evidence accumulates.
        return int(round(pw * conf)), ka

    pw, ka = windows(own_hist, n_own)

    active = np.flatnonzero(np.asarray(invocations)[lo_e:hi_e] > 0) + lo_e
    prev = None
    # seed the online update with the last active minute before evaluation begins
    pre = np.flatnonzero(np.asarray(invocations)[history_window[0]:history_window[1]] > 0)
    if len(pre):
        prev = int(pre[-1] + history_window[0])

    since = 0
    for s in active:
        if online and prev is not None:
            gap = int(s - prev)
            if gap > 0:
                own_hist[min(gap, hist_range)] += 1.0
                n_own += 1
                since += 1
                if since >= recompute_every:
                    pw, ka = windows(own_hist, n_own)
                    since = 0
        prev = int(s)

        start = min(max(s + pw, s + 1), T)
        end = min(start + ka, T)
        if start < end:
            warm[start:end] = True
    return warm
