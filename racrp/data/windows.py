"""Situations, features and cold-start risk labels.

A **situation** is a pair ``(app a, minute t)``. It carries everything the
scheduler knows at the instant it must decide whether to keep app ``a`` warm
through minute ``t+1``:

  * the invocation history ``counts[a, t-W+1 : t+1]``  (never anything after t),
  * cyclical time-of-day / day-of-week encodings of ``t``,
  * static per-app descriptors (trigger, memory, duration, idle-time profile).

The **label** for horizon ``dt`` is

    y = 1  iff  sum(counts[a, t+1 : t+1+dt]) > 0

i.e. "if the instance is dropped now, an invocation arrives within dt minutes and
pays a cold start". This is policy-independent, which is what makes it a
prediction target rather than a policy evaluation.

Windows are produced on the fly from the shared count matrix; materialising them
would cost tens of GB for no benefit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from racrp.config import (
    MIN_PER_DAY, WINDOW, HORIZONS, PRIMARY_HORIZON,
    TRAIN_SLICE, VAL_SLICE, TEST_SLICE, OUTCOME_LEN,
)
from racrp.data.trace import TRIGGERS, idle_times

# multi-scale summaries of the raw window, in minutes
SCALES = (5, 15, 60, 240)


def static_features(meta: pd.DataFrame, rows: np.ndarray,
                    counts: np.ndarray, train_slice: tuple[int, int],
                    *, history_minutes: int | None = None,
                    history_end: int | None = None) -> np.ndarray:
    """Per-app descriptors. Behavioural statistics use observable history only.

    Returns float32 (n_rows, D_STATIC).

    **This function is where the first version of the experiment went wrong.** The
    behavioural block below (idle-time quantiles, duty cycle, invocation rate) is
    a compressed per-application memory. Computing it over the full training
    period hands every application its own memory for free, which makes a
    *shared* memory redundant by construction -- and that is exactly the null
    result we measured before diagnosing it.

    ``history_minutes`` limits how much of an application's own past may be used:
    statistics are computed over the ``history_minutes`` immediately preceding
    ``history_end`` (default: the start of the training slice's successor, i.e.
    the evaluation boundary). ``history_minutes=0`` models a function on the day
    it is deployed -- the behavioural block is zeroed and only deployment-time
    metadata (trigger, memory, duration, function count) survives, which is
    precisely the information a platform actually has at that moment.
    """
    if history_minutes is None:
        lo, hi = train_slice
    else:
        hi = train_slice[1] if history_end is None else history_end
        lo = max(hi - history_minutes, 0)
    observed = hi > lo
    sub = meta.loc[rows]
    trig = np.zeros((len(rows), len(TRIGGERS)), dtype=np.float32)
    for j, t in enumerate(TRIGGERS):
        trig[:, j] = (sub["trigger"].to_numpy() == t).astype(np.float32)

    feats = [trig]
    log = lambda x: np.log1p(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0))
    feats.append(log(sub["memory_mb"]).reshape(-1, 1))
    feats.append(log(sub["avg_duration_ms"]).reshape(-1, 1))
    feats.append(log(sub["p99_duration_ms"]).reshape(-1, 1))
    feats.append(log(sub["n_functions"]).reshape(-1, 1))

    # ---- behavioural block: the per-application memory, from observable history
    it_stats = np.zeros((len(rows), 5), dtype=np.float64)
    duty = np.zeros(len(rows))
    rate = np.zeros(len(rows))
    if observed:
        for i, r in enumerate(rows):
            its = idle_times(counts[r, lo:hi])
            if len(its):
                it_stats[i] = [
                    np.log1p(np.median(its)),
                    np.log1p(np.percentile(its, 5)),
                    np.log1p(np.percentile(its, 95)),
                    np.log1p(its.mean()),
                    np.log1p(len(its)),
                ]
        span = hi - lo
        duty = (counts[rows, lo:hi] > 0).sum(axis=1) / span
        rate = counts[rows, lo:hi].sum(axis=1) / span
    feats.append(it_stats)
    feats.append(duty.reshape(-1, 1))
    feats.append(log(rate).reshape(-1, 1))

    return np.concatenate([np.asarray(f, dtype=np.float32) for f in feats], axis=1)


D_STATIC = len(TRIGGERS) + 4 + 5 + 2   # 18


def time_features(t: np.ndarray) -> np.ndarray:
    """Cyclical encodings of absolute minute index. float32 (n, 6)."""
    tod = (t % MIN_PER_DAY) / MIN_PER_DAY
    dow = ((t // MIN_PER_DAY) % 7) / 7.0
    woy = (t // MIN_PER_DAY) / 14.0
    return np.stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
        woy, np.ones_like(tod),
    ], axis=1).astype(np.float32)


D_TIME = 6


def make_labels(counts: np.ndarray, rows: np.ndarray,
                horizons=HORIZONS) -> dict[int, np.ndarray]:
    """``labels[dt][i, t] = 1`` iff app rows[i] is invoked in ``(t, t+dt]``.

    Implemented as a suffix cumulative sum so all horizons are O(T) each.
    """
    active = (counts[rows] > 0).astype(np.int32)
    T = active.shape[1]
    csum = np.concatenate([np.zeros((len(rows), 1), np.int64),
                           np.cumsum(active, axis=1, dtype=np.int64)], axis=1)
    out = {}
    for dt in horizons:
        hi = np.minimum(np.arange(1, T + 1) + dt, T)
        lab = (csum[:, hi] - csum[:, 1:T + 1]) > 0
        out[dt] = lab.astype(np.int8)
    return out


def decision_mask(counts: np.ndarray, rows: np.ndarray,
                  window: int = WINDOW) -> np.ndarray:
    """Minutes where a keep-warm decision is non-trivial.

    We require (a) a full history window is available, and (b) the app is idle at
    ``t`` -- when an invocation is being served the instance is warm regardless of
    what any policy decides, so those minutes would only inflate the metrics.
    """
    m = counts[rows] == 0
    m[:, :window] = False
    return m


class SituationDataset(Dataset):
    """Indexes into the shared count matrix; never materialises the windows.

    Yields ``(hist, tfeat, sfeat, y, index)`` where ``hist`` is the log1p
    invocation window ordered oldest -> newest.
    """

    def __init__(self, counts: np.ndarray, rows: np.ndarray, meta: pd.DataFrame,
                 time_slice: tuple[int, int], *, horizon: int = PRIMARY_HORIZON,
                 stride: int = 1, window: int = WINDOW,
                 decision_points_only: bool = True,
                 train_slice: tuple[int, int] = TRAIN_SLICE,
                 max_samples: int | None = None, seed: int = 0,
                 history_minutes: int | None = None,
                 history_end: int | None = None,
                 state_features: bool = False):
        self.counts = counts
        self.rows = np.asarray(rows)
        self.window = window
        self.horizon = horizon

        self.sfeat = static_features(meta, self.rows, counts, train_slice,
                                     history_minutes=history_minutes,
                                     history_end=history_end)
        self.labels = make_labels(counts, self.rows, (horizon,))[horizon]

        lo, hi = time_slice
        lo = max(lo, window)
        # a label at t needs counts up to t+horizon
        hi = min(hi, counts.shape[1] - horizon)

        mask = np.zeros((len(self.rows), counts.shape[1]), dtype=bool)
        mask[:, lo:hi] = True
        if decision_points_only:
            mask &= decision_mask(counts, self.rows, window)
        else:
            mask[:, :window] = False
        if stride > 1:
            keep = np.zeros(counts.shape[1], dtype=bool)
            keep[lo:hi:stride] = True
            mask &= keep

        ai, ti = np.nonzero(mask)
        if max_samples is not None and len(ai) > max_samples:
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(ai), size=max_samples, replace=False)
            sel.sort()
            ai, ti = ai[sel], ti[sel]
        self.app_idx = ai.astype(np.int32)
        self.t_idx = ti.astype(np.int32)
        self.tfeat_all = None   # computed lazily per item

        # Explicit renewal state. The 240-minute window cannot represent a gap
        # longer than itself, so without these the model is blind to the most
        # informative variable in the problem at exactly the decision points
        # where the window has saturated. See racrp/data/statefeat.py.
        self.state = None
        if state_features:
            from racrp.data.statefeat import state_features as _sf
            self.state = _sf(counts, self.rows, self.app_idx, self.t_idx)

    def __len__(self) -> int:
        return len(self.app_idx)

    def positive_rate(self) -> float:
        return float(self.labels[self.app_idx, self.t_idx].mean())

    def __getitem__(self, i: int):
        a = int(self.app_idx[i]); t = int(self.t_idx[i])
        r = int(self.rows[a])
        hist = self.counts[r, t - self.window + 1: t + 1].astype(np.float32)
        hist = np.log1p(hist)
        tf = time_features(np.array([t]))[0]
        sf = self.sfeat[a]
        y = float(self.labels[a, t])
        return (torch.from_numpy(hist),
                torch.from_numpy(tf),
                torch.from_numpy(sf),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(i, dtype=torch.long))


def multiscale(hist: torch.Tensor, scales=SCALES) -> torch.Tensor:
    """Mean-pool the raw window at several resolutions -> (B, len(scales)+1, W).

    Gives the encoder an explicit view of both the last few minutes and the last
    few hours without making it learn long-range pooling from scratch.
    """
    B, W = hist.shape
    chans = [hist]
    for s in scales:
        pad = torch.nn.functional.pad(hist.unsqueeze(1), (s - 1, 0), mode="replicate")
        chans.append(torch.nn.functional.avg_pool1d(pad, kernel_size=s, stride=1).squeeze(1))
    return torch.stack(chans, dim=1)


def outcome_profile(counts: np.ndarray, row: int, t: int,
                    length: int = OUTCOME_LEN) -> np.ndarray:
    """What actually happened after minute ``t``: binary activity for the next
    ``length`` minutes. This is the *value* a retrieved memory contributes -- the
    episode's consequence, not just its shape."""
    seg = counts[row, t + 1: t + 1 + length]
    out = np.zeros(length, dtype=np.float32)
    out[:len(seg)] = (seg > 0).astype(np.float32)
    return out
