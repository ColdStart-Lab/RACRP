"""Warm-state policies.

Every policy maps an invocation trace (and optionally a per-minute risk signal)
to a boolean ``keep_warm`` array consumed by :func:`racrp.sim.simulator.simulate_app`.

Causality rule: ``keep_warm[t]`` may depend only on invocations strictly before
``t`` (and on model outputs computed from the same). The oracle policy is the one
deliberate exception and is labelled as an upper bound, never as a method.
"""
from __future__ import annotations
import numpy as np


def fixed_keepalive(invocations: np.ndarray, keepalive: int) -> np.ndarray:
    """Classic platform default: stay warm for ``keepalive`` minutes after activity.

    ``warm[t]`` is True iff some minute in ``[t-keepalive, t-1]`` had invocations.
    A rolling-window OR, computed as a difference of cumulative sums so a full
    keep-alive sweep over thousands of apps stays cheap.
    """
    T = len(invocations)
    if keepalive <= 0:
        return np.zeros(T, dtype=bool)
    active = (np.asarray(invocations) > 0).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(active)])      # c[i] = active[:i].sum()
    t = np.arange(T)
    hi = t                                             # exclusive end: t-1 inclusive
    lo = np.maximum(t - keepalive, 0)
    return (c[hi] - c[lo]) > 0


def prewarm_and_keepalive(
    invocations: np.ndarray, prewarm: np.ndarray, keepalive: np.ndarray
) -> np.ndarray:
    """ATC'20-style two-parameter policy with *per-minute* windows.

    After an active minute at ``s`` the instance is dropped immediately and
    re-created at minute ``s + prewarm[s]``, then held for ``keepalive[s]``
    minutes. The offset is measured from ``s`` because an idle time of ``IT``
    puts the next invocation at ``s + IT``. ``prewarm[s] == 0`` degenerates to
    plain keep-alive from ``s``.
    """
    T = len(invocations)
    warm = np.zeros(T, dtype=bool)
    active_idx = np.flatnonzero(invocations > 0)
    for s in active_idx:
        pw = int(prewarm[s])
        ka = int(keepalive[s])
        lo = max(s + pw, s + 1) if pw > 0 else s + 1
        hi = lo + ka
        hi = min(hi, T)
        if lo < hi:
            warm[lo:hi] = True
    return warm


def risk_threshold(
    invocations: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    *,
    horizon: int = 5,
    min_keepalive: int = 1,
) -> np.ndarray:
    """Risk-aware policy: a high risk at minute ``s`` holds the instance warm
    through the horizon that risk refers to.

    ``risk[s]`` is the model's P(an invocation arrives in ``(s, s+horizon]``),
    computed from data strictly before ``s``. The decision that probability
    licenses is therefore about the minutes **after** ``s``, so a high risk warms
    ``[s+1, s+horizon]``.

    An earlier version set ``warm = risk >= threshold``, warming minute ``s``
    itself. That is not merely off by one, it is inert: the risk grid is populated
    only at *idle* decision minutes, while a cold start can only occur at an
    *active* minute, where the grid is zero by construction. The policy therefore
    warmed minutes that were never used and never warmed the ones that mattered,
    producing an identical cold-start count at every threshold while the memory
    cost varied. Nothing but an end-to-end run surfaces that.

    ``min_keepalive`` is a short unconditional grace period after activity, since
    a just-used container is nearly free to hold for one tick.
    """
    T = len(invocations)
    warm = np.zeros(T, dtype=bool)
    hot = np.flatnonzero(np.asarray(risk) >= threshold)
    for s in hot:
        lo = min(s + 1, T)
        hi = min(s + 1 + horizon, T)
        if lo < hi:
            warm[lo:hi] = True
    if min_keepalive > 0:
        warm |= fixed_keepalive(invocations, min_keepalive)
    return warm


def oracle(invocations: np.ndarray, horizon: int) -> np.ndarray:
    """Upper bound: keep warm only on minutes that are actually about to be used.

    Warm at ``t`` iff ``invocations[t] > 0`` -- i.e. perfect foresight with zero
    idle waste. Not achievable; used only to bound the Pareto plot.
    """
    return invocations > 0


def never_warm(invocations: np.ndarray) -> np.ndarray:
    """Scale-to-zero: every active minute after an idle one is a cold start."""
    return np.zeros(len(invocations), dtype=bool)
