"""Minute-tick serverless simulator.

Model (faithful to the granularity of the Azure 2019 trace and to how ATC'20 /
FaasCache / IceBreaker evaluate keep-alive policies):

  * Resources are allocated per **application**, so each app has at most one
    warm instance; concurrency within a minute is aggregated.
  * At minute t the app receives ``inv[t]`` invocations.
      - if the app is warm at the start of t  -> all of them are warm starts
      - otherwise                             -> the app must be brought up:
        one cold start is charged, the remaining ``inv[t]-1`` invocations in that
        minute reuse the now-warm instance.
  * After an active minute the instance stays warm while the policy says so.
  * A policy may also **pre-warm**: declare the app warm at minute t before any
    invocation arrives, paying idle memory-time for the privilege.

Accounting:
  * ``cold_starts``      - number of cold starts charged
  * ``cold_start_ms``    - total latency penalty inflicted on users
  * ``warm_minutes``     - minutes an instance existed
  * ``idle_warm_minutes``- minutes an instance existed with zero invocations
                           (this is the waste the policy pays for)
  * ``gb_minutes``       - idle_warm_minutes weighted by app memory
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from racrp.config import (
    COLD_START_BASE_MS,
    COLD_START_PER_GB_MS,
    USD_PER_GB_SECOND,
)


def cold_start_ms(memory_mb: float) -> float:
    """Latency of a cold start for an app of the given size."""
    return COLD_START_BASE_MS + COLD_START_PER_GB_MS * (memory_mb / 1024.0)


@dataclass
class SimResult:
    cold_starts: int = 0
    warm_starts: int = 0
    invocations: int = 0
    active_minutes: int = 0
    cold_start_ms: float = 0.0
    warm_minutes: int = 0
    idle_warm_minutes: int = 0
    gb_minutes: float = 0.0
    idle_gb_minutes: float = 0.0
    per_app: dict = field(default_factory=dict)

    @property
    def cold_start_rate(self) -> float:
        """Fraction of invocations that paid a cold start."""
        return self.cold_starts / self.invocations if self.invocations else 0.0

    @property
    def cold_start_rate_per_active_minute(self) -> float:
        return self.cold_starts / self.active_minutes if self.active_minutes else 0.0

    @property
    def mean_added_latency_ms(self) -> float:
        """Cold-start latency amortised over every invocation served."""
        return self.cold_start_ms / self.invocations if self.invocations else 0.0

    @property
    def wasted_usd(self) -> float:
        return self.idle_gb_minutes * 60.0 * USD_PER_GB_SECOND

    def as_dict(self) -> dict:
        return {
            "cold_starts": self.cold_starts,
            "warm_starts": self.warm_starts,
            "invocations": self.invocations,
            "active_minutes": self.active_minutes,
            "cold_start_rate": self.cold_start_rate,
            "cold_start_rate_per_active_minute": self.cold_start_rate_per_active_minute,
            "cold_start_ms": self.cold_start_ms,
            "mean_added_latency_ms": self.mean_added_latency_ms,
            "warm_minutes": self.warm_minutes,
            "idle_warm_minutes": self.idle_warm_minutes,
            "gb_minutes": self.gb_minutes,
            "idle_gb_minutes": self.idle_gb_minutes,
            "wasted_usd": self.wasted_usd,
        }

    def merge(self, other: "SimResult") -> "SimResult":
        for f in (
            "cold_starts", "warm_starts", "invocations", "active_minutes",
            "cold_start_ms", "warm_minutes", "idle_warm_minutes",
            "gb_minutes", "idle_gb_minutes",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.per_app.update(other.per_app)
        return self


def simulate_app(
    invocations: np.ndarray,
    keep_warm: np.ndarray,
    memory_mb: float,
    *,
    trace_decisions: bool = False,
):
    """Simulate one app over ``T`` minutes (vectorised).

    Parameters
    ----------
    invocations : int array (T,)
        Invocation count per minute.
    keep_warm : bool array (T,)
        ``keep_warm[t]`` is the policy's decision, made **before** minute ``t``'s
        invocations are observed, to hold an instance warm through minute ``t``.
    memory_mb : float
        Memory footprint of the app.

    Because ``keep_warm`` is precomputed from the (exogenous) invocation trace,
    no minute's outcome feeds back into a later decision, so the whole
    simulation reduces to array algebra. :func:`simulate_app_reference` is the
    plain loop this must agree with, and the test suite checks that it does.

    Returns
    -------
    SimResult, and (if ``trace_decisions``) a per-minute state array where
    0 = idle-cold, 1 = idle-warm (wasted), 2 = warm hit, 3 = cold start.
    """
    inv = np.asarray(invocations)
    warm = np.asarray(keep_warm, dtype=bool)
    assert len(warm) == len(inv), "keep_warm must cover every minute"

    gb = memory_mb / 1024.0
    cs_ms = cold_start_ms(memory_mb)

    active = inv > 0
    cold = active & ~warm            # active minute with no instance -> cold start
    hot = active & warm

    res = SimResult()
    res.invocations = int(inv.sum())
    res.active_minutes = int(active.sum())
    res.cold_starts = int(cold.sum())
    # every invocation except the one that triggered each cold start is warm
    res.warm_starts = res.invocations - res.cold_starts
    res.cold_start_ms = float(res.cold_starts * cs_ms)

    # an instance exists on every active minute (created if need be) and on every
    # minute the policy paid to keep it
    exists = warm | active
    res.warm_minutes = int(exists.sum())
    res.idle_warm_minutes = int((warm & ~active).sum())
    res.gb_minutes = float(res.warm_minutes * gb)
    res.idle_gb_minutes = float(res.idle_warm_minutes * gb)

    if trace_decisions:
        states = np.zeros(len(inv), dtype=np.int8)
        states[warm & ~active] = 1
        states[hot] = 2
        states[cold] = 3
        return res, states
    return res


def simulate_app_reference(invocations, keep_warm, memory_mb):
    """Plain per-minute loop. Slow, obvious, and the ground truth the vectorised
    implementation is tested against."""
    T = len(invocations)
    gb = memory_mb / 1024.0
    cs_ms = cold_start_ms(memory_mb)
    res = SimResult()
    for t in range(T):
        warm = bool(keep_warm[t])
        n = int(invocations[t])
        if n > 0:
            res.invocations += n
            res.active_minutes += 1
            if warm:
                res.warm_starts += n
            else:
                res.cold_starts += 1
                res.cold_start_ms += cs_ms
                res.warm_starts += n - 1
            res.warm_minutes += 1
            res.gb_minutes += gb
        elif warm:
            res.warm_minutes += 1
            res.idle_warm_minutes += 1
            res.gb_minutes += gb
            res.idle_gb_minutes += gb
    return res
