"""The ATC'20 baseline must behave like the policy described in the paper."""
import numpy as np
import pytest

from racrp.baselines.histogram import (
    hybrid_histogram_policy, _percentile_from_hist, _ar_forecast,
)
from racrp.sim.simulator import simulate_app


def test_percentile_from_histogram():
    h = np.zeros(11, dtype=np.int64)
    h[[2, 5, 9]] = [50, 30, 20]           # 50% at 2, 30% at 5, 20% at 9
    assert _percentile_from_hist(h, 50) == 2
    assert _percentile_from_hist(h, 75) == 5
    assert _percentile_from_hist(h, 99) == 9


def test_perfectly_periodic_app_is_learned_and_prewarmed():
    """An app invoked exactly every 20 minutes is the easy case: the histogram
    should concentrate at 20 and the policy should stop paying for the gap."""
    T = 4000
    inv = np.zeros(T, dtype=int)
    inv[::20] = 1
    warm = hybrid_histogram_policy(inv, recompute_every=1)
    r = simulate_app(inv, warm, 1024.0)
    n_invocations = int((inv > 0).sum())
    # after the warm-up it should catch nearly every invocation ...
    assert r.cold_starts < 0.1 * n_invocations
    # ... without holding the instance through the whole idle gap
    assert r.idle_warm_minutes < 0.5 * T


def test_sparse_app_uses_the_forecast_branch_not_a_permanent_warm_instance():
    """Idle times far beyond the 240-min histogram range trigger the forecast
    fallback. It should place a band around the predicted next invocation --
    catching most of them -- without degenerating into keeping the app warm
    through the whole gap, which is what an unbounded keep-alive would do."""
    T = 20000
    inv = np.zeros(T, dtype=int)
    inv[::1500] = 1                       # gap of 1500 min >> 240-min range
    warm = hybrid_histogram_policy(inv)
    r = simulate_app(inv, warm, 1024.0)
    n_inv = int((inv > 0).sum())
    always_warm_cost = T - n_inv
    assert r.cold_starts <= 0.4 * n_inv, "forecast branch should catch most invocations"
    assert r.idle_warm_minutes < 0.5 * always_warm_cost,         "policy should cost far less than simply staying warm"


def test_policy_never_looks_at_the_future():
    """Changing the trace only after minute t must not change any decision up to t."""
    rng = np.random.default_rng(3)
    T = 3000
    inv = (rng.random(T) < 0.05).astype(int)
    cut = 1500
    inv2 = inv.copy()
    inv2[cut:] = (rng.random(T - cut) < 0.5).astype(int)
    w1 = hybrid_histogram_policy(inv)
    w2 = hybrid_histogram_policy(inv2)
    assert w1[:cut + 1].tolist() == w2[:cut + 1].tolist()


def test_per_invocation_refresh_is_the_stronger_configuration():
    """The default (refresh after every idle-time sample, as in the paper) must
    not be worse than a lazily-refreshed variant -- we run the baseline in its
    strongest configuration so that beating it means something."""
    rng = np.random.default_rng(5)
    inv = (rng.random(6000) < 0.03).astype(int)
    eager = simulate_app(inv, hybrid_histogram_policy(inv, recompute_every=1), 512.0)
    lazy = simulate_app(inv, hybrid_histogram_policy(inv, recompute_every=32), 512.0)
    assert eager.cold_starts <= lazy.cold_starts


def test_warmup_slice_carries_history_into_the_evaluation_period():
    T = 6000
    inv = np.zeros(T, dtype=int)
    inv[::30] = 1
    cold = hybrid_histogram_policy(inv)
    warmed = hybrid_histogram_policy(inv, warmup_slice=(0, 3000))
    # with history preloaded the policy is already tuned at the start of the
    # evaluation window, so it should do no worse there
    r_cold = simulate_app(inv[3000:], cold[3000:], 512.0)
    r_warm = simulate_app(inv[3000:], warmed[3000:], 512.0)
    assert r_warm.cold_starts <= r_cold.cold_starts + 1
