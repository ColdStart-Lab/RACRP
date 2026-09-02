"""Hand-computed checks on the simulator and policies.

These are the ground truth for every number the project reports, so they are
verified against traces small enough to reason about by hand.
"""
import numpy as np
import pytest

from racrp.sim.simulator import simulate_app, cold_start_ms
from racrp.sim import policies as P


MB = 1024.0  # 1 GB app -> cold start = 500 + 1000 = 1500 ms


def test_never_warm_every_active_minute_is_cold():
    inv = np.array([1, 0, 3, 0, 0, 2])
    warm = P.never_warm(inv)
    r = simulate_app(inv, warm, MB)
    assert r.cold_starts == 3          # three active minutes, none warm
    assert r.invocations == 6
    assert r.warm_starts == 3          # 0 + 2 + 1 extra invocations in cold minutes
    assert r.idle_warm_minutes == 0
    assert r.cold_start_ms == pytest.approx(3 * cold_start_ms(MB))


def test_oracle_is_zero_cold_starts_and_zero_waste():
    inv = np.array([1, 0, 3, 0, 0, 2])
    warm = P.oracle(inv, horizon=5)
    r = simulate_app(inv, warm, MB)
    assert r.cold_starts == 0
    assert r.idle_warm_minutes == 0
    assert r.warm_starts == 6


def test_fixed_keepalive_window_semantics():
    # active at t=0 only; keepalive=2 => warm at t=1,2 and cold thereafter
    inv = np.zeros(6, dtype=int)
    inv[0] = 1
    warm = P.fixed_keepalive(inv, 2)
    assert warm.tolist() == [False, True, True, False, False, False]


def test_fixed_keepalive_saves_the_second_burst():
    inv = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1])
    #   ka=3 -> warm covers t=1..3 (saves t=3) and t=4..6 (misses t=8)
    warm = P.fixed_keepalive(inv, 3)
    r = simulate_app(inv, warm, MB)
    assert r.cold_starts == 2           # t=0 and t=8
    # warm minutes are 1,2,3(used),4,5,6 -> idle ones are 1,2,4,5,6
    assert r.idle_warm_minutes == 5


def test_idle_gb_minutes_scales_with_memory():
    inv = np.zeros(10, dtype=int); inv[0] = 1
    warm = P.fixed_keepalive(inv, 5)
    r_small = simulate_app(inv, warm, 512.0)
    r_big = simulate_app(inv, warm, 2048.0)
    assert r_big.idle_gb_minutes == pytest.approx(4 * r_small.idle_gb_minutes)
    assert r_small.idle_warm_minutes == r_big.idle_warm_minutes


def test_multiple_invocations_in_a_cold_minute_pay_one_cold_start():
    inv = np.array([100])
    r = simulate_app(inv, P.never_warm(inv), MB)
    assert r.cold_starts == 1
    assert r.warm_starts == 99
    assert r.invocations == 100
    assert r.cold_start_rate == pytest.approx(0.01)


def test_prewarm_policy_skips_the_idle_gap():
    # active at t=0, next activity at t=10. prewarm=9, keepalive=3
    inv = np.zeros(20, dtype=int); inv[0] = 1; inv[10] = 1
    pw = np.zeros(20, dtype=int); ka = np.zeros(20, dtype=int)
    pw[0] = 9; ka[0] = 3
    pw[10] = 9; ka[10] = 3
    warm = P.prewarm_and_keepalive(inv, pw, ka)
    assert warm[9] and warm[10] and warm[11]
    assert not warm[1:9].any()          # nothing paid during the known idle gap
    r = simulate_app(inv, warm, MB)
    assert r.cold_starts == 1           # only the very first invocation
    assert r.idle_warm_minutes == 3     # t=9, and t=11,12 (t=10 was used)


def test_risk_threshold_warms_the_horizon_the_risk_refers_to():
    """A high risk at minute s licenses warming s+1..s+horizon, not s itself.

    risk[s] means "an invocation arrives in (s, s+horizon]", so that is the window
    the probability is about.
    """
    inv = np.zeros(12, dtype=int)
    inv[0] = 1
    risk = np.zeros(12)
    risk[5] = 0.9
    warm = P.risk_threshold(inv, risk, threshold=0.5, horizon=3, min_keepalive=0)
    assert warm.tolist() == [False] * 6 + [True, True, True] + [False] * 3


def test_risk_threshold_can_actually_prevent_a_cold_start():
    """The bug this guards against was inert rather than merely wrong.

    Risk is only ever populated at IDLE decision minutes, while a cold start can
    only occur at an ACTIVE minute. An implementation that warms minute s itself
    therefore warms minutes that are never used and never warms the ones that
    matter -- giving an identical cold-start count at every threshold while the
    memory cost varies. The policy must be able to convert a high risk into a
    saved cold start.
    """
    inv = np.zeros(20, dtype=int)
    inv[0] = 1
    inv[10] = 1                       # the invocation we want to protect
    risk = np.zeros(20)
    risk[8] = 0.9                     # an idle minute, predicting the arrival
    warm = P.risk_threshold(inv, risk, threshold=0.5, horizon=5, min_keepalive=1)
    assert warm[10], "a high risk before the arrival must warm the arrival minute"
    saved = simulate_app(inv, warm, MB)
    unwarmed = simulate_app(inv, P.risk_threshold(inv, np.zeros(20), 0.5,
                                                  horizon=5, min_keepalive=1), MB)
    assert saved.cold_starts < unwarmed.cold_starts, (
        "the risk signal did not prevent any cold start")


def test_risk_threshold_responds_to_the_threshold():
    """Cold starts must actually change as the threshold sweeps."""
    rng = np.random.default_rng(0)
    T = 2000
    inv = (rng.random(T) < 0.03).astype(int)
    risk = rng.random(T)
    counts = [simulate_app(inv, P.risk_threshold(inv, risk, thr, horizon=5,
                                                 min_keepalive=1), MB).cold_starts
              for thr in (0.1, 0.5, 0.9)]
    assert len(set(counts)) > 1, (
        f"cold starts identical across thresholds ({counts}): the policy is inert")


def test_causality_keep_warm_at_t_cannot_use_invocations_at_t():
    """fixed_keepalive must not peek at the current minute."""
    inv = np.zeros(5, dtype=int); inv[3] = 1
    warm = P.fixed_keepalive(inv, 10)
    assert not warm[3], "policy warmed the very minute it first saw traffic"
    assert warm[4], "instance should survive the minute after activity"


def test_vectorised_simulator_matches_the_reference_loop():
    """The fast path must be numerically identical to the obvious loop."""
    from racrp.sim.simulator import simulate_app_reference
    rng = np.random.default_rng(0)
    for trial in range(50):
        T = int(rng.integers(20, 400))
        inv = (rng.random(T) < rng.uniform(0.02, 0.4)) * rng.integers(1, 40, T)
        warm = rng.random(T) < rng.uniform(0.0, 0.9)
        mem = float(rng.uniform(64, 3000))
        a = simulate_app(inv, warm, mem).as_dict()
        b = simulate_app_reference(inv, warm, mem).as_dict()
        for k in a:
            assert a[k] == pytest.approx(b[k]), f"trial {trial}, field {k}"


def test_vectorised_keepalive_matches_naive_definition():
    rng = np.random.default_rng(1)
    for _ in range(30):
        T = int(rng.integers(20, 300))
        inv = (rng.random(T) < 0.15).astype(int)
        k = int(rng.integers(0, 30))
        got = P.fixed_keepalive(inv, k)
        want = np.zeros(T, dtype=bool)
        for t in range(T):
            lo = max(t - k, 0)
            want[t] = bool(inv[lo:t].sum() > 0)
        assert got.tolist() == want.tolist()
