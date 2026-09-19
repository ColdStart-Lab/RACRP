"""The state-conditioned estimator: correctness, causality, and its limits."""
import numpy as np
import pytest

from racrp.baselines.conditional import conditional_policy, gap_bucket, tod_bucket
from racrp.sim.simulator import simulate_app
from racrp.config import MIN_PER_DAY


def test_gap_buckets_are_log_spaced_and_bounded():
    assert gap_bucket(1) == 0
    assert gap_bucket(4) == 1
    assert gap_bucket(16) == 2
    assert gap_bucket(10 ** 6) == 5          # clipped to the top bucket
    assert (gap_bucket([1, 4, 16, 64]) == np.array([0, 1, 2, 3])).all()


def test_tod_buckets_partition_the_day():
    assert tod_bucket(0) == 0
    assert tod_bucket(6 * 60) == 1
    assert tod_bucket(23 * 60) == 3
    assert tod_bucket(MIN_PER_DAY) == 0      # wraps


def test_key_none_reproduces_the_marginal_baseline():
    """The baseline must be the same code with the conditioning switched off."""
    rng = np.random.default_rng(0)
    inv = (rng.random(20000) < 0.05).astype(int)
    ev = (10000, 20000)
    a = conditional_policy(inv, eval_slice=ev, key="none")
    b = conditional_policy(inv, eval_slice=ev, key="prev_gap", kappa=1e9)
    # an infinite prior weight collapses the conditional onto the marginal
    assert a.tolist() == b.tolist()


def test_policy_never_uses_a_gap_that_has_not_closed():
    """Changing the trace only after minute c must not change any decision up to c."""
    rng = np.random.default_rng(3)
    T, cut = 20000, 15000
    inv = (rng.random(T) < 0.04).astype(int)
    inv2 = inv.copy()
    inv2[cut:] = (rng.random(T - cut) < 0.5).astype(int)
    ev = (10000, T)
    w1 = conditional_policy(inv, eval_slice=ev, key="prev_gap")
    w2 = conditional_policy(inv2, eval_slice=ev, key="prev_gap")
    assert w1[:cut].tolist() == w2[:cut].tolist()


def test_conditioning_helps_a_two_regime_application():
    """The case the marginal provably cannot represent.

    Alternating regimes: long runs of 4-minute gaps, then long runs of 128-minute
    gaps. The marginal histogram blends them into one distribution that is wrong
    in both regimes. Conditioning on the gap that just closed identifies which
    regime we are in, so it should win at comparable memory.
    """
    inv = np.zeros(40000, dtype=int)
    t = 0
    while t < 40000:
        for _ in range(40):                      # dense regime
            t += 4
            if t < 40000:
                inv[t] = 1
        for _ in range(12):                      # sparse regime
            t += 128
            if t < 40000:
                inv[t] = 1
    ev = (20000, 40000)
    marg = simulate_app(inv[ev[0]:ev[1]],
                        conditional_policy(inv, eval_slice=ev, key="none",
                                           tail_margin=92.0)[ev[0]:ev[1]], 512.0)
    cond = simulate_app(inv[ev[0]:ev[1]],
                        conditional_policy(inv, eval_slice=ev, key="prev_gap",
                                           tail_margin=92.0)[ev[0]:ev[1]], 512.0)
    # conditioning should not be worse on both axes at once
    assert not (cond.cold_starts > marg.cold_starts
                and cond.idle_gb_minutes > marg.idle_gb_minutes), (
        f"conditional dominated on both axes: {cond.as_dict()} vs {marg.as_dict()}")
    # and it should spend materially less memory for comparable cold starts
    assert cond.idle_gb_minutes < marg.idle_gb_minutes


def test_conditioning_cannot_help_a_memoryless_application():
    """Where there is no structure, conditioning must not invent any.

    Geometric (memoryless) gaps: the next gap is independent of the last, so the
    conditional carries no information and can only add variance. It must not
    produce a large spurious gain -- if it did, the gain in the real experiment
    would be an artefact of splitting data rather than of real structure.
    """
    rng = np.random.default_rng(11)
    inv = (rng.random(40000) < 0.02).astype(int)
    ev = (20000, 40000)
    marg = simulate_app(inv[ev[0]:ev[1]],
                        conditional_policy(inv, eval_slice=ev, key="none",
                                           tail_margin=92.0)[ev[0]:ev[1]], 512.0)
    cond = simulate_app(inv[ev[0]:ev[1]],
                        conditional_policy(inv, eval_slice=ev, key="prev_gap",
                                           tail_margin=92.0)[ev[0]:ev[1]], 512.0)
    ratio = cond.cold_starts / max(marg.cold_starts, 1)
    assert 0.75 < ratio < 1.35, (
        f"conditioning changed a memoryless app too much: {ratio:.2f}")


def test_partial_pooling_is_monotone_in_kappa():
    """Larger kappa must move the estimate towards the marginal, not away."""
    rng = np.random.default_rng(5)
    inv = (rng.random(30000) < 0.03).astype(int)
    ev = (15000, 30000)
    base = conditional_policy(inv, eval_slice=ev, key="none")
    diffs = []
    # kappa is measured in "equivalent observed gaps", so collapsing onto the
    # marginal needs a kappa that dwarfs an active application's per-cell counts,
    # which run into the thousands -- 1000 is not large here.
    for kap in (0.5, 8.0, 1e9):
        w = conditional_policy(inv, eval_slice=ev, key="prev_gap", kappa=kap)
        diffs.append(int((w != base).sum()))
    assert diffs[0] >= diffs[1] >= diffs[2], f"not monotone in kappa: {diffs}"
    assert diffs[2] == 0, "a huge kappa must reproduce the marginal exactly"
