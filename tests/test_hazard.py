"""The derived stopping rule: does it minimise what it claims to minimise?"""
import numpy as np
import pytest

from racrp.baselines.hazard import (
    optimal_keepalive, optimal_window, hazard_policy, hazard_at,
)
from racrp.sim.simulator import simulate_app
from racrp.config import TEST_SLICE


def _hist(pairs, n=241):
    h = np.zeros(n)
    for g, c in pairs:
        h[min(g, n - 1)] += c
    return h


def brute_force_optimum(hist, ratio, smooth=1.0):
    """The same objective, evaluated the obvious way.

    ``smooth`` is total pseudo-mass spread over the support, matching the
    estimator: a per-bin count would be a 20-30% uniform prior on a 241-minute
    support and is what the estimator was fixed not to do.
    """
    h = np.asarray(hist, float) + smooth / len(hist)
    pmf = h / h.sum()
    best, best_c = 1, None
    for T in range(1, len(pmf)):
        mem = sum(max(1.0 - np.cumsum(pmf)[u - 1] if u > 0 else 1.0, 0.0)
                  for u in range(1, T + 1))
        surv = max(1.0 - np.cumsum(pmf)[T], 0.0)
        c = ratio * mem + surv
        if best_c is None or c < best_c - 1e-12:
            best, best_c = T, c
    return best


def test_matches_a_brute_force_search_of_the_same_objective():
    for pairs in ([(20, 200)], [(3, 50), (150, 50)], [(1, 80), (7, 20), (200, 5)]):
        h = _hist(pairs)
        for r in (0.5, 0.05, 0.005):
            for sm in (0.0, 1.0, 25.0):
                assert (optimal_keepalive(h, r, smooth=sm)
                        == brute_force_optimum(h, r, smooth=sm)), (pairs, r, sm)


def test_a_periodic_application_is_held_to_its_period():
    """The failure that motivated minimising directly rather than solving the
    first-order condition: hazard is zero before the period, so a first-crossing
    rule stops at minute 1 and cold-starts on every invocation."""
    h = _hist([(20, 200)])
    assert optimal_keepalive(h, 0.01) == 20


def test_cheaper_cold_starts_mean_shorter_windows():
    """Monotone in the price ratio: memory dearer relative to a cold start must
    never make the policy hold on longer."""
    h = _hist([(2, 60), (40, 30), (200, 10)])
    ts = [optimal_keepalive(h, r) for r in (0.0005, 0.005, 0.05, 0.5)]
    assert ts == sorted(ts, reverse=True), ts


def test_the_hazard_identity_holds_at_an_interior_optimum():
    """The first-order condition is the right *interpretation* of the rule.

    `hazard(T) = m/K` is what the derivative says, and where the minimum is
    interior the chosen stopping time should sit near it. The test is stated for
    an interior optimum on purpose: when the minimum is at the edge of the
    support the derivative is not zero there and the identity simply does not
    apply, which is the same reason the identity cannot be used *as* the
    algorithm (see `test_a_periodic_application_is_held_to_its_period`).
    """
    u = np.arange(1, 241, dtype=float)
    h = np.zeros(241)
    h[1:] = 4000.0 * u ** -2.0          # decreasing hazard, not rounded to zero
    checked = 0
    for r in (0.5, 0.3, 0.15, 0.08):
        t = optimal_keepalive(h, r)
        if not (1 < t < 239):           # interior only
            continue
        checked += 1
        hz = hazard_at(h, t)
        assert 0.3 * r < hz < 4.0 * r, (r, t, hz)
    assert checked >= 2, "no interior optimum was exercised"


def test_policy_is_causal():
    rng = np.random.default_rng(0)
    T, cut = 20000, 15000
    inv = (rng.random(T) < 0.04).astype(int)
    inv2 = inv.copy()
    inv2[cut:] = (rng.random(T - cut) < 0.5).astype(int)
    ev = (10000, T)
    a = hazard_policy(inv, eval_slice=ev, price_ratio=0.05, key="prev_gap")
    b = hazard_policy(inv2, eval_slice=ev, price_ratio=0.05, key="prev_gap")
    assert a[:cut].tolist() == b[:cut].tolist()


def test_policy_responds_to_the_price_ratio():
    rng = np.random.default_rng(3)
    inv = (rng.random(20000) < 0.03).astype(int)
    ev = (10000, 20000)
    res = [simulate_app(inv[ev[0]:ev[1]],
                        hazard_policy(inv, eval_slice=ev, price_ratio=r,
                                      key="prev_gap+tod")[ev[0]:ev[1]], 512.0)
           for r in (0.5, 0.05, 0.005)]
    colds = [r.cold_starts for r in res]
    gbs = [r.idle_gb_minutes for r in res]
    assert len(set(colds)) > 1, f"policy inert across prices: {colds}"
    assert gbs == sorted(gbs), f"cheaper cold starts must buy more memory: {gbs}"


# --- the joint (offset, length) optimiser -----------------------------------

def test_joint_window_matches_brute_force():
    """The O(range) telescoping optimiser must equal an explicit O(range^2) search.

    This is the property that lets the window be recomputed online. The vectorised
    form was a 43x speedup over the naive one and is exactly the kind of rewrite
    that silently changes an answer, so it is pinned against the definition.
    """
    rng = np.random.default_rng(0)
    for trial in range(20):
        R = 120
        hist = rng.poisson(1.5, R + 1).astype(float)
        hist[rng.integers(0, R)] += 40           # a mode somewhere
        ratio = float(rng.choice([0.3, 0.12, 0.05, 0.02, 0.008]))
        sm = float(rng.choice([0.0, 1.0, 25.0]))

        a_fast, L_fast = optimal_window(hist, ratio, smooth=sm)

        h = hist + sm / (R + 1)           # `smooth` is total mass, not per-bin
        pmf = h / h.sum()
        g = np.arange(R + 1)
        best, best_c = None, np.inf
        q = np.cumsum(pmf)
        cand = sorted({0} | {int(np.searchsorted(q, p))
                             for p in (0.01, 0.05, 0.10, 0.25, 0.50)})
        for a in cand:
            if a > R:
                continue
            for L in range(1, R - a + 1):
                overlap = np.clip(np.minimum(a + L, g) - a, 0, None)
                mem = float((pmf * overlap).sum())
                miss = float(pmf[(g < a) | (g >= a + L)].sum())
                c = ratio * mem + miss
                if c < best_c - 1e-12:
                    best, best_c = (a, L), c
        assert best is not None
        fast_cost = None
        # compare achieved cost, not the argmin: ties are legitimate
        overlap = np.clip(np.minimum(a_fast + L_fast, g) - a_fast, 0, None)
        fast_cost = (ratio * float((pmf * overlap).sum())
                     + float(pmf[(g < a_fast) | (g >= a_fast + L_fast)].sum()))
        assert fast_cost <= best_c + 1e-9, (trial, fast_cost, best_c)


def test_joint_window_recovers_a_periodic_application():
    """A perfectly periodic app should be released and brought back at its period."""
    hist = np.zeros(241)
    hist[20] = 500.0
    a, L = optimal_window(hist, 0.05)
    assert a == 20 and L == 1, (a, L)


def test_joint_window_never_costs_more_than_holding_from_zero():
    """Optimising the pair is a superset of optimising the length alone."""
    rng = np.random.default_rng(7)
    for _ in range(15):
        hist = rng.poisson(2.0, 121).astype(float)
        ratio = float(rng.choice([0.2, 0.05, 0.01]))
        sm = float(rng.choice([0.0, 1.0, 25.0]))
        h = hist + sm / len(hist)
        pmf = h / h.sum()
        g = np.arange(len(hist))

        def cost(a, L):
            ov = np.clip(np.minimum(a + L, g) - a, 0, None)
            return (ratio * float((pmf * ov).sum())
                    + float(pmf[(g < a) | (g >= a + L)].sum()))

        a, L = optimal_window(hist, ratio, smooth=sm)
        T = optimal_keepalive(hist, ratio, smooth=sm)
        assert cost(a, L) <= cost(0, T) + 1e-9


# --- the donor control -------------------------------------------------------

def test_donor_histogram_actually_replaces_the_evidence():
    """The control must change the policy, or it controls for nothing.

    A silently-ignored control argument would report the treated arm twice and
    look like a clean null. This asserts the substitution bites.
    """
    inv = np.zeros(3000, dtype=np.int64)
    inv[::15] = 1                                   # period 15
    donor = np.zeros(241)
    donor[180] = 60.0                               # a donor that is nothing like it

    own = hazard_policy(inv, eval_slice=(1500, 3000), key="none",
                        price_ratio=0.05, joint=True)
    lent = hazard_policy(inv, eval_slice=(1500, 3000), key="none",
                         price_ratio=0.05, joint=True, donor_hist=donor)
    assert own.sum() != lent.sum()

    # and the donor must make it worse on this application, since it is wrong
    from racrp.sim.simulator import simulate_app
    lo, hi = 1500, 3000
    r_own = simulate_app(inv[lo:hi], own[lo:hi], 1024.0)
    r_lent = simulate_app(inv[lo:hi], lent[lo:hi], 1024.0)
    assert r_own.cold_starts < r_lent.cold_starts


def test_the_prior_strength_is_the_one_validation_chose():
    """DEFAULT_SMOOTH must match what scripts/tune_smoothing.py selected.

    This is the guard on the result that mattered most here: the prior strength
    that scores best on the test period is not the one that scores best on
    validation, so the constant has to stay pinned to the validation record and
    not drift towards whatever flatters the headline.
    """
    import json
    from racrp.config import TABLES
    from racrp.baselines.hazard import DEFAULT_SMOOTH

    p = TABLES / "smoothing_tuning.json"
    if not p.exists():
        pytest.skip("run scripts/tune_smoothing.py")
    rec = json.loads(p.read_text())
    assert rec["chosen_on"].startswith("validation")
    assert DEFAULT_SMOOTH == rec["chosen_smooth_total"], (
        DEFAULT_SMOOTH, rec["chosen_smooth_total"])
