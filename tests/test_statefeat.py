"""Renewal-state features: correctness and, above all, causality."""
import numpy as np
import pytest

from racrp.data.statefeat import state_features, D_STATE
from racrp.baselines.conditional import gap_bucket, N_GAP_BUCKETS


def test_shapes_and_flag():
    counts = np.zeros((2, 500), dtype=np.uint16)
    counts[0, [10, 40, 100]] = 1
    rows = np.array([0, 1])
    f = state_features(counts, rows, np.array([0, 1]), np.array([200, 200]))
    assert f.shape == (2, D_STATE)
    assert f[0, -1] == 1.0, "app with gaps must be flagged valid"
    assert f[1, -1] == 0.0, "app with no activity must be flagged invalid"


def test_elapsed_age_and_previous_gap_are_correct():
    counts = np.zeros((1, 1000), dtype=np.uint16)
    counts[0, [100, 160, 300]] = 1          # gaps: 60 then 140
    rows = np.array([0])
    # at t=400 the last active minute is 300, and the gap that closed there is 140
    f = state_features(counts, rows, np.array([0]), np.array([400]))
    assert np.isclose(f[0, 0], np.log1p(100), atol=1e-4)   # age = 400-300
    assert np.isclose(f[0, 1], np.log1p(140), atol=1e-4)   # previous gap
    b = int(gap_bucket(140))
    assert f[0, 3 + b] == 1.0
    assert f[0, 3:3 + N_GAP_BUCKETS].sum() == 1.0


def test_age_is_unbounded_by_the_window():
    """The whole point: for a gap far longer than the 240-minute window, the age
    must still be represented, not saturated."""
    counts = np.zeros((1, 20000), dtype=np.uint16)
    counts[0, [10, 5000]] = 1
    rows = np.array([0])
    f = state_features(counts, rows, np.array([0]), np.array([15000]))
    assert np.isclose(f[0, 0], np.log1p(10000), atol=1e-3)
    assert f[0, 0] > np.log1p(240), "age must exceed what a 240-min window can show"


def test_no_future_information_leaks_in():
    """Changing the trace only at or after minute t must not change the features."""
    rng = np.random.default_rng(0)
    base = (rng.random(4000) < 0.05).astype(np.uint16)
    counts_a = base.reshape(1, -1).copy()
    counts_b = base.reshape(1, -1).copy()
    t = 2500
    counts_b[0, t:] = (rng.random(4000 - t) < 0.9).astype(np.uint16)
    rows = np.array([0])
    ts = np.array([t])
    fa = state_features(counts_a, rows, np.array([0]), ts)
    fb = state_features(counts_b, rows, np.array([0]), ts)
    assert np.allclose(fa, fb), "features changed when only the future changed"


def test_activity_at_t_itself_is_not_used():
    """A decision at minute t may not consult minute t."""
    counts = np.zeros((1, 1000), dtype=np.uint16)
    counts[0, [100, 200]] = 1
    rows = np.array([0])
    f1 = state_features(counts, rows, np.array([0]), np.array([500]))
    counts[0, 500] = 1                      # activity exactly at the decision minute
    f2 = state_features(counts, rows, np.array([0]), np.array([500]))
    assert np.allclose(f1, f2), "the decision minute itself leaked into the features"


def test_multiple_applications_are_not_mixed_up():
    counts = np.zeros((3, 2000), dtype=np.uint16)
    counts[0, [10, 30]] = 1                 # gap 20
    counts[1, [10, 110]] = 1                # gap 100
    counts[2, [10, 810]] = 1                # gap 800
    rows = np.arange(3)
    f = state_features(counts, rows, np.array([2, 0, 1]), np.array([1000, 1000, 1000]))
    # rows are returned in the order given, not in sorted-app order
    assert np.isclose(f[0, 1], np.log1p(800), atol=1e-4)
    assert np.isclose(f[1, 1], np.log1p(20), atol=1e-4)
    assert np.isclose(f[2, 1], np.log1p(100), atol=1e-4)
