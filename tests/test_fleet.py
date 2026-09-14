"""The fleet estimator: pooling, shrinkage, and the limits that define it."""
import numpy as np
import pytest

from racrp.baselines.fleet import (
    hist_to_quantiles, quantiles_to_windows, gap_histogram, QGRID,
)


def _hist(gaps, n=241):
    h = np.zeros(n)
    for g in gaps:
        h[min(g, n - 1)] += 1
    return h


def test_quantile_function_of_a_periodic_app_is_flat():
    q = hist_to_quantiles(_hist([20] * 50))
    assert np.interp(0.05, QGRID, q) == pytest.approx(20, abs=1)
    assert np.interp(0.95, QGRID, q) == pytest.approx(20, abs=1)
    assert quantiles_to_windows(q, 5, 92, 10) == (20, 1)


def test_quantile_pooling_gives_a_typical_value_not_the_union_of_supports():
    """This is the failure that made the first version of the estimator lose.

    Two analogues, one idling ~5 min and one ~200 min. Averaging their DENSITIES
    makes a mixture whose 92nd percentile sits at 200, so the policy holds a
    container warm for the worst analogue's gap. Averaging their QUANTILE
    FUNCTIONS gives the typical gap, which is what the percentile is meant to be.
    """
    a, b = _hist([5] * 20), _hist([200] * 20)
    mixture = (a / a.sum() + b / b.sum()) / 2
    barycenter = (hist_to_quantiles(a) + hist_to_quantiles(b)) / 2

    p92_mixture = np.interp(0.92, QGRID, hist_to_quantiles(mixture))
    p92_bary = np.interp(0.92, QGRID, barycenter)

    assert p92_mixture == pytest.approx(200, abs=2)
    assert p92_bary == pytest.approx(102.5, abs=2)
    assert p92_bary < p92_mixture / 1.5, "quantile pooling must not inflate the tail"


def test_shrinkage_recovers_the_own_histogram_in_the_limit():
    """With plenty of own evidence the estimate IS the ATC'20 histogram.

    This is the property that makes the method a generalisation of the deployed
    policy rather than a competitor, so it is worth pinning explicitly.
    """
    own = hist_to_quantiles(_hist([30] * 500))
    prior = hist_to_quantiles(_hist([180] * 500))
    kappa = 8.0
    for n, expect in ((0, 180), (8, 105), (10_000, 30)):
        q = ((n * own + kappa * prior) / (n + kappa)) if n else prior
        got = np.interp(0.5, QGRID, q)
        assert got == pytest.approx(expect, abs=2), f"n_own={n}"


def test_shrinkage_is_monotone_in_own_evidence():
    own = hist_to_quantiles(_hist([10] * 100))
    prior = hist_to_quantiles(_hist([200] * 100))
    vals = []
    for n in (0, 1, 4, 16, 64, 256):
        q = (n * own + 8.0 * prior) / (n + 8.0)
        vals.append(np.interp(0.5, QGRID, q))
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), \
        "estimate must move monotonically from the prior towards own evidence"


def test_gap_histogram_counts_only_the_observable_window():
    inv = np.zeros(1000, dtype=int)
    inv[[100, 130, 160, 800, 860]] = 1
    h, n = gap_histogram(inv, (0, 500))
    assert n == 2 and h[30] == 2          # only the gaps inside [0,500)
    h2, n2 = gap_histogram(inv, (0, 0))
    assert n2 == 0 and h2.sum() == 0      # a just-deployed app has no evidence


def test_out_of_range_gaps_are_kept_as_evidence_in_the_top_bin():
    """A 10-hour gap must not be silently dropped -- that would make a sparse app
    look dense and produce a far too aggressive keep-alive."""
    inv = np.zeros(2000, dtype=int)
    inv[[10, 610, 1210]] = 1              # gaps of 600 min, beyond the 240 range
    h, n = gap_histogram(inv, (0, 2000))
    assert n == 2
    assert h[240] == 2 and h[:240].sum() == 0
