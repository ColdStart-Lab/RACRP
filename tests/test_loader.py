"""The fast batched loader must produce exactly what the reference Dataset does.

The loader builds torch tensors, so these are the only tests in the suite that
need torch. It is an optional extra -- nothing in the headline result depends on
it -- so they skip rather than fail where it is absent, which is what lets CI
run the rest without an 800 MB install.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip(
    "torch", reason="torch is optional: pip install -e '.[negative-results]'")

from racrp.data.windows import SituationDataset
from racrp.data.loader import BatchedSituations


def _toy(n_apps=6, T=3000, seed=0):
    rng = np.random.default_rng(seed)
    counts = (rng.random((n_apps, T)) < 0.08).astype(np.uint16) * rng.integers(1, 9, (n_apps, T)).astype(np.uint16)
    meta = pd.DataFrame({
        "HashApp": [f"a{i}" for i in range(n_apps)],
        "HashOwner": ["o"] * n_apps,
        "trigger": ["http", "timer", "queue", "event", "storage", "others"][:n_apps],
        "n_functions": rng.integers(1, 5, n_apps),
        "total_invocations": counts.sum(1),
        "active_minutes": (counts > 0).sum(1),
        "avg_duration_ms": rng.uniform(50, 3000, n_apps),
        "p99_duration_ms": rng.uniform(100, 9000, n_apps),
        "memory_mb": rng.uniform(64, 2048, n_apps),
    })
    return counts, meta


def test_batched_loader_matches_the_reference_dataset_exactly():
    counts, meta = _toy()
    rows = np.arange(len(meta))
    ds = SituationDataset(counts, rows, meta, time_slice=(0, 2000), horizon=5,
                          window=240, stride=7, train_slice=(0, 1500))
    assert len(ds) > 200, "toy dataset should not be empty"

    loader = BatchedSituations(ds, batch_size=64, shuffle=False)
    seen = 0
    for hist, tf, sf, y, idx in loader:
        for j in range(len(idx)):
            i = int(idx[j])
            h0, t0, s0, y0, _ = ds[i]
            assert torch.allclose(hist[j], h0), f"history mismatch at sample {i}"
            assert torch.allclose(tf[j], t0), f"time features mismatch at {i}"
            assert torch.allclose(sf[j], s0), f"static features mismatch at {i}"
            assert float(y[j]) == float(y0), f"label mismatch at {i}"
            seen += 1
    assert seen == len(ds)


def test_label_matches_its_definition():
    """y = 1 iff an invocation lands in (t, t+horizon]."""
    counts, meta = _toy()
    rows = np.arange(len(meta))
    dt = 5
    ds = SituationDataset(counts, rows, meta, time_slice=(0, 2000), horizon=dt,
                          window=240, stride=11, train_slice=(0, 1500))
    for i in range(0, len(ds), 37):
        a = int(ds.app_idx[i]); t = int(ds.t_idx[i])
        expect = float((counts[rows[a], t + 1:t + 1 + dt] > 0).any())
        assert float(ds.labels[a, t]) == expect


def test_decision_points_are_idle_minutes_only():
    counts, meta = _toy()
    rows = np.arange(len(meta))
    ds = SituationDataset(counts, rows, meta, time_slice=(0, 2000), horizon=5,
                          window=240, stride=3, train_slice=(0, 1500))
    a, t = ds.app_idx, ds.t_idx
    assert (counts[rows[a], t] == 0).all(), "active minutes must not be decision points"
    assert (t >= 240).all(), "every sample needs a full history window"


def test_no_future_leaks_into_the_history_window():
    """The last element of the window is minute t itself; nothing beyond."""
    counts, meta = _toy()
    rows = np.arange(len(meta))
    ds = SituationDataset(counts, rows, meta, time_slice=(0, 2000), horizon=5,
                          window=240, stride=13, train_slice=(0, 1500))
    loader = BatchedSituations(ds, batch_size=32)
    hist, _, _, _, idx = next(iter(loader))
    for j in range(len(idx)):
        i = int(idx[j])
        a = int(ds.app_idx[i]); t = int(ds.t_idx[i])
        want = np.log1p(counts[rows[a], t - 239:t + 1].astype(np.float32))
        assert np.allclose(hist[j].numpy(), want)
