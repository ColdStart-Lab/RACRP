"""Batched loader for situations.

A per-item ``Dataset`` + ``DataLoader`` would spend most of training in Python:
seven million individual window slices per run, each costing more than the
forward pass on a batch of them. Instead we build each batch with one fancy-index
gather into the shared count matrix, which is a single vectorised memory read.

Yields the same tuple as ``SituationDataset`` so models and the training loop are
unchanged: ``(hist, tfeat, sfeat, y, idx)``.
"""
from __future__ import annotations

import numpy as np
import torch

from racrp.config import BATCH_SIZE, SEED
from racrp.data.windows import time_features


class BatchedSituations:
    """Iterable of batches gathered directly from the count matrix."""

    def __init__(self, ds, batch_size: int = BATCH_SIZE, shuffle: bool = False,
                 device: str = "cpu", seed: int = SEED, outcome_len: int = 0):
        self.ds = ds
        self.bs = batch_size
        self.shuffle = shuffle
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.counts = ds.counts
        self.rows = np.asarray(ds.rows)
        self.window = ds.window
        self.app_idx = ds.app_idx
        self.t_idx = ds.t_idx
        self.sfeat = ds.sfeat
        # per-situation renewal state, appended to the per-application statics
        self.state = getattr(ds, "state", None)
        self.labels = ds.labels

        # absolute matrix row for each sample, and the window offset template
        self.abs_row = self.rows[self.app_idx].astype(np.int64)
        self.offsets = np.arange(-self.window + 1, 1, dtype=np.int64)

        # time features depend only on the minute, so cache them per sample
        self.tfeat = time_features(self.t_idx.astype(np.int64))
        self.y = self.labels[self.app_idx, self.t_idx].astype(np.float32)

        # optional auxiliary target: the activity profile that FOLLOWS minute t.
        # Used to shape the embedding space by trajectory rather than by risk
        # level alone -- see racrp/models/train.py.
        self.outcome_len = outcome_len
        self.out_offsets = (np.arange(1, outcome_len + 1, dtype=np.int64)
                            if outcome_len else None)

    def __len__(self) -> int:
        return (len(self.app_idx) + self.bs - 1) // self.bs

    @property
    def n(self) -> int:
        return len(self.app_idx)

    def positive_rate(self) -> float:
        return float(self.y.mean())

    def __iter__(self):
        order = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(order)
        for s in range(0, self.n, self.bs):
            sel = order[s:s + self.bs]
            r = self.abs_row[sel][:, None]
            t = self.t_idx[sel].astype(np.int64)[:, None]
            hist = self.counts[r, t + self.offsets[None, :]].astype(np.float32)
            np.log1p(hist, out=hist)
            batch = [
                torch.from_numpy(hist).to(self.device),
                torch.from_numpy(self.tfeat[sel]).to(self.device),
                torch.from_numpy(
                    self.sfeat[self.app_idx[sel]] if self.state is None
                    else np.concatenate(
                        [self.sfeat[self.app_idx[sel]], self.state[sel]], axis=1)
                ).to(self.device),
                torch.from_numpy(self.y[sel]).to(self.device),
                torch.from_numpy(sel.astype(np.int64)).to(self.device),
            ]
            if self.outcome_len:
                cols = np.clip(t + self.out_offsets[None, :],
                               0, self.counts.shape[1] - 1)
                fut = (self.counts[r, cols] > 0).astype(np.float32)
                batch.append(torch.from_numpy(fut).to(self.device))
            yield tuple(batch)
