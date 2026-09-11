"""Episodic workload memory: the memory bank and its retrieval index.

This is the component that has no counterpart in existing serverless schedulers.
A memory entry is a *past situation together with what happened next*:

    entry = ( embedding e_j          # how the situation looked
            , outcome  o_j           # the next 60 minutes of activity that
                                     #   actually followed it
            , label    y_j           # whether a cold start would have occurred
            , meta                   # app row, minute, for explanation in the demo
            )

Retrieval returns the ``k`` nearest entries by cosine similarity in embedding
space. Crucially, the fusion block downstream attends over the **outcomes**, not
the embeddings alone -- the scheduler is reading off what happened last time, not
merely noting that something looked similar.

Leakage contract: the bank is built exclusively from the training slice, and a
query is never allowed to retrieve an entry from its own app *at a nearby
minute* (``exclude_self_window``), which would otherwise let a situation match a
near-duplicate of itself.

Embeddings are cached to ``artifacts/embeddings`` so a rebuild is never needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from racrp.config import EMBED, TOP_K, OUTCOME_LEN


@dataclass
class MemoryBank:
    embeddings: np.ndarray      # (N, d) float32, L2-normalised
    outcomes: np.ndarray        # (N, OUTCOME_LEN) float32, binary activity profile
    labels: np.ndarray          # (N,) float32, risk label at the primary horizon
    app_rows: np.ndarray        # (N,) int32, index into the cohort
    minutes: np.ndarray         # (N,) int32, absolute minute of the situation

    def __len__(self) -> int:
        return len(self.embeddings)

    def save(self, name: str) -> Path:
        p = EMBED / f"{name}.npz"
        np.savez_compressed(
            p, embeddings=self.embeddings, outcomes=self.outcomes,
            labels=self.labels, app_rows=self.app_rows, minutes=self.minutes)
        return p

    @classmethod
    def load(cls, name: str) -> "MemoryBank | None":
        p = EMBED / f"{name}.npz"
        if not p.exists():
            return None
        z = np.load(p)
        return cls(z["embeddings"], z["outcomes"], z["labels"],
                   z["app_rows"], z["minutes"])


def l2_normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


class RetrievalIndex:
    """Cosine kNN over the memory bank.

    Uses FAISS when available (inner product on normalised vectors == cosine) and
    falls back to a blocked torch matmul on the GPU, which is fast enough for
    banks of a few hundred thousand entries and keeps the dependency optional.
    """

    def __init__(self, bank: MemoryBank, device: str = "cpu", use_faiss: bool = True):
        self.bank = bank
        self.device = device
        self.emb = l2_normalise(bank.embeddings.astype(np.float32))
        self._faiss = None
        if use_faiss:
            try:
                import faiss  # type: ignore
                idx = faiss.IndexFlatIP(self.emb.shape[1])
                idx.add(self.emb)
                self._faiss = idx
            except Exception:
                self._faiss = None
        if self._faiss is None:
            self._torch_emb = torch.from_numpy(self.emb).to(device)

    def search(self, queries: np.ndarray, k: int = TOP_K,
               *, exclude_app: np.ndarray | None = None,
               exclude_self_window: int = 0,
               query_minutes: np.ndarray | None = None,
               overfetch: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, similarities)`` of shape ``(B, k)``.

        ``exclude_app`` / ``exclude_self_window`` drop neighbours drawn from the
        same app within +/- ``exclude_self_window`` minutes of the query, which
        would be a near-duplicate of the query itself rather than an analogue.
        """
        q = l2_normalise(np.asarray(queries, dtype=np.float32))
        need = k * overfetch if exclude_self_window > 0 else k
        need = min(need, len(self.bank))

        if self._faiss is not None:
            sims, idx = self._faiss.search(q, need)
        else:
            qt = torch.from_numpy(q).to(self.device)
            sims_t, idx_t = torch.topk(qt @ self._torch_emb.T, k=need, dim=1)
            sims, idx = sims_t.cpu().numpy(), idx_t.cpu().numpy()

        if exclude_self_window <= 0 or exclude_app is None or query_minutes is None:
            return idx[:, :k], sims[:, :k]

        out_i = np.zeros((len(q), k), dtype=np.int64)
        out_s = np.zeros((len(q), k), dtype=np.float32)
        bank_app = self.bank.app_rows
        bank_min = self.bank.minutes
        for b in range(len(q)):
            cand = idx[b]
            bad = (bank_app[cand] == exclude_app[b]) & (
                np.abs(bank_min[cand].astype(np.int64) - int(query_minutes[b]))
                <= exclude_self_window)
            good = cand[~bad][:k]
            gs = sims[b][~bad][:k]
            if len(good) < k:                      # pad with the best available
                pad = k - len(good)
                good = np.concatenate([good, cand[:pad]])
                gs = np.concatenate([gs, sims[b][:pad]])
            out_i[b] = good[:k]
            out_s[b] = gs[:k]
        return out_i, out_s

    def gather(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Neighbour ``(embeddings, outcomes, labels)`` for a batch of indices."""
        return (self.emb[indices],
                self.bank.outcomes[indices],
                self.bank.labels[indices])
