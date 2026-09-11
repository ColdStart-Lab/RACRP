"""Precomputed retrieval: frozen retriever, trained reader.

Why the retriever is frozen
---------------------------
If the query encoder keeps changing during training, every step needs a fresh
k-NN search over the whole memory bank, and the bank's keys go stale the moment
the encoder updates. Both problems disappear if the *lookup* representation is
fixed: we train the no-retrieval transformer (B3) first, freeze a copy of its
encoder, and use that frozen encoder for both the bank keys and the queries.

This is not a shortcut, it is the standard retrieval-augmented design (a frozen
retriever with a trained reader), and it buys three things this project needs:

  * neighbour ids can be computed **once** for every situation and cached, so
    training the fusion model costs no more than training the ablation twin --
    which is precisely what makes the ablation fair;
  * the memory bank is a fixed artefact that the web demo can serve directly;
  * RACRP's trunk is still free to adapt, so any gain is attributable to having
    the retrieved outcomes, not to a second encoder's extra capacity.

Everything is cached under ``artifacts/embeddings`` keyed by split and settings,
so a rerun never repeats the search.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from racrp.config import EMBED, TOP_K


def cache_path(split: str, horizon: int, k: int, excl: int, tag: str = "") -> Path:
    t = f"_{tag}" if tag else ""
    return EMBED / f"neighbours_{split}_h{horizon}_k{k}_x{excl}{t}.npz"


def build_neighbour_cache(index, query_emb: np.ndarray, app_idx: np.ndarray,
                          t_idx: np.ndarray, *, split: str, horizon: int,
                          k: int = TOP_K, exclude_self_window: int = 0,
                          tag: str = "", batch: int = 4096,
                          force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Top-k neighbour ids and similarities for every situation in a split.

    Returns ``(idx (N,k) int32, sims (N,k) float32)`` and caches them to disk.
    """
    p = cache_path(split, horizon, k, exclude_self_window, tag)
    if p.exists() and not force:
        z = np.load(p)
        if len(z["idx"]) == len(query_emb):
            print(f"[retrieval] loaded cached neighbours for {split}: {p.name}")
            return z["idx"], z["sims"]

    n = len(query_emb)
    out_i = np.zeros((n, k), dtype=np.int32)
    out_s = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, batch):
        e = slice(s, min(s + batch, n))
        i, sim = index.search(query_emb[e], k=k,
                              exclude_app=app_idx[e],
                              exclude_self_window=exclude_self_window,
                              query_minutes=t_idx[e])
        out_i[e] = i.astype(np.int32)
        out_s[e] = sim.astype(np.float32)
        if (s // batch) % 10 == 0:
            print(f"[retrieval] {split}: {min(s + batch, n):,}/{n:,}", flush=True)
    np.savez_compressed(p, idx=out_i, sims=out_s)
    print(f"[retrieval] cached {n:,} neighbour sets -> {p.name}")
    return out_i, out_s


def build_random_neighbour_cache(bank, query_emb: np.ndarray, k: int,
                                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Control condition: random memories, with their **true** similarities.

    The point of this control is to separate "retrieved episodes carry useful
    information" from "an extra attention pathway helps". For that to be a fair
    control, every input to the fusion block must be internally consistent: a
    random memory has to arrive with the similarity it actually has to the query,
    not with the similarity of the genuine neighbour it replaced. Reusing the
    real neighbours' similarity scores would leak a signal (how well this query
    matches the bank at all) into the control and understate the true effect of
    retrieval in either direction.
    """
    rng = np.random.default_rng(seed)
    n = len(query_emb)
    idx = rng.integers(0, len(bank.labels), (n, k)).astype(np.int32)
    emb = bank.embeddings
    sims = np.einsum("nd,nkd->nk", query_emb.astype(np.float32),
                     emb[idx].astype(np.float32)).astype(np.float32)
    return idx, sims


class CachedRetriever:
    """Feeds precomputed neighbours into a RACRP forward pass.

    ``disable`` skips fusion entirely. For the random-memory control, pass the
    output of :func:`build_random_neighbour_cache` as ``nb_idx``/``nb_sims``
    rather than shuffling here, so that similarities stay consistent with the
    memories they describe.
    """

    def __init__(self, bank, nb_idx: np.ndarray, nb_sims: np.ndarray, device: str,
                 *, disable: bool = False, top_k: int | None = None):
        import torch
        self.torch = torch
        self.device = device
        self.disable = disable
        self.k = top_k or nb_idx.shape[1]
        self.nb_idx = nb_idx[:, : self.k]
        self.nb_sims = nb_sims[:, : self.k]
        self.emb = torch.from_numpy(bank.embeddings)
        self.out = torch.from_numpy(bank.outcomes)
        self.lab = torch.from_numpy(bank.labels)

    def __call__(self, model, hist, tf, sf, idx):
        q = model.embed(hist, tf, sf)
        if self.disable:
            return model.head(q)
        i = idx.detach().cpu().numpy()
        nb = self.torch.from_numpy(self.nb_idx[i].astype(np.int64))
        sims = self.torch.from_numpy(self.nb_sims[i]).to(self.device)
        z = model.fuse(q,
                       self.emb[nb].to(self.device),
                       self.out[nb].to(self.device),
                       self.lab[nb].to(self.device),
                       sims)
        return model.head(z)
