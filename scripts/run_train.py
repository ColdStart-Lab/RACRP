"""Phases 2-3 - train the learned predictors and RACRP, then evaluate.

Pipeline
--------
  1. build datasets over the frozen temporal splits (train d1-8, val d9-10, test d11-14)
  2. train B2 (LSTM) and B3 (transformer, no retrieval)
  3. freeze a copy of B3's encoder as the **retriever**; encode the TRAIN split
     into the episodic memory bank; cache top-k neighbours for every situation
  4. train RACRP -- same trunk (warm-started from B3), retrieval on
  5. calibrate on validation, score on test, dump per-situation probabilities

Ablation arms, all trained through the identical loop so the comparison is fair:
  * ``RACRP``                 retrieved neighbours + learned fusion
  * ``RACRP-shuffled``        random memories instead of retrieved ones
                              (separates "retrieval helps" from "more input helps")
  * ``B3-Transformer``        no retrieval at all -- the twin
  * ``Retrieval-only-vote``   parameter-free similarity-weighted vote over the
                              retrieved outcomes: does the fusion earn its keep?

Everything expensive is cached under ``artifacts/`` and reused.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from racrp.config import (
    SEED, DEVICE, BATCH_SIZE, EPOCHS, PRIMARY_HORIZON, WINDOW, TOP_K, OUTCOME_LEN,
    TRAIN_SLICE, VAL_SLICE, TEST_SLICE, STRIDE_TRAIN, CKPT, TABLES, PROCESSED,
)
from racrp.data.trace import build_app_matrix
from racrp.data.windows import SituationDataset, outcome_profile
from racrp.data.loader import BatchedSituations
from racrp.models.encoder import LSTMRiskModel, TransformerRiskModel
from racrp.models.racrp import RACRP, outcome_prior
from racrp.models.memory import MemoryBank, RetrievalIndex, l2_normalise
from racrp.models.retrieval_cache import (
    build_neighbour_cache, build_random_neighbour_cache, CachedRetriever,
)
from racrp.models.train import train_model, set_seed
from racrp.eval.metrics import (
    prediction_metrics, TemperatureScaler, paired_bootstrap_pvalue,
)

COHORT_FILE = PROCESSED / "cohort.npy"


def build_datasets(counts, meta, rows, horizon, max_train, max_eval):
    common = dict(meta=meta, horizon=horizon, window=WINDOW,
                  decision_points_only=True, train_slice=TRAIN_SLICE)
    tr = SituationDataset(counts, rows, time_slice=TRAIN_SLICE, stride=STRIDE_TRAIN,
                          max_samples=max_train, seed=SEED, **common)
    va = SituationDataset(counts, rows, time_slice=VAL_SLICE, stride=3,
                          max_samples=max_eval, seed=SEED + 1, **common)
    te = SituationDataset(counts, rows, time_slice=TEST_SLICE, stride=1,
                          max_samples=max_eval, seed=SEED + 2, **common)
    return tr, va, te


@torch.no_grad()
def embed_all(model, ds, device, batch=2048) -> np.ndarray:
    model.eval()
    out = []
    for hist, tf, sf, y, idx in BatchedSituations(ds, batch_size=batch):
        out.append(model.embed(hist.to(device), tf.to(device),
                               sf.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def build_bank(emb, counts, rows, ds, name, force=False) -> MemoryBank:
    """Store every TRAIN situation together with what happened next."""
    cached = None if force else MemoryBank.load(name)
    if cached is not None and len(cached) == len(ds.app_idx):
        print(f"[memory] loaded cached bank: {len(cached):,} episodes")
        return cached
    t0 = time.time()
    n = len(ds.app_idx)
    outcomes = np.zeros((n, OUTCOME_LEN), dtype=np.float32)
    for i in range(n):
        outcomes[i] = outcome_profile(counts, int(rows[int(ds.app_idx[i])]),
                                      int(ds.t_idx[i]), OUTCOME_LEN)
    bank = MemoryBank(
        embeddings=l2_normalise(emb).astype(np.float32),
        outcomes=outcomes,
        labels=ds.labels[ds.app_idx, ds.t_idx].astype(np.float32),
        app_rows=ds.app_idx.astype(np.int32),
        minutes=ds.t_idx.astype(np.int32),
    )
    p = bank.save(name)
    print(f"[memory] built {len(bank):,} episodes in {time.time()-t0:.1f}s -> {p.name}")
    return bank


@torch.no_grad()
def logits_of(model, loader, device, retriever=None):
    model.eval()
    ls, ys = [], []
    for hist, tf, sf, y, idx in loader:
        lg = retriever(model, hist, tf, sf, idx) if retriever else model(hist, tf, sf)
        ls.append(lg.float().cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ls), np.concatenate(ys)


class _BankSubset:
    """View of the train dataset used as the memory bank (mirrors the fields
    BatchedSituations reads, so the bank is encoded through the same path)."""

    def __init__(self, ds, sel):
        self.ds, self.sel = ds, sel
        self.counts, self.rows, self.window = ds.counts, ds.rows, ds.window
        self.sfeat, self.labels = ds.sfeat, ds.labels
        self.app_idx, self.t_idx = ds.app_idx[sel], ds.t_idx[sel]

    def __len__(self):
        return len(self.sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train", type=int, default=400_000)
    ap.add_argument("--max-eval", type=int, default=300_000)
    ap.add_argument("--bank-size", type=int, default=150_000)
    ap.add_argument("--topk", type=int, default=TOP_K)
    ap.add_argument("--exclude-self-window", type=int, default=1440,
                    help="drop neighbours from the same app within +/- this many "
                         "minutes of the query -- those are near-duplicates of the "
                         "query itself, not analogous past episodes")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-shuffle-arm", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    tag = f"_{args.tag}" if args.tag else ""
    print(f"[train] device={device} horizon={args.horizon} min")
    set_seed(SEED)

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    tr, va, te = build_datasets(counts, meta, rows, args.horizon,
                                args.max_train, args.max_eval)
    print(f"[train] cohort={len(rows)} apps | situations "
          f"train={len(tr):,} val={len(va):,} test={len(te):,}")
    print(f"[train] positive rate  train={tr.positive_rate():.4f} "
          f"val={va.positive_rate():.4f} test={te.positive_rate():.4f}")

    tr_l = BatchedSituations(tr, BATCH_SIZE, shuffle=True, device=device)
    va_l = BatchedSituations(va, BATCH_SIZE, device=device)
    te_l = BatchedSituations(te, BATCH_SIZE, device=device)

    reports, results, dumps = {}, [], {}

    # ------------------------------------------------------------ B2: LSTM
    lstm = LSTMRiskModel()
    reports["b2_lstm"] = train_model(lstm, tr_l, va_l, name=f"b2_lstm{tag}",
                                     epochs=args.epochs, device=device)

    # ------------------------------- B3: transformer, no retrieval (the twin)
    tfm = TransformerRiskModel()
    reports["b3_transformer"] = train_model(tfm, tr_l, va_l,
                                            name=f"b3_transformer{tag}",
                                            epochs=args.epochs, device=device)

    # ------------------------------------------- episodic memory + retrieval
    sel = np.arange(len(tr))
    if args.bank_size and len(tr) > args.bank_size:
        sel = np.sort(np.random.default_rng(SEED).choice(
            len(tr), args.bank_size, replace=False))
    bank_ds = _BankSubset(tr, sel)

    print("[memory] encoding bank with the frozen stage-1 encoder ...")
    bank_emb = embed_all(tfm, bank_ds, device)
    bank = build_bank(bank_emb, counts, rows, bank_ds,
                      f"memory_bank_h{args.horizon}{tag}", force=args.force)
    index = RetrievalIndex(bank, device=device)
    print(f"[memory] index backend = {'faiss' if index._faiss else 'torch matmul'}")

    nbrs, rand_nbrs = {}, {}
    for split, ds in (("train", tr), ("val", va), ("test", te)):
        q = l2_normalise(embed_all(tfm, ds, device))
        nbrs[split] = build_neighbour_cache(
            index, q, ds.app_idx, ds.t_idx, split=split + tag,
            horizon=args.horizon, k=args.topk,
            exclude_self_window=args.exclude_self_window, force=args.force)
        # control arm: random memories carrying their own true similarities
        rand_nbrs[split] = build_random_neighbour_cache(
            bank, q, args.topk, seed=SEED + hash(split) % 1000)

    mk = lambda split, **kw: CachedRetriever(
        bank, nbrs[split][0], nbrs[split][1], device, **kw)
    mk_rand = lambda split, **kw: CachedRetriever(
        bank, rand_nbrs[split][0], rand_nbrs[split][1], device, **kw)

    # --------------------------------------------------------------- RACRP
    racrp = RACRP(use_retrieval=True)
    racrp.encoder.load_state_dict(tfm.encoder.state_dict())
    reports["racrp"] = train_model(
        racrp, tr_l, va_l, name=f"racrp{tag}", epochs=args.epochs, device=device,
        retriever=_SplitSwitch(mk("train"), mk("val")))

    # ---- capacity/step-matched control: RACRP is warm-started from B3 and then
    # trained for a further `epochs`, so a bare RACRP-vs-B3 comparison confounds
    # retrieval with extra optimisation. This arm gives B3 the same second run.
    import copy as _copy
    tfm_cont = TransformerRiskModel()
    tfm_cont.load_state_dict(_copy.deepcopy(tfm.state_dict()))
    reports["b3_continued"] = train_model(
        tfm_cont, tr_l, va_l, name=f"b3_continued{tag}", epochs=args.epochs,
        device=device)

    arms = {
        "B2-LSTM": (lstm, None, None),
        "B3-Transformer-noRetrieval": (tfm, None, None),
        "B3-Transformer-continued": (tfm_cont, None, None),
        "RACRP": (racrp, mk("val"), mk("test")),
    }

    # ------------- control arm: same architecture, random memories not retrieved
    if not args.skip_shuffle_arm:
        shuf = RACRP(use_retrieval=True)
        shuf.encoder.load_state_dict(tfm.encoder.state_dict())
        reports["racrp_shuffled"] = train_model(
            shuf, tr_l, va_l, name=f"racrp_shuffled{tag}", epochs=args.epochs,
            device=device,
            retriever=_SplitSwitch(mk_rand("train"), mk_rand("val")))
        arms["RACRP-shuffledMemory"] = (
            shuf, mk_rand("val"), mk_rand("test"))

    # --------------------------------------------- calibrate + score on test
    y_test = None
    for name, (m, rv, rt) in arms.items():
        lv, yv = logits_of(m, va_l, device, rv)
        lt, yt = logits_of(m, te_l, device, rt)
        y_test = yt
        cal = TemperatureScaler().fit(lv, yv)
        p = cal.transform(lt)
        met = prediction_metrics(p, yt, name)
        met["temperature"], met["bias"] = cal.T, cal.b
        met["params"] = reports.get(
            {"B2-LSTM": "b2_lstm", "B3-Transformer-noRetrieval": "b3_transformer",
             "B3-Transformer-continued": "b3_continued",
             "RACRP": "racrp", "RACRP-shuffledMemory": "racrp_shuffled"}[name],
            None).params if name != "" else 0
        results.append(met)
        dumps[name] = p
        print(f"[eval] {name:28s} AP={met['average_precision']:.4f} "
              f"AUC={met['auc_roc']:.4f} ECE={met['ece']:.4f}")

    # ------------- parameter-free reference: the memories alone, no learning
    ni, ns = nbrs["test"]
    prior = outcome_prior(torch.from_numpy(bank.outcomes[ni.astype(np.int64)]),
                          torch.from_numpy(bank.labels[ni.astype(np.int64)]),
                          torch.from_numpy(ns), args.horizon).numpy()
    results.append(prediction_metrics(prior, y_test, "Retrieval-only-vote"))
    dumps["Retrieval-only-vote"] = prior
    print(f"[eval] {'Retrieval-only-vote':28s} AP={results[-1]['average_precision']:.4f}")

    # ------------------------------------------------------ the ablation test
    ablations = {}
    for a, b in (("RACRP", "B3-Transformer-noRetrieval"),
                 ("RACRP", "B3-Transformer-continued"),
                 ("RACRP", "RACRP-shuffledMemory"),
                 ("RACRP", "Retrieval-only-vote")):
        if a in dumps and b in dumps:
            ablations[f"{a}_vs_{b}"] = paired_bootstrap_pvalue(dumps[a], dumps[b], y_test)
            d = ablations[f"{a}_vs_{b}"]
            print(f"[ablation] {a} - {b}: dAP={d['mean_delta']:+.4f} "
                  f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
                  f"p(not better)={d['p_value_a_not_better']:.4f}")

    (TABLES / f"prediction_metrics_h{args.horizon}{tag}.json").write_text(json.dumps({
        "results": results, "ablations": ablations,
        "reports": {k: v.__dict__ for k, v in reports.items()},
        "config": vars(args),
        "bank_size": len(bank),
    }, indent=2))
    np.savez_compressed(
        TABLES / f"test_predictions_h{args.horizon}{tag}.npz",
        y=y_test, app_idx=te.app_idx, t_idx=te.t_idx, rows=rows,
        **{k.replace("-", "_"): v for k, v in dumps.items()})
    print(f"[train] wrote metrics and test predictions (horizon {args.horizon})")


class _SplitSwitch:
    """Batch indices refer to the train dataset during training and to the
    validation dataset during the eval pass, so the retriever picks the matching
    neighbour cache from ``model.training``."""

    def __init__(self, train_ret, val_ret):
        self.tr, self.va = train_ret, val_ret

    def __call__(self, model, hist, tf, sf, idx):
        return (self.tr if model.training else self.va)(model, hist, tf, sf, idx)


if __name__ == "__main__":
    main()
