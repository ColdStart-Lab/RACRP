"""Diagnostic: how much could episodic memory possibly be worth here?

The smoke run showed RACRP beating its no-retrieval twin by +0.017 AP but beating
a random-memory control by only +0.002. Two very different explanations fit that:

  (H1) The key space is bad. Neighbours retrieved by an encoder trained on the
       binary risk label are situations the model already agrees with, so they
       carry no new information. Fixable.
  (H2) The task is saturated. The recent 240-minute window plus per-app statistics
       already determine the answer, so no memory of any kind can add much.
       Not fixable by modelling -- it would mean changing the task.

This script separates them **without training anything**, by measuring the
information carried by retrieved outcomes under three key spaces:

  * ``random``   -- random memories                       (floor)
  * ``learned``  -- the encoder's actual embedding space  (what we have)
  * ``ORACLE``   -- memories chosen by how close their outcome is to the query's
                    TRUE future                           (ceiling)

The oracle is deliberate cheating: it uses the label. It is never a model, only a
bound. If oracle retrieval scores far above the learned key space, the memory
*content* is informative and H1 holds -- the fix is metric learning. If even the
oracle barely beats random, H2 holds and the honest response is to change the
task, not the model.

We also report the same three under a **hard-case** restriction (long idle age),
because that is where the recent window is least informative and where any real
episodic-memory effect must show up if it exists at all.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from racrp.config import TABLES, PROCESSED, PRIMARY_HORIZON, OUTCOME_LEN, EMBED
from racrp.data.trace import build_app_matrix
from racrp.models.memory import MemoryBank, l2_normalise
from racrp.eval.metrics import prediction_metrics

COHORT_FILE = PROCESSED / "cohort.npy"


def idle_age_at(counts_row: np.ndarray, t: np.ndarray) -> np.ndarray:
    active = np.flatnonzero(counts_row > 0)
    if len(active) == 0:
        return np.full(len(t), np.inf)
    pos = np.searchsorted(active, t, side="right") - 1
    return np.where(pos >= 0, t - active[np.clip(pos, 0, len(active) - 1)], np.inf)


def vote(nb_labels: np.ndarray, sims: np.ndarray, temp: float = 10.0) -> np.ndarray:
    """Similarity-weighted vote of the neighbours' realised labels."""
    w = np.exp(temp * (sims - sims.max(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    return np.clip((w * nb_labels).sum(axis=1), 1e-6, 1 - 1e-6)


def topk_by(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores per row."""
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(
        idx, np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1), axis=1)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=8000)
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    preds = np.load(TABLES / f"test_predictions_h{args.horizon}{tag}.npz")
    bank = MemoryBank.load(f"memory_bank_h{args.horizon}{tag}")
    if bank is None:
        raise SystemExit("no memory bank; run scripts/run_train.py first")

    y = preds["y"].astype(np.float64)
    app_idx, t_idx = preds["app_idx"], preds["t_idx"]
    n = len(y)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(n, min(args.max_queries, n), replace=False))
    y, app_idx, t_idx = y[sel], app_idx[sel], t_idx[sel]

    # the query's TRUE future profile (used only by the oracle key space)
    q_out = np.zeros((len(sel), OUTCOME_LEN), dtype=np.float32)
    for i, (a, t) in enumerate(zip(app_idx, t_idx)):
        seg = counts[rows[a], t + 1: t + 1 + OUTCOME_LEN]
        q_out[i, :len(seg)] = (seg > 0).astype(np.float32)

    # query embeddings in the learned key space
    nb_files = sorted(EMBED.glob(f"neighbours_test{tag}_h{args.horizon}_*.npz"))
    if not nb_files:
        raise SystemExit("no cached test neighbours; run scripts/run_train.py first")
    nb = np.load(nb_files[0])
    learned_idx = nb["idx"][sel][:, :args.k]
    learned_sim = nb["sims"][sel][:, :args.k]

    bank_lab = bank.labels.astype(np.float64)
    bank_out = bank.outcomes.astype(np.float32)
    B = len(bank_lab)

    results = {}

    # ---- floor: random memories
    r_idx = rng.integers(0, B, (len(sel), args.k))
    r_sim = np.zeros_like(r_idx, dtype=np.float32)
    results["random"] = vote(bank_lab[r_idx], r_sim)

    # ---- what we actually have
    results["learned"] = vote(bank_lab[learned_idx], learned_sim)

    # ---- ceiling: neighbours whose LATER future matches the query's later future.
    #
    # Critically, the first ``horizon`` minutes of the outcome ARE the label, so
    # ranking by the full profile would let the oracle read the answer directly
    # and the "bound" would be tautological. We therefore rank only on minutes
    # AFTER the labelled window. The oracle still uses future information the
    # model cannot have -- it is a bound, never a method -- but it now measures
    # something meaningful: whether trajectory similarity is a useful retrieval
    # key, rather than whether knowing the label helps predict the label.
    h = args.horizon
    on = l2_normalise(bank_out[:, h:])
    qn = l2_normalise(q_out[:, h:])
    oracle_idx = np.zeros((len(sel), args.k), dtype=np.int64)
    oracle_sim = np.zeros((len(sel), args.k), dtype=np.float32)
    chunk = 512
    for s in range(0, len(sel), chunk):
        e = slice(s, min(s + chunk, len(sel)))
        sc = qn[e] @ on.T
        oi = topk_by(sc, args.k)
        oracle_idx[e] = oi
        oracle_sim[e] = np.take_along_axis(sc, oi, axis=1)
    results["oracle_outcome_space"] = vote(bank_lab[oracle_idx], oracle_sim)

    # ---- a second, fully honest reference: retrieve on the OBSERVABLE past only
    #      (raw history window cosine), no learning, no future information. If
    #      this already beats the learned key space, the learned space is not
    #      merely imperfect -- it is worse than a trivial baseline.
    W = 120
    q_hist = np.zeros((len(sel), W), dtype=np.float32)
    for i, (a, t) in enumerate(zip(app_idx, t_idx)):
        q_hist[i] = np.log1p(counts[rows[a], t - W + 1: t + 1].astype(np.float32))
    b_hist = np.zeros((B, W), dtype=np.float32)
    for i in range(B):
        r = int(rows[int(bank.app_rows[i])]); t = int(bank.minutes[i])
        b_hist[i] = np.log1p(counts[r, t - W + 1: t + 1].astype(np.float32))
    qh, bh = l2_normalise(q_hist), l2_normalise(b_hist)
    raw_idx = np.zeros((len(sel), args.k), dtype=np.int64)
    raw_sim = np.zeros((len(sel), args.k), dtype=np.float32)
    for s in range(0, len(sel), chunk):
        e = slice(s, min(s + chunk, len(sel)))
        sc = qh[e] @ bh.T
        oi = topk_by(sc, args.k)
        raw_idx[e] = oi
        raw_sim[e] = np.take_along_axis(sc, oi, axis=1)
    results["raw_history_cosine"] = vote(bank_lab[raw_idx], raw_sim)

    # ---------------------------------------------------------------- reporting
    ages = np.zeros(len(sel))
    for a in np.unique(app_idx):
        m = app_idx == a
        ages[m] = idle_age_at(counts[rows[a]], t_idx[m])

    out = {"n_queries": int(len(sel)), "bank_size": int(B), "k": args.k,
           "positive_rate": float(y.mean()), "regimes": {}}

    regimes = {"all": np.ones(len(sel), bool),
               "idle<15min": ages < 15,
               "idle 15-60min": (ages >= 15) & (ages < 60),
               "idle>60min": ages >= 60}

    print(f"\nqueries={len(sel):,}  bank={B:,}  k={args.k}  "
          f"base rate={y.mean():.4f}\n")
    print(f"{'regime':<16}{'n':>8}{'base':>8}{'random':>9}{'rawHist':>9}"
          f"{'learned':>9}{'ORACLE':>9}{'headroom':>10}")
    print("-" * 80)
    for rname, mask in regimes.items():
        if mask.sum() < 100 or y[mask].max() == 0 or y[mask].min() == 1:
            continue
        row = {"n": int(mask.sum()), "base_rate": float(y[mask].mean())}
        for kname, p in results.items():
            row[kname] = prediction_metrics(p[mask], y[mask])["average_precision"]
        row["headroom_oracle_minus_learned"] = (
            row["oracle_outcome_space"] - row["learned"])
        row["learned_minus_random"] = row["learned"] - row["random"]
        out["regimes"][rname] = row
        print(f"{rname:<16}{row['n']:>8,}{row['base_rate']:>8.3f}"
              f"{row['random']:>9.4f}{row['raw_history_cosine']:>9.4f}"
              f"{row['learned']:>9.4f}{row['oracle_outcome_space']:>9.4f}"
              f"{row['headroom_oracle_minus_learned']:>10.4f}")

    # how often does the learned key space retrieve a neighbour whose realised
    # label matches the query's? (retrieval precision, not prediction quality)
    agree_learned = (bank_lab[learned_idx] == y[:, None]).mean()
    agree_random = (bank_lab[r_idx] == y[:, None]).mean()
    agree_oracle = (bank_lab[oracle_idx] == y[:, None]).mean()
    out["neighbour_label_agreement"] = {
        "random": float(agree_random), "learned": float(agree_learned),
        "oracle": float(agree_oracle)}
    print(f"\nneighbour label agreement:  random={agree_random:.4f}  "
          f"learned={agree_learned:.4f}  oracle={agree_oracle:.4f}")

    p = TABLES / f"diag_retrieval_ceiling{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")

    lm = out["regimes"].get("all", {})
    print("\nREAD-OUT")
    if lm:
        if lm["headroom_oracle_minus_learned"] > 0.03:
            print("  Large oracle headroom -> the memory CONTENT is informative and "
                  "the learned key space is the bottleneck (H1). Fix retrieval, "
                  "not the task.")
        else:
            print("  Small oracle headroom -> retrieved outcomes carry little the "
                  "model cannot already infer (H2). Changing the task or the "
                  "evaluation regime is the honest response.")


if __name__ == "__main__":
    main()
