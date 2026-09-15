"""Objective O4: turn the calibrated risk into a scheduling policy and score it.

Why this needs its own script
-----------------------------
``run_train.py`` evaluates prediction quality on a *sample* of test situations,
which is the right thing for average precision. It is the wrong thing for a
policy: a scheduler must decide at **every** idle minute, and scattering a sample
onto the (application x minute) grid leaves almost every minute at risk zero, so
the policy degenerates to its unconditional floor. (That failure is real and was
observed; it is why this script exists.)

So here we take a smaller cohort of applications and run inference over **all** of
their decision minutes, then:

  1. calibrate with the temperature fitted on validation during training,
  2. sweep the cost knob lambda to trace a cost/benefit curve,
  3. compare against the ATC'20 histogram and the conditional estimator on the
     same applications, at matched memory cost.

The decision rule is the cost-aware one: hold the instance warm while

    risk  >  lambda * memory_cost / cold_start_penalty
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from racrp.config import (
    TABLES, CKPT, PROCESSED, TEST_SLICE, TRAIN_SLICE, VAL_SLICE, PRIMARY_HORIZON,
    WINDOW, BATCH_SIZE, LAMBDA_SWEEP, SEED, DEVICE,
)
from racrp.data.trace import build_app_matrix
from racrp.data.windows import SituationDataset
from racrp.data.loader import BatchedSituations
from racrp.models.encoder import LSTMRiskModel, TransformerRiskModel
from racrp.sim import policies as P
from racrp.sim.simulator import cold_start_ms, simulate_app, SimResult
from racrp.baselines.conditional import conditional_policy
from racrp.eval.metrics import prediction_metrics

COHORT_FILE = PROCESSED / "cohort.npy"
TAILS = (60.0, 75.0, 85.0, 92.0, 96.0, 99.0)


def interp_at(curve, budget):
    pts = sorted(((c, y) for c, y in curve), key=lambda t: t[0])
    if not pts or budget < pts[0][0] or budget > pts[-1][0]:
        return None
    return float(np.interp(budget, [a for a, _ in pts], [b for _, b in pts]))


@torch.no_grad()
def infer(model, ds, device):
    model.eval()
    out = []
    for hist, tf, sf, y, idx in BatchedSituations(ds, BATCH_SIZE, device=device):
        out.append(model(hist, tf, sf).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=400,
                    help="applications to evaluate over ALL their decision minutes")
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    device = "cpu" if not torch.cuda.is_available() else args.device

    counts, meta = build_app_matrix()
    rows_all = np.load(COHORT_FILE)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rows_all[rng.choice(len(rows_all), min(args.apps, len(rows_all)),
                                       replace=False)])
    mem = meta.loc[rows, "memory_mb"].to_numpy()
    print(f"[policy] {len(rows)} applications, ALL decision minutes on the test slice")

    ds = SituationDataset(counts, rows, meta, time_slice=TEST_SLICE,
                          horizon=args.horizon, stride=1, window=WINDOW,
                          decision_points_only=True, train_slice=TRAIN_SLICE,
                          max_samples=None)
    print(f"[policy] {len(ds):,} decision points (no sampling)")

    # ---- load whatever models trained successfully
    models = {}
    for name, ctor, ck in (("B2-LSTM", LSTMRiskModel, f"b2_lstm{tag}"),
                           ("B3-Transformer", TransformerRiskModel,
                            f"b3_transformer{tag}")):
        p = CKPT / f"{ck}.pt"
        if p.exists():
            m = ctor()
            m.load_state_dict(torch.load(p, map_location="cpu"))
            models[name] = m.to(device)
    if not models:
        raise SystemExit("no checkpoints found; run scripts/run_train.py first")
    print(f"[policy] models: {list(models)}")

    # ---- calibration fitted during training, on validation (never on test)
    temps = {}
    mp = TABLES / f"prediction_metrics_h{args.horizon}{tag}.json"
    if mp.exists():
        for r in json.loads(mp.read_text())["results"]:
            if "temperature" in r:
                temps[r["model"]] = (r["temperature"], r["bias"])

    n_min = counts.shape[1]
    records = []

    def score_curve(name, policy_fn, sweep):
        curve = []
        for knob in sweep:
            total = SimResult()
            for i, r in enumerate(rows):
                inv = counts[r].astype(np.int64)
                warm = np.asarray(policy_fn(inv, i, knob))
                total.merge(simulate_app(inv[TEST_SLICE[0]:TEST_SLICE[1]],
                                         warm[TEST_SLICE[0]:TEST_SLICE[1]],
                                         float(mem[i])))
            rec = total.as_dict()
            rec["method"] = name
            rec["knob"] = knob
            records.append(rec)
            curve.append((rec["idle_gb_minutes"], rec["cold_starts"]))
            print(f"  {name:26s} knob={knob:6.2f} cold={rec['cold_starts']:8,} "
                  f"gb-min={rec['idle_gb_minutes']:11,.0f}", flush=True)
        return curve

    # ---- reference curves on the SAME applications
    base_curve = score_curve(
        "ATC'20 marginal", lambda inv, i, t: conditional_policy(
            inv, eval_slice=TEST_SLICE, key="none", tail_margin=t), TAILS)
    cond_curve = score_curve(
        "conditional (ours)", lambda inv, i, t: conditional_policy(
            inv, eval_slice=TEST_SLICE, key="prev_gap+tod", tail_margin=t), TAILS)

    # ---- learned-risk policies
    gb = mem / 1024.0
    penalty = np.array([cold_start_ms(m) for m in mem])
    base_ratio = gb / (penalty / 1000.0)
    learned_curves = {}
    ytrue = ds.labels[ds.app_idx, ds.t_idx].astype(np.float64)

    for name, model in models.items():
        logits = infer(model, ds, device)
        T, b = temps.get(name.replace("B3-Transformer", "B3-Transformer-noRetrieval"),
                         (1.0, 0.0))
        prob = 1.0 / (1.0 + np.exp(-(logits / T + b)))
        met = prediction_metrics(prob, ytrue, name)
        print(f"[policy] {name}: AP={met.get('average_precision'):.4f} "
              f"ECE={met.get('ece'):.4f} over {len(prob):,} decision points")
        grid = np.zeros((len(rows), n_min), dtype=np.float32)
        grid[ds.app_idx, ds.t_idx] = prob
        learned_curves[name] = score_curve(
            f"risk policy: {name}",
            lambda inv, i, lam, g=grid: P.risk_threshold(
                inv, g[i], float(np.clip(lam * base_ratio[i], 1e-6, 0.999)),
                horizon=args.horizon, min_keepalive=1),
            LAMBDA_SWEEP)

    # ---- matched-cost comparison against the baseline's operating points
    matched = []
    for cost, cold in base_curve:
        row = {"budget_idle_gb_min": round(cost, 1), "baseline_cold_starts": cold}
        v = interp_at(cond_curve, cost)
        row["conditional"] = None if v is None else round(v, 1)
        if v is not None and cold:
            row["conditional_reduction_pct"] = round(100 * (1 - v / cold), 2)
        for name, curve in learned_curves.items():
            v = interp_at(curve, cost)
            key = "risk_" + name.replace("-", "_")
            row[key] = None if v is None else round(v, 1)
            if v is not None and cold:
                row[key + "_reduction_pct"] = round(100 * (1 - v / cold), 2)
        matched.append(row)

    (TABLES / f"policy_eval_sweep{tag}.json").write_text(json.dumps(records, indent=2))
    (TABLES / f"policy_eval_matched{tag}.json").write_text(json.dumps(matched, indent=2))
    pd.set_option("display.width", 240)
    print("\ncold starts at MATCHED memory cost, same applications")
    print(pd.DataFrame(matched).to_string(index=False))
    print(f"\nwrote policy_eval_matched{tag}.json")


if __name__ == "__main__":
    main()
