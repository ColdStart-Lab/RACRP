"""Phase 8: does making the renewal state explicit fix the predictor?

The predictor's only view of recent history is a 240-minute invocation window.
The information audit found the most informative context for the next idle gap is
the length of the gap that just closed, but the window cannot represent a gap
longer than itself, and 54% of decision points are more than an hour idle. So the
model is blind to that variable exactly where the window has saturated.

Two arms, identical in every other respect -- same architecture, same optimiser,
same seed, same splits, same number of steps -- differing only in whether the
explicit renewal-state features are present.

**The falsifiable prediction, stated before running:** the improvement must
concentrate at long idle age, where the window is uninformative and the added
features are the only source of the signal. A uniform improvement across
idle-age bands would mean the features are helping for some other reason and the
explanation is wrong. Results are therefore reported by stratum, not only in
aggregate -- the aggregate is dominated by the easy third of decision points and
would hide the effect either way.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from racrp.config import (
    TABLES, PROCESSED, CKPT, TRAIN_SLICE, VAL_SLICE, TEST_SLICE, PRIMARY_HORIZON,
    WINDOW, BATCH_SIZE, STRIDE_TRAIN, SEED, DEVICE,
)
from racrp.data.trace import build_app_matrix
from racrp.data.windows import SituationDataset
from racrp.data.loader import BatchedSituations
from racrp.data.statefeat import D_STATE
from racrp.models.encoder import TransformerRiskModel
from racrp.models.train import train_model, set_seed
from racrp.eval.metrics import (
    prediction_metrics, TemperatureScaler, paired_bootstrap_pvalue,
)

COHORT_FILE = PROCESSED / "cohort.npy"
AGE_BINS = [0, 15, 60, 240, 1440, np.inf]
AGE_LABELS = ["<15 min", "15-60 min", "1-4 h", "4-24 h", ">24 h"]


def idle_age(counts, rows, app_idx, t_idx):
    """Minutes since the last invocation, per situation."""
    out = np.full(len(app_idx), np.inf)
    for a in np.unique(app_idx):
        m = app_idx == a
        act = np.flatnonzero(counts[rows[a]] > 0)
        if len(act) == 0:
            continue
        t = t_idx[m].astype(np.int64)
        pos = np.searchsorted(act, t, side="left") - 1
        out[m] = np.where(pos >= 0, t - act[np.clip(pos, 0, len(act) - 1)], np.inf)
    return out


@torch.no_grad()
def logits_of(model, loader, device):
    model.eval()
    ls, ys = [], []
    for hist, tf, sf, y, idx in loader:
        ls.append(model(hist, tf, sf).float().cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ls), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=80_000)
    ap.add_argument("--max-eval", type=int, default=80_000)
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain even if a checkpoint exists")
    args = ap.parse_args()
    device = "cpu" if not torch.cuda.is_available() else args.device
    set_seed(SEED)

    counts, meta = build_app_matrix()
    rows = np.load(COHORT_FILE)
    print(f"[phase8] device={device}, cohort={len(rows)}")

    results, dumps = [], {}
    ytest = None
    ages = None

    for use_state in (False, True):
        name = "with renewal state" if use_state else "window only (current)"
        set_seed(SEED)
        common = dict(meta=meta, horizon=args.horizon, window=WINDOW,
                      decision_points_only=True, train_slice=TRAIN_SLICE,
                      state_features=use_state)
        tr = SituationDataset(counts, rows, time_slice=TRAIN_SLICE,
                              stride=STRIDE_TRAIN, max_samples=args.max_train,
                              seed=SEED, **common)
        va = SituationDataset(counts, rows, time_slice=VAL_SLICE, stride=3,
                              max_samples=args.max_eval, seed=SEED + 1, **common)
        te = SituationDataset(counts, rows, time_slice=TEST_SLICE, stride=1,
                              max_samples=args.max_eval, seed=SEED + 2, **common)
        d_extra = D_STATE if use_state else 0
        print(f"[phase8] {name}: train={len(tr):,} test={len(te):,} "
              f"(+{d_extra} state features)")

        tr_l = BatchedSituations(tr, BATCH_SIZE, shuffle=True, device=device)
        va_l = BatchedSituations(va, BATCH_SIZE, device=device)
        te_l = BatchedSituations(te, BATCH_SIZE, device=device)

        tag = "state" if use_state else "window"
        model = TransformerRiskModel(d_static_extra=d_extra)
        ck = CKPT / f"phase8_{tag}.pt"
        if ck.exists() and not args.retrain:
            # reuse a completed arm: the analysis below is cheap to redo, the
            # training is not, and both arms are already seeded identically
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model = model.to(device)
            rep = type("R", (), {"params": sum(p_.numel() for p_ in model.parameters())})()
            print(f"[phase8] {name}: reusing checkpoint {ck.name}")
        else:
            rep = train_model(model, tr_l, va_l, name=f"phase8_{tag}",
                              epochs=args.epochs, device=device)

        lv, yv = logits_of(model, va_l, device)
        lt, yt = logits_of(model, te_l, device)
        cal = TemperatureScaler().fit(lv, yv)
        p = cal.transform(lt)
        ytest = yt
        if ages is None:
            ages = idle_age(counts, rows, te.app_idx, te.t_idx)

        met = prediction_metrics(p, yt, name)
        met["params"] = rep.params
        results.append(met)
        dumps[name] = p
        print(f"[phase8] {name}: AP={met['average_precision']:.4f} "
              f"AUC={met['auc_roc']:.4f} ECE={met['ece']:.4f}")

    # ---------------------------------------------- by idle age, the real test
    band = pd.Series(pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS, right=False))
    rows_out = []
    a, b = "window only (current)", "with renewal state"
    for lab in AGE_LABELS:
        m = (band == lab).to_numpy()
        if m.sum() < 200 or ytest[m].max() == 0 or ytest[m].min() == 1:
            continue
        ma = prediction_metrics(dumps[a][m], ytest[m])["average_precision"]
        mb = prediction_metrics(dumps[b][m], ytest[m])["average_precision"]
        rows_out.append({"idle_age": lab, "n": int(m.sum()),
                         "positive_rate": float(ytest[m].mean()),
                         "window_only_ap": ma, "with_state_ap": mb,
                         "gain": mb - ma})
    df = pd.DataFrame(rows_out)
    pd.set_option("display.width", 200)
    print("\naverage precision by how long the application had been idle")
    print(df.to_string(index=False))

    ab = paired_bootstrap_pvalue(dumps[b], dumps[a], ytest)
    print(f"\noverall: dAP={ab['mean_delta']:+.4f} "
          f"[{ab['ci_low']:+.4f}, {ab['ci_high']:+.4f}] "
          f"p(not better)={ab['p_value_a_not_better']:.4f}")

    out = {"results": results, "by_idle_age": rows_out, "ablation": ab,
           "config": vars(args)}
    (TABLES / "predictor_ablation.json").write_text(json.dumps(out, indent=2))

    print("\nREAD-OUT")
    if len(df) >= 2:
        short, long = df.iloc[0]["gain"], df.iloc[-1]["gain"]
        if long > short and long > 0.01:
            print("  The gain concentrates at long idle age, where the window has "
                  "saturated. That is the predicted signature: the features are "
                  "supplying information the window cannot represent.")
        else:
            print("  The gain does NOT concentrate at long idle age. Whatever the "
                  "features are doing, it is not the explanation we gave, and it "
                  "must be reported that way.")
    print(f"\nwrote {TABLES / 'predictor_ablation.json'}")


if __name__ == "__main__":
    main()
