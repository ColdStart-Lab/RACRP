"""Prediction-quality metrics for calibrated cold-start risk.

The risk label is rare, so accuracy and ROC-AUC both flatter a model that has
learned little. Average precision (area under precision-recall) is the headline;
calibration is reported too, because the scheduling policy thresholds the number
directly -- an uncalibrated 0.7 that really means 0.2 produces the wrong decision
even if the ranking is perfect.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve,
)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """ECE with equal-mass bins (equal-width bins are useless when p is skewed)."""
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p)
    p, y = p[order], y[order]
    edges = np.linspace(0, n, bins + 1).astype(int)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        ece += (b - a) / n * abs(p[a:b].mean() - y[a:b].mean())
    return float(ece)


def max_f1(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Best achievable F1 and the threshold that attains it."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1))
    return float(f1[i]), float(thr[min(i, len(thr) - 1)])


def prediction_metrics(p: np.ndarray, y: np.ndarray, name: str = "") -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = {"model": name, "n": int(len(y)), "positive_rate": float(y.mean())}
    if 0 < y.mean() < 1:
        out["average_precision"] = float(average_precision_score(y, p))
        out["auc_roc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["ece"] = expected_calibration_error(p, y)
        f1, thr = max_f1(p, y)
        out["max_f1"] = f1
        out["max_f1_threshold"] = thr
        # lift over the trivial "always predict the base rate" model
        out["ap_lift_over_prior"] = out["average_precision"] / max(y.mean(), 1e-12)
    return out


def bootstrap_ci(p: np.ndarray, y: np.ndarray, metric=average_precision_score,
                 n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI -- so improvements are reported with uncertainty
    rather than as bare point estimates."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].max() == 0:
            continue
        vals.append(metric(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_bootstrap_pvalue(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                            metric=average_precision_score,
                            n_boot: int = 1000, seed: int = 0) -> dict:
    """Is model A better than model B on the same samples?

    Paired resampling of the shared evaluation set; the p-value is the fraction of
    resamples in which A does not beat B. Used for the retrieval ablation, where
    a small difference must not be reported as a finding without this.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].max() == 0:
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return {
        "mean_delta": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value_a_not_better": float((diffs <= 0).mean()),
    }


class TemperatureScaler:
    """Single-parameter calibration fitted on the validation split.

    Keeps the ranking intact while making the probability mean what it says, which
    is what the risk-threshold policy needs.
    """

    def __init__(self):
        self.T = 1.0
        self.b = 0.0

    def fit(self, logits: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.05):
        import torch
        lg = torch.tensor(logits, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        logT = torch.zeros(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([logT, b], lr=lr)
        for _ in range(iters):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                lg / torch.exp(logT) + b, yt)
            opt.zero_grad(); loss.backward(); opt.step()
        self.T = float(torch.exp(logT).item())
        self.b = float(b.item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64) / self.T + self.b
        return 1.0 / (1.0 + np.exp(-z))
