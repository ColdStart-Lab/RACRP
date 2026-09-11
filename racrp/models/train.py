"""Training loop shared by every learned model in the project.

One loop, one objective, one seed policy -- so that differences between the LSTM
baseline, the no-retrieval transformer and RACRP come from the models and not
from unequal training care.

The risk label is heavily imbalanced (most idle minutes are not followed by an
invocation), so the loss is a focal-weighted BCE and model selection is on
validation **average precision**, not accuracy or AUC, both of which are easy to
look good on under imbalance.

Two-stage schedule for RACRP:
  stage 1 -- train encoder + head with retrieval disabled, so the embedding space
             is organised by cold-start outcome before it is ever used as a
             retrieval key (a randomly-initialised key space retrieves noise);
  stage 2 -- freeze nothing, build the memory bank from the stage-1 encoder, and
             fine-tune the whole model with retrieval on.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from racrp.config import (
    SEED, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE, CKPT, DEVICE,
)


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              alpha: float = 0.5, gamma: float = 1.5) -> torch.Tensor:
    """Focal BCE: down-weights the flood of easy negatives so the rare positive
    (an invocation is about to arrive after a quiet stretch) still drives the
    gradient. gamma=0 recovers plain BCE."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt).pow(gamma) * bce).mean()


@dataclass
class TrainReport:
    model: str
    best_epoch: int = -1
    best_val_ap: float = -1.0
    history: list[dict] = field(default_factory=list)
    params: int = 0
    train_seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


@torch.no_grad()
def predict(model, loader, device: str = DEVICE, retriever=None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities, labels)`` for a whole loader."""
    model.eval()
    ps, ys = [], []
    for batch in loader:
        hist, tf, sf, y, idx = [b.to(device) if torch.is_tensor(b) else b
                                for b in batch[:5]]
        if retriever is not None:
            logits = retriever(model, hist, tf, sf, idx)
        else:
            logits = model(hist, tf, sf)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.float().cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)


def train_model(model, train_loader, val_loader, *, name: str,
                epochs: int = EPOCHS, lr: float = LR, wd: float = WEIGHT_DECAY,
                patience: int = PATIENCE, device: str = DEVICE,
                retriever=None, gamma: float = 1.5, verbose: bool = True,
                aux_weight: float = 0.0) -> TrainReport:
    """Train, early-stop on validation average precision, checkpoint the best.

    ``retriever`` is an optional callable ``(model, hist, tf, sf, idx) -> logits``
    that injects retrieved neighbours; when it is None the model is called
    directly. This keeps the retrieval-on and retrieval-off runs on identical
    code paths.

    ``aux_weight > 0`` adds the auxiliary future-trajectory reconstruction loss
    (requires a loader built with ``outcome_len`` set and a model exposing
    ``outcome_head``). It shapes the embedding space so that nearest neighbours
    are situations with similar *futures*, which is what retrieval needs.
    """
    set_seed()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(epochs * len(train_loader), 1), pct_start=0.25)

    rep = TrainReport(model=name, params=n_params)
    best_state, bad = None, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            hist, tf, sf, y, idx = [b.to(device) if torch.is_tensor(b) else b
                                    for b in batch[:5]]
            fut = batch[5].to(device) if len(batch) > 5 else None
            logits = retriever(model, hist, tf, sf, idx) if retriever else model(hist, tf, sf)
            loss = focal_bce(logits, y, gamma=gamma)
            head = getattr(model, "outcome_head", None)
            if aux_weight > 0 and fut is not None and head is not None:
                e = model.embed(hist, tf, sf)
                loss = loss + aux_weight * nn.functional.binary_cross_entropy_with_logits(
                    head(e), fut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item()); nb += 1

        p, y = predict(model, val_loader, device, retriever)
        ap = float(average_precision_score(y, p)) if y.max() > 0 else float("nan")
        auc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "val_ap": ap, "val_auc": auc}
        rep.history.append(rec)
        if verbose:
            print(f"  [{name}] ep{ep:2d} loss={rec['train_loss']:.5f} "
                  f"val_AP={ap:.4f} val_AUC={auc:.4f}")

        if ap > rep.best_val_ap:
            rep.best_val_ap, rep.best_epoch = ap, ep
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [{name}] early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, CKPT / f"{name}.pt")
    rep.train_seconds = round(time.time() - t0, 1)
    (CKPT / f"{name}_report.json").write_text(rep.to_json())
    return rep
