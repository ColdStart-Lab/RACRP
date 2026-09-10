"""Situation encoder + prediction heads.

Three models share this file so that the ablation is airtight -- the retrieval
and no-retrieval variants use the *same* encoder class with the same parameter
count in the trunk, and differ only in whether the fusion block is present.

  * ``SituationEncoder``  window + time + static  ->  d-dim situation embedding
  * ``RiskHead``          embedding               ->  logit of P(cold start in dt)
  * ``LSTMRiskModel``     AWU-style recurrent baseline [Xu et al. ICPADS'19]
  * ``TransformerRiskModel``  encoder + head, no retrieval  (ablation twin)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from racrp.config import (
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, EMB_DIM, WINDOW, OUTCOME_LEN,
)
from racrp.data.windows import multiscale, D_STATIC, D_TIME, SCALES

N_CHANNELS = len(SCALES) + 1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                      # (B, L, D)
        return x + self.pe[:, : x.size(1)]


class SituationEncoder(nn.Module):
    """Encode (history window, time features, static features) into one vector.

    The window is first reduced by a strided conv stack -- 240 raw minutes is too
    long for full self-attention on a 4 GB GPU, and the first few strides are
    exactly the local smoothing a hand-built feature set would do anyway.
    """

    def __init__(self, window: int = WINDOW, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 d_ff: int = D_FF, dropout: float = DROPOUT, emb_dim: int = EMB_DIM,
                 d_static_extra: int = 0):
        super().__init__()
        self.window = window
        # extra per-situation inputs appended to the static block (renewal state)
        self.d_static_extra = d_static_extra
        self.conv = nn.Sequential(
            nn.Conv1d(N_CHANNELS, d_model // 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ctx = nn.Sequential(
            nn.Linear(D_TIME + D_STATIC + d_static_extra, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.proj = nn.Linear(2 * d_model, emb_dim)
        self.ctx_proj = nn.Linear(d_model, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, hist, tfeat, sfeat) -> torch.Tensor:
        x = multiscale(hist)                       # (B, C, W)
        h = self.conv(x).transpose(1, 2)           # (B, W/8, D)
        h = self.tf(self.pos(h))
        pooled = torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)   # mean + most recent
        e = self.proj(pooled)
        c = self.ctx_proj(self.ctx(torch.cat([tfeat, sfeat], dim=-1)))
        # context is injected additively so embeddings stay comparable across
        # apps -- the retrieval index depends on that comparability.
        return self.norm(e + c)


class RiskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = D_FF, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerRiskModel(nn.Module):
    """B3: encoder + risk head. RACRP with retrieval switched off.

    This is the ablation twin -- any gain RACRP shows over this model is
    attributable to retrieval and nothing else.

    The optional ``outcome_head`` predicts the activity profile of the next
    ``OUTCOME_LEN`` minutes from the embedding. It exists to shape the *retrieval
    key space*: trained on the binary risk label alone, the encoder maps every
    situation with a similar risk level to a similar vector, so nearest
    neighbours are situations the model already agrees with and retrieval tells
    it nothing new. Asking the same embedding to reconstruct the future
    trajectory forces it to encode *which* future is coming, not merely how
    likely one is -- which is what makes a retrieved episode informative.

    The head is used only as an auxiliary training signal; it is not part of any
    reported prediction, and the risk path is identical with or without it, so it
    does not give the ablation twin an unfair advantage (both arms get it).
    """

    def __init__(self, outcome_len: int = OUTCOME_LEN, **kw):
        super().__init__()
        self.encoder = SituationEncoder(**kw)
        self.head = RiskHead(EMB_DIM)
        self.outcome_head = (nn.Sequential(
            nn.LayerNorm(EMB_DIM), nn.Linear(EMB_DIM, D_FF), nn.GELU(),
            nn.Linear(D_FF, outcome_len))
            if outcome_len else None)

    def embed(self, hist, tfeat, sfeat):
        return self.encoder(hist, tfeat, sfeat)

    def forward(self, hist, tfeat, sfeat):
        return self.head(self.embed(hist, tfeat, sfeat))

    def forward_with_outcome(self, hist, tfeat, sfeat):
        e = self.embed(hist, tfeat, sfeat)
        return self.head(e), (self.outcome_head(e) if self.outcome_head else None)


class LSTMRiskModel(nn.Module):
    """B2: AWU-style LSTM baseline [Xu et al., ICPADS 2019].

    The original predicts the next invocation *time*; we give it the identical
    risk objective so the comparison isolates the architecture rather than the
    task. Same feature stack, comparable parameter budget.
    """

    def __init__(self, hidden: int = D_MODEL, layers: int = 2,
                 dropout: float = DROPOUT, window: int = WINDOW):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)   # 240 -> 60 steps
        self.lstm = nn.LSTM(N_CHANNELS, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.ctx = nn.Sequential(
            nn.Linear(D_TIME + D_STATIC, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.head = RiskHead(2 * hidden)

    def forward(self, hist, tfeat, sfeat):
        x = multiscale(hist)                 # (B, C, W)
        x = self.pool(x).transpose(1, 2)     # (B, W/4, C)
        out, _ = self.lstm(x)
        c = self.ctx(torch.cat([tfeat, sfeat], dim=-1))
        return self.head(torch.cat([out[:, -1], c], dim=-1))
