"""RACRP: the retrieval-augmented cold-start risk predictor.

Architecture
------------
                 current situation                retrieved episodes (k)
                        |                                  |
                SituationEncoder                  (e_j, o_j, y_j) from the
                        |                          episodic memory bank
                   query q (d)                            |
                        |                          value_j = MLP([e_j ; o_j ; y_j])
                        +------ cross-attention ----------+
                                    |
                        fused = q + Attn(q -> values)
                                    |
                                 RiskHead
                                    |
                          logit P(cold start within dt)

The single most important design point: the attention **values** are built from
the neighbour's *outcome* (what the workload actually did in the 60 minutes after
that historical moment) and its realised label -- not from the neighbour's input
pattern alone. That is what makes this episodic memory rather than a
nearest-neighbour smoother: the model reads off consequences.

A learned gate decides how much to trust the memory for each query, so situations
with no good analogue fall back gracefully onto the encoder's own judgement. The
gate value is also what the demo shows the user as "how much did memory matter
here", and it is what we inspect when a claim fails.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from racrp.config import EMB_DIM, D_FF, DROPOUT, N_HEADS, OUTCOME_LEN, TOP_K
from racrp.models.encoder import SituationEncoder, RiskHead


class NeighbourValueEncoder(nn.Module):
    """Turn a retrieved episode into an attention key/value pair.

    key   -- from the neighbour's embedding (what it looked like)
    value -- from its outcome profile and realised label (what it led to)
    """

    def __init__(self, emb_dim: int = EMB_DIM, outcome_len: int = OUTCOME_LEN,
                 d_model: int = EMB_DIM):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(emb_dim, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
        self.value = nn.Sequential(
            nn.Linear(outcome_len + 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, nb_emb, nb_out, nb_lab, sims):
        k = self.key(nb_emb)
        v = self.value(torch.cat([nb_out, nb_lab.unsqueeze(-1),
                                  sims.unsqueeze(-1)], dim=-1))
        return k, v


class RACRP(nn.Module):
    """Retrieval-augmented risk predictor.

    ``use_retrieval=False`` reduces this to the ablation twin while keeping the
    exact same trunk, so the comparison measures retrieval and not capacity.
    """

    def __init__(self, *, emb_dim: int = EMB_DIM, n_heads: int = N_HEADS,
                 outcome_len: int = OUTCOME_LEN, dropout: float = DROPOUT,
                 use_retrieval: bool = True, **enc_kw):
        super().__init__()
        self.use_retrieval = use_retrieval
        self.encoder = SituationEncoder(emb_dim=emb_dim, dropout=dropout, **enc_kw)
        self.nbenc = NeighbourValueEncoder(emb_dim, outcome_len, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(emb_dim)
        # gate: how much to trust memory for this query
        self.gate = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, 1), nn.Sigmoid(),
        )
        self.head = RiskHead(emb_dim)
        self._last_gate: torch.Tensor | None = None
        self._last_attn: torch.Tensor | None = None

    def embed(self, hist, tfeat, sfeat):
        return self.encoder(hist, tfeat, sfeat)

    def fuse(self, q, nb_emb, nb_out, nb_lab, sims):
        """q: (B, d); neighbours: (B, k, *)."""
        k, v = self.nbenc(nb_emb, nb_out, nb_lab, sims)
        ctx, attn_w = self.attn(q.unsqueeze(1), k, v, need_weights=True,
                                average_attn_weights=True)
        ctx = self.attn_norm(ctx.squeeze(1))
        g = self.gate(torch.cat([q, ctx], dim=-1))
        self._last_gate = g.detach()
        self._last_attn = attn_w.detach().squeeze(1)
        return q + g * ctx

    def forward(self, hist, tfeat, sfeat,
                nb_emb=None, nb_out=None, nb_lab=None, sims=None):
        q = self.embed(hist, tfeat, sfeat)
        if self.use_retrieval and nb_emb is not None:
            z = self.fuse(q, nb_emb, nb_out, nb_lab, sims)
        else:
            self._last_gate = torch.zeros(q.size(0), 1, device=q.device)
            self._last_attn = None
            z = q
        return self.head(z)

    @property
    def last_gate(self) -> torch.Tensor | None:
        """Per-query memory reliance in [0,1] from the most recent forward pass."""
        return self._last_gate

    @property
    def last_attention(self) -> torch.Tensor | None:
        """Per-neighbour attention weights from the most recent forward pass."""
        return self._last_attn


def outcome_prior(nb_out: torch.Tensor, nb_lab: torch.Tensor,
                  sims: torch.Tensor, horizon: int) -> torch.Tensor:
    """Similarity-weighted vote of the retrieved outcomes.

    A parameter-free retrieval baseline: what would you predict using *only* the
    memory, with no learned fusion at all? If RACRP cannot beat this, the fusion
    block is not earning its keep -- so it is reported alongside the model.
    """
    w = torch.softmax(sims * 10.0, dim=-1)
    voted = (w * nb_out[:, :, :horizon].amax(dim=-1)).sum(dim=-1)
    return voted.clamp(1e-6, 1 - 1e-6)
