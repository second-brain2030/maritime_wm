"""Shared Re-ID head (spec sections 11/12).

Token sequence [B, T, D] -> LayerNorm -> 2-layer TransformerEncoder ->
temporal pooling (attention or mean) -> projection to embed_dim ->
BatchNorm -> L2 normalization -> identity classifier (training only).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .temporal_pooling import get_pooler


class SharedReIDHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_dim: int = 512,
        pooler: str = "attention",
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(input_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = get_pooler(pooler, input_dim)
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor | None = None,
        return_logits: bool = True,
    ) -> dict[str, Tensor | None]:
        """Return dict with 'embedding' [B, embed_dim] (L2-unit) and
        'logits' [B, num_classes] (None when ``return_logits`` is False)."""
        # TransformerEncoder's key_padding_mask semantics: True = ignore,
        # so flip the True=valid token mask.
        src_key_padding_mask = None if token_mask is None else ~token_mask
        x = self.norm(tokens)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pool(x, token_mask)
        x = self.proj(x)
        x = self.bn(x)
        embedding = F.normalize(x, dim=-1)
        out: dict[str, Tensor | None] = {"embedding": embedding}
        out["logits"] = self.classifier(embedding) if return_logits else None
        return out

    @torch.no_grad()
    def get_embedding(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        """Convenience: return only the L2-normalized embedding, no grad."""
        return self(tokens, token_mask=mask, return_logits=False)["embedding"]
