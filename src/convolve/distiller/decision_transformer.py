from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = object
    F = None
    TORCH_AVAILABLE = False

from convolve.distiller.dataset import ABSTAIN, VOCAB_SIZE


@dataclass
class DTConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_len: int = 128


if TORCH_AVAILABLE:

    class DecisionTransformer(nn.Module):
        """Minimal Decision Transformer for action-token prediction.

        Inputs
        ------
        tokens : (B, L) int64
        rtg    : (B, L) float32 - return-to-go, one value per position
        mask   : (B, L) float32 - 1 for real tokens, 0 for padding

        Output
        ------
        logits : (B, L, vocab_size)
        """

        def __init__(self, cfg: DTConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
            self.rtg_proj = nn.Sequential(
                nn.Linear(1, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
            )
            self.ln = nn.LayerNorm(cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(self, tokens, rtg, mask=None):
            B, L = tokens.shape
            device = tokens.device
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            x = self.tok_emb(tokens) + self.pos_emb(pos)
            x = x + self.rtg_proj(rtg.unsqueeze(-1))
            x = self.ln(x)
            # Causal mask
            causal = torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
            )
            pad_mask = None
            if mask is not None:
                pad_mask = mask == 0  # True = ignore
            h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
            return self.head(h)

        @torch.no_grad()
        def score_abstain(self, tokens, rtg, mask=None):
            """Probability mass assigned to the ABSTAIN token at each position."""
            logits = self.forward(tokens, rtg, mask)
            probs = F.softmax(logits, dim=-1)
            return probs[..., ABSTAIN]


def is_available() -> bool:
    return TORCH_AVAILABLE


def build_model(cfg: DTConfig | None = None):
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is not installed. Install with: pip install 'convolve[torch]'"
        )
    return DecisionTransformer(cfg or DTConfig())


def num_parameters(model) -> int:
    if not TORCH_AVAILABLE:  # pragma: no cover
        return 0
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "DTConfig",
    "DecisionTransformer",
    "build_model",
    "is_available",
    "num_parameters",
    "TORCH_AVAILABLE",
    "np",
]
