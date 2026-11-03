"""Single gain value head for actor-critic training."""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleCritic(nn.Module):
    """Single value head that estimates expected gain."""

    def __init__(self, state_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"gain": self.network(state).squeeze(-1)}

