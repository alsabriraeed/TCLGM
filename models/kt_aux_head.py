from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KTAuxHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(state).squeeze(-1))

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            self.network(state).squeeze(-1), target.float())

