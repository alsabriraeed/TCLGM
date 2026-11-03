"""Sequence encoder for correctness, elapsed time, and forgetting lag."""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnerStateEncoder(nn.Module):
    def __init__(self, concept_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.answer_embedding = nn.Embedding(2, 8)
        self.input_projection = nn.Sequential(
            nn.Linear(concept_dim + 8 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cold_start = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, concept_embeddings: torch.Tensor, answers: torch.Tensor,
                elapsed: torch.Tensor, lengths: torch.Tensor,
                lag: torch.Tensor | None = None) -> torch.Tensor:
        if lag is None:
            lag = torch.zeros_like(elapsed)
        numeric = torch.stack((torch.log1p(elapsed), torch.log1p(lag)), dim=-1)
        tokens = torch.cat((concept_embeddings,
                            self.answer_embedding(answers.long()), numeric), dim=-1)
        encoded, _ = self.gru(self.input_projection(tokens))
        state = self.cold_start.unsqueeze(0).expand(encoded.shape[0], -1).clone()
        nonempty = lengths > 0
        if nonempty.any():
            last = (lengths[nonempty] - 1).long()
            state[nonempty] = encoded[nonempty, last]
        return state

