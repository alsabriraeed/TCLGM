"""Heteroscedastic personalized lognormal time model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class PersonalizedTimeCostModel(nn.Module):
    def __init__(self, learner_dim: int, concept_dim: int, num_concepts: int,
                 hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.concept_bias = nn.Embedding(num_concepts, 1)
        nn.init.zeros_(self.concept_bias.weight)
        self.speed = nn.Sequential(
            nn.Linear(learner_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.input = nn.Sequential(
            nn.Linear(learner_dim + concept_dim, hidden), nn.GELU())
        self.trunk = nn.Sequential(
            ResidualBlock(hidden, dropout), ResidualBlock(hidden, dropout))
        self.mean_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        self.register_buffer("log_time_mean", torch.tensor(0.0))
        self.register_buffer("log_time_std", torch.tensor(1.0))

    @torch.no_grad()
    def set_target_stats(self, durations: torch.Tensor) -> None:
        log_time = torch.log(durations.clamp_min(1e-3))
        self.log_time_mean.copy_(log_time.mean())
        self.log_time_std.copy_(log_time.std().clamp_min(1e-3))

    def distribution(self, learner: torch.Tensor, concept: torch.Tensor,
                     concept_ids: torch.Tensor):
        hidden = self.trunk(self.input(torch.cat((learner, concept), dim=-1)))
        standardized_mean = (
            self.mean_head(hidden).squeeze(-1)
            - self.speed(learner).squeeze(-1)
            + self.concept_bias(concept_ids).squeeze(-1))
        mean = standardized_mean * self.log_time_std + self.log_time_mean
        scale = F.softplus(self.scale_head(hidden).squeeze(-1))
        scale = (scale * self.log_time_std).clamp(0.05, 3.0)
        return mean, scale

    def forward(self, learner: torch.Tensor, concept: torch.Tensor,
                cidx: torch.Tensor | None = None):
        if cidx is None:
            raise ValueError("cidx is required for the concept time-intensity effect")
        mean, scale = self.distribution(learner, concept, cidx)
        expected = torch.exp(mean + 0.5 * scale.square())
        return expected

    def loss(self, learner: torch.Tensor, concept: torch.Tensor,
             true_time: torch.Tensor, cidx: torch.Tensor):
        mean, scale = self.distribution(learner, concept, cidx)
        target = torch.log(true_time.clamp_min(1e-3))
        nll = (0.5 * ((target - mean) / scale).square() + torch.log(scale)).mean()
        return nll


