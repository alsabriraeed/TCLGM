"""Directed dual-view knowledge topology encoder.

The explicit graph preserves prerequisite direction. The implicit graph is
used for representation and diversity only, never as a hard prerequisite.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)


def build_false_negative_mask(prerequisite: torch.Tensor,
                              collaborative: torch.Tensor) -> torch.Tensor:
    mask = (prerequisite + prerequisite.T + collaborative) > 0
    mask = mask.clone()
    mask.fill_diagonal_(False)
    return mask


class RGCNLayer(nn.Module):
    def __init__(self, dim: int, n_relations: int, dropout: float = 0.1):
        super().__init__()
        self.self_transform = nn.Linear(dim, dim)
        self.relation_weights = nn.Parameter(torch.empty(n_relations, dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_weights)

    def forward(self, features: torch.Tensor,
                relations: Sequence[torch.Tensor]) -> torch.Tensor:
        value = self.self_transform(features)
        for relation, weight in zip(relations, self.relation_weights):
            value = value + relation @ (features @ weight)
        value = self.norm(value)
        return self.dropout(F.gelu(value))


class KnowledgeTopologyEncoderV2(nn.Module):
    def __init__(self, n_concepts: int, dim: int = 64, n_layers: int = 2,
                 tau: float = 0.2, initial_features: Optional[torch.Tensor] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.n_concepts = n_concepts
        self.dim = dim
        self.tau = tau
        if initial_features is None:
            self.base = nn.Parameter(torch.randn(n_concepts, dim) * 0.05)
            self.feature_projector = None
        else:
            if initial_features.shape[0] != n_concepts:
                raise ValueError("initial feature rows must match n_concepts")
            self.register_buffer("initial_features", initial_features.float())
            self.feature_projector = nn.Sequential(
                nn.Linear(initial_features.shape[1], dim), nn.LayerNorm(dim))
            self.base = nn.Parameter(torch.randn(n_concepts, dim) * 0.01)
        self.explicit_layers = nn.ModuleList(
            RGCNLayer(dim, 2, dropout) for _ in range(n_layers))
        self.implicit_layers = nn.ModuleList(
            RGCNLayer(dim, 1, dropout) for _ in range(n_layers))
        self.fusion = nn.Linear(dim, 1, bias=False)
        for name in ("forward_graph", "reverse_graph", "collaborative_graph", "negative_mask"):
            self.register_buffer(name, None, persistent=False)

    def set_graphs(self, prerequisite: torch.Tensor,
                   collaborative: torch.Tensor) -> None:
        prerequisite = prerequisite.float()
        collaborative = collaborative.float()
        self.forward_graph = row_normalize(prerequisite)
        self.reverse_graph = row_normalize(prerequisite.T)
        self.collaborative_graph = row_normalize(collaborative)
        self.negative_mask = build_false_negative_mask(prerequisite, collaborative)

    def _initial(self) -> torch.Tensor:
        if self.feature_projector is None:
            return self.base
        return self.feature_projector(self.initial_features) + self.base

    def forward(self, prerequisite: Optional[torch.Tensor] = None,
                collaborative: Optional[torch.Tensor] = None):
        if prerequisite is not None and collaborative is not None:
            self.set_graphs(prerequisite, collaborative)
        if self.forward_graph is None:
            raise RuntimeError("call set_graphs before forward")
        explicit = implicit = self._initial()
        for layer in self.explicit_layers:
            explicit = layer(explicit, (self.forward_graph, self.reverse_graph))
        for layer in self.implicit_layers:
            implicit = layer(implicit, (self.collaborative_graph,))
        pair = torch.stack((explicit, implicit), dim=1)
        weights = torch.softmax(self.fusion(pair).squeeze(-1), dim=1)
        fused = (weights.unsqueeze(-1) * pair).sum(dim=1)
        return explicit, implicit, fused

    def contrastive_loss(self, explicit: torch.Tensor,
                         implicit: torch.Tensor) -> torch.Tensor:
        left = F.normalize(explicit, dim=-1)
        right = F.normalize(implicit, dim=-1)
        labels = torch.arange(self.n_concepts, device=explicit.device)
        logits_lr = (left @ right.T) / self.tau
        logits_rl = (right @ left.T) / self.tau
        if self.negative_mask is not None:
            logits_lr = logits_lr.masked_fill(self.negative_mask, float("-inf"))
            logits_rl = logits_rl.masked_fill(self.negative_mask.T, float("-inf"))
        return 0.5 * (
            F.cross_entropy(logits_lr, labels) + F.cross_entropy(logits_rl, labels))


KnowledgeTopologyEncoder = KnowledgeTopologyEncoderV2

