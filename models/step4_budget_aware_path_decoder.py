"""Budget-, prerequisite-, goal-, and difficulty-aware pointer decoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .multi_critic import SingleCritic


class BudgetAwarePathDecoder(nn.Module):
    def __init__(self, concept_dim: int, learner_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1,
                 stop_bias: float = -1.5,
                 zero_budget_features: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.zero_budget_features = bool(zero_budget_features)
        self.candidate_projection = nn.Linear(concept_dim, hidden_dim)
        heads = 4 if hidden_dim % 4 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.candidate_encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False)
        self.initial_state = nn.Linear(learner_dim, hidden_dim)
        self.state_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.state_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_score = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.target_score = nn.Linear(concept_dim, hidden_dim, bias=False)
        self.feature_score = nn.Linear(6, hidden_dim, bias=False)
        self.pointer = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1))
        nn.init.constant_(self.stop_head[-1].bias, stop_bias)
        self.critic = SingleCritic(hidden_dim, hidden_dim, dropout)

    @staticmethod
    def _prerequisite_ready(local_prerequisite: torch.Tensor,
                            selected: torch.Tensor) -> torch.Tensor:
        requirements = local_prerequisite.bool()
        return ((~requirements) | selected.unsqueeze(1)).all(dim=0)

    def forward(self, cand_emb: torch.Tensor, cand_cost: torch.Tensor,
                learner_state: torch.Tensor, target_context: torch.Tensor,
                budget: float, local_prerequisite: torch.Tensor | None = None,
                goal_scores: torch.Tensor | None = None,
                difficulties: torch.Tensor | None = None,
                min_steps: int = 1, max_steps: int | None = None,
                greedy: bool = False,
                generator: torch.Generator | None = None):
        device = cand_emb.device
        count = cand_emb.shape[0]
        if count == 0:
            return {"path": [], "decisions": [], "predicted_total": 0.0,
                    "remaining_budget": float(budget), "budget_trace": [float(budget)]}
        if local_prerequisite is None:
            local_prerequisite = torch.zeros(count, count, device=device)
        if goal_scores is None:
            goal_scores = torch.ones(count, device=device)
        if difficulties is None:
            difficulties = torch.full((count,), 0.5, device=device)

        encoded = self.candidate_encoder(
            self.candidate_projection(cand_emb).unsqueeze(0)).squeeze(0)
        state = torch.tanh(self.initial_state(learner_state))
        selected = torch.zeros(count, dtype=torch.bool, device=device)
        remaining = float(budget)
        initial_budget = max(float(budget), 1e-6)
        path: list[int] = []
        decisions: list[dict] = []
        budget_trace = [remaining]
        limit = count if max_steps is None else min(int(max_steps), count)
        normalized_goal = goal_scores / goal_scores.max().clamp_min(1e-6)

        for _ in range(limit + 1):
            if len(path) >= limit:
                break
            ready = self._prerequisite_ready(local_prerequisite, selected)
            if len(path) < min_steps:
                feasible = (~selected) & ready
            else:
                feasible = (~selected) & ready & (cand_cost <= remaining + 1e-9)
            if not feasible.any():
                break

            remaining_tensor = torch.tensor(remaining, device=device)
            unmet_count = (local_prerequisite.bool() & (~selected).unsqueeze(1)).sum(dim=0).float()
            if self.zero_budget_features:
                f0 = torch.zeros_like(cand_cost)
                f1 = torch.zeros_like(cand_cost)
            else:
                f0 = cand_cost / remaining_tensor.clamp_min(1e-6)
                f1 = torch.full_like(cand_cost, math.log(max(remaining, 1e-6)))
            features = torch.stack((
                f0,
                f1,
                normalized_goal,
                difficulties,
                unmet_count / max(count, 1),
                torch.full_like(cand_cost, float(selected.float().mean())),
            ), dim=-1)
            pointer_hidden = (
                self.state_score(state).unsqueeze(0)
                + self.candidate_score(encoded)
                + self.target_score(target_context).unsqueeze(0)
                + self.feature_score(features))
            logits = self.pointer(self.dropout(torch.tanh(pointer_hidden))).squeeze(-1)
            logits = logits.masked_fill(~feasible, float("-inf"))

            feasible_goal = normalized_goal[feasible].max()
            stop_features = torch.stack((
                remaining_tensor / initial_budget,
                feasible_goal,
                selected.float().mean()))
            stop_logit = self.stop_head(torch.cat((state, stop_features))).squeeze()
            if len(path) < min_steps:
                stop_logit = torch.full_like(stop_logit, float("-inf"))
            action_logits = torch.cat((logits, stop_logit.unsqueeze(0)))
            if not torch.isfinite(action_logits).any():
                action_logits[-1] = 0.0
            probabilities = torch.softmax(action_logits, dim=0)
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
            prob_sum = probabilities.sum()
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
            else:
                probabilities = torch.zeros_like(action_logits)
                probabilities[-1] = 1.0
            if greedy:
                action = int(probabilities.argmax())
            else:
                action = int(torch.multinomial(probabilities, 1, generator=generator))
            log_probability = torch.log(probabilities[action].clamp_min(1e-12))
            entropy = -(probabilities[probabilities > 0]
                        * torch.log(probabilities[probabilities > 0])).sum()
            values = self.critic(state)
            decision = {
                "action": -1 if action == count else action,
                "log_prob": log_probability,
                "entropy": entropy,
                "state": state,
                "values": values,
            }
            decisions.append(decision)
            if action == count:
                break

            path.append(action)
            selected[action] = True
            remaining -= float(cand_cost[action].detach())
            budget_trace.append(remaining)
            state = self.state_update(encoded[action], state)

        return {
            "path": path,
            "decisions": decisions,
            "predicted_total": float(budget) - remaining,
            "remaining_budget": remaining,
            "budget_trace": budget_trace,
        }

    def forward_batch(self, instances: list[dict]) -> list[dict]:
        """Correct dynamic-mask batch wrapper.

        Instances may have different prerequisite graphs and goal sets, so the
        wrapper keeps episode semantics explicit. Training accumulates their
        losses before one optimizer step.
        """
        return [self(**instance) for instance in instances]


