"""PyKT DKT cognitive simulator with an explicit online-update interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dkt import DKT


@dataclass
class SimulationResult:
    final_mastery: np.ndarray
    realized_times: list[float]
    mastery_trace: list[np.ndarray]
    correctness_trace: list[list[int]]


class TimeEmittingDKT(nn.Module):
    """Cognitive simulator powered by the pyKT DKT architecture.

    Maintains learner mastery states across recommendations, supporting both
    historical replaying and online single-pass/stochastic learning rollouts.
    """

    def __init__(
        self,
        n_concepts: int,
        hidden_dim: int = 64,
        max_emitted_seconds: float = 300.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = int(n_concepts)
        self.num_c = int(n_concepts)
        self.hidden = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_emitted_seconds = float(max_emitted_seconds)

        # Core pyKT DKT model
        self.dkt = DKT(num_c=self.K, emb_size=self.hidden, dropout=dropout)

        # Fallback time head for legacy interface compatibility
        self.time_head = nn.Linear(self.hidden, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int = 1):
        shape = (1, batch_size, self.hidden)
        return (
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
        )

    def forward(
        self,
        concepts: torch.Tensor,
        answers: torch.Tensor,
        state=None,
        return_logits: bool = False,
    ):
        y, next_state = self.dkt(concepts, answers, state=state, return_state=True)
        # Reconstruct output hidden states for compatibility if needed
        output = y  # (B, T, K)
        if return_logits:
            eps = 1e-7
            y_clamped = torch.clamp(y, eps, 1.0 - eps)
            logits = torch.log(y_clamped / (1.0 - y_clamped))
            return logits, y, output, next_state
        return y, output, next_state

    def mastery_from_state(self, state) -> torch.Tensor:
        """Derives mastery prediction vector (batch, K) directly from recurrent state h."""
        h = state[0][-1]  # (batch, hidden)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.dkt.out_layer(h)
        return torch.sigmoid(logits)

    def predict_time(self, hidden: torch.Tensor, next_concept: torch.Tensor) -> torch.Tensor:
        """Predicts estimated time in seconds (fallback for legacy callers)."""
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        log_time = self.time_head(hidden).squeeze(-1)
        return torch.exp(log_time).clamp(1.0, self.max_emitted_seconds)

    @torch.no_grad()
    def mastery_after_history(
        self, q_hist: np.ndarray | Sequence[int], a_hist: np.ndarray | Sequence[int]
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        """Replays historical interactions to yield student start mastery and recurrent state."""
        self.eval()
        if len(q_hist) == 0:
            state = self.initial_state(1)
            return self.mastery_from_state(state)[0].cpu().numpy(), state

        q = torch.as_tensor(q_hist, dtype=torch.long, device=self.device).unsqueeze(0)
        r = torch.as_tensor(a_hist, dtype=torch.long, device=self.device).unsqueeze(0)
        y, state = self.dkt(q, r, return_state=True)
        return y[0, -1].cpu().numpy(), state

    @torch.no_grad()
    def observe(self, state, concept: int, correct: int):
        """Advances simulator state by one interaction."""
        self.eval()
        q = torch.tensor([[concept]], dtype=torch.long, device=self.device)
        r = torch.tensor([[correct]], dtype=torch.long, device=self.device)
        y, new_state = self.dkt(q, r, state=state, return_state=True)
        return new_state, y[0, -1].cpu().numpy()

    @torch.no_grad()
    def simulate_path_src_style(self, state, path: list[int]) -> SimulationResult:
        """Single-pass study simulation.

        Each concept in the recommended path is processed sequentially as a
        positive learning event (correct=1). The recurrent state is updated once
        per concept without stochastic failure, matching standard instructional evaluation.
        """
        self.eval()
        current = (state[0].clone(), state[1].clone())
        mastery = self.mastery_from_state(current)
        mastery_trace = [mastery[0].cpu().numpy().copy()]
        realized_times: list[float] = []

        for concept in path:
            current, mastery_np = self.observe(current, concept, correct=1)
            mastery_trace.append(mastery_np.copy())
            realized_times.append(120.0)  # nominal default; overridden by time model / dataset durations

        return SimulationResult(
            final_mastery=mastery_trace[-1],
            realized_times=realized_times,
            mastery_trace=mastery_trace,
            correctness_trace=[[1] for _ in path],
        )

    @torch.no_grad()
    def simulate_path_detailed(
        self, state, path: list[int], seed: int = 0, max_attempts: int = 4
    ) -> SimulationResult:
        """Detailed stochastic simulation rollout."""
        self.eval()
        rng = np.random.default_rng(seed)
        current = (state[0].clone(), state[1].clone())
        mastery = self.mastery_from_state(current)
        mastery_trace = [mastery[0].cpu().numpy().copy()]
        realized_times: list[float] = []
        correctness_trace: list[list[int]] = []

        for concept in path:
            concept_time = 0.0
            attempts: list[int] = []
            for attempt_idx in range(max_attempts):
                probability = float(mastery_trace[-1][concept])
                is_final = attempt_idx == max_attempts - 1
                correct = 1 if is_final else int(rng.random() < probability)
                concept_time += 120.0
                current, mastery_np = self.observe(current, concept, correct)
                mastery_trace.append(mastery_np.copy())
                attempts.append(correct)
                if correct:
                    break
            realized_times.append(concept_time)
            correctness_trace.append(attempts)

        return SimulationResult(
            final_mastery=mastery_trace[-1],
            realized_times=realized_times,
            mastery_trace=mastery_trace,
            correctness_trace=correctness_trace,
        )

    @torch.no_grad()
    def simulate_path(
        self, state, path: list[int], seed: int = 0, max_attempts: int = 4, n_rollouts: int = 1
    ):
        results = [
            self.simulate_path_detailed(
                state, path, seed=seed + 1009 * run, max_attempts=max_attempts
            )
            for run in range(n_rollouts)
        ]
        final = np.mean([result.final_mastery for result in results], axis=0)
        times = (
            np.mean([result.realized_times for result in results], axis=0).tolist()
            if path
            else []
        )
        return final, times

    def load_state_dict(self, state_dict, strict: bool = False):
        """Flexible loader accommodating pure DKT weights or full simulator state."""
        dkt_dict = {}
        for k, v in state_dict.items():
            if k.startswith("dkt."):
                dkt_dict[k[4:]] = v
            elif (
                k.startswith("interaction_emb.")
                or k.startswith("lstm_layer.")
                or k.startswith("out_layer.")
                or k.startswith("dropout_layer.")
            ):
                dkt_dict[k] = v

        if dkt_dict:
            self.dkt.load_state_dict(dkt_dict, strict=False)
            if "time_head.weight" in state_dict:
                self.time_head.weight.data.copy_(state_dict["time_head.weight"])
            if "time_head.bias" in state_dict:
                self.time_head.bias.data.copy_(state_dict["time_head.bias"])
            return self
        return super().load_state_dict(state_dict, strict=False)


# Aliases for unified usage across modules
CognitiveSimulator = TimeEmittingDKT
MasterySimulator = TimeEmittingDKT

E_SUP = 1.0


def learning_gain(start: float, end: float, ceiling: float = E_SUP) -> float:
    return float((end - start) / max(ceiling - start, 1e-6))
