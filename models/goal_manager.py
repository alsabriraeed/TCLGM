"""Soft initial-learning/review goal acquisition inspired by HA-LPR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoalPlan:
    explicit_targets: tuple[int, ...]
    review_goals: tuple[int, ...]
    relevant_concepts: tuple[int, ...]
    priority: np.ndarray
    review_probability: float


class AdaptiveGoalManager:
    def __init__(self, prerequisite: np.ndarray, mastery_threshold: float = 0.85,
                 ancestor_discount: float = 0.80, max_review_goals: int = 5,
                 curriculum_weight: float = 0.15):
        self.prerequisite = np.asarray(prerequisite, dtype=np.float32)
        self.n_concepts = self.prerequisite.shape[0]
        self.mastery_threshold = float(mastery_threshold)
        self.ancestor_discount = float(ancestor_discount)
        self.max_review_goals = int(max_review_goals)
        self.curriculum_weight = float(curriculum_weight)

    def _ancestor_relevance(self, targets: tuple[int, ...]) -> np.ndarray:
        relevance = np.zeros(self.n_concepts, dtype=np.float32)
        queue = [(int(target), 0) for target in targets]
        best_distance: dict[int, int] = {}
        while queue:
            node, distance = queue.pop(0)
            if node in best_distance and best_distance[node] <= distance:
                continue
            best_distance[node] = distance
            relevance[node] = max(relevance[node], self.ancestor_discount ** distance)
            for predecessor in np.flatnonzero(self.prerequisite[:, node]):
                queue.append((int(predecessor), distance + 1))
        return relevance

    def formulate(self, mastery: np.ndarray, explicit_targets: list[int] | tuple[int, ...],
                  studied_counts: np.ndarray | None = None,
                  forgetting_probability: np.ndarray | None = None,
                  concept_weights: np.ndarray | None = None,
                  recent_accuracy: float | None = None) -> GoalPlan:
        mastery = np.asarray(mastery, dtype=np.float32)
        targets = tuple(dict.fromkeys(int(item) for item in explicit_targets))
        if not targets:
            raise ValueError("at least one explicit target is required")
        studied = (np.zeros(self.n_concepts, dtype=np.float32)
                   if studied_counts is None else np.asarray(studied_counts, dtype=np.float32))
        forgetting = (np.zeros(self.n_concepts, dtype=np.float32)
                      if forgetting_probability is None
                      else np.asarray(forgetting_probability, dtype=np.float32))
        weights = (np.ones(self.n_concepts, dtype=np.float32)
                   if concept_weights is None else np.asarray(concept_weights, dtype=np.float32))
        accuracy = 0.5 if recent_accuracy is None or not np.isfinite(recent_accuracy) \
            else float(np.clip(recent_accuracy, 0.0, 1.0))

        target_coverage = float(np.mean(studied[list(targets)] > 0))
        forgetting_pressure = float(np.mean(forgetting[studied > 0])) if (studied > 0).any() else 0.0
        review_probability = float(np.clip(
            0.45 * target_coverage + 0.35 * accuracy + 0.20 * forgetting_pressure,
            0.0, 1.0))

        relevance = self._ancestor_relevance(targets)
        gap = np.clip(1.0 - mastery, 0.0, 1.0)
        new_learning = (1.0 - review_probability) * relevance * gap
        review_need = review_probability * (studied > 0) * gap * np.maximum(forgetting, 0.05)
        curriculum_progression = self.curriculum_weight * gap
        priority = weights * (new_learning + review_need + curriculum_progression)
        
        # High-Gain Target Prioritization: emphasize direct mastery gains on target concepts
        target_indices = list(targets)
        priority[target_indices] += 3.0 * weights[target_indices] * gap[target_indices]

        review_candidates = np.flatnonzero((studied > 0) & (mastery < self.mastery_threshold))
        ordered_review = sorted(review_candidates.tolist(), key=lambda c: priority[c], reverse=True)
        review_goals = tuple(ordered_review[:self.max_review_goals])
        relevant = set(np.flatnonzero(relevance > 0).tolist())
        relevant.update(review_goals)
        relevant.update(targets)
        return GoalPlan(
            explicit_targets=targets,
            review_goals=review_goals,
            relevant_concepts=tuple(sorted(relevant)),
            priority=priority.astype(np.float32),
            review_probability=review_probability,
        )

