"""Goal-driven, prerequisite-aware, budget-safe candidate construction.

Target and prerequisite closure are retained first; optional review and curriculum
expansion concepts are prioritized based on cost-normalized pedagogical priority density.
"""
from __future__ import annotations

import numpy as np

from .goal_manager import GoalPlan


class BudgetAdaptiveCandidateConstructor:
    def __init__(self, fused_emb: np.ndarray, prerequisite: np.ndarray,
                 prerequisite_confidence: np.ndarray | None = None,
                 hard_edge_confidence: float = 0.80,
                 mastery_threshold: float = 0.85,
                 max_candidates: int = 40,
                 enable_curriculum_expansion: bool = True):
        self.embeddings = np.asarray(fused_emb, dtype=np.float32)
        self.prerequisite = np.asarray(prerequisite, dtype=np.float32)
        self.confidence = (
            self.prerequisite.copy() if prerequisite_confidence is None
            else np.asarray(prerequisite_confidence, dtype=np.float32))
        self.hard_graph = (
            (self.prerequisite > 0) & (self.confidence >= hard_edge_confidence)).astype(np.float32)
        self.mastery_threshold = float(mastery_threshold)
        self.max_candidates = int(max_candidates)
        self.enable_curriculum_expansion = bool(enable_curriculum_expansion)
        self.n_concepts = self.embeddings.shape[0]

    def _ancestors(self, node: int) -> set[int]:
        result: set[int] = set()
        stack = np.flatnonzero(self.hard_graph[:, node]).tolist()
        while stack:
            current = int(stack.pop())
            if current in result:
                continue
            result.add(current)
            stack.extend(np.flatnonzero(self.hard_graph[:, current]).tolist())
        return result

    def _closure(self, concepts: set[int], mastery: np.ndarray,
                 selected: set[int]) -> set[int]:
        closure = set(concepts)
        for concept in list(concepts):
            closure.update(
                ancestor for ancestor in self._ancestors(concept)
                if mastery[ancestor] < self.mastery_threshold and ancestor not in selected)
        return closure

    def construct(self, budget: float, personalized_costs: np.ndarray,
                  mastery: np.ndarray, goal_plan: GoalPlan,
                  selected: set[int] | None = None) -> dict:
        selected = set() if selected is None else set(selected)
        cost = np.asarray(personalized_costs, dtype=np.float32)
        mastery = np.asarray(mastery, dtype=np.float32)
        affordable = set(np.flatnonzero(np.isfinite(cost) & (cost <= budget + 1e-9)).tolist())
        unmastered = set(np.flatnonzero(mastery < self.mastery_threshold).tolist())

        closure_unmastered = (self._closure(set(goal_plan.explicit_targets), mastery, selected) & unmastered) - selected
        affordable_mandatory = closure_unmastered & affordable
        mandatory = affordable_mandatory if len(affordable_mandatory) >= 5 else closure_unmastered
        optional_seed = (set(goal_plan.relevant_concepts) & affordable & unmastered) - selected
        optional_seed -= mandatory

        # Rank mandatory candidates (targets and prerequisites) by priority density
        mandatory_order = sorted(
            mandatory,
            key=lambda c: goal_plan.priority[c] / max(np.sqrt(cost[c]), 1e-3),
            reverse=True)
        room = max(0, self.max_candidates - len(mandatory_order))
        optional_order = sorted(
            list(optional_seed),
            key=lambda c: goal_plan.priority[c] / max(np.sqrt(cost[c]), 1e-3),
            reverse=True)[:room]
        candidates = mandatory_order + optional_order

        # Curriculum expansion: populate remaining slots from high-priority unmastered concepts
        if self.enable_curriculum_expansion and len(candidates) < self.max_candidates:
            curriculum_pool = (affordable & unmastered) - selected - set(candidates)
            curriculum_order = sorted(
                list(curriculum_pool),
                key=lambda c: goal_plan.priority[c] / max(cost[c], 1e-6),
                reverse=True)
            remaining_room = max(0, self.max_candidates - len(candidates))
            expanded_pool: set[int] = set()
            for c in curriculum_order:
                c_closure = (self._closure({c}, mastery, selected) & unmastered) - selected
                if (c_closure & affordable) == c_closure:
                    expanded_pool.update(c_closure - set(candidates))
                if len(expanded_pool) >= remaining_room * 2:
                    break
            if not expanded_pool:
                expanded_pool = set(curriculum_order[:remaining_room * 2])
            expanded_order = sorted(
                list(expanded_pool),
                key=lambda c: goal_plan.priority[c] / max(cost[c], 1e-6),
                reverse=True)[:remaining_room]
            candidates = candidates + expanded_order

        # If an unmet hard prerequisite is unaffordable or was pruned, its
        # dependent cannot be exposed as a false frontier item. Iterate until
        # every retained candidate has all unmet hard predecessors retained.
        retained = set(candidates)
        changed = True
        while changed:
            changed = False
            for concept in list(retained):
                predecessors = set(np.flatnonzero(self.hard_graph[:, concept]).tolist())
                missing = {
                    predecessor for predecessor in predecessors
                    if mastery[predecessor] < self.mastery_threshold
                    and predecessor not in selected and predecessor not in retained
                }
                if missing:
                    retained.remove(concept)
                    changed = True
        candidates = [concept for concept in candidates if concept in retained]

        if not candidates:
            return {
                "candidates": [], "frontier": [],
                "local_prerequisites": np.zeros((0, 0), dtype=np.float32),
                "goal_scores": np.zeros(0, dtype=np.float32),
                "costs": np.zeros(0, dtype=np.float32),
            }

        local = self.hard_graph[np.ix_(candidates, candidates)].copy()
        # A prerequisite already mastered or completed need not be selected again.
        for row, concept in enumerate(candidates):
            if mastery[concept] >= self.mastery_threshold or concept in selected:
                local[row, :] = 0.0
        incoming = local.sum(axis=0)
        frontier = [candidates[i] for i in np.flatnonzero(incoming == 0)]
        return {
            "candidates": candidates,
            "frontier": frontier,
            "local_prerequisites": local,
            "goal_scores": goal_plan.priority[candidates].astype(np.float32),
            "costs": cost[candidates].astype(np.float32),
            "mandatory": sorted(mandatory),
            "review_probability": goal_plan.review_probability,
        }
