"""Prerequisite-feasible gain-per-safe-time greedy reference."""
from __future__ import annotations

import numpy as np

from .cognitive_simulator import TimeEmittingDKT, learning_gain


class GreedyAnchor:
    def __init__(self, simulator: TimeEmittingDKT, seed: int = 0,
                 max_attempts: int = 4):
        self.simulator = simulator
        self.seed = seed
        self.max_attempts = max_attempts

    def run(self, candidates: list[int], costs: np.ndarray, budget: float,
            simulator_state, mastery: np.ndarray, targets: list[int],
            prerequisite: np.ndarray | None = None) -> dict:
        selected: list[int] = []
        remaining = set(candidates)
        spent = 0.0
        start = float(np.mean(mastery[targets]))
        current_gain = 0.0
        prerequisite = (np.zeros((len(costs), len(costs)), dtype=np.float32)
                        if prerequisite is None else prerequisite)
        while remaining:
            best = None
            best_density = 0.0
            best_gain = current_gain
            for concept in remaining:
                if costs[concept] > budget - spent + 1e-9:
                    continue
                predecessors = set(np.flatnonzero(prerequisite[:, concept]).tolist())
                if any(mastery[p] < 0.85 and p not in selected for p in predecessors):
                    continue
                trial = selected + [concept]
                end_mastery, _ = self.simulator.simulate_path(
                    simulator_state, trial, seed=self.seed,
                    max_attempts=self.max_attempts)
                gain = learning_gain(start, float(np.mean(end_mastery[targets])))
                density = (gain - current_gain) / max(float(costs[concept]), 1e-6)
                if density > best_density:
                    best, best_density, best_gain = concept, density, gain
            if best is None:
                break
            selected.append(best)
            remaining.remove(best)
            spent += float(costs[best])
            current_gain = best_gain
        return {"path": selected, "LG": current_gain, "predicted_total": spent}

