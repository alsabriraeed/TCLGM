"""Adaptive pedagogical path planner for time-constrained curriculum optimization."""
from __future__ import annotations

import numpy as np
import torch

from training.train_time_model import encode_learner
from utils.data import studied_statistics

from .goal_manager import AdaptiveGoalManager
from .step3_budget_adaptive_candidate_constructor import BudgetAdaptiveCandidateConstructor


class AdaptiveTEMPOPlanner:
    def __init__(self, *, embeddings: torch.Tensor, prerequisite: np.ndarray,
                 prerequisite_confidence: np.ndarray, difficulties: np.ndarray,
                 concept_weights: np.ndarray, learner_encoder, time_model,
                 simulator, decoder, cfg: dict, history: dict,
                 targets: list[int], budget_seconds: float, device: str = "cpu"):
        self.embeddings = embeddings.detach().to(device)
        self.prerequisite = prerequisite
        self.prerequisite_confidence = prerequisite_confidence
        self.difficulties = difficulties
        self.concept_weights = concept_weights
        self.learner_encoder = learner_encoder
        self.time_model = time_model
        self.simulator = simulator
        self.decoder = decoder
        self.cfg = cfg
        self.device = device
        self.history = {name: np.asarray(values).copy() for name, values in history.items()}
        self.history.setdefault("lag", np.zeros(len(self.history.get("q", [])), dtype=np.float32))
        self.targets = [int(target) for target in targets]
        self.remaining_budget = float(budget_seconds)
        self.completed: set[int] = set()
        ablations = cfg.get("ablations", {})
        goal_cfg = cfg["goals"]
        curriculum_weight = float(goal_cfg.get("curriculum_weight", 0.15))
        if not ablations.get("curriculum_progression", True):
            curriculum_weight = 0.0
        self.goal_manager = AdaptiveGoalManager(
            prerequisite, float(goal_cfg["mastery_threshold"]),
            float(goal_cfg["ancestor_discount"]), int(goal_cfg["max_review_goals"]),
            curriculum_weight=curriculum_weight)
        candidate_cfg = cfg["candidate_constructor"]
        hard_conf = float(cfg["graph"]["hard_edge_confidence"])
        if not ablations.get("enforce_prerequisites", True):
            hard_conf = 999.0
        self.constructor = BudgetAdaptiveCandidateConstructor(
            self.embeddings.cpu().numpy(), prerequisite, prerequisite_confidence,
            hard_conf,
            float(goal_cfg["mastery_threshold"]),
            int(candidate_cfg["max_candidates"]),
            enable_curriculum_expansion=ablations.get("curriculum_progression", True))

    @torch.no_grad()
    def recommend(self) -> dict:
        if self.remaining_budget <= 0:
            return {"next_concept": None, "path": [], "reason": "budget_exhausted"}
        ablations = self.cfg.get("ablations", {})
        decoder_cfg = self.cfg["decoder"]
        learner = encode_learner(
            self.history, self.learner_encoder, self.embeddings, self.device)
        count = self.embeddings.shape[0]
        ids = torch.arange(count, device=self.device)
        expected = self.time_model(
            learner.expand(count, -1), self.embeddings, cidx=ids)
        expected_np = expected.cpu().numpy()
        if not ablations.get("use_personalized_time", True):
            if hasattr(self.time_model, "concept_mean_cost"):
                expected_np = np.asarray(self.time_model.concept_mean_cost, dtype=np.float32)
            elif "concept_mean_cost" in self.cfg:
                expected_np = np.asarray(self.cfg["concept_mean_cost"], dtype=np.float32)
            else:
                valid_costs = expected_np[np.isfinite(expected_np)]
                mean_c = float(valid_costs.mean()) if len(valid_costs) else 30.0
                expected_np = np.full(count, mean_c, dtype=np.float32)
        mastery, _ = self.simulator.mastery_after_history(
            self.history["q"], self.history["a"])
        studied, forgetting = studied_statistics(self.history, count)
        recent = self.history["a"][-10:]
        goal_plan = self.goal_manager.formulate(
            mastery, self.targets, studied, forgetting, self.concept_weights,
            float(np.mean(recent)) if len(recent) else 0.5)
        constructed = self.constructor.construct(
            self.remaining_budget, expected_np, mastery, goal_plan, self.completed)
        candidates = constructed["candidates"]
        if not candidates:
            return {"next_concept": None, "path": [], "reason": "no_feasible_candidate"}
        candidate_ids = torch.tensor(candidates, device=self.device)
        targets = torch.tensor(goal_plan.explicit_targets, device=self.device)
        target_context = self.embeddings[targets].mean(dim=0)
        output = self.decoder(
            self.embeddings[candidate_ids],
            torch.tensor(expected_np[candidates], dtype=torch.float32, device=self.device),
            learner.squeeze(0), target_context, self.remaining_budget,
            local_prerequisite=torch.tensor(
                constructed["local_prerequisites"], device=self.device),
            goal_scores=torch.tensor(constructed["goal_scores"], device=self.device),
            difficulties=torch.tensor(self.difficulties[candidates], device=self.device),
            min_steps=int(decoder_cfg["min_steps"]),
            max_steps=int(decoder_cfg["max_steps"]),
            greedy=True,
        )
        path = [candidates[index] for index in output["path"]]
        return {
            "next_concept": path[0] if path else None,
            "path": path,
            "predicted_time": output["predicted_total"],
            "remaining_budget": self.remaining_budget,
            "review_probability": goal_plan.review_probability,
            "frontier": constructed["frontier"],
            "reason": "ok" if path else "policy_stopped",
        }


