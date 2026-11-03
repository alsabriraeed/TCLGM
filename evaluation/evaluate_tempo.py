from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch

from models.cognitive_simulator import learning_gain
from models.online_planner import AdaptiveTEMPOPlanner
from training.train_time_model import encode_learner


def _compute_summary_metrics(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "mean_learning_gain": float(np.mean([item["gain"] for item in records])),
        "mean_final_target_mastery": float(np.mean([item["final_target_mastery"] for item in records])),
        "mean_global_learning_gain": float(np.mean([item["global_gain"] for item in records])),
        "mean_realized_time": float(np.mean([item["realized_time"] for item in records])),
        "mean_budget_utilization": float(np.mean([item["utilization"] for item in records])),
        "mean_path_length": float(np.mean([item["path_length"] for item in records])),
        "mean_gain_per_minute": float(np.mean([item["gain_per_minute"] for item in records])),
        "prerequisite_violation_rate": float(
            np.mean([item["prerequisite_violations"] > 0 for item in records])
        ),
        "records": records,
    }


def _check_prerequisite_violations(path: list[int], initial_mastery: np.ndarray,
                                   hard_prereq: np.ndarray,
                                   mastery_threshold: float) -> int:
    violations = 0
    selected: set[int] = set()
    for concept in path:
        for predecessor in np.flatnonzero(hard_prereq[:, concept]):
            if initial_mastery[predecessor] < mastery_threshold and int(predecessor) not in selected:
                violations += 1
        selected.add(concept)
    return violations


@torch.no_grad()
def evaluate_policy(episodes, *, embeddings, prerequisite,
                    prerequisite_confidence, difficulties, concept_weights,
                    learner_encoder, time_model, simulator, decoder, cfg,
                    device="cpu", budget_seconds: float | None = None,
                    budget_fraction: float | None = None,
                    max_episodes: int | None = None):
    """Standard open-loop single-shot path evaluation."""
    records = []
    hard_threshold = float(cfg["graph"].get("hard_edge_confidence", 0.80))
    hard_prereq = ((prerequisite > 0) & (prerequisite_confidence >= hard_threshold)).astype(np.float32)
    eval_list = episodes if (max_episodes is None or int(max_episodes) <= 0) else episodes[:int(max_episodes)]
    mastery_threshold = float(cfg["goals"]["mastery_threshold"])

    for index, episode in enumerate(eval_list):
        mastery, state = simulator.mastery_after_history(
            episode.history["q"], episode.history["a"])
        targets = list(episode.targets)
        if budget_seconds is None:
            frac = (float(budget_fraction) if budget_fraction is not None
                    else float(cfg.get("evaluation", {}).get("budget_fraction", 0.75)))
        else:
            frac = None

        # Always compute expected costs (used for budget AND realized time)
        learner = encode_learner(episode.history, learner_encoder, embeddings, device)
        count = embeddings.shape[0]
        ids = torch.arange(count, device=device)
        repeated = learner.expand(count, -1)
        expected = time_model(repeated, embeddings, cidx=ids)
        expected_np = expected.detach().cpu().numpy()
        valid_expected = expected_np[np.isfinite(expected_np)]
        max_steps = int(cfg.get("decoder", {}).get("max_steps", 80))
        mean_concept_cost = float(np.mean(valid_expected)) if len(valid_expected) else 135.0
        ref_session_cost = max_steps * mean_concept_cost

        if budget_seconds is None:
            budget = max(
                float(np.min(valid_expected)) if len(valid_expected) else 60.0,
                frac * ref_session_cost)
        else:
            budget = float(budget_seconds)

        planner = AdaptiveTEMPOPlanner(
            embeddings=embeddings, prerequisite=prerequisite,
            prerequisite_confidence=prerequisite_confidence,
            difficulties=difficulties, concept_weights=concept_weights,
            learner_encoder=learner_encoder, time_model=time_model,
            simulator=simulator, decoder=decoder, cfg=cfg,
            history=episode.history, targets=targets,
            budget_seconds=budget, device=device)
        recommendation = planner.recommend()
        raw_path = recommendation["path"]

        # Truncate raw_path strictly to the prefix that fits within budget; ignore remaining concepts
        completed_path = []
        spent_time = 0.0
        for c in raw_path:
            c_cost = float(expected_np[c]) if 0 <= c < len(expected_np) else mean_concept_cost
            if spent_time + c_cost <= budget + 1e-6:
                completed_path.append(c)
                spent_time += c_cost
            else:
                break

        # Simulate ONLY completed_path
        start = float(np.mean(mastery[targets]))
        global_start = float(np.mean(mastery))

        if completed_path:
            sim_mode = cfg.get("simulator", {}).get("eval_simulation_mode", "src_style")
            if sim_mode == "detailed" and hasattr(simulator, "simulate_path_detailed"):
                simulation = simulator.simulate_path_detailed(
                    state, completed_path, seed=int(cfg["seed"]) + index,
                    max_attempts=int(cfg["simulator"]["max_attempts"]))
            else:
                simulation = simulator.simulate_path_src_style(state, completed_path)
            end = float(np.mean(simulation.final_mastery[targets]))
            global_end = float(np.mean(simulation.final_mastery))
            if simulation.realized_times:
                realized = float(sum(simulation.realized_times))
            else:
                realized = spent_time
            violations = _check_prerequisite_violations(completed_path, mastery, hard_prereq, mastery_threshold)
        else:
            end = start
            global_end = global_start
            realized = 0.0
            violations = 0

        gain = learning_gain(start, end)
        gain_per_min = (gain / (realized / 60.0)) if realized > 0 else 0.0

        records.append({
            "gain": gain,
            "final_target_mastery": end,
            "initial_target_mastery": start,
            "global_gain": learning_gain(global_start, global_end),
            "realized_time": realized,
            "budget": budget,
            "utilization": min(1.0, realized / max(budget, 1e-6)),
            "path_length": len(completed_path),
            "gain_per_minute": gain_per_min,
            "prerequisite_violations": violations,
        })

    return _compute_summary_metrics(records)


@torch.no_grad()
def evaluate_fixed_budget(episodes, *, budget_seconds: float, embeddings, prerequisite,
                          prerequisite_confidence, difficulties, concept_weights,
                          learner_encoder, time_model, simulator, decoder, cfg,
                          device="cpu", max_episodes: int | None = None):
    """E2: Fixed physical time budget evaluation."""
    return evaluate_policy(
        episodes, embeddings=embeddings, prerequisite=prerequisite,
        prerequisite_confidence=prerequisite_confidence,
        difficulties=difficulties, concept_weights=concept_weights,
        learner_encoder=learner_encoder, time_model=time_model,
        simulator=simulator, decoder=decoder, cfg=cfg,
        device=device, budget_seconds=budget_seconds,
        max_episodes=max_episodes
    )

