"""REINFORCE policy training for Adaptive TEMPO, aligned with SRC's formulation.

The time model and its learner encoder are frozen together. The policy cannot
reduce its apparent cost by changing the representation on which time
estimates depend.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.goal_manager import AdaptiveGoalManager
from models.kt_aux_head import KTAuxHead
from models.step3_budget_adaptive_candidate_constructor import (
    BudgetAdaptiveCandidateConstructor,
)
from models.step4_budget_aware_path_decoder import BudgetAwarePathDecoder
from training.train_time_model import encode_learner
from utils.data import PolicyEpisode, studied_statistics
from utils.logging_utils import get_logger


log = get_logger("adaptive_tempo.policy")


@dataclass
class EpisodeOutcome:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    kt_loss: torch.Tensor
    gain: float
    utilization: float
    path_length: int
    target_gain: float = 0.0
    global_gain: float = 0.0


@torch.no_grad()
def _expected_costs(history, learner_encoder, time_model, embeddings,
                    device: str, ablations: dict | None = None):
    learner = encode_learner(history, learner_encoder, embeddings, device)
    count = embeddings.shape[0]
    ids = torch.arange(count, device=device)
    repeated = learner.expand(count, -1)
    expected = time_model(repeated, embeddings, cidx=ids)
    expected_np = expected.cpu().numpy()
    if ablations and not ablations.get("use_personalized_time", True):
        if hasattr(time_model, "concept_mean_cost"):
            expected_np = np.asarray(time_model.concept_mean_cost, dtype=np.float32)
        else:
            valid_costs = expected_np[np.isfinite(expected_np)]
            mean_c = float(valid_costs.mean()) if len(valid_costs) else 30.0
            expected_np = np.full(count, mean_c, dtype=np.float32)
    return learner.squeeze(0), expected_np


def _episode_reward(initial_score: float, final_score: float) -> float:
    """SRC-style normalized learning gain: (final - initial) / (1 - initial + eps)."""
    return float((final_score - initial_score) / (1.0 - initial_score + 1e-9))


def _episode_outcome(
    episode: PolicyEpisode,
    budget_fraction: float,
    decoder: BudgetAwarePathDecoder,
    kt_head: KTAuxHead,
    embeddings: torch.Tensor,
    prerequisite: np.ndarray,
    prerequisite_confidence: np.ndarray,
    difficulties: np.ndarray,
    weights: np.ndarray,
    learner_encoder,
    time_model,
    simulator,
    goal_manager: AdaptiveGoalManager,
    constructor: BudgetAdaptiveCandidateConstructor,
    cfg: dict,
    device: str,
    generator: torch.Generator,
    seed: int,
) -> EpisodeOutcome | None:
    decoder_cfg = cfg["decoder"]
    max_steps = int(decoder_cfg["max_steps"])
    learner_state, expected_cost = _expected_costs(
        episode.history, learner_encoder, time_model, embeddings, device,
        ablations=cfg.get("ablations", {}))
    mastery, simulator_state = simulator.mastery_after_history(
        episode.history["q"], episode.history["a"])
    studied, forgetting = studied_statistics(episode.history, len(mastery))
    recent = episode.history["a"][-10:]
    recent_accuracy = float(np.mean(recent)) if len(recent) else 0.5
    goal_plan = goal_manager.formulate(
        mastery, episode.targets, studied, forgetting, weights, recent_accuracy)

    valid_costs = expected_cost[np.isfinite(expected_cost)]
    if not len(valid_costs):
        return None
    mean_concept_cost = float(valid_costs.mean())
    ref_session_cost = max_steps * mean_concept_cost
    budget = max(float(valid_costs.min()), budget_fraction * ref_session_cost)
    constructed = constructor.construct(
        budget, expected_cost, mastery, goal_plan, selected=set())
    candidates = constructed["candidates"]
    if not candidates:
        return None

    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    target_tensor = torch.tensor(goal_plan.explicit_targets, dtype=torch.long, device=device)
    target_weight = torch.tensor(weights[list(goal_plan.explicit_targets)], device=device)
    target_context = (
        embeddings[target_tensor] * target_weight.unsqueeze(-1)).sum(dim=0)
    target_context = target_context / target_weight.sum().clamp_min(1e-6)
    local_prerequisite = torch.tensor(
        constructed["local_prerequisites"], dtype=torch.float32, device=device)
    goal_scores = torch.tensor(constructed["goal_scores"], device=device)
    local_difficulty = torch.tensor(difficulties[candidates], device=device)

    output = decoder(
        embeddings[candidate_tensor],
        torch.tensor(expected_cost[candidates], dtype=torch.float32, device=device),
        learner_state, target_context, budget,
        local_prerequisite=local_prerequisite,
        goal_scores=goal_scores,
        difficulties=local_difficulty,
        min_steps=int(decoder_cfg["min_steps"]),
        max_steps=max_steps,
        greedy=False,
        generator=generator,
    )
    if not output["decisions"]:
        return None
    global_path = [candidates[index] for index in output["path"]]
    if not global_path:
        return None

    # --- SRC-style single-pass simulation (correct=1 for each concept) ---
    simulation = simulator.simulate_path_src_style(simulator_state, global_path)
    mastery_trace = simulation.mastery_trace
    realized_times = simulation.realized_times

    # --- Learning Gain Computation (Target & Global) ---
    targets = list(goal_plan.explicit_targets)
    initial_target = float(np.mean(mastery_trace[0][targets]))
    final_target = float(np.mean(mastery_trace[-1][targets]))
    target_gain = _episode_reward(initial_target, final_target)

    initial_global = float(np.mean(mastery_trace[0]))
    final_global = float(np.mean(mastery_trace[-1]))
    global_gain = _episode_reward(initial_global, final_global)

    # --- Blended & Scaled Policy Optimization Reward ---
    train_cfg = cfg.get("training", {})
    target_w = float(train_cfg.get("target_gain_weight", 0.70))
    global_w = float(train_cfg.get("global_gain_weight", 0.30))
    scale = float(train_cfg.get("reward_scale", 100.0))

    raw_reward = target_w * target_gain + global_w * global_gain
    reward = raw_reward
    scaled_reward = raw_reward * scale

    # --- REINFORCE loss: -scaled_reward * sum(log_probs) ---
    decisions = output["decisions"]
    log_probs = torch.stack([d["log_prob"] for d in decisions])
    entropy = torch.stack([d["entropy"] for d in decisions]).mean()
    policy_loss = -(scaled_reward * log_probs.sum())

    # --- KT auxiliary loss (optional regularizer, same as SRC's withKT) ---
    kt_states, kt_targets = [], []
    for index, decision in enumerate(decisions):
        if decision["action"] < 0 or index >= len(global_path):
            continue
        kt_states.append(decision["state"])
        global_concept = global_path[index]
        kt_targets.append(float(mastery_trace[index][global_concept]))
    if kt_states:
        kt_loss = kt_head.loss(
            torch.stack(kt_states),
            torch.tensor(kt_targets, device=embeddings.device))
    else:
        kt_loss = policy_loss * 0.0

    realized_times = simulation.realized_times
    if realized_times and sum(realized_times) > 0.0:
        realized_total = float(sum(realized_times))
    else:
        realized_total = float(sum(
            float(expected_cost[c]) for c in global_path
            if 0 <= c < len(expected_cost)
        ))
    return EpisodeOutcome(
        policy_loss=policy_loss,
        entropy=entropy,
        kt_loss=kt_loss,
        gain=raw_reward,
        utilization=realized_total / max(budget, 1e-6),
        path_length=len(global_path),
        target_gain=target_gain,
        global_gain=global_gain,
    )


def train_constrained_policy(
    episodes: list[PolicyEpisode],
    embeddings: torch.Tensor,
    prerequisite: np.ndarray,
    prerequisite_confidence: np.ndarray,
    difficulties: np.ndarray,
    concept_weights: np.ndarray,
    learner_encoder,
    time_model,
    simulator,
    cfg: dict,
    device: str = "cpu",
    checkpoint_dir: Path | str | None = None,
):
    if not episodes:
        raise ValueError("policy training requires at least one episode")
    embeddings = embeddings.detach().to(device)
    ablations = cfg.get("ablations", {})
    decoder_cfg = cfg["decoder"]
    zero_budget_features = not ablations.get("use_budget_features", True)
    decoder = BudgetAwarePathDecoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        int(decoder_cfg["hidden_dim"]), float(decoder_cfg["dropout"]),
        float(decoder_cfg["stop_bias"]),
        zero_budget_features=zero_budget_features).to(device)
    kt_head = KTAuxHead(int(decoder_cfg["hidden_dim"]),
                        float(decoder_cfg["dropout"])).to(device)

    best_score, best_state = -float("inf"), None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        decoder_file = ckpt_path / "path_decoder.pt"
        kt_file = ckpt_path / "kt_aux_head.pt"
        loaded_decoder = loaded_kt = False
        if decoder_file.exists():
            try:
                decoder.load_state_dict(torch.load(decoder_file, map_location=device, weights_only=True))
                log.info("Loaded existing policy checkpoint from %s", decoder_file)
                loaded_decoder = True
            except Exception as e:
                log.warning("Could not load policy checkpoint from %s (%s). Reinitializing fresh decoder.", decoder_file, e)
        if kt_file.exists():
            try:
                kt_head.load_state_dict(torch.load(kt_file, map_location=device, weights_only=True))
                log.info("Loaded existing KT head checkpoint from %s", kt_file)
                loaded_kt = True
            except Exception as e:
                log.warning("Could not load KT head checkpoint from %s (%s). Reinitializing fresh KT head.", kt_file, e)
        if loaded_decoder and loaded_kt:
            best_state = (deepcopy(decoder.state_dict()), deepcopy(kt_head.state_dict()))

    # Preserve calibrated cost semantics during policy learning.
    for frozen_model in (learner_encoder, time_model, simulator):
        if hasattr(frozen_model, "eval"):
            frozen_model.eval()
        if hasattr(frozen_model, "parameters"):
            for parameter in frozen_model.parameters():
                parameter.requires_grad_(False)

    goal_cfg = cfg["goals"]
    curriculum_weight = float(goal_cfg.get("curriculum_weight", 0.15))
    if not ablations.get("curriculum_progression", True):
        curriculum_weight = 0.0

    goal_manager = AdaptiveGoalManager(
        prerequisite, float(goal_cfg["mastery_threshold"]),
        float(goal_cfg["ancestor_discount"]), int(goal_cfg["max_review_goals"]),
        curriculum_weight=curriculum_weight)

    candidate_cfg = cfg["candidate_constructor"]
    hard_conf = float(cfg["graph"]["hard_edge_confidence"])
    if not ablations.get("enforce_prerequisites", True):
        hard_conf = 999.0  # Disables hard graph edges

    constructor = BudgetAdaptiveCandidateConstructor(
        embeddings.cpu().numpy(), prerequisite, prerequisite_confidence,
        hard_edge_confidence=hard_conf,
        mastery_threshold=float(goal_cfg["mastery_threshold"]),
        max_candidates=int(candidate_cfg["max_candidates"]),
        enable_curriculum_expansion=ablations.get("curriculum_progression", True))

    kt_loss_weight = float(decoder_cfg["kt_weight"])
    if not ablations.get("use_kt_aux_loss", True):
        kt_loss_weight = 0.0

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(kt_head.parameters()),
        lr=float(decoder_cfg["lr"]), weight_decay=float(decoder_cfg["weight_decay"]))
    rng = np.random.default_rng(cfg["seed"])
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    fractions = [float(value) for value in cfg["training"]["budget_fractions"]]
    batch_size = int(decoder_cfg["batch_size"])
    ep_per_ep = cfg.get("training", {}).get("episodes_per_epoch", 500)
    if ep_per_ep is None or (isinstance(ep_per_ep, (int, float)) and int(ep_per_ep) <= 0) or str(ep_per_ep).lower() == "all":
        episodes_per_epoch = len(episodes)
    else:
        episodes_per_epoch = min(int(ep_per_ep), len(episodes))
    history = {key: [] for key in ("gain", "target_gain", "global_gain", "utilization", "path_length")}

    for epoch in range(int(decoder_cfg["epochs"])):
        decoder.train(); kt_head.train()
        sampled = rng.choice(len(episodes), size=episodes_per_epoch, replace=False)
        epoch_outcomes: list[EpisodeOutcome] = []
        for start in range(0, len(sampled), batch_size):
            optimizer.zero_grad()
            batch_outcomes = []
            for episode_index in sampled[start:start + batch_size]:
                outcome = _episode_outcome(
                    episodes[int(episode_index)], float(rng.choice(fractions)),
                    decoder, kt_head, embeddings, prerequisite,
                    prerequisite_confidence, difficulties, concept_weights,
                    learner_encoder, time_model, simulator, goal_manager,
                    constructor, cfg, device,
                    generator, int(cfg["seed"] + epoch * 10000 + episode_index))
                if outcome is not None:
                    batch_outcomes.append(outcome)
            if not batch_outcomes:
                continue
            loss = torch.stack([
                item.policy_loss
                - float(decoder_cfg["entropy_weight"]) * item.entropy
                + kt_loss_weight * item.kt_loss
                for item in batch_outcomes]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) + list(kt_head.parameters()),
                float(decoder_cfg["gradient_clip"]))
            optimizer.step()
            epoch_outcomes.extend(batch_outcomes)

        if not epoch_outcomes:
            log.warning("epoch %d produced no feasible training episodes", epoch + 1)
            continue
        mean_gain = float(np.mean([item.gain for item in epoch_outcomes]))
        mean_target_gain = float(np.mean([item.target_gain for item in epoch_outcomes]))
        mean_global_gain = float(np.mean([item.global_gain for item in epoch_outcomes]))
        utilization = float(np.mean([item.utilization for item in epoch_outcomes]))
        path_length = float(np.mean([item.path_length for item in epoch_outcomes]))
        for key, value in {
            "gain": mean_gain,
            "target_gain": mean_target_gain,
            "global_gain": mean_global_gain,
            "utilization": utilization,
            "path_length": path_length,
        }.items():
            history[key].append(value)
        score = mean_gain
        if score > best_score:
            best_score = score
            best_state = (deepcopy(decoder.state_dict()), deepcopy(kt_head.state_dict()))
        log.info(
            "epoch %d reward=%.4f (global=%.4f, target=%.4f) util=%.2f len=%.2f",
            epoch + 1, mean_gain, mean_global_gain, mean_target_gain, utilization, path_length)
    if best_state is not None:
        decoder.load_state_dict(best_state[0])
        kt_head.load_state_dict(best_state[1])
    decoder.eval(); kt_head.eval()
    return decoder, kt_head, history


