"""Configuration helpers for Adaptive TEMPO."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    # Root directory that contains the tempo dataset (raw/, processed/, splits/).
    # Override in config.yaml or pass --data explicitly to the train command.
    "data_dir": "/home/raed/tempo/data",
    "data": {
        "split_strategy": "user",
        "train_fraction": 0.80,
        "val_fraction": 0.00,
        "test_fraction": 0.20,
        "max_elapsed_seconds": 300.0,
        "max_history": 100,
        "policy_target_horizon": 5,
        "min_history": 4,
    },
    "graph": {
        "embedding_dim": 64,
        "gcn_layers": 2,
        "tau": 0.2,
        "topology_epochs": 20,
        "lr": 1e-3,
        "transition_confidence": 0.70,
        "collaborative_similarity": 0.85,
        "min_shared_learners": 5,
        "hard_edge_confidence": 0.80,
    },
    "learner_encoder": {"hidden_dim": 64, "dropout": 0.1},
    "time_model": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "epochs": 30,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stopping_patience": 5,
    },
    "simulator": {
        "hidden_dim": 64,
        "epochs": 25,
        "batch_size": 64,
        "lr": 1e-3,
        "early_stopping_patience": 5,
        "max_attempts": 4,
        "reward_rollouts": 2,
    },
    "goals": {
        "mastery_threshold": 0.85,
        "ancestor_discount": 0.80,
        "max_review_goals": 5,
    },
    "candidate_constructor": {
        "max_candidates": 30,
    },
    "decoder": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "epochs": 50,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "gamma": 0.99,
        "entropy_weight": 0.01,
        "critic_weight": 0.5,
        "kt_weight": 0.1,
        "gradient_clip": 5.0,
        "max_steps": 40,
        "min_steps": 1,
        "stop_bias": -1.5,
    },
    "constraints": {},
    "ablations": {
        "use_personalized_time": True,      # A1: False -> concept-average costs
        "use_collaborative_graph": True,    # A2: False -> explicit-only topology
        "enforce_prerequisites": True,      # A3: False -> remove hard masking
        "curriculum_progression": True,     # A4: False -> no curriculum expansion
        "use_budget_features": True,        # A5: False -> zero budget features
        "use_kt_aux_loss": True,            # Optional: False -> kt_weight = 0.0
    },
    "training": {
        "budget_fractions": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "episodes_per_epoch": 500,
    },
    "evaluation": {
        "budget_fraction": 0.75,
        "eval_fractions": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        "max_episodes": 100,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML and recursively apply it over documented defaults."""
    if path is None:
        cfg = deepcopy(DEFAULT_CONFIG)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            cfg = _merge(DEFAULT_CONFIG, yaml.safe_load(handle) or {})
    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    return cfg


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested

