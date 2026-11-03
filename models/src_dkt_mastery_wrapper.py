"""Unified Mastery Wrapper (formerly MindSpore SRCDKTMasteryWrapper).

Directly wraps the trained PyTorch pyKT DKT mastery simulator, completely eliminating
any dependency on MindSpore or SRC code while preserving backwards compatibility.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from models.cognitive_simulator import TimeEmittingDKT, SimulationResult

log = logging.getLogger("adaptive_tempo.src_dkt")


def init_mindspore(device_target: str = "GPU") -> None:
    """No-op stub for backwards compatibility (MindSpore is replaced by PyTorch pyKT DKT)."""
    log.info("MindSpore bypassed: using native PyTorch pyKT DKT mastery simulator.")


def load_kes_env(src_data_dir: str | Path, dataset_name: str):
    """Backwards compatibility stub returning dataset name."""
    return dataset_name


class SRCDKTMasteryWrapper:
    """Mastery wrapper delegating to the unified PyTorch pyKT DKT simulator.

    Provides identical interface:
        mastery_after_history(q_hist, a_hist) -> (mastery_np[K], state)
        simulate_path_src_style(state, path)   -> SimulationResult
    """

    def __init__(self, kes_env=None, n_concepts: int = 196, max_history: int = 100, device: str = "cpu"):
        self.K = int(n_concepts)
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.sim = TimeEmittingDKT(n_concepts=self.K, hidden_dim=64).to(self.device)

        # Attempt to load trained checkpoint if present
        root = Path(__file__).resolve().parents[1]
        ckpt_paths = [
            root / f"artifacts_ASSISST{'2017' if self.K == 102 else '2012'}/checkpoints/cognitive_simulator.pt",
            root / f"models/checkpoints/dkt_assist{'17' if self.K == 102 else '12'}.pt",
        ]
        loaded = False
        for p in ckpt_paths:
            if p.exists():
                try:
                    self.sim.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                    log.info("SRCDKTMasteryWrapper successfully loaded PyTorch pyKT DKT from %s", p)
                    loaded = True
                    break
                except Exception as e:
                    log.warning("Could not load %s: %s", p, e)
        if not loaded:
            log.warning("SRCDKTMasteryWrapper using uninitialized PyTorch DKT model.")

        self.sim.eval()

    def eval(self):
        self.sim.eval()
        return self

    def parameters(self):
        return self.sim.parameters()

    def mastery_after_history(
        self,
        q_hist: np.ndarray | Sequence[int],
        a_hist: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        return self.sim.mastery_after_history(q_hist, a_hist)

    def simulate_path_src_style(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        path: list[int] | np.ndarray,
    ) -> SimulationResult:
        return self.sim.simulate_path_src_style(state, list(path))

    def simulate_path(self, state, path: list[int], seed: int = 0, max_attempts: int = 4, n_rollouts: int = 1):
        return self.sim.simulate_path(state, path, seed=seed, max_attempts=max_attempts, n_rollouts=n_rollouts)
