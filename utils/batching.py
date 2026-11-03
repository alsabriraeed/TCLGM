"""History tensorization shared by time and policy components."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


EMPTY_HISTORY = {
    "q": np.asarray([], dtype=np.int64),
    "a": np.asarray([], dtype=np.int64),
    "t": np.asarray([], dtype=np.float32),
    "lag": np.asarray([], dtype=np.float32),
}


def collate_histories(samples: Iterable[dict], concept_embeddings: torch.Tensor,
                      device: str | torch.device):
    samples = list(samples)
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_len = max(1, max(len(s.get("hist", EMPTY_HISTORY)["q"]) for s in samples))
    batch = len(samples)
    dim = concept_embeddings.shape[1]
    ce = torch.zeros(batch, max_len, dim, device=device)
    answers = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    elapsed = torch.zeros(batch, max_len, device=device)
    lag = torch.zeros(batch, max_len, device=device)
    lengths = torch.zeros(batch, dtype=torch.long, device=device)

    for row, sample in enumerate(samples):
        hist = sample.get("hist", EMPTY_HISTORY)
        q = np.asarray(hist.get("q", []), dtype=np.int64)
        n = len(q)
        lengths[row] = n
        if n == 0:
            continue
        q_t = torch.as_tensor(q, dtype=torch.long, device=device)
        ce[row, :n] = concept_embeddings[q_t]
        answers[row, :n] = torch.as_tensor(hist["a"], dtype=torch.long, device=device)
        elapsed[row, :n] = torch.as_tensor(hist["t"], dtype=torch.float32, device=device)
        lag_values = hist.get("lag", np.zeros(n, dtype=np.float32))
        lag[row, :n] = torch.as_tensor(lag_values, dtype=torch.float32, device=device)

    concept_ids = torch.as_tensor(
        [int(s.get("cidx", 0)) for s in samples], dtype=torch.long, device=device)
    targets = torch.as_tensor(
        [float(s.get("t_true", 1.0)) for s in samples], dtype=torch.float32, device=device)
    return ce, answers, elapsed, lag, lengths, concept_ids, targets

