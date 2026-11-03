"""Construction of explicit prerequisite and implicit collaborative graphs."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def hashed_name_features(names: list[str], dim: int = 128) -> np.ndarray:
    """Offline deterministic character n-gram features; no model download."""
    features = np.zeros((len(names), dim), dtype=np.float32)
    for row, raw_name in enumerate(names):
        normalized = f"  {str(raw_name).lower().strip()}  "
        grams = [normalized[i:i + 3] for i in range(max(1, len(normalized) - 2))]
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            column = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[row, column] += sign
        norm = np.linalg.norm(features[row])
        if norm > 0:
            features[row] /= norm
    return features


def _has_path(adjacency: np.ndarray, start: int, goal: int) -> bool:
    stack = [start]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(np.flatnonzero(adjacency[node]).tolist())
    return False


def _add_edges_acyclic(n_concepts: int, edges: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    confidence = np.zeros_like(adjacency)
    for source, target, score in sorted(edges, key=lambda item: item[2], reverse=True):
        if source == target or _has_path(adjacency, target, source):
            continue
        adjacency[source, target] = 1.0
        confidence[source, target] = float(score)
    return adjacency, confidence


def explicit_graph(df_train: pd.DataFrame, n_concepts: int,
                   confidence_threshold: float = 0.70,
                   prerequisite_csv: str | Path | None = None,
                   concept_to_idx: dict[str, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a confidence-scored DAG.

    Expert edges, when supplied, receive confidence 1.0. Log-derived edges are
    directional associations and therefore remain confidence-scored; only
    high-confidence edges should later be used as hard prerequisites.
    """
    transition = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    for _, group in df_train.groupby("uidx"):
        sequence = group.sort_values("event_order")["cidx"].to_numpy(dtype=int)
        for source, target in zip(sequence[:-1], sequence[1:]):
            if source != target:
                transition[source, target] += 1.0

    edges: list[tuple[int, int, float]] = []
    for source in range(n_concepts):
        for target in range(n_concepts):
            if source == target or transition[source, target] < 2:
                continue
            forward = transition[source, target]
            reverse = transition[target, source]
            score = (forward + 1.0) / (forward + reverse + 2.0)
            if score >= confidence_threshold:
                edges.append((source, target, float(score)))

    if prerequisite_csv is not None:
        expert = pd.read_csv(prerequisite_csv)
        if not {"prerequisite", "concept"}.issubset(expert.columns):
            raise ValueError("prerequisite CSV must contain prerequisite and concept columns")
        for row in expert.itertuples(index=False):
            source_raw, target_raw = str(row.prerequisite), str(row.concept)
            if concept_to_idx is None:
                source, target = int(source_raw), int(target_raw)
            else:
                if source_raw not in concept_to_idx or target_raw not in concept_to_idx:
                    continue
                source, target = concept_to_idx[source_raw], concept_to_idx[target_raw]
            edges.append((source, target, 1.0))
    return _add_edges_acyclic(n_concepts, edges)


def collaborative_graph(df_train: pd.DataFrame, n_concepts: int,
                        similarity_threshold: float = 0.85,
                        min_shared_learners: int = 5) -> np.ndarray:
    response = df_train.pivot_table(
        index="uidx", columns="cidx", values="correct", aggfunc="mean")
    matrix = response.reindex(columns=np.arange(n_concepts)).to_numpy(dtype=np.float64)
    adjacency = np.zeros((n_concepts, n_concepts), dtype=np.float32)
    for left in range(n_concepts):
        for right in range(left + 1, n_concepts):
            mask = np.isfinite(matrix[:, left]) & np.isfinite(matrix[:, right])
            if int(mask.sum()) < min_shared_learners:
                continue
            a, b = matrix[mask, left], matrix[mask, right]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            similarity = float(a.dot(b) / denominator) if denominator > 0 else 0.0
            if similarity >= similarity_threshold:
                adjacency[left, right] = adjacency[right, left] = 1.0
    return adjacency


def build_graphs(df: pd.DataFrame, n_concepts: int, cfg: dict,
                 prerequisite_csv: str | Path | None = None,
                 concept_to_idx: dict[str, int] | None = None):
    graph_cfg = cfg["graph"]
    train = df[df["split"] == "train"]
    prerequisite, confidence = explicit_graph(
        train, n_concepts,
        confidence_threshold=float(graph_cfg["transition_confidence"]),
        prerequisite_csv=prerequisite_csv,
        concept_to_idx=concept_to_idx,
    )
    collaborative = collaborative_graph(
        train, n_concepts,
        similarity_threshold=float(graph_cfg["collaborative_similarity"]),
        min_shared_learners=int(graph_cfg["min_shared_learners"]),
    )
    return prerequisite, confidence, collaborative


def save_graphs(directory: str | Path, prerequisite: np.ndarray,
                confidence: np.ndarray, collaborative: np.ndarray) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "prerequisite.npy", prerequisite)
    np.save(directory / "prerequisite_confidence.npy", confidence)
    np.save(directory / "collaborative.npy", collaborative)
