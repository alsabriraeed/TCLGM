"""Dataset preparation for interaction logs.

Required semantic fields are learner, concept, correctness, elapsed time, and
event order. Common column aliases are accepted and normalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ALIASES = {
    "user_id": ("user_id", "student_id", "studentid", "learner_id", "user", "userid", "student"),
    "skill": ("skill", "concept", "concept_id", "conceptid", "skill_name", "skillname", "skill_id", "skillid"),
    "correct": ("correct", "is_correct", "iscorrect", "score", "answer"),
    "elapsed_sec": (
        "elapsed_sec", "elapsed_time", "elapsedtime", "duration", "time_taken", "timetaken",
        "response_time", "responsetime", "time_spent", "timespent", "ms_first_response", "overlap_time"
    ),
    "timestamp": ("timestamp", "start_time", "starttime", "event_time", "eventtime", "order", "time", "date"),
}


@dataclass(frozen=True)
class PolicyEpisode:
    learner_id: int
    history: dict[str, np.ndarray]
    targets: tuple[int, ...]
    future_concepts: tuple[int, ...]


def _find_column(frame: pd.DataFrame, semantic: str, required: bool = True) -> str | None:
    # First exact match
    norm_map = {c.lower().replace("_", ""): c for c in frame.columns}
    for candidate in ALIASES.get(semantic, (semantic,)):
        if candidate in frame.columns:
            return candidate
        cand_norm = candidate.lower().replace("_", "")
        if cand_norm in norm_map:
            return norm_map[cand_norm]
    if required:
        raise ValueError(
            f"missing {semantic!r}; accepted columns: {', '.join(ALIASES[semantic])}")
    return None


def subsample_by_learners(
    df: pd.DataFrame,
    sample_fraction: float | None = None,
    max_learners: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsamples an interactions dataframe at the learner level.

    Preserves all chronological interactions and split assignments per sampled learner,
    allowing for rapid prototyping, quick training, and smoke testing without data fragmentation.

    Args:
        df: DataFrame containing student interactions (must have 'uidx' or 'user_id').
        sample_fraction: Fraction of learners to retain (e.g. 0.1 for 10%).
        max_learners: Maximum number of distinct learners to retain.
        seed: Random seed for reproducible sampling.

    Returns:
        Subsampled DataFrame containing all interactions for the sampled learners.
    """
    u_col = "uidx" if "uidx" in df.columns else ("user_id" if "user_id" in df.columns else None)
    if u_col is None:
        return df

    unique_users = df[u_col].unique()
    n_users = len(unique_users)

    if sample_fraction is not None and 0.0 < float(sample_fraction) < 1.0:
        n_keep = max(1, int(n_users * float(sample_fraction)))
    elif max_learners is not None and int(max_learners) > 0:
        n_keep = min(n_users, int(max_learners))
    else:
        return df

    rng = np.random.default_rng(seed)
    sampled_users = set(rng.choice(unique_users, size=n_keep, replace=False))
    return df[df[u_col].isin(sampled_users)].copy()


def load_interactions(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    user_col = _find_column(raw, "user_id", required=True)
    skill_col = _find_column(raw, "skill", required=True)
    correct_col = _find_column(raw, "correct", required=True)
    elapsed_col = _find_column(raw, "elapsed_sec", required=False)
    time_col = _find_column(raw, "timestamp", required=False)

    df = pd.DataFrame()
    df["user_id"] = raw[user_col]
    df["skill"] = raw[skill_col]
    df["correct"] = pd.to_numeric(raw[correct_col], errors="coerce").fillna(0).clip(0, 1).round().astype(int)

    # Process timestamps and event_order
    if time_col and time_col in raw.columns:
        parsed_time = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        if parsed_time.notna().any():
            fallback = pd.Series(np.arange(len(raw)), index=raw.index, dtype=float)
            seconds = parsed_time.map(lambda v: v.timestamp() if pd.notna(v) else np.nan)
            df["event_order"] = pd.to_numeric(seconds, errors="coerce").fillna(fallback)
            df["timestamp"] = raw[time_col]
        else:
            df["event_order"] = pd.to_numeric(raw[time_col], errors="coerce").fillna(
                pd.Series(np.arange(len(raw)), index=raw.index, dtype=float))
            df["timestamp"] = df["event_order"]
    else:
        df["event_order"] = pd.Series(np.arange(len(raw)), index=raw.index, dtype=float)
        df["timestamp"] = df["event_order"]

    # Process elapsed_sec if column exists
    max_elapsed = float(cfg["data"].get("max_elapsed_seconds", 300.0))
    if elapsed_col:
        elapsed = pd.to_numeric(raw[elapsed_col], errors="coerce")
        if elapsed_col in ("ms_first_response", "overlap_time") and elapsed.median() > 1000.0:
            elapsed = elapsed / 1000.0
        df["elapsed_sec"] = elapsed.clip(lower=0.1, upper=max_elapsed)
    else:
        df["elapsed_sec"] = np.nan

    df = df.dropna(subset=["user_id", "skill", "correct"]).copy()

    users = sorted(df["user_id"].astype(str).unique())
    concepts = sorted(df["skill"].astype(str).unique())
    user_to_idx = {name: idx for idx, name in enumerate(users)}
    concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
    df["uidx"] = df["user_id"].astype(str).map(user_to_idx).astype(int)
    df["cidx"] = df["skill"].astype(str).map(concept_to_idx).astype(int)
    df = df.sort_values(["uidx", "event_order"]).reset_index(drop=True)

    diff_order = df["event_order"].diff().fillna(0.0)
    same_user_prev = (df["uidx"] == df["uidx"].shift(1))
    df["lag_sec"] = np.where(same_user_prev, diff_order.clip(lower=0.0), 0.0).astype(np.float32)

    if elapsed_col is None or df["elapsed_sec"].isna().any():
        same_user_next = (df["uidx"] == df["uidx"].shift(-1))
        diff_next = (df["event_order"].shift(-1) - df["event_order"]).clip(lower=1.0, upper=max_elapsed)
        derived = np.where(same_user_next, diff_next, 30.0)
        df["elapsed_sec"] = df["elapsed_sec"].fillna(pd.Series(derived, index=df.index)).fillna(30.0).astype(np.float32)

    # Splitting strategy
    if "split" in raw.columns and set(raw["split"].dropna().unique()).issubset({"train", "val", "test"}):
        df["split"] = raw["split"].values[:len(df)]
    else:
        train_fraction = float(cfg["data"]["train_fraction"])
        val_fraction = float(cfg["data"].get("val_fraction", 0.0))
        split_strategy = str(cfg["data"].get("split_strategy", "user")).lower()

        if split_strategy == "user" or val_fraction == 0.0:
            rng = np.random.default_rng(int(cfg.get("seed", 42)))
            shuffled = rng.permutation(len(users))
            train_end = int(np.floor(len(users) * train_fraction))
            val_end = int(np.floor(len(users) * (train_fraction + val_fraction)))
            user_split = {}
            for pos, uidx in enumerate(shuffled):
                if pos < train_end:
                    user_split[uidx] = "train"
                elif pos < val_end:
                    user_split[uidx] = "val"
                else:
                    user_split[uidx] = "test"
            df["split"] = df["uidx"].map(user_split).fillna("train")
        else:
            split = np.full(len(df), "train", dtype=object)
            for _, indices in df.groupby("uidx", sort=False).groups.items():
                indices = np.asarray(list(indices))
                n = len(indices)
                train_end = max(1, int(np.floor(n * train_fraction)))
                val_end = max(train_end, int(np.floor(n * (train_fraction + val_fraction))))
                val_end = min(val_end, n)
                split[indices[train_end:val_end]] = "val"
                split[indices[val_end:]] = "test"
            df["split"] = split
    metadata = {
        "user_to_idx": user_to_idx,
        "concept_to_idx": concept_to_idx,
        "idx_to_concept": concepts,
        "n_users": len(users),
        "n_concepts": len(concepts),
        "n_interactions": len(df),
    }
    return df, metadata


def history_from_frame(frame: pd.DataFrame, max_history: int | None = None) -> dict:
    ordered = frame.sort_values("event_order")
    if max_history is not None:
        ordered = ordered.tail(max_history)
    return {
        "q": ordered["cidx"].to_numpy(dtype=np.int64),
        "a": ordered["correct"].to_numpy(dtype=np.int64),
        "t": ordered["elapsed_sec"].to_numpy(dtype=np.float32),
        "lag": ordered["lag_sec"].to_numpy(dtype=np.float32),
    }


def learner_histories(df: pd.DataFrame, splits: Iterable[str],
                      max_history: int | None = None) -> dict[int, dict]:
    sub = df[df["split"].isin(list(splits))]
    return {
        int(uid): history_from_frame(group, max_history)
        for uid, group in sub.groupby("uidx")
    }


def time_samples(df: pd.DataFrame, target_split: str, max_history: int,
                 max_samples: int | None = None, seed: int = 42) -> list[dict]:
    """Create leakage-safe (past history -> next duration) examples with fast numpy slicing."""
    sub = df[df["split"] == target_split].sort_values(["uidx", "event_order"])
    if sub.empty:
        return []

    uidx_arr = sub["uidx"].to_numpy(dtype=np.int64)
    cidx_arr = sub["cidx"].to_numpy(dtype=np.int64)
    elapsed_arr = sub["elapsed_sec"].to_numpy(dtype=np.float32)
    correct_arr = sub["correct"].to_numpy(dtype=np.int64)
    lag_arr = sub["lag_sec"].to_numpy(dtype=np.float32)

    change_mask = np.concatenate(([True], uidx_arr[1:] != uidx_arr[:-1], [True]))
    user_boundaries = np.where(change_mask)[0]

    all_positions = []
    for i in range(len(user_boundaries) - 1):
        u_start = user_boundaries[i]
        u_end = user_boundaries[i + 1]
        if u_end - u_start > 1:
            for pos in range(u_start + 1, u_end):
                all_positions.append((u_start, pos))

    if not all_positions:
        return []

    if max_samples is not None and len(all_positions) > max_samples:
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(all_positions), size=max_samples, replace=False)
        selected_positions = [all_positions[j] for j in chosen_idx]
    else:
        selected_positions = all_positions

    samples: list[dict] = []
    for u_start, pos in selected_positions:
        hist_start = max(u_start, pos - max_history)
        samples.append({
            "uidx": int(uidx_arr[pos]),
            "cidx": int(cidx_arr[pos]),
            "t_true": float(elapsed_arr[pos]),
            "hist": {
                "q": cidx_arr[hist_start:pos],
                "a": correct_arr[hist_start:pos],
                "t": elapsed_arr[hist_start:pos],
                "lag": lag_arr[hist_start:pos],
            },
        })
    return samples


def policy_episodes(df: pd.DataFrame, cfg: dict, split: str = "train",
                     n_targets: int = 3) -> list[PolicyEpisode]:
    min_history = int(cfg["data"]["min_history"])
    max_history = int(cfg["data"]["max_history"])
    n_concepts = int(df["cidx"].max()) + 1
    episodes: list[PolicyEpisode] = []
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    sub = df[df["split"] == split]
    for uid, group in sub.groupby("uidx"):
        group = group.sort_values("event_order").reset_index(drop=True)
        if len(group) < min_history + 1:
            continue
        prefix = group.tail(min(max_history, len(group)))
        history = history_from_frame(prefix)
        # SRC-style: sample n_targets random concepts as targets
        targets = tuple(rng.choice(n_concepts, size=n_targets, replace=False).tolist())
        episodes.append(PolicyEpisode(
            learner_id=int(uid),
            history=history,
            targets=targets,
            future_concepts=targets,
        ))
    return episodes



def studied_statistics(history: dict, n_concepts: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(n_concepts, dtype=np.float32)
    last_lag = np.zeros(n_concepts, dtype=np.float32)
    for q, lag in zip(history.get("q", []), history.get("lag", [])):
        counts[int(q)] += 1
        last_lag[int(q)] = float(lag)
    forgetting = 1.0 - np.exp(-last_lag / (7.0 * 24.0 * 3600.0 + 1e-9))
    return counts, forgetting.astype(np.float32)


def concept_weights(df: pd.DataFrame, n_concepts: int) -> np.ndarray:
    frequency = df[df["split"] == "train"]["cidx"].value_counts()
    values = np.asarray([frequency.get(i, 0) for i in range(n_concepts)], dtype=np.float32)
    values = np.log1p(values)
    if values.max() > 0:
        values /= values.max()
    return 0.5 + 0.5 * values


def concept_difficulties(df: pd.DataFrame, n_concepts: int) -> np.ndarray:
    train = df[df["split"] == "train"]
    correct = train.groupby("cidx")["correct"].agg(["sum", "count"])
    difficulty = np.full(n_concepts, 0.5, dtype=np.float32)
    for concept in range(n_concepts):
        if concept in correct.index:
            row = correct.loc[concept]
            probability = (float(row["sum"]) + 1.0) / (float(row["count"]) + 2.0)
            difficulty[concept] = 1.0 - probability
    return difficulty
