"""Train and validate the PyKT DKT Cognitive Simulator on learner interaction logs."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

from models.cognitive_simulator import TimeEmittingDKT
from utils.logging_utils import get_logger

log = get_logger("adaptive_tempo.simulator")


def _sequences(df, scored_split: str, conditioning_splits: tuple[str, ...] = ()):
    sequences = []
    allowed = set(conditioning_splits) | {scored_split}
    sub = df[df["split"].isin(allowed)]
    order_col = "order" if "order" in sub.columns else "start_time"
    for _, group in sub.groupby("uidx"):
        group = group.sort_values(order_col)
        if len(group) < 2 or not (group["split"] == scored_split).any():
            continue
        sequences.append({
            "q": group["cidx"].to_numpy(dtype=np.int64),
            "a": group["correct"].to_numpy(dtype=np.int64),
            "score": (group["split"] == scored_split).to_numpy(dtype=bool),
        })
    return sequences


def _run(
    model: TimeEmittingDKT,
    sequences: list[dict],
    device: str,
    batch_size: int,
    optimizer=None,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    order = np.arange(len(sequences))
    if is_train:
        np.random.shuffle(order)

    losses, all_probs, all_labels = [], [], []

    for start in range(0, len(order), batch_size):
        batch = [sequences[index] for index in order[start : start + batch_size]]
        q_list = [torch.as_tensor(item["q"], dtype=torch.long) for item in batch]
        a_list = [torch.as_tensor(item["a"], dtype=torch.long) for item in batch]
        lengths = torch.tensor([len(item) for item in q_list], device=device)

        q = pad_sequence(q_list, batch_first=True, padding_value=0).to(device)
        a = pad_sequence(a_list, batch_first=True, padding_value=0).to(device)

        B, T = q.shape
        if T < 2:
            continue

        q_in = q[:, :-1]
        a_in = a[:, :-1]
        next_q = q[:, 1:]
        next_a = a[:, 1:].float()

        # pyKT DKT forward
        y = model.dkt(q_in, a_in)  # (B, T-1, K)
        pred = y.gather(2, next_q.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        positions = torch.arange(T - 1, device=device).unsqueeze(0)
        valid = positions < (lengths.unsqueeze(1) - 1)

        if not valid.any():
            continue

        pred_valid = pred[valid]
        target_valid = next_a[valid]

        pred_clamped = torch.clamp(pred_valid, 1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred_clamped, target_valid)

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        n_valid = int(valid.sum())
        losses.append(float(loss.detach()) * n_valid)

        with torch.no_grad():
            all_probs.extend(pred_valid.detach().cpu().numpy().tolist())
            all_labels.extend(target_valid.detach().cpu().numpy().astype(int).tolist())

    total = sum(len(all_labels) for _ in [1]) if all_labels else 0
    mean_loss = float(sum(losses) / max(len(all_labels), 1))
    if len(set(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_probs))
        acc = float(accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int)))
    else:
        auc = 0.5
        acc = 0.5

    return {
        "loss": mean_loss,
        "auc": auc,
        "acc": acc,
        "n": len(all_labels),
    }


def train_simulator(df, cfg: dict, device: str = "cpu"):
    """Trains the pyKT DKT cognitive simulator with next-step mastery BCE loss."""
    simulator_cfg = cfg["simulator"]
    n_concepts = int(df["cidx"].max()) + 1
    emb_size = int(simulator_cfg.get("emb_size", simulator_cfg.get("hidden_dim", 64)))
    dropout = float(simulator_cfg.get("dropout", 0.1))
    lr = float(simulator_cfg.get("lr", 0.001))
    epochs = int(simulator_cfg.get("epochs", 25))
    batch_size = int(simulator_cfg.get("batch_size", 128))
    patience_limit = int(simulator_cfg.get("early_stopping_patience", 5))

    model = TimeEmittingDKT(
        n_concepts=n_concepts,
        hidden_dim=emb_size,
        dropout=dropout,
        max_emitted_seconds=float(cfg.get("data", {}).get("max_elapsed_seconds", 300.0)),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_sequences = _sequences(df, "train")
    val_sequences = _sequences(df, "val", ("train",))
    if not val_sequences:
        val_sequences = _sequences(df, "test")
    validation = val_sequences or train_sequences

    best_auc = 0.0
    best_state = None
    patience = 0

    log.info("Starting pyKT DKT simulator training (K=%d, dim=%d, epochs=%d)", n_concepts, emb_size, epochs)
    for epoch in range(epochs):
        train_metrics = _run(model, train_sequences, device, batch_size, optimizer)
        with torch.no_grad():
            val_metrics = _run(model, validation, device, batch_size)

        log.info(
            "epoch %02d/%02d | train_loss=%.4f train_auc=%.4f | val_loss=%.4f val_auc=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            train_metrics["loss"],
            train_metrics["auc"],
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["acc"],
        )

        if val_metrics["auc"] > best_auc + 1e-4:
            best_auc = val_metrics["auc"]
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                log.info("Early stopping triggered at epoch %d (best_val_auc=%.4f)", epoch + 1, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    final_metrics = _run(model, validation, device, batch_size)
    return model, final_metrics
