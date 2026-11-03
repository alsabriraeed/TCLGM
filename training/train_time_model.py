from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from models.learner_state_encoder import LearnerStateEncoder
from models.step2_personalized_time_cost_model import PersonalizedTimeCostModel
from utils.batching import collate_histories
from utils.logging_utils import get_logger


log = get_logger("adaptive_tempo.time")


def _run(samples: list[dict], learner_encoder: LearnerStateEncoder,
         time_model: PersonalizedTimeCostModel, embeddings: torch.Tensor,
         batch_size: int, device: str, optimizer=None, rng=None):
    indices = np.arange(len(samples))
    if optimizer is not None and rng is not None:
        rng.shuffle(indices)
    total_loss = total_items = 0
    absolute_errors: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch = [samples[index] for index in indices[start:start + batch_size]]
        ce, answers, elapsed, lag, lengths, concept_ids, true_time = collate_histories(
            batch, embeddings, device)
        learner = learner_encoder(ce, answers, elapsed, lengths, lag)
        expected = time_model(learner, embeddings[concept_ids], cidx=concept_ids)
        loss = time_model.loss(
            learner, embeddings[concept_ids], true_time, cidx=concept_ids)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(learner_encoder.parameters()) + list(time_model.parameters()), 5.0)
            optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
        total_items += len(batch)
        absolute_errors.append((expected.detach().cpu() - true_time.cpu()).abs().numpy())
    errors = np.concatenate(absolute_errors) if absolute_errors else np.asarray([])
    return {
        "loss": total_loss / max(total_items, 1),
        "mae_sec": float(errors.mean()) if len(errors) else float("nan"),
        "n": total_items,
    }


def train_time_cost_model(samples_train: list[dict], samples_val: list[dict],
                          embeddings: torch.Tensor, cfg: dict,
                          device: str = "cpu"):
    if not samples_train:
        raise ValueError("time model requires at least one training sample")
    time_cfg = cfg["time_model"]
    learner_encoder = LearnerStateEncoder(
        embeddings.shape[1], int(cfg["learner_encoder"]["hidden_dim"]),
        float(cfg["learner_encoder"]["dropout"])).to(device)
    time_model = PersonalizedTimeCostModel(
        learner_dim=int(cfg["learner_encoder"]["hidden_dim"]),
        concept_dim=embeddings.shape[1], num_concepts=embeddings.shape[0],
        hidden=int(time_cfg["hidden_dim"]),
        dropout=float(time_cfg["dropout"])).to(device)
    embeddings = embeddings.detach().to(device)
    time_model.set_target_stats(torch.tensor(
        [sample["t_true"] for sample in samples_train], device=device))
    optimizer = torch.optim.AdamW(
        list(learner_encoder.parameters()) + list(time_model.parameters()),
        lr=float(time_cfg["lr"]), weight_decay=float(time_cfg["weight_decay"]))
    rng = np.random.default_rng(cfg["seed"])
    validation = samples_val or samples_train
    best = float("inf")
    best_state = None
    patience = 0
    for epoch in range(int(time_cfg["epochs"])):
        learner_encoder.train(); time_model.train()
        train_metrics = _run(
            samples_train, learner_encoder, time_model, embeddings,
            int(time_cfg["batch_size"]), device, optimizer, rng)
        learner_encoder.eval(); time_model.eval()
        with torch.no_grad():
            val_metrics = _run(
                validation, learner_encoder, time_model, embeddings,
                int(time_cfg["batch_size"]), device)
        if val_metrics["loss"] < best - 1e-5:
            best = val_metrics["loss"]
            best_state = (deepcopy(learner_encoder.state_dict()),
                          deepcopy(time_model.state_dict()))
            patience = 0
        else:
            patience += 1
        log.info("epoch %d/%d train=%.4f val=%.4f val_MAE=%.1fs",
                 epoch + 1, int(time_cfg["epochs"]), train_metrics["loss"],
                 val_metrics["loss"], val_metrics["mae_sec"])
        if patience >= int(time_cfg["early_stopping_patience"]):
            log.info("time model early stopping at epoch %d", epoch + 1)
            break
    if best_state is not None:
        learner_encoder.load_state_dict(best_state[0])
        time_model.load_state_dict(best_state[1])
    learner_encoder.eval(); time_model.eval()
    metrics = _run(validation, learner_encoder, time_model, embeddings,
                   int(time_cfg["batch_size"]), device)
    return learner_encoder, time_model, metrics


@torch.no_grad()
def encode_learner(history: dict, learner_encoder: LearnerStateEncoder,
                   embeddings: torch.Tensor, device: str):
    sample = [{"hist": history, "cidx": 0, "t_true": 1.0}]
    ce, answers, elapsed, lag, lengths, _, _ = collate_histories(sample, embeddings, device)
    return learner_encoder(ce, answers, elapsed, lengths, lag)


@torch.no_grad()
def predict_all_costs(history: dict, learner_encoder: LearnerStateEncoder,
                      time_model: PersonalizedTimeCostModel,
                      embeddings: torch.Tensor, device: str):
    embeddings = embeddings.to(device)
    learner = encode_learner(history, learner_encoder, embeddings, device)
    count = embeddings.shape[0]
    repeated = learner.expand(count, -1)
    ids = torch.arange(count, device=device)
    expected = time_model(repeated, embeddings, cidx=ids)
    return expected.cpu().numpy(), learner.squeeze(0)


