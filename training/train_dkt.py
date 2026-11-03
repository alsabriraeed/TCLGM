"""High-performance training pipeline for pyKT Deep Knowledge Tracing (DKT).

Trains the PyTorch DKT mastery model on ASSISTments 2012 (196 concepts) and
ASSISTments 2017 (102 concepts), saves checkpoints, and evaluates ROC-AUC / Accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

# Add repository root to path
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.dkt import DKT
from utils.data import subsample_by_learners


def build_sequences_from_df(
    df: pd.DataFrame, max_seq_len: int = 100, min_seq_len: int = 3
) -> List[Dict[str, np.ndarray]]:
    """Groups interaction dataframe by user and chunks into windows of max_seq_len."""
    sequences = []
    # Sort interactions by user and chronological event order
    order_col = "order" if "order" in df.columns else "start_time"
    sorted_df = df.sort_values(["uidx", order_col])

    for uidx, group in sorted_df.groupby("uidx"):
        q_all = group["cidx"].to_numpy(dtype=np.int64)
        r_all = group["correct"].to_numpy(dtype=np.int64)

        if len(q_all) < min_seq_len:
            continue

        # Chunk longer histories into non-overlapping windows of max_seq_len
        for start in range(0, len(q_all), max_seq_len):
            q_chunk = q_all[start : start + max_seq_len]
            r_chunk = r_all[start : start + max_seq_len]
            if len(q_chunk) >= min_seq_len:
                sequences.append({
                    "uidx": int(uidx),
                    "q": q_chunk,
                    "r": r_chunk,
                })
    return sequences


def create_splits(
    df: pd.DataFrame, max_seq_len: int = 100, val_user_ratio: float = 0.10, seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Creates train, val, and test sequence sets respecting user disjointness."""
    rng = np.random.default_rng(seed)

    # Use native split column if available
    if "split" in df.columns:
        train_users = df[df["split"] == "train"]["uidx"].unique()
        test_users = df[df["split"] == "test"]["uidx"].unique()

        # Hold out a portion of train users for early stopping validation
        rng.shuffle(train_users)
        n_val = max(1, int(len(train_users) * val_user_ratio))
        val_users = set(train_users[:n_val])
        actual_train_users = set(train_users[n_val:])
        test_users_set = set(test_users)

        train_df = df[df["uidx"].isin(actual_train_users)]
        val_df = df[df["uidx"].isin(val_users)]
        test_df = df[df["uidx"].isin(test_users_set)]
    else:
        # User-level split
        users = df["uidx"].unique()
        rng.shuffle(users)
        n_total = len(users)
        n_test = int(n_total * 0.20)
        n_val = int(n_total * 0.10)
        test_users_set = set(users[:n_test])
        val_users = set(users[n_test : n_test + n_val])
        actual_train_users = set(users[n_test + n_val:])

        train_df = df[df["uidx"].isin(actual_train_users)]
        val_df = df[df["uidx"].isin(val_users)]
        test_df = df[df["uidx"].isin(test_users_set)]

    train_seqs = build_sequences_from_df(train_df, max_seq_len=max_seq_len)
    val_seqs = build_sequences_from_df(val_df, max_seq_len=max_seq_len)
    test_seqs = build_sequences_from_df(test_df, max_seq_len=max_seq_len)

    return train_seqs, val_seqs, test_seqs


def run_epoch(
    model: DKT,
    sequences: List[dict],
    device: str | torch.device,
    batch_size: int = 128,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float, float]:
    """Runs one forward-backward training or evaluation epoch over sequence batches."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    indices = np.arange(len(sequences))
    if is_train:
        np.random.shuffle(indices)

    total_loss = 0.0
    total_samples = 0
    all_preds: List[float] = []
    all_targets: List[int] = []

    for b0 in range(0, len(indices), batch_size):
        batch_idx = indices[b0 : b0 + batch_size]
        batch = [sequences[i] for i in batch_idx]

        q_tensors = [torch.as_tensor(s["q"], dtype=torch.long) for s in batch]
        r_tensors = [torch.as_tensor(s["r"], dtype=torch.long) for s in batch]
        lengths = torch.tensor([len(q) for q in q_tensors], dtype=torch.long, device=device)

        q_pad = pad_sequence(q_tensors, batch_first=True, padding_value=0).to(device)  # (B, T)
        r_pad = pad_sequence(r_tensors, batch_first=True, padding_value=0).to(device)  # (B, T)

        B, T = q_pad.shape
        if T < 2:
            continue

        # Forward through pyKT DKT on interactions 0 .. T-2
        q_in = q_pad[:, :-1]
        r_in = r_pad[:, :-1]
        next_q = q_pad[:, 1:]
        next_r = r_pad[:, 1:].float()

        # y shape: (B, T-1, num_c)
        y = model(q_in, r_in)

        # Gather prediction for the concept tested at step t+1
        pred_gathered = y.gather(2, next_q.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # Valid mask: step t+1 < length of student interaction sequence
        step_indices = torch.arange(T - 1, device=device).unsqueeze(0)  # (1, T-1)
        valid_mask = step_indices < (lengths.unsqueeze(1) - 1)  # (B, T-1)

        if not valid_mask.any():
            continue

        pred_valid = pred_gathered[valid_mask]
        target_valid = next_r[valid_mask]

        pred_clamped = torch.clamp(pred_valid, 1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred_clamped, target_valid)

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        n_valid = int(valid_mask.sum().item())
        total_loss += float(loss.item()) * n_valid
        total_samples += n_valid

        with torch.no_grad():
            all_preds.extend(pred_valid.detach().cpu().numpy().tolist())
            all_targets.extend(target_valid.detach().cpu().numpy().astype(int).tolist())

    mean_loss = total_loss / max(1, total_samples)
    if len(set(all_targets)) > 1:
        auc = float(roc_auc_score(all_targets, all_preds))
        acc = float(accuracy_score(all_targets, (np.array(all_preds) >= 0.5).astype(int)))
    else:
        auc = 0.5
        acc = 0.5

    return mean_loss, auc, acc


def train_dkt_dataset(
    dataset_name: str,
    data_path: str | Path,
    output_dir: str | Path,
    num_concepts: int,
    emb_size: int = 64,
    dropout: float = 0.1,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 25,
    batch_size: int = 128,
    patience: int = 5,
    device_str: str = "cuda:0",
    max_seq_len: int = 100,
    sample_fraction: float | None = None,
    max_learners: int | None = None,
    seed: int = 42,
) -> Tuple[DKT, dict]:
    """Trains DKT on the specified dataset and saves all checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"TRAINING PYKT DKT: {dataset_name} (K={num_concepts})")
    print(f"Device: {device} | Emb Size: {emb_size} | Epochs: {epochs}")
    print(f"Data Path: {data_path}")
    if sample_fraction is not None or max_learners is not None:
        print(f"Subsampling: fraction={sample_fraction}, max_learners={max_learners}")
    print(f"=======================================================")

    df = pd.read_csv(data_path)
    if sample_fraction is not None or max_learners is not None:
        n_orig = len(df["uidx"].unique())
        df = subsample_by_learners(df, sample_fraction=sample_fraction, max_learners=max_learners, seed=seed)
        n_sub = len(df["uidx"].unique())
        print(f"[QuickTrain] Subsampled dataset: {n_sub}/{n_orig} learners ({len(df)} interactions)")
    train_seqs, val_seqs, test_seqs = create_splits(df, max_seq_len=max_seq_len, seed=seed)
    print(f"Loaded sequences: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")

    model = DKT(
        num_c=num_concepts,
        emb_size=emb_size,
        dropout=dropout,
        emb_type="qid",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    history = []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_auc, train_acc = run_epoch(
            model, train_seqs, device=device, batch_size=batch_size, optimizer=optimizer
        )
        val_loss, val_auc, val_acc = run_epoch(
            model, val_seqs, device=device, batch_size=batch_size, optimizer=None
        )
        scheduler.step(val_auc)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_auc": round(train_auc, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_auc": round(val_auc, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} ACC: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} ACC: {val_acc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs (Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}).")
                break

    # Load best model for final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_auc, test_acc = run_epoch(
        model, test_seqs, device=device, batch_size=batch_size, optimizer=None
    )
    print(f"\n---> FINAL TEST EVALUATION on {dataset_name}:")
    print(f"     Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test ACC: {test_acc:.4f}")

    metrics = {
        "dataset": dataset_name,
        "n_concepts": num_concepts,
        "emb_size": emb_size,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_val_auc, 4),
        "test_loss": round(test_loss, 4),
        "test_auc": round(test_auc, 4),
        "test_acc": round(test_acc, 4),
        "history": history,
    }

    # Save metrics and checkpoints
    (output_path / "dkt_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(best_state, output_path / "dkt_pykt.pt")
    torch.save(best_state, output_path / "cognitive_simulator.pt")
    print(f"Checkpoints saved to: {output_path / 'cognitive_simulator.pt'}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train pyKT DKT Mastery Simulator")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ASSISST2012", "assist12", "ASSISST2017", "assist17", "both"],
        default="both",
        help="Dataset to train on",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--emb-size", type=int, default=64, help="Embedding and hidden size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--sample-fraction", type=float, default=None, help="Fraction of learners to use for quick train/test (e.g. 0.1 for 10%)")
    parser.add_argument("--max-learners", type=int, default=None, help="Maximum number of learners to use for quick train/test (e.g. 500)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    datasets_to_run = []
    if args.dataset in ("ASSISST2012", "assist12", "both"):
        datasets_to_run.append({
            "name": "ASSISST2012",
            "data_path": project_root / "Light/data/processed/interactions_processed.csv",
            "output_dir": project_root / "artifacts_ASSISST2012/checkpoints",
            "additional_ckpts": [
                project_root / "Light/checkpoints/assist12/cognitive_simulator.pt",
                project_root / "models/checkpoints/dkt_assist12.pt",
            ],
            "n_concepts": 196,
        })

    if args.dataset in ("ASSISST2017", "assist17", "both"):
        datasets_to_run.append({
            "name": "ASSISST2017",
            "data_path": project_root / "Light/data/ASSISST2017/processed/interactions_processed.csv",
            "output_dir": project_root / "artifacts_ASSISST2017/checkpoints",
            "additional_ckpts": [
                project_root / "Light/checkpoints/ASSISST2017/cognitive_simulator.pt",
                project_root / "models/checkpoints/dkt_assist17.pt",
            ],
            "n_concepts": 102,
        })

    for dinfo in datasets_to_run:
        model, metrics = train_dkt_dataset(
            dataset_name=dinfo["name"],
            data_path=dinfo["data_path"],
            output_dir=dinfo["output_dir"],
            num_concepts=dinfo["n_concepts"],
            emb_size=args.emb_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device_str=args.device,
            sample_fraction=args.sample_fraction,
            max_learners=args.max_learners,
        )

        # Mirror best checkpoint to additional locations
        best_ckpt = dinfo["output_dir"] / "cognitive_simulator.pt"
        for extra_p in dinfo["additional_ckpts"]:
            extra_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.load(best_ckpt, map_location="cpu"), extra_p)
            print(f"Mirrored checkpoint to: {extra_p}")


if __name__ == "__main__":
    main()
