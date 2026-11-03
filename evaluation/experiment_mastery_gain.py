"""
Publication-Grade Experiment: Learner-Concept Mastery Before and After Heatmap (N=500 Learners).

This module:
1. Evaluates 500 learners under realistic time budgets (e.g. 30 min).
2. Records the exact continuous mastery state before and after the recommended learning path for all learners and concepts.
3. Generates ONLY the side-by-side Before and After Mastery Heatmap Matrix (without any numbers inside cells).
4. Saves high-resolution publication-ready outputs (300 DPI PNG & vector PDF) and structured CSV/JSON data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import load_config
from main import _load_model_bundle
from models.cognitive_simulator import learning_gain
from models.online_planner import AdaptiveTEMPOPlanner
from utils.data import policy_episodes


# Styling for scientific papers (IEEE / ACM / Springer style)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "figure.dpi": 300,
})


def run_mastery_experiment(
    artifacts_dir: str | Path = "artifacts/ASSISST2017",
    budget_minutes: float = 30.0,
    n_learners: int = 500,
    device: str = "auto",
    output_dir: str | Path = "experiments/mastery_gain_500",
    split: str = "test",
    sim_mode: str = "src_style",
) -> dict[str, Any]:
    """Runs the experiment on 500 learners and generates the Before vs. After Heatmap."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_path = Path(artifacts_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Target device: {device}")
    print(f"[*] Loading model bundle from: {artifacts_path}")
    bundle = _load_model_bundle(artifacts_path, device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner_encoder, time_model, simulator, decoder) = bundle

    names = metadata["idx_to_concept"]
    n_concepts = len(names)
    mastery_threshold = float(cfg.get("goals", {}).get("mastery_threshold", 0.85))
    hard_threshold = float(cfg.get("graph", {}).get("hard_edge_confidence", 0.80))
    hard_prereq = ((prerequisite > 0) & (confidence >= hard_threshold)).astype(np.float32)
    budget_seconds = float(budget_minutes) * 60.0

    # Load episodes
    episodes_list = []
    prof_file = artifacts_path / "student_profiles.json"
    csv_file = artifacts_path / "interactions_processed.csv"

    if csv_file.exists():
        print(f"[*] Loading interactions from CSV for {split} split...")
        df = pd.read_csv(csv_file)
        raw_episodes = policy_episodes(df, cfg, split)
        if len(raw_episodes) < n_learners:
            val_episodes = policy_episodes(df, cfg, "val")
            raw_episodes = raw_episodes + val_episodes
            if len(raw_episodes) < n_learners:
                train_episodes = policy_episodes(df, cfg, "train")
                raw_episodes = raw_episodes + train_episodes

        for ep in raw_episodes:
            episodes_list.append({
                "learner_id": ep.learner_id,
                "label": f"Learner #{ep.learner_id}",
                "history": ep.history,
                "targets": list(ep.targets),
            })
    elif prof_file.exists():
        print(f"[*] Loading student profiles from: {prof_file}")
        with open(prof_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        for p in profiles:
            episodes_list.append({
                "learner_id": p.get("uidx", p.get("user_id", 0)),
                "label": p.get("label", f"Student #{p.get('uidx', 0)}"),
                "history": {
                    "q": np.asarray(p["concepts"], dtype=np.int64),
                    "a": np.asarray(p["correct"], dtype=np.int64),
                    "t": np.asarray(p["elapsed_sec"], dtype=np.float32),
                    "lag": np.asarray(p.get("lag_sec", [0.0] * len(p["concepts"])), dtype=np.float32),
                },
                "targets": p.get("targets", None),
            })

    total_available = len(episodes_list)
    print(f"[*] Total available learner instances: {total_available}")
    n_eval = min(n_learners, total_available)
    selected_episodes = episodes_list[:n_eval]
    print(f"[*] Evaluating our model on {n_eval} learners under {budget_minutes} min budget...")

    learner_records = []
    concept_step_records = []

    for idx, ep in enumerate(selected_episodes):
        hist = ep["history"]
        init_mastery, sim_state = simulator.mastery_after_history(hist["q"], hist["a"])

        # Determine target concepts (unmastered frontier)
        targets = ep["targets"]
        if not targets:
            studied_set = set(int(c) for c in hist["q"])
            unmastered = [c for c in studied_set if float(init_mastery[c]) < mastery_threshold]
            unmastered.sort(key=lambda c: float(init_mastery[c]))
            targets = unmastered[:3] if unmastered else list(np.argsort(init_mastery)[:3])

        # TCLGM Planner
        planner = AdaptiveTEMPOPlanner(
            embeddings=embeddings,
            prerequisite=prerequisite,
            prerequisite_confidence=confidence,
            difficulties=difficulties,
            concept_weights=weights,
            learner_encoder=learner_encoder,
            time_model=time_model,
            simulator=simulator,
            decoder=decoder,
            cfg=cfg,
            history=hist,
            targets=targets,
            budget_seconds=budget_seconds,
            device=device,
        )

        plan_res = planner.recommend()
        path = plan_res["path"]

        if not path:
            continue

        # Simulate learning path
        if sim_mode == "detailed":
            sim_res = simulator.simulate_path_detailed(sim_state, path, seed=42 + idx)
        else:
            sim_res = simulator.simulate_path_src_style(sim_state, path)

        final_mastery = sim_res.final_mastery
        mastery_trace = sim_res.mastery_trace
        realized_times = sim_res.realized_times

        # Record concept-level details
        for step_i, c in enumerate(path, start=1):
            pre_m = float(init_mastery[c])
            post_m = float(final_mastery[c])
            abs_gain = post_m - pre_m
            norm_gain = learning_gain(pre_m, post_m)
            c_name = names[c] if c < len(names) else f"Concept_{c}"
            time_sec = realized_times[step_i - 1] if (step_i - 1) < len(realized_times) else 0.0

            concept_step_records.append({
                "learner_idx": idx,
                "learner_id": ep["learner_id"],
                "step": step_i,
                "concept_idx": int(c),
                "concept_name": c_name,
                "mastery_before": pre_m,
                "mastery_after": post_m,
                "absolute_gain": abs_gain,
                "learning_gain": norm_gain,
                "time_seconds": time_sec,
                "is_target": int(c) in targets,
            })

        # Summary metrics
        target_pre = float(np.mean([init_mastery[t] for t in targets]))
        target_post = float(np.mean([final_mastery[t] for t in targets]))
        target_norm_gain = learning_gain(target_pre, target_post)

        path_pre = float(np.mean([init_mastery[c] for c in path]))
        path_post = float(np.mean([final_mastery[c] for c in path]))
        path_norm_gain = learning_gain(path_pre, path_post)

        global_pre = float(np.mean(init_mastery))
        global_post = float(np.mean(final_mastery))
        global_norm_gain = learning_gain(global_pre, global_post)
        total_time_min = float(sum(realized_times)) / 60.0

        learner_records.append({
            "learner_idx": idx,
            "learner_id": ep["learner_id"],
            "path_length": len(path),
            "total_time_minutes": total_time_min,
            "path_mastery_before": path_pre,
            "path_mastery_after": path_post,
            "path_learning_gain": path_norm_gain,
            "target_mastery_before": target_pre,
            "target_mastery_after": target_post,
            "target_learning_gain": target_norm_gain,
            "global_mastery_before": global_pre,
            "global_mastery_after": global_post,
            "global_learning_gain": global_norm_gain,
            "gain_per_minute": (target_norm_gain / total_time_min) if total_time_min > 0 else 0.0,
            "initial_all_mastery": init_mastery.tolist(),
            "final_all_mastery": final_mastery.tolist(),
            "path_indices": [int(c) for c in path],
            "target_indices": [int(t) for t in targets],
        })

        if (idx + 1) % 100 == 0 or idx == n_eval - 1:
            print(f"  [+] Processed {idx + 1}/{n_eval} learners (Latest: #{ep['learner_id']}, Path Gain={path_norm_gain:.2%}, Target Gain={target_norm_gain:.2%})")

    # Save JSON & CSV
    results_json = {
        "model": "TCLGM (Proposed Adaptive Model)",
        "dataset": artifacts_path.name,
        "budget_minutes": budget_minutes,
        "mastery_threshold": mastery_threshold,
        "n_learners_evaluated": len(learner_records),
        "mean_path_mastery_before": float(np.mean([r["path_mastery_before"] for r in learner_records])),
        "mean_path_mastery_after": float(np.mean([r["path_mastery_after"] for r in learner_records])),
        "mean_path_learning_gain": float(np.mean([r["path_learning_gain"] for r in learner_records])),
        "mean_target_mastery_before": float(np.mean([r["target_mastery_before"] for r in learner_records])),
        "mean_target_mastery_after": float(np.mean([r["target_mastery_after"] for r in learner_records])),
        "mean_target_learning_gain": float(np.mean([r["target_learning_gain"] for r in learner_records])),
        "mean_global_learning_gain": float(np.mean([r["global_learning_gain"] for r in learner_records])),
        "mean_gain_per_minute": float(np.mean([r["gain_per_minute"] for r in learner_records])),
        "mean_path_length": float(np.mean([r["path_length"] for r in learner_records])),
        "mean_realized_time_minutes": float(np.mean([r["total_time_minutes"] for r in learner_records])),
        "learners": learner_records,
    }
    with open(out_path / "mastery_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    df_steps = pd.DataFrame(concept_step_records)
    df_steps.to_csv(out_path / "concept_mastery_before_after.csv", index=False)

    df_learners = pd.DataFrame([{
        "learner_id": r["learner_id"],
        "path_length": r["path_length"],
        "time_minutes": r["total_time_minutes"],
        "path_mastery_before": r["path_mastery_before"],
        "path_mastery_after": r["path_mastery_after"],
        "path_learning_gain": r["path_learning_gain"],
        "target_mastery_before": r["target_mastery_before"],
        "target_mastery_after": r["target_mastery_after"],
        "target_learning_gain": r["target_learning_gain"],
        "global_learning_gain": r["global_learning_gain"],
        "gain_per_minute": r["gain_per_minute"],
    } for r in learner_records])
    df_learners.to_csv(out_path / "learner_summary_results.csv", index=False)

    print(f"[*] Results and tables saved to: {out_path}")

    # Generate ONLY the Before vs. After Heatmap
    plot_mastery_heatmap_before_after(learner_records, names, out_path)
    return results_json


def plot_mastery_heatmap_before_after(
    learners: list[dict[str, Any]],
    idx_to_concept: list[str],
    output_dir: Path,
):
    """Generates a clean, publication-ready side-by-side Before and After Mastery Heatmap (no numbers inside)."""
    print("[*] Generating Before and After Heatmap (no internal numbers)...")
    if not learners:
        return

    # Select concepts that appear most frequently across recommended paths
    concept_counts = {}
    for l in learners:
        for c in l["path_indices"]:
            concept_counts[c] = concept_counts.get(c, 0) + 1

    sorted_concepts = sorted(concept_counts.keys(), key=lambda c: concept_counts[c], reverse=True)
    # Take top 25 most frequent concepts for a clear, readable X-axis
    selected_concepts = sorted_concepts[:25] if len(sorted_concepts) >= 25 else sorted_concepts

    c_labels = [idx_to_concept[c][:20] for c in selected_concepts]

    n_learners = len(learners)
    pre_matrix = np.zeros((n_learners, len(selected_concepts)))
    post_matrix = np.zeros((n_learners, len(selected_concepts)))

    for r_i, l in enumerate(learners):
        for c_i, c_idx in enumerate(selected_concepts):
            pre_matrix[r_i, c_i] = l["initial_all_mastery"][c_idx]
            post_matrix[r_i, c_i] = l["final_all_mastery"][c_idx]

    # Sort learners by mean post mastery or learner ID for clean visual stratification
    sort_idx = np.argsort(pre_matrix.mean(axis=1))
    pre_matrix = pre_matrix[sort_idx]
    post_matrix = post_matrix[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)

    # Panel (a): Before Path (Pre-Mastery)
    im1 = sns.heatmap(
        pre_matrix,
        ax=axes[0],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=False,  # NO numbers inside cells
        xticklabels=c_labels,
        yticklabels=False,
        cbar_kws={"label": "Mastery Level $[0, 1]$", "shrink": 0.85},
    )
    axes[0].set_title("(a) Initial Concept Mastery Matrix (Before Path)", fontweight="bold", fontsize=12, pad=12)
    axes[0].set_xlabel("Knowledge Concepts", fontweight="bold", fontsize=11)
    axes[0].set_ylabel(f"Learners ($N = {n_learners}$)", fontweight="bold", fontsize=11)
    axes[0].set_xticklabels(c_labels, rotation=45, ha="right", fontsize=8.5)

    # Panel (b): After Path (Post-Mastery)
    im2 = sns.heatmap(
        post_matrix,
        ax=axes[1],
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        annot=False,  # NO numbers inside cells
        xticklabels=c_labels,
        yticklabels=False,
        cbar_kws={"label": "Mastery Level $[0, 1]$", "shrink": 0.85},
    )
    axes[1].set_title("(b) Final Concept Mastery Matrix (After Path)", fontweight="bold", fontsize=12, pad=12)
    axes[1].set_xlabel("Knowledge Concepts", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(c_labels, rotation=45, ha="right", fontsize=8.5)

    plt.suptitle(
        f"Learner–Concept Mastery State Before vs. After Recommended Paths ($N = {n_learners}$ Learners)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    # Save PNG and vector PDF
    png_path = output_dir / "learner_concept_mastery_heatmap_before_after.png"
    pdf_path = output_dir / "learner_concept_mastery_heatmap_before_after.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  [+] Saved Heatmap PNG: {png_path}")
    print(f"  [+] Saved Heatmap PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Learner-Concept Mastery Before and After Heatmap (N=500 Learners).")
    parser.add_argument("--artifacts", default="artifacts/ASSISST2017", help="Path to artifacts directory")
    parser.add_argument("--budget-minutes", type=float, default=30.0, help="Time budget in minutes (default: 30.0)")
    parser.add_argument("--n-learners", type=int, default=500, help="Number of learners to evaluate (default: 500)")
    parser.add_argument("--output", default="experiments/mastery_gain_500", help="Output directory for heatmap")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--sim-mode", default="src_style", choices=["src_style", "detailed"], help="Simulation mode")
    args = parser.parse_args()

    run_mastery_experiment(
        artifacts_dir=args.artifacts,
        budget_minutes=args.budget_minutes,
        n_learners=args.n_learners,
        device=args.device,
        output_dir=args.output,
        sim_mode=args.sim_mode,
    )


if __name__ == "__main__":
    main()
