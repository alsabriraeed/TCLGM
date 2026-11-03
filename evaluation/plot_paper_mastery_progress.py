"""
Publication-Grade Step-by-Step Global Mastery Progress Plot for ASSIST12 and ASSIST2017.

Replicates the exact reference paper figure:
- Subplot (a) ASSIST12: Blue trajectory curve with open circles + shaded standard deviation band + n_20 annotation.
- Subplot (b) ASSIST2017: Orange trajectory curve with open squares + shaded standard deviation band + n_20 annotation.
- Shared Y-axis: 'Mean global mastery' [0.38, 0.83] with ticks [0.4, 0.5, 0.6, 0.7, 0.8].
- X-axis: 'Recommendation step $t$' [0, 20] with ticks [0, 4, 8, 12, 16, 20].
- Generates publication-ready 300 DPI PNG, vector PDF, and structured CSV of trajectory data.
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
import torch

from main import _load_model_bundle
from models.online_planner import AdaptiveTEMPOPlanner
from utils.data import policy_episodes


# Styling for scientific papers (IEEE / ACM / Springer style)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def compute_stepwise_mastery_trajectories(
    artifacts_dir: str | Path,
    dataset_name: str,
    n_learners: int = 500,
    budget_minutes: float = 30.0,
    max_steps: int = 20,
    device: str = "auto",
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, int, list[list[float]]]:
    """Computes mean global mastery at each recommendation step t in [0, max_steps]."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_path = Path(artifacts_dir)
    print(f"\n[*] Computing step trajectories for {dataset_name} ({artifacts_path.name}) on {device} (N = {n_learners})...")

    bundle = _load_model_bundle(artifacts_path, device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner_encoder, time_model, simulator, decoder) = bundle

    mastery_threshold = float(cfg.get("goals", {}).get("mastery_threshold", 0.85))
    budget_seconds = float(budget_minutes) * 60.0

    # Load episodes
    episodes_list = []
    csv_file = artifacts_path / "interactions_processed.csv"
    if csv_file.exists():
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
                "history": ep.history,
                "targets": list(ep.targets),
            })
    else:
        prof_file = artifacts_path / "student_profiles.json"
        with open(prof_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        for p in profiles:
            episodes_list.append({
                "learner_id": p.get("uidx", p.get("user_id", 0)),
                "history": {
                    "q": np.asarray(p["concepts"], dtype=np.int64),
                    "a": np.asarray(p["correct"], dtype=np.int64),
                    "t": np.asarray(p["elapsed_sec"], dtype=np.float32),
                    "lag": np.asarray(p.get("lag_sec", [0.0] * len(p["concepts"])), dtype=np.float32),
                },
                "targets": p.get("targets", None),
            })

    n_eval = min(n_learners, len(episodes_list))
    selected = episodes_list[:n_eval]

    # Matrix of shape [n_eval, max_steps + 1] representing global mastery at t = 0, 1, ..., max_steps
    trajectories = np.zeros((n_eval, max_steps + 1), dtype=np.float32)
    path_lengths = []
    raw_traces = []

    for idx, ep in enumerate(selected):
        hist = ep["history"]
        init_mastery, sim_state = simulator.mastery_after_history(hist["q"], hist["a"])

        targets = ep["targets"]
        if not targets:
            studied_set = set(int(c) for c in hist["q"])
            unmastered = [c for c in studied_set if float(init_mastery[c]) < mastery_threshold]
            unmastered.sort(key=lambda c: float(init_mastery[c]))
            targets = unmastered[:3] if unmastered else list(np.argsort(init_mastery)[:3])

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
        path_lengths.append(len(path))

        if path:
            sim_res = simulator.simulate_path_src_style(sim_state, path)
            trace = sim_res.mastery_trace  # trace[0] is step 0, trace[t] is after step t
        else:
            trace = [init_mastery.copy()]

        # Global mastery at each step t
        mean_trace = [float(np.mean(m)) for m in trace]
        raw_traces.append(mean_trace)

        # Fill up to max_steps (t = 0 ... max_steps)
        for t in range(max_steps + 1):
            if t < len(mean_trace):
                trajectories[idx, t] = mean_trace[t]
            else:
                # If learner path finished earlier, mastery persists at final achieved state
                trajectories[idx, t] = mean_trace[-1]

        if (idx + 1) % 100 == 0 or idx == n_eval - 1:
            print(f"  [+] {dataset_name}: {idx + 1}/{n_eval} learners simulated.")

    # Count how many learners had path length >= max_steps
    n_at_max = sum(1 for pl in path_lengths if pl >= max_steps)

    mean_trajectory = trajectories.mean(axis=0)
    std_trajectory = trajectories.std(axis=0)

    print(f"  [+] {dataset_name} Step 0 Mastery: {mean_trajectory[0]:.4f} ± {std_trajectory[0]:.4f}")
    print(f"  [+] {dataset_name} Step {max_steps} Mastery: {mean_trajectory[max_steps]:.4f} ± {std_trajectory[max_steps]:.4f}")
    print(f"  [+] Learners with path length >= {max_steps}: {n_at_max}/{n_eval}")

    return mean_trajectory, std_trajectory, n_at_max, raw_traces


def plot_paper_mastery_progress(
    mean_12: np.ndarray,
    std_12: np.ndarray,
    n_20_12: int,
    mean_17: np.ndarray,
    std_17: np.ndarray,
    n_20_17: int,
    output_path: Path,
    max_steps: int = 20,
):
    """Generates the side-by-side progression plot matching the reference paper figure."""
    print("\n[*] Generating publication mastery progression plot...")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})

    steps = np.arange(max_steps + 1)
    marker_steps = np.arange(0, max_steps + 1, 2)  # Mark every 2 steps (0, 2, 4, ..., 20)

    # -------------------------------------------------------------
    # Subplot (a): ASSIST12 (Blue Palette)
    # -------------------------------------------------------------
    color_12 = "#0072B2"       # Classic Deep Blue
    shade_12 = "#56B4E9"       # Soft Light Blue

    axes[0].plot(
        steps,
        mean_12,
        color=color_12,
        linewidth=2.4,
        marker="o",
        markevery=2,
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=color_12,
        markeredgewidth=1.8,
        zorder=4,
    )
    axes[0].fill_between(
        steps,
        mean_12 - std_12,
        mean_12 + std_12,
        color=shade_12,
        alpha=0.28,
        zorder=2,
    )

    axes[0].set_title("(a) ASSIST12", fontweight="bold", fontsize=12.5, pad=10)
    axes[0].set_xlabel("Recommendation step $t$", fontsize=11.5)
    axes[0].set_ylabel("Mean global mastery", fontsize=11.5)
    axes[0].set_xlim(0, max_steps)
    axes[0].set_ylim(0.38, 0.83)
    axes[0].set_xticks([0, 4, 8, 12, 16, 20])
    axes[0].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axes[0].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)
    axes[0].tick_params(direction="out", length=4, width=0.9)

    # Annotation top-right: n_20 = 235 (or actual count)
    axes[0].text(
        0.95, 0.93,
        f"$n_{{20}} = {n_20_12}$",
        transform=axes[0].transAxes,
        ha="right", va="top",
        fontsize=10.5,
        fontstyle="italic",
        color="#374151",
    )

    # -------------------------------------------------------------
    # Subplot (b): ASSIST2017 (Orange Palette)
    # -------------------------------------------------------------
    color_17 = "#D55E00"       # Rich Rust / Terracotta Orange
    shade_17 = "#E69F00"       # Warm Amber Shade

    axes[1].plot(
        steps,
        mean_17,
        color=color_17,
        linewidth=2.4,
        marker="s",
        markevery=2,
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor=color_17,
        markeredgewidth=1.8,
        zorder=4,
    )
    axes[1].fill_between(
        steps,
        mean_17 - std_17,
        mean_17 + std_17,
        color=shade_17,
        alpha=0.28,
        zorder=2,
    )

    axes[1].set_title("(b) ASSIST2017", fontweight="bold", fontsize=12.5, pad=10)
    axes[1].set_xlabel("Recommendation step $t$", fontsize=11.5)
    axes[1].set_xlim(0, max_steps)
    axes[1].set_xticks([0, 4, 8, 12, 16, 20])
    axes[1].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)
    axes[1].tick_params(direction="out", length=4, width=0.9)

    # Annotation bottom/middle-right: n_20 = 48 (or actual count)
    axes[1].text(
        0.95, 0.48,
        f"$n_{{20}} = {n_20_17}$",
        transform=axes[1].transAxes,
        ha="right", va="center",
        fontsize=10.5,
        fontstyle="italic",
        color="#374151",
    )

    output_path.mkdir(parents=True, exist_ok=True)
    png_file = output_path / "paper_mastery_progress.png"
    pdf_file = output_path / "paper_mastery_progress.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved Progress Plot PNG: {png_file}")
    print(f"[+] Saved Progress Plot PDF: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot Step-by-Step Global Mastery Progression for ASSIST12 and ASSIST17.")
    parser.add_argument("--artifacts-12", default="artifacts/ASSISST2012", help="Path to ASSIST12 artifacts")
    parser.add_argument("--artifacts-17", default="artifacts/ASSISST2017", help="Path to ASSIST2017 artifacts")
    parser.add_argument("--budget-minutes", type=float, default=30.0, help="Time budget in minutes (default: 30.0)")
    parser.add_argument("--n-learners", type=int, default=500, help="Number of learners to evaluate (default: 500)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max recommendation steps to plot (default: 20)")
    parser.add_argument("--output", default="experiments/paper_plots_progress", help="Output directory for plots and CSVs")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute for ASSIST12
    mean_12, std_12, n20_12, raw_12 = compute_stepwise_mastery_trajectories(
        artifacts_dir=args.artifacts_12,
        dataset_name="ASSIST12",
        n_learners=args.n_learners,
        budget_minutes=args.budget_minutes,
        max_steps=args.max_steps,
        device=args.device,
    )

    # 2. Compute for ASSIST2017
    mean_17, std_17, n20_17, raw_17 = compute_stepwise_mastery_trajectories(
        artifacts_dir=args.artifacts_17,
        dataset_name="ASSIST2017",
        n_learners=args.n_learners,
        budget_minutes=args.budget_minutes,
        max_steps=args.max_steps,
        device=args.device,
    )

    # 3. Export CSV table of mean trajectories
    steps = np.arange(args.max_steps + 1)
    df_progress = pd.DataFrame({
        "step": steps,
        "assist12_mean": np.round(mean_12, 4),
        "assist12_std": np.round(std_12, 4),
        "assist17_mean": np.round(mean_17, 4),
        "assist17_std": np.round(std_17, 4),
    })
    csv_path = out_dir / "stepwise_mastery_progress.csv"
    df_progress.to_csv(csv_path, index=False)
    print(f"[+] Saved Stepwise Progress CSV: {csv_path}")

    # 4. Plot Side-by-Side Figure
    plot_paper_mastery_progress(
        mean_12=mean_12,
        std_12=std_12,
        n_20_12=n20_12,
        mean_17=mean_17,
        std_17=std_17,
        n_20_17=n20_17,
        output_path=out_dir,
        max_steps=args.max_steps,
    )

    print("\n" + "=" * 70)
    print("        STEPWISE GLOBAL MASTERY PROGRESSION SUMMARY (STEPS 0 TO 20)    ")
    print("=" * 70)
    print(f"Step | ASSIST12 (Mean ± Std)       | ASSIST2017 (Mean ± Std)")
    print("-" * 70)
    for s in range(0, args.max_steps + 1, 2):
        print(f" t={s:<2} | {mean_12[s]:.4f} ± {std_12[s]:.4f}          | {mean_17[s]:.4f} ± {std_17[s]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
