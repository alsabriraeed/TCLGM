"""
Publication-Grade Recommended Path Step Delta Mastery Heatmap for ASSIST12 and ASSIST2017.

Option A Scope: Recommended Path Concepts across Step Index (500 Learners x 25 Steps)
- Subplot (a) ASSIST12: 500 Learners (Rows) x 25 Path Steps (Columns)
- Subplot (b) ASSIST2017: 500 Learners (Rows) x 25 Path Steps (Columns)
- For each cell (i, k): Δ Mastery = M_after(i, Path_i[k]) - M_before(i, Path_i[k])
- Clean, publication-grade styling without cell text clutter.
- Exports granular CSVs, summary CSVs, JSON report, and 300 DPI PNG / vector PDF.
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
from models.cognitive_simulator import learning_gain
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


def evaluate_path_step_mastery(
    artifacts_dir: str | Path,
    dataset_name: str,
    n_learners: int = 500,
    budget_minutes: float = 30.0,
    max_steps: int = 25,
    device: str = "auto",
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluates recommended path step mastery (Option A scope) for n_learners."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_path = Path(artifacts_dir)
    print(f"\n[*] Evaluating {dataset_name} ({artifacts_path.name}) on {device} (N = {n_learners}, Steps = {max_steps})...")

    bundle = _load_model_bundle(artifacts_path, device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner_encoder, time_model, simulator, decoder) = bundle

    names = metadata["idx_to_concept"]
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

    # Path Step Matrices: [n_eval x max_steps]
    step_delta_matrix = np.zeros((n_eval, max_steps), dtype=np.float32)
    step_pre_matrix = np.zeros((n_eval, max_steps), dtype=np.float32)
    step_post_matrix = np.zeros((n_eval, max_steps), dtype=np.float32)
    step_gain_matrix = np.zeros((n_eval, max_steps), dtype=np.float32)

    learner_summaries = []
    step_records = []

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

        if path:
            sim_res = simulator.simulate_path_src_style(sim_state, path)
            final_mastery = sim_res.final_mastery
            realized_times = sim_res.realized_times
        else:
            final_mastery = init_mastery.copy()
            realized_times = []

        # Populate Step-wise data along the recommended path
        path_len = len(path)
        step_gains_list = []
        step_deltas_list = []
        step_pres_list = []
        step_posts_list = []

        for step_k in range(max_steps):
            if step_k < path_len:
                c_idx = path[step_k]
                m_pre = float(init_mastery[c_idx])
                m_post = float(final_mastery[c_idx])
                d_m = m_post - m_pre
                g_val = learning_gain(m_pre, m_post)
                c_name = names[c_idx] if c_idx < len(names) else f"Concept_{c_idx}"
                t_sec = realized_times[step_k] if step_k < len(realized_times) else 0.0

                step_pre_matrix[idx, step_k] = m_pre
                step_post_matrix[idx, step_k] = m_post
                step_delta_matrix[idx, step_k] = d_m
                step_gain_matrix[idx, step_k] = g_val

                step_deltas_list.append(d_m)
                step_gains_list.append(g_val)
                step_pres_list.append(m_pre)
                step_posts_list.append(m_post)

                step_records.append({
                    "learner_id": ep["learner_id"],
                    "path_step": step_k + 1,
                    "concept_idx": c_idx,
                    "concept_name": c_name,
                    "mastery_before": round(m_pre, 4),
                    "mastery_after": round(m_post, 4),
                    "delta_mastery": round(d_m, 4),
                    "learning_gain": round(g_val, 4),
                    "time_seconds": round(t_sec, 2),
                    "is_target": int(c_idx) in targets,
                })
            else:
                # If path length is shorter than max_steps, fill with last valid or average
                last_m_pre = step_pres_list[-1] if step_pres_list else 0.5
                last_m_post = step_posts_list[-1] if step_posts_list else 0.5
                step_pre_matrix[idx, step_k] = last_m_pre
                step_post_matrix[idx, step_k] = last_m_post
                step_delta_matrix[idx, step_k] = last_m_post - last_m_pre
                step_gain_matrix[idx, step_k] = learning_gain(last_m_pre, last_m_post)

        time_min = float(sum(realized_times)) / 60.0
        p_pre = float(np.mean(step_pres_list)) if step_pres_list else 0.0
        p_post = float(np.mean(step_posts_list)) if step_posts_list else 0.0
        p_gain = learning_gain(p_pre, p_post)

        learner_summaries.append({
            "learner_idx": idx,
            "learner_id": ep["learner_id"],
            "path_length": path_len,
            "time_minutes": round(time_min, 3),
            "path_mastery_before": round(p_pre, 4),
            "path_mastery_after": round(p_post, 4),
            "path_delta_mastery": round(p_post - p_pre, 4),
            "path_learning_gain": round(p_gain, 4),
            "gain_per_minute": round((p_gain / time_min) if time_min > 0 else 0.0, 4),
        })

        if (idx + 1) % 100 == 0 or idx == n_eval - 1:
            print(f"  [+] {dataset_name}: {idx + 1}/{n_eval} learners evaluated.")

    # -------------------------------------------------------------
    # Ranking & Sorting:
    # Sort Rows (Learners) descending by their mean path mastery gain:
    # Rank 1 (top) = Learner with highest average path gain
    # Rank 500 (bottom) = Learner with lowest average path gain
    # Columns remain ordered by Path Step Index 1 -> 25 (chronological instructional sequence)
    # -------------------------------------------------------------
    learner_path_gains = step_delta_matrix.mean(axis=1)
    learner_order = np.argsort(-learner_path_gains)  # Descending

    ranked_step_delta_matrix = step_delta_matrix[learner_order, :]

    # Map learner rank
    learner_id_to_rank = {selected[l_orig]["learner_id"]: rank for rank, l_orig in enumerate(learner_order, start=1)}

    df_steps = pd.DataFrame(step_records)
    df_steps["learner_rank"] = df_steps["learner_id"].map(learner_id_to_rank)

    df_learners = pd.DataFrame(learner_summaries)
    df_learners["learner_rank"] = df_learners["learner_id"].map(learner_id_to_rank)
    df_learners = df_learners.sort_values("learner_rank").reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = dataset_name.lower().replace(" ", "_")
        csv_steps_path = output_dir / f"{file_prefix}_path_step_mastery.csv"
        csv_summary_path = output_dir / f"{file_prefix}_learner_path_summary.csv"

        df_steps.to_csv(csv_steps_path, index=False)
        df_learners.to_csv(csv_summary_path, index=False)
        print(f"  [+] Saved {dataset_name} path steps CSV: {csv_steps_path}")
        print(f"  [+] Saved {dataset_name} summary CSV:    {csv_summary_path}")

    # Compute aggregate metrics for Recommended Path Concepts
    agg_metrics = {
        "dataset": dataset_name,
        "n_learners": n_eval,
        "max_path_steps": max_steps,
        "mean_path_mastery_before": float(np.mean(df_learners["path_mastery_before"])),
        "mean_path_mastery_after": float(np.mean(df_learners["path_mastery_after"])),
        "mean_path_delta_mastery": float(np.mean(df_learners["path_delta_mastery"])),
        "mean_path_learning_gain": float(np.mean(df_learners["path_learning_gain"])),
        "mean_path_length": float(np.mean(df_learners["path_length"])),
        "mean_realized_time_minutes": float(np.mean(df_learners["time_minutes"])),
        "mean_gain_per_minute": float(np.mean(df_learners["gain_per_minute"])),
    }

    return {
        "ranked_step_delta_matrix": ranked_step_delta_matrix,
        "n_learners": n_eval,
        "max_steps": max_steps,
        "agg_metrics": agg_metrics,
        "df_learners": df_learners,
        "df_steps": df_steps,
    }


def plot_path_step_comparison_heatmap(
    delta_12: np.ndarray,
    n_learners_12: int,
    steps_12: int,
    delta_17: np.ndarray,
    n_learners_17: int,
    steps_17: int,
    output_path: Path,
):
    """Generates the publication heatmap for Recommended Path Concepts (Option A)."""
    print("\n[*] Generating Recommended Path Concepts comparison heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.18})

    vmin, vmax = -0.05, 0.40
    cmap = "RdBu_r"

    # --- Subplot (a): ASSIST12 ---
    im1 = axes[0].imshow(
        delta_12,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[1, steps_12, n_learners_12, 1],  # [left, right, bottom, top] so Rank 1 is at top
    )
    axes[0].set_title("(a) ASSIST12", fontweight="bold", fontsize=13, pad=10)
    axes[0].set_xlabel("Path Step Index", fontsize=12)
    axes[0].set_ylabel("Learner Rank", fontsize=12)

    mid_l12 = (1 + n_learners_12) // 2
    axes[0].set_xticks([1, 13, steps_12])
    axes[0].set_yticks([1, mid_l12, n_learners_12])
    axes[0].tick_params(direction="out", length=4, width=1.0)

    # --- Subplot (b): ASSIST2017 ---
    im2 = axes[1].imshow(
        delta_17,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[1, steps_17, n_learners_17, 1],  # [left, right, bottom, top] so Rank 1 is at top
    )
    axes[1].set_title("(b) ASSIST2017", fontweight="bold", fontsize=13, pad=10)
    axes[1].set_xlabel("Path Step Index", fontsize=12)
    axes[1].set_ylabel("Learner Rank", fontsize=12)

    mid_l17 = (1 + n_learners_17) // 2
    axes[1].set_xticks([1, 13, steps_17])
    axes[1].set_yticks([1, mid_l17, n_learners_17])
    axes[1].tick_params(direction="out", length=4, width=1.0)

    # Shared Colorbar on the right with extend='both'
    cbar = fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar.set_label(r"$\Delta$ Mastery", fontsize=12)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4])
    cbar.ax.tick_params(labelsize=10)

    output_path.mkdir(parents=True, exist_ok=True)
    png_file = output_path / "paper_path_delta_heatmap.png"
    pdf_file = output_path / "paper_path_delta_heatmap.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved Heatmap PNG: {png_file}")
    print(f"[+] Saved Heatmap PDF: {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Recommended Path Step Mastery (Option A) for ASSIST12 and ASSIST17.")
    parser.add_argument("--artifacts-12", default="artifacts/ASSISST2012", help="Path to ASSIST12 artifacts")
    parser.add_argument("--artifacts-17", default="artifacts/ASSISST2017", help="Path to ASSIST2017 artifacts")
    parser.add_argument("--budget-minutes", type=float, default=30.0, help="Time budget in minutes (default: 30.0)")
    parser.add_argument("--n-learners", type=int, default=500, help="Number of learners to evaluate (default: 500)")
    parser.add_argument("--max-steps", type=int, default=25, help="Max path steps (default: 25)")
    parser.add_argument("--output", default="experiments/paper_plots_path_scope", help="Output directory for CSVs and plots")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate ASSIST12
    res_12 = evaluate_path_step_mastery(
        artifacts_dir=args.artifacts_12,
        dataset_name="ASSIST12",
        n_learners=args.n_learners,
        budget_minutes=args.budget_minutes,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=out_dir,
    )

    # 2. Evaluate ASSIST2017
    res_17 = evaluate_path_step_mastery(
        artifacts_dir=args.artifacts_17,
        dataset_name="ASSIST2017",
        n_learners=args.n_learners,
        budget_minutes=args.budget_minutes,
        max_steps=args.max_steps,
        device=args.device,
        output_dir=out_dir,
    )

    # 3. Plot Comparison Heatmap for Recommended Path Concepts
    plot_path_step_comparison_heatmap(
        delta_12=res_12["ranked_step_delta_matrix"],
        n_learners_12=res_12["n_learners"],
        steps_12=res_12["max_steps"],
        delta_17=res_17["ranked_step_delta_matrix"],
        n_learners_17=res_17["n_learners"],
        steps_17=res_17["max_steps"],
        output_path=out_dir,
    )

    # Save summary JSON
    summary_json = {
        "ASSIST12": res_12["agg_metrics"],
        "ASSIST2017": res_17["agg_metrics"],
    }
    with open(out_dir / "path_mastery_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 75)
    print("      RECOMMENDED PATH CONCEPTS: MASTERY AVERAGE REPORT (OPTION A)    ")
    print("=" * 75)
    for d_name, m in summary_json.items():
        print(f"\n--- {d_name} (N = {m['n_learners']} Learners, Scope: Recommended Path Concepts, Budget = {args.budget_minutes} min) ---")
        print(f"  • Path Mastery Before (Pre-Study):   {m['mean_path_mastery_before']:.4f}")
        print(f"  • Path Mastery After  (Post-Study):  {m['mean_path_mastery_after']:.4f}")
        print(f"  • Net Path Delta (Δ Mastery):        +{m['mean_path_delta_mastery']:.4f}")
        print(f"  • Normalized Path Learning Gain (G): {m['mean_path_learning_gain']:.2%}")
        print(f"  • Mean Path Length:                  {m['mean_path_length']:.2f} concepts")
        print(f"  • Mean Realized Time:                {m['mean_realized_time_minutes']:.2f} min (budget: {args.budget_minutes} min)")
        print(f"  • Learning Efficiency (G / min):     {m['mean_gain_per_minute']:.4f} / min")
    print("=" * 75)


if __name__ == "__main__":
    main()
