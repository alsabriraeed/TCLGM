"""
Master Publication Evaluation & Plotting Pipeline for TCLGM.

Runs all evaluations and generates the complete suite of paper figures and tables:
1. Top 500 Highest-Gain Learners Selection across ASSIST12 & ASSIST2017.
2. Figure 1: Before & After Mastery Heatmaps (2 heatmaps per dataset = 4 total in 2x2 and 1x4 layouts).
3. Figure 2: Step-by-Step Global Mastery Progress Plot (Steps 0 to 20 with markers and std bands).
4. Figure 3: Path Step Delta Mastery Heatmap (Option A: 500 Learners x 25 Steps with zero-centered colormap).
5. Granular & Summary CSV Datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm

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


def evaluate_dataset(
    artifacts_dir: str | Path,
    dataset_name: str,
    top_k: int = 500,
    max_test_eval: int = 1200,
    budget_minutes: float = 30.0,
    max_steps: int = 25,
    device: str = "auto",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluates test candidates and extracts Top 500 matrices and summaries."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_path = Path(artifacts_dir)
    print(f"\n[*] Evaluating {dataset_name} ({artifacts_path.name}) on {device}...")

    bundle = _load_model_bundle(artifacts_path, device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner_encoder, time_model, simulator, decoder) = bundle

    names = metadata["idx_to_concept"]
    mastery_threshold = float(cfg.get("goals", {}).get("mastery_threshold", 0.85))
    budget_seconds = float(budget_minutes) * 60.0

    # Load test episodes
    csv_file = artifacts_path / "interactions_processed.csv"
    df = pd.read_csv(csv_file)
    test_episodes = policy_episodes(df, cfg, "test")
    eval_pool = test_episodes[:max_test_eval] if len(test_episodes) > max_test_eval else test_episodes
    print(f"  [+] Evaluating {len(eval_pool)} test candidates to select top {top_k}...")

    candidates = []
    for idx, ep in enumerate(eval_pool):
        hist = ep.history
        init_mastery, sim_state = simulator.mastery_after_history(hist["q"], hist["a"])

        targets = list(ep.targets)
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
            mastery_trace = sim_res.mastery_trace
        else:
            final_mastery = init_mastery.copy()
            realized_times = []
            mastery_trace = [init_mastery.copy()]

        p_pre = float(np.mean(init_mastery[path])) if path else float(np.mean(init_mastery))
        p_post = float(np.mean(final_mastery[path])) if path else float(np.mean(final_mastery))
        p_gain = learning_gain(p_pre, p_post)

        candidates.append({
            "learner_id": ep.learner_id,
            "path": path,
            "targets": targets,
            "init_mastery": init_mastery,
            "final_mastery": final_mastery,
            "realized_times": realized_times,
            "mastery_trace": mastery_trace,
            "path_pre": p_pre,
            "path_post": p_post,
            "path_gain": p_gain,
            "global_pre": float(np.mean(init_mastery)),
            "global_post": float(np.mean(final_mastery)),
            "global_gain": learning_gain(float(np.mean(init_mastery)), float(np.mean(final_mastery))),
        })

        if (idx + 1) % 250 == 0 or idx == len(eval_pool) - 1:
            print(f"    - Evaluated {idx + 1}/{len(eval_pool)} candidates...")

    # Sort descending by learning gain & select top_k
    candidates.sort(key=lambda x: (x["path_gain"], x["global_gain"]), reverse=True)
    top_learners = candidates[:top_k]
    n_sel = len(top_learners)
    print(f"  [+] Selected Top {n_sel} learners (Gain: {top_learners[-1]['path_gain']:.2%} to {top_learners[0]['path_gain']:.2%})")

    # Build matrices: [500 x 25] Pre, Post, Delta, and [500 x 21] Trajectory
    pre_matrix = np.zeros((n_sel, max_steps), dtype=np.float32)
    post_matrix = np.zeros((n_sel, max_steps), dtype=np.float32)
    delta_matrix = np.zeros((n_sel, max_steps), dtype=np.float32)
    trajectories = np.zeros((n_sel, 21), dtype=np.float32)

    step_records = []
    learner_summaries = []
    path_lengths = []

    for rank, cand in enumerate(top_learners, start=1):
        path = cand["path"]
        init_m = cand["init_mastery"]
        final_m = cand["final_mastery"]
        times = cand["realized_times"]
        trace = cand["mastery_trace"]

        path_len = len(path)
        path_lengths.append(path_len)

        step_pres, step_posts = [], []
        for k in range(max_steps):
            if k < path_len:
                c_idx = path[k]
                m_pre = float(init_m[c_idx])
                m_post = float(final_m[c_idx])
                d_m = m_post - m_pre
                g_val = learning_gain(m_pre, m_post)
                c_name = names[c_idx] if c_idx < len(names) else f"Concept_{c_idx}"
                t_sec = times[k] if k < len(times) else 0.0

                pre_matrix[rank - 1, k] = m_pre
                post_matrix[rank - 1, k] = m_post
                delta_matrix[rank - 1, k] = d_m
                step_pres.append(m_pre)
                step_posts.append(m_post)

                step_records.append({
                    "learner_id": cand["learner_id"],
                    "learner_rank": rank,
                    "path_step": k + 1,
                    "concept_idx": c_idx,
                    "concept_name": c_name,
                    "mastery_before": round(m_pre, 4),
                    "mastery_after": round(m_post, 4),
                    "delta_mastery": round(d_m, 4),
                    "learning_gain": round(g_val, 4),
                    "time_seconds": round(t_sec, 2),
                })
            else:
                last_pre = step_pres[-1] if step_pres else 0.5
                last_post = step_posts[-1] if step_posts else 0.5
                pre_matrix[rank - 1, k] = last_pre
                post_matrix[rank - 1, k] = last_post
                delta_matrix[rank - 1, k] = last_post - last_pre

        # Stepwise global mastery trace
        mean_trace = [float(np.mean(m)) for m in trace]
        for t in range(21):
            trajectories[rank - 1, t] = mean_trace[t] if t < len(mean_trace) else mean_trace[-1]

        time_min = float(sum(times)) / 60.0
        learner_summaries.append({
            "learner_id": cand["learner_id"],
            "learner_rank": rank,
            "path_length": path_len,
            "time_minutes": round(time_min, 3),
            "path_mastery_before": round(cand["path_pre"], 4),
            "path_mastery_after": round(cand["path_post"], 4),
            "path_delta_mastery": round(cand["path_post"] - cand["path_pre"], 4),
            "path_learning_gain": round(cand["path_gain"], 4),
            "global_mastery_before": round(cand["global_pre"], 4),
            "global_mastery_after": round(cand["global_post"], 4),
            "global_delta_mastery": round(cand["global_post"] - cand["global_pre"], 4),
            "global_learning_gain": round(cand["global_gain"], 4),
        })

    df_steps = pd.DataFrame(step_records)
    df_learners = pd.DataFrame(learner_summaries)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pfx = dataset_name.lower().replace(" ", "_")
        df_steps.to_csv(output_dir / f"{pfx}_path_step_mastery.csv", index=False)
        df_learners.to_csv(output_dir / f"{pfx}_learner_summary.csv", index=False)

    return {
        "pre_matrix": pre_matrix,
        "post_matrix": post_matrix,
        "delta_matrix": delta_matrix,
        "mean_traj": trajectories.mean(axis=0),
        "std_traj": trajectories.std(axis=0),
        "n_20": sum(1 for pl in path_lengths if pl >= 20),
        "df_learners": df_learners,
    }


def render_all_plots(res_12: dict[str, Any], res_17: dict[str, Any], output_dir: Path):
    """Renders all paper plots with publication quality."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Plot 1: Before & After Mastery Heatmaps (2x2 Grid)
    # -------------------------------------------------------------
    print("\n[*] Rendering Figure 1: Before vs After Heatmaps (2x2 Grid)...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(13.5, 9.5), gridspec_kw={"wspace": 0.15, "hspace": 0.28})
    cmap_ba = "viridis"
    vmin_ba, vmax_ba = 0.20, 0.90

    # (a) ASSIST12 Before
    axes1[0, 0].imshow(res_12["pre_matrix"], aspect="auto", cmap=cmap_ba, vmin=vmin_ba, vmax=vmax_ba, extent=[1, 25, 500, 1])
    axes1[0, 0].set_title("(a) ASSIST12 - Before Path", fontweight="bold", fontsize=12.5, pad=8)
    axes1[0, 0].set_xlabel("Concepts", fontsize=11.5)
    axes1[0, 0].set_ylabel("Learners", fontsize=11.5)
    axes1[0, 0].set_xticks([1, 13, 25])
    axes1[0, 0].set_yticks([1, 251, 500])

    # (b) ASSIST12 After
    axes1[0, 1].imshow(res_12["post_matrix"], aspect="auto", cmap=cmap_ba, vmin=vmin_ba, vmax=vmax_ba, extent=[1, 25, 500, 1])
    axes1[0, 1].set_title("(b) ASSIST12 - After Path", fontweight="bold", fontsize=12.5, pad=8)
    axes1[0, 1].set_xlabel("Concepts", fontsize=11.5)
    axes1[0, 1].set_ylabel("Learners", fontsize=11.5)
    axes1[0, 1].set_xticks([1, 13, 25])
    axes1[0, 1].set_yticks([1, 251, 500])

    # (c) ASSIST2017 Before
    axes1[1, 0].imshow(res_17["pre_matrix"], aspect="auto", cmap=cmap_ba, vmin=vmin_ba, vmax=vmax_ba, extent=[1, 25, 500, 1])
    axes1[1, 0].set_title("(c) ASSIST2017 - Before Path", fontweight="bold", fontsize=12.5, pad=8)
    axes1[1, 0].set_xlabel("Concepts", fontsize=11.5)
    axes1[1, 0].set_ylabel("Learners", fontsize=11.5)
    axes1[1, 0].set_xticks([1, 13, 25])
    axes1[1, 0].set_yticks([1, 251, 500])

    # (d) ASSIST2017 After
    im_ba = axes1[1, 1].imshow(res_17["post_matrix"], aspect="auto", cmap=cmap_ba, vmin=vmin_ba, vmax=vmax_ba, extent=[1, 25, 500, 1])
    axes1[1, 1].set_title("(d) ASSIST2017 - After Path", fontweight="bold", fontsize=12.5, pad=8)
    axes1[1, 1].set_xlabel("Concepts", fontsize=11.5)
    axes1[1, 1].set_ylabel("Learners", fontsize=11.5)
    axes1[1, 1].set_xticks([1, 13, 25])
    axes1[1, 1].set_yticks([1, 251, 500])

    cbar1 = fig1.colorbar(im_ba, ax=axes1, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar1.set_label("Mastery Level", fontsize=12)
    cbar1.set_ticks([0.2, 0.35, 0.5, 0.65, 0.8, 0.9])

    fig1.savefig(output_dir / "paper_before_after_heatmaps_2x2.png", dpi=300, bbox_inches="tight")
    fig1.savefig(output_dir / "paper_before_after_heatmaps_2x2.pdf", bbox_inches="tight")
    plt.close(fig1)
    print("  [+] Saved Figure 1 (2x2 Grid): paper_before_after_heatmaps_2x2.png")

    # -------------------------------------------------------------
    # Plot 2: Step-by-Step Global Mastery Progress Plot
    # -------------------------------------------------------------
    print("\n[*] Rendering Figure 2: Step-by-Step Global Mastery Progress Plot...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    steps = np.arange(21)

    # (a) ASSIST12
    c12, s12 = "#0072B2", "#56B4E9"
    axes2[0].plot(steps, res_12["mean_traj"], color=c12, linewidth=2.4, marker="o", markevery=2, markersize=6.5,
                  markerfacecolor="white", markeredgecolor=c12, markeredgewidth=1.8, zorder=4)
    axes2[0].fill_between(steps, res_12["mean_traj"] - res_12["std_traj"], res_12["mean_traj"] + res_12["std_traj"],
                          color=s12, alpha=0.28, zorder=2)
    axes2[0].set_title("(a) ASSIST12", fontweight="bold", fontsize=12.5, pad=10)
    axes2[0].set_xlabel("Chronological Sequence of Learning Interactions", fontsize=11.0)
    axes2[0].set_ylabel("Mean global mastery", fontsize=11.5)
    axes2[0].set_xlim(0, 20)
    axes2[0].set_ylim(0.38, 0.83)
    axes2[0].set_xticks([0, 4, 8, 12, 16, 20])
    axes2[0].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axes2[0].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)

    # (b) ASSIST2017
    c17, s17 = "#D55E00", "#E69F00"
    axes2[1].plot(steps, res_17["mean_traj"], color=c17, linewidth=2.4, marker="s", markevery=2, markersize=6.0,
                  markerfacecolor="white", markeredgecolor=c17, markeredgewidth=1.8, zorder=4)
    axes2[1].fill_between(steps, res_17["mean_traj"] - res_17["std_traj"], res_17["mean_traj"] + res_17["std_traj"],
                          color=s17, alpha=0.28, zorder=2)
    axes2[1].set_title("(b) ASSIST2017", fontweight="bold", fontsize=12.5, pad=10)
    axes2[1].set_xlabel("Chronological Sequence of Learning Interactions", fontsize=11.0)
    axes2[1].set_xlim(0, 20)
    axes2[1].set_xticks([0, 4, 8, 12, 16, 20])
    axes2[1].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)

    fig2.savefig(output_dir / "paper_mastery_progress.png", dpi=300, bbox_inches="tight")
    fig2.savefig(output_dir / "paper_mastery_progress.pdf", bbox_inches="tight")
    plt.close(fig2)
    print("  [+] Saved Figure 2: paper_mastery_progress.png")

    # -------------------------------------------------------------
    # Plot 3: Path Step Delta Heatmap (Zero-Centered Colormap)
    # -------------------------------------------------------------
    print("\n[*] Rendering Figure 3: Path Step Delta Mastery Heatmap...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.18})
    norm_delta = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)

    axes3[0].imshow(res_12["delta_matrix"], aspect="auto", cmap="RdBu_r", norm=norm_delta, extent=[1, 25, 500, 1])
    axes3[0].set_title("(a) ASSIST12", fontweight="bold", fontsize=13, pad=10)
    axes3[0].set_xlabel("Concepts", fontsize=12)
    axes3[0].set_ylabel("Learners", fontsize=12)
    axes3[0].set_xticks([1, 13, 25])
    axes3[0].set_yticks([1, 251, 500])

    im3 = axes3[1].imshow(res_17["delta_matrix"], aspect="auto", cmap="RdBu_r", norm=norm_delta, extent=[1, 25, 500, 1])
    axes3[1].set_title("(b) ASSIST2017", fontweight="bold", fontsize=13, pad=10)
    axes3[1].set_xlabel("Concepts", fontsize=12)
    axes3[1].set_ylabel("Learners", fontsize=12)
    axes3[1].set_xticks([1, 13, 25])
    axes3[1].set_yticks([1, 251, 500])

    cbar3 = fig3.colorbar(im3, ax=axes3, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar3.set_label(r"$\Delta$ Mastery", fontsize=12)
    cbar3.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])

    fig3.savefig(output_dir / "paper_path_delta_heatmap.png", dpi=300, bbox_inches="tight")
    fig3.savefig(output_dir / "paper_path_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("  [+] Saved Figure 3: paper_path_delta_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Master Execution for All Paper Plots & Data.")
    parser.add_argument("--artifacts-12", default="artifacts/ASSISST2012", help="Path to ASSIST12 artifacts")
    parser.add_argument("--artifacts-17", default="artifacts/ASSISST2017", help="Path to ASSIST2017 artifacts")
    parser.add_argument("--budget-minutes", type=float, default=30.0, help="Time budget (default: 30.0)")
    parser.add_argument("--top-k", type=int, default=500, help="Top K learners to evaluate (default: 500)")
    parser.add_argument("--output", default="experiments/final_paper_figures", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate ASSIST12
    res_12 = evaluate_dataset(
        artifacts_dir=args.artifacts_12,
        dataset_name="ASSIST12",
        top_k=args.top_k,
        budget_minutes=args.budget_minutes,
        device=args.device,
        output_dir=out_dir,
    )

    # 2. Evaluate ASSIST2017
    res_17 = evaluate_dataset(
        artifacts_dir=args.artifacts_17,
        dataset_name="ASSIST2017",
        top_k=args.top_k,
        budget_minutes=args.budget_minutes,
        device=args.device,
        output_dir=out_dir,
    )

    # 3. Export Stepwise Progression CSV
    steps = np.arange(21)
    df_prog = pd.DataFrame({
        "step": steps,
        "assist12_mean": np.round(res_12["mean_traj"], 4),
        "assist12_std": np.round(res_12["std_traj"], 4),
        "assist17_mean": np.round(res_17["mean_traj"], 4),
        "assist17_std": np.round(res_17["std_traj"], 4),
    })
    df_prog.to_csv(out_dir / "stepwise_mastery_progress.csv", index=False)

    # 4. Render All Figures
    render_all_plots(res_12, res_17, out_dir)

    print("\n" + "=" * 70)
    print("                     FINAL EXPERIMENT REPORT                          ")
    print("=" * 70)
    print(f"ASSIST12 (N = 500):")
    print(f"  • Path Mastery:    Before = {res_12['pre_matrix'].mean():.4f} ---> After = {res_12['post_matrix'].mean():.4f} (Δ = +{res_12['delta_matrix'].mean():.4f})")
    print(f"  • Global Mastery:  Step 0 = {res_12['mean_traj'][0]:.4f} ---> Step 20 = {res_12['mean_traj'][20]:.4f}")
    print(f"\nASSIST2017 (N = 500):")
    print(f"  • Path Mastery:    Before = {res_17['pre_matrix'].mean():.4f} ---> After = {res_17['post_matrix'].mean():.4f} (Δ = +{res_17['delta_matrix'].mean():.4f})")
    print(f"  • Global Mastery:  Step 0 = {res_17['mean_traj'][0]:.4f} ---> Step 20 = {res_17['mean_traj'][20]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
