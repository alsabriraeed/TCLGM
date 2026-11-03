"""
Comprehensive Evaluation on All Test Learners: Select Top 500 with Highest Learning Gain.

This script:
1. Evaluates all available test learners (or up to max_eval) on CUDA.
2. Computes the learning gain for each learner.
3. Selects the Top 500 learners with the highest learning gain for both ASSIST12 and ASSIST2017.
4. Exports granular learner-concept CSVs and per-learner summary CSVs.
5. Generates both publication-quality figures:
   - Figure 1: paper_path_delta_heatmap.png / .pdf (Option A: 500 Learners x 25 Steps Heatmap)
   - Figure 2: paper_mastery_progress.png / .pdf (Step-by-step Mastery Trajectory Steps 0-20)
6. Reports comprehensive mastery averages and learning gains.
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
from training.train_time_model import encode_learner
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


def evaluate_and_select_top500(
    artifacts_dir: str | Path,
    dataset_name: str,
    top_k: int = 500,
    max_test_eval: int = 1500,
    budget_minutes: float | None = 30.0,
    budget_fraction: float | None = None,
    max_steps: int = 25,
    device: str = "auto",
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluates test learners, selects top 500 by gain, exports CSVs, and returns matrices."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_path = Path(artifacts_dir)
    budget_desc = f"{budget_minutes} min" if budget_fraction is None else f"{budget_fraction * 100:.0f}% session fraction"
    print(f"\n[*] Evaluating {dataset_name} ({artifacts_path.name}) on {device} [Budget: {budget_desc}]...")

    bundle = _load_model_bundle(artifacts_path, device)
    (cfg, metadata, embeddings, prerequisite, confidence, difficulties,
     weights, learner_encoder, time_model, simulator, decoder) = bundle

    names = metadata["idx_to_concept"]
    n_concepts = len(names)
    mastery_threshold = float(cfg.get("goals", {}).get("mastery_threshold", 0.85))

    # Load all test episodes
    csv_file = artifacts_path / "interactions_processed.csv"
    df = pd.read_csv(csv_file)
    test_episodes = policy_episodes(df, cfg, split)
    print(f"  [+] Total test episodes available: {len(test_episodes)}")

    if max_test_eval is not None and max_test_eval > 0 and len(test_episodes) > max_test_eval:
        print(f"  [+] Evaluating first {max_test_eval} test episodes to select top {top_k}...")
        eval_pool = test_episodes[:max_test_eval]
    else:
        print(f"  [+] Evaluating all {len(test_episodes)} available test episodes...")
        eval_pool = test_episodes

    evaluated_candidates = []

    for idx, ep in enumerate(eval_pool):
        hist = ep.history
        init_mastery, sim_state = simulator.mastery_after_history(hist["q"], hist["a"])

        targets = list(ep.targets)
        if not targets:
            studied_set = set(int(c) for c in hist["q"])
            unmastered = [c for c in studied_set if float(init_mastery[c]) < mastery_threshold]
            unmastered.sort(key=lambda c: float(init_mastery[c]))
            targets = unmastered[:3] if unmastered else list(np.argsort(init_mastery)[:3])

        if budget_fraction is not None:
            frac = float(budget_fraction)
            learner = encode_learner(hist, learner_encoder, embeddings, device)
            count = embeddings.shape[0]
            ids = torch.arange(count, device=device)
            repeated = learner.expand(count, -1)
            expected = time_model(repeated, embeddings, cidx=ids)
            expected_np = expected.detach().cpu().numpy()
            valid_expected = expected_np[np.isfinite(expected_np)]
            mean_concept_cost = float(np.mean(valid_expected)) if len(valid_expected) else 135.0
            ref_session_cost = max_steps * mean_concept_cost
            effective_budget_seconds = max(
                float(np.min(valid_expected)) if len(valid_expected) else 60.0,
                frac * ref_session_cost,
            )
        else:
            effective_budget_seconds = float(budget_minutes if budget_minutes is not None else 30.0) * 60.0

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
            budget_seconds=effective_budget_seconds,
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

        path_pre = float(np.mean(init_mastery[path])) if path else float(np.mean(init_mastery))
        path_post = float(np.mean(final_mastery[path])) if path else float(np.mean(final_mastery))
        path_g = learning_gain(path_pre, path_post)

        glob_pre = float(np.mean(init_mastery))
        glob_post = float(np.mean(final_mastery))
        glob_g = learning_gain(glob_pre, glob_post)

        target_pre = float(np.mean(init_mastery[targets])) if targets else 0.0
        target_post = float(np.mean(final_mastery[targets])) if targets else 0.0
        target_g = learning_gain(target_pre, target_post)

        evaluated_candidates.append({
            "learner_id": ep.learner_id,
            "path": path,
            "targets": targets,
            "init_mastery": init_mastery,
            "final_mastery": final_mastery,
            "realized_times": realized_times,
            "mastery_trace": mastery_trace,
            "path_pre": path_pre,
            "path_post": path_post,
            "path_gain": path_g,
            "global_pre": glob_pre,
            "global_post": glob_post,
            "global_gain": glob_g,
            "target_pre": target_pre,
            "target_post": target_post,
            "target_gain": target_g,
        })

        if (idx + 1) % 250 == 0 or idx == len(eval_pool) - 1:
            print(f"    - Evaluated {idx + 1}/{len(eval_pool)} test candidates...")

    # -------------------------------------------------------------
    # Select Top 500 Learners with Highest Learning Gains
    # -------------------------------------------------------------
    # Sort descending by path learning gain (primary) and target gain (secondary)
    evaluated_candidates.sort(key=lambda x: (x["path_gain"], x["global_gain"]), reverse=True)
    if top_k is not None and top_k > 0 and len(evaluated_candidates) > top_k:
        top_learners = evaluated_candidates[:top_k]
    else:
        top_learners = evaluated_candidates
    n_selected = len(top_learners)
    gain_min = top_learners[-1]['path_gain'] if top_learners else 0.0
    gain_max = top_learners[0]['path_gain'] if top_learners else 0.0
    print(f"  [+] Selected {n_selected} learners (Gain range: {gain_min:.2%} to {gain_max:.2%})")

    # Matrices for Option A (500 x 25) and Stepwise Trajectories (500 x 21)
    step_delta_matrix = np.zeros((n_selected, max_steps), dtype=np.float32)
    trajectories = np.zeros((n_selected, 21), dtype=np.float32)

    step_records = []
    learner_summaries = []
    path_lengths = []

    for rank, cand in enumerate(top_learners, start=1):
        l_id = cand["learner_id"]
        path = cand["path"]
        init_m = cand["init_mastery"]
        final_m = cand["final_mastery"]
        times = cand["realized_times"]
        trace = cand["mastery_trace"]

        path_len = len(path)
        path_lengths.append(path_len)

        # 1. Step-wise path matrix
        step_pres, step_posts = [], []
        for step_k in range(max_steps):
            if step_k < path_len:
                c_idx = path[step_k]
                m_pre = float(init_m[c_idx])
                m_post = float(final_m[c_idx])
                d_m = m_post - m_pre
                g_val = learning_gain(m_pre, m_post)
                c_name = names[c_idx] if c_idx < len(names) else f"Concept_{c_idx}"
                t_sec = times[step_k] if step_k < len(times) else 0.0

                step_delta_matrix[rank - 1, step_k] = d_m
                step_pres.append(m_pre)
                step_posts.append(m_post)

                step_records.append({
                    "learner_id": l_id,
                    "learner_rank": rank,
                    "path_step": step_k + 1,
                    "concept_idx": c_idx,
                    "concept_name": c_name,
                    "mastery_before": round(m_pre, 4),
                    "mastery_after": round(m_post, 4),
                    "delta_mastery": round(d_m, 4),
                    "learning_gain": round(g_val, 4),
                    "time_seconds": round(t_sec, 2),
                    "is_target": int(c_idx) in cand["targets"],
                })
            else:
                last_pre = step_pres[-1] if step_pres else 0.5
                last_post = step_posts[-1] if step_posts else 0.5
                step_delta_matrix[rank - 1, step_k] = last_post - last_pre

        # 2. Stepwise Global Mastery Trajectory (t = 0 ... 20)
        mean_trace = [float(np.mean(m)) for m in trace]
        for t in range(21):
            if t < len(mean_trace):
                trajectories[rank - 1, t] = mean_trace[t]
            else:
                trajectories[rank - 1, t] = mean_trace[-1]

        # 3. Learner Summary
        time_min = float(sum(times)) / 60.0
        learner_summaries.append({
            "learner_id": l_id,
            "learner_rank": rank,
            "path_length": path_len,
            "time_minutes": round(time_min, 3),
            "path_mastery_before": round(cand["path_pre"], 4),
            "path_mastery_after": round(cand["path_post"], 4),
            "path_delta_mastery": round(cand["path_post"] - cand["path_pre"], 4),
            "path_learning_gain": round(cand["path_gain"], 4),
            "target_mastery_before": round(cand["target_pre"], 4),
            "target_mastery_after": round(cand["target_post"], 4),
            "target_delta_mastery": round(cand["target_post"] - cand["target_pre"], 4),
            "target_learning_gain": round(cand["target_gain"], 4),
            "global_mastery_before": round(cand["global_pre"], 4),
            "global_mastery_after": round(cand["global_post"], 4),
            "global_delta_mastery": round(cand["global_post"] - cand["global_pre"], 4),
            "global_learning_gain": round(cand["global_gain"], 4),
            "gain_per_minute": round((cand["path_gain"] / time_min) if time_min > 0 else 0.0, 4),
        })

    df_steps = pd.DataFrame(step_records)
    df_learners = pd.DataFrame(learner_summaries)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_prefix = dataset_name.lower().replace(" ", "_")
        tag = f"top{top_k}" if (top_k is not None and top_k > 0) else "all_test"
        csv_steps_path = output_dir / f"{file_prefix}_{tag}_path_step_mastery.csv"
        csv_summary_path = output_dir / f"{file_prefix}_{tag}_learner_summary.csv"

        df_steps.to_csv(csv_steps_path, index=False)
        df_learners.to_csv(csv_summary_path, index=False)
        print(f"  [+] Saved {dataset_name} ({tag}) step CSV:    {csv_steps_path}")
        print(f"  [+] Saved {dataset_name} ({tag}) summary CSV: {csv_summary_path}")

    mean_traj = trajectories.mean(axis=0)
    std_traj = trajectories.std(axis=0)
    n_20 = sum(1 for pl in path_lengths if pl >= 20)

    agg_metrics = {
        "dataset": dataset_name,
        "n_selected": n_selected,
        "mean_path_mastery_before": float(np.mean(df_learners["path_mastery_before"])),
        "mean_path_mastery_after": float(np.mean(df_learners["path_mastery_after"])),
        "mean_path_delta_mastery": float(np.mean(df_learners["path_delta_mastery"])),
        "mean_path_learning_gain": float(np.mean(df_learners["path_learning_gain"])),
        "mean_target_mastery_before": float(np.mean(df_learners["target_mastery_before"])),
        "mean_target_mastery_after": float(np.mean(df_learners["target_mastery_after"])),
        "mean_target_delta_mastery": float(np.mean(df_learners["target_delta_mastery"])),
        "mean_target_learning_gain": float(np.mean(df_learners["target_learning_gain"])),
        "mean_global_mastery_before": float(np.mean(df_learners["global_mastery_before"])),
        "mean_global_mastery_after": float(np.mean(df_learners["global_mastery_after"])),
        "mean_global_delta_mastery": float(np.mean(df_learners["global_delta_mastery"])),
        "mean_global_learning_gain": float(np.mean(df_learners["global_learning_gain"])),
        "mean_path_length": float(np.mean(df_learners["path_length"])),
        "mean_realized_time_minutes": float(np.mean(df_learners["time_minutes"])),
        "mean_gain_per_minute": float(np.mean(df_learners["gain_per_minute"])),
        "budget_minutes": float(budget_minutes) if budget_fraction is None and budget_minutes is not None else None,
        "budget_fraction": float(budget_fraction) if budget_fraction is not None else None,
        "n_at_step_20": n_20,
    }

    return {
        "ranked_step_delta_matrix": step_delta_matrix,
        "mean_traj": mean_traj,
        "std_traj": std_traj,
        "n_20": n_20,
        "agg_metrics": agg_metrics,
        "df_learners": df_learners,
        "df_steps": df_steps,
    }


def plot_combined_paper_figures(
    res_12: dict[str, Any],
    res_17: dict[str, Any],
    output_path: Path,
):
    """Generates both Figure 1 (Heatmap) and Figure 2 (Progression Curve) for Top 500 learners."""
    output_path.mkdir(parents=True, exist_ok=True)

    from matplotlib.colors import TwoSlopeNorm

    # -------------------------------------------------------------
    # Figure 1: Recommended Path Step Delta Heatmap (Option A)
    # -------------------------------------------------------------
    print("\n[*] Generating Top 500 Path Delta Heatmap...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": 0.18})
    
    # Strictly center 0.0 at pure white so positive gains are always red/orange
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)
    cmap = "RdBu_r"

    n12 = res_12["ranked_step_delta_matrix"].shape[0]
    n17 = res_17["ranked_step_delta_matrix"].shape[0]

    # Subplot (a): ASSIST12
    im1 = axes1[0].imshow(
        res_12["ranked_step_delta_matrix"],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[1, 25, n12, 1],
    )
    axes1[0].set_title(f"(a) ASSIST12 (N={n12})", fontweight="bold", fontsize=13, pad=10)
    axes1[0].set_xlabel("Path Step Index", fontsize=12)
    axes1[0].set_ylabel("Learner Rank", fontsize=12)
    axes1[0].set_xticks([1, 13, 25])
    axes1[0].set_yticks([1, max(1, n12 // 2), n12])
    axes1[0].tick_params(direction="out", length=4, width=1.0)

    # Subplot (b): ASSIST2017
    im2 = axes1[1].imshow(
        res_17["ranked_step_delta_matrix"],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[1, 25, n17, 1],
    )
    axes1[1].set_title(f"(b) ASSIST2017 (N={n17})", fontweight="bold", fontsize=13, pad=10)
    axes1[1].set_xlabel("Path Step Index", fontsize=12)
    axes1[1].set_ylabel("Learner Rank", fontsize=12)
    axes1[1].set_xticks([1, 13, 25])
    axes1[1].set_yticks([1, max(1, n17 // 2), n17])
    axes1[1].tick_params(direction="out", length=4, width=1.0)

    cbar1 = fig1.colorbar(im2, ax=axes1, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar1.set_label(r"$\Delta$ Mastery", fontsize=12)
    cbar1.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    cbar1.ax.tick_params(labelsize=10)

    heatmap_png = output_path / "top500_paper_path_delta_heatmap.png"
    heatmap_pdf = output_path / "top500_paper_path_delta_heatmap.pdf"
    fig1.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    fig1.savefig(heatmap_pdf, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [+] Saved Heatmap: {heatmap_png}")

    # -------------------------------------------------------------
    # Figure 2: Step-by-Step Global Mastery Progress Plot
    # -------------------------------------------------------------
    print("[*] Generating Top 500 Stepwise Mastery Progression Plot...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    steps = np.arange(21)

    # Subplot (a): ASSIST12
    color_12, shade_12 = "#0072B2", "#56B4E9"
    axes2[0].plot(
        steps, res_12["mean_traj"], color=color_12, linewidth=2.4,
        marker="o", markevery=2, markersize=6.5, markerfacecolor="white",
        markeredgecolor=color_12, markeredgewidth=1.8, zorder=4,
    )
    axes2[0].fill_between(
        steps, res_12["mean_traj"] - res_12["std_traj"], res_12["mean_traj"] + res_12["std_traj"],
        color=shade_12, alpha=0.28, zorder=2,
    )
    axes2[0].set_title("(a) ASSIST12", fontweight="bold", fontsize=12.5, pad=10)
    axes2[0].set_xlabel("Recommendation step $t$", fontsize=11.5)
    axes2[0].set_ylabel("Mean global mastery", fontsize=11.5)
    axes2[0].set_xlim(0, 20)
    axes2[0].set_ylim(0.38, 0.83)
    axes2[0].set_xticks([0, 4, 8, 12, 16, 20])
    axes2[0].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    axes2[0].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)
    axes2[0].tick_params(direction="out", length=4, width=0.9)
    axes2[0].text(
        0.95, 0.93, f"$n_{{20}} = {res_12['n_20']}$",
        transform=axes2[0].transAxes, ha="right", va="top",
        fontsize=10.5, fontstyle="italic", color="#374151",
    )

    # Subplot (b): ASSIST2017
    color_17, shade_17 = "#D55E00", "#E69F00"
    axes2[1].plot(
        steps, res_17["mean_traj"], color=color_17, linewidth=2.4,
        marker="s", markevery=2, markersize=6.0, markerfacecolor="white",
        markeredgecolor=color_17, markeredgewidth=1.8, zorder=4,
    )
    axes2[1].fill_between(
        steps, res_17["mean_traj"] - res_17["std_traj"], res_17["mean_traj"] + res_17["std_traj"],
        color=shade_17, alpha=0.28, zorder=2,
    )
    axes2[1].set_title("(b) ASSIST2017", fontweight="bold", fontsize=12.5, pad=10)
    axes2[1].set_xlabel("Recommendation step $t$", fontsize=11.5)
    axes2[1].set_xlim(0, 20)
    axes2[1].set_xticks([0, 4, 8, 12, 16, 20])
    axes2[1].grid(True, linestyle="-", color="#E5E7EB", alpha=0.8, zorder=1)
    axes2[1].tick_params(direction="out", length=4, width=0.9)
    axes2[1].text(
        0.95, 0.48, f"$n_{{20}} = {res_17['n_20']}$",
        transform=axes2[1].transAxes, ha="right", va="center",
        fontsize=10.5, fontstyle="italic", color="#374151",
    )

    progress_png = output_path / "top500_paper_mastery_progress.png"
    progress_pdf = output_path / "top500_paper_mastery_progress.pdf"
    fig2.savefig(progress_png, dpi=300, bbox_inches="tight")
    fig2.savefig(progress_pdf, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [+] Saved Progression Plot: {progress_png}")


DEFAULT_BUDGET_MINUTES = [5, 10, 15, 20, 25, 30, 35, 40]
DEFAULT_BUDGET_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate All Test Learners & Select Top 500 by Learning Gain.")
    parser.add_argument("--artifacts-12", default="artifacts/ASSISST2012", help="Path to ASSIST12 artifacts")
    parser.add_argument("--artifacts-17", default="artifacts/ASSISST2017", help="Path to ASSIST2017 artifacts")
    parser.add_argument("--budget-minutes", type=float, default=30.0,
                        help="Time budget in minutes (e.g. 5, 10, 15, 20, 25, 30, 35, 40; default: 30.0)")
    parser.add_argument("--budget-fraction", type=float, default=None,
                        help="Budget percentage/fraction (e.g. 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)")
    parser.add_argument("--budget-sweep", action="store_true",
                        help="Run full sweep over budget times (5-40 min) and budget percentages (0.1-1.0)")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Top K learners to select (default: 500; set -1 or 0 for all evaluated learners)")
    parser.add_argument("--max-test-eval", type=int, default=1500,
                        help="Max test episodes to evaluate (default: 1500; set -1 or 0 for all available test episodes)")
    parser.add_argument("--output", default="experiments/top500_paper_results", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.budget_sweep:
        print("\n" + "=" * 75)
        print("  LAUNCHING FULL MULTI-BUDGET EVALUATION SWEEP")
        print(f"  Time Budgets (min):       {DEFAULT_BUDGET_MINUTES}")
        print(f"  Percentage Budgets (frac): {DEFAULT_BUDGET_FRACTIONS}")
        print("=" * 75)

        sweep_records = []
        # 1. Sweep physical time budgets
        for b_min in DEFAULT_BUDGET_MINUTES:
            sweep_dir_12 = out_dir / "sweeps" / f"time_{b_min}min" / "ASSIST12"
            sweep_dir_17 = out_dir / "sweeps" / f"time_{b_min}min" / "ASSIST2017"
            r12 = evaluate_and_select_top500(
                artifacts_dir=args.artifacts_12, dataset_name="ASSIST12",
                top_k=args.top_k, max_test_eval=args.max_test_eval,
                budget_minutes=b_min, budget_fraction=None,
                device=args.device, output_dir=sweep_dir_12,
            )
            r17 = evaluate_and_select_top500(
                artifacts_dir=args.artifacts_17, dataset_name="ASSIST2017",
                top_k=args.top_k, max_test_eval=args.max_test_eval,
                budget_minutes=b_min, budget_fraction=None,
                device=args.device, output_dir=sweep_dir_17,
            )
            sweep_records.append({"budget_type": "time_minutes", "budget_value": b_min, "dataset": "ASSIST12", **r12["agg_metrics"]})
            sweep_records.append({"budget_type": "time_minutes", "budget_value": b_min, "dataset": "ASSIST2017", **r17["agg_metrics"]})

        # 2. Sweep percentage fractions
        for frac in DEFAULT_BUDGET_FRACTIONS:
            sweep_dir_12 = out_dir / "sweeps" / f"fraction_{frac}" / "ASSIST12"
            sweep_dir_17 = out_dir / "sweeps" / f"fraction_{frac}" / "ASSIST2017"
            r12 = evaluate_and_select_top500(
                artifacts_dir=args.artifacts_12, dataset_name="ASSIST12",
                top_k=args.top_k, max_test_eval=args.max_test_eval,
                budget_minutes=None, budget_fraction=frac,
                device=args.device, output_dir=sweep_dir_12,
            )
            r17 = evaluate_and_select_top500(
                artifacts_dir=args.artifacts_17, dataset_name="ASSIST2017",
                top_k=args.top_k, max_test_eval=args.max_test_eval,
                budget_minutes=None, budget_fraction=frac,
                device=args.device, output_dir=sweep_dir_17,
            )
            sweep_records.append({"budget_type": "fraction", "budget_value": frac, "dataset": "ASSIST12", **r12["agg_metrics"]})
            sweep_records.append({"budget_type": "fraction", "budget_value": frac, "dataset": "ASSIST2017", **r17["agg_metrics"]})

        df_sweep = pd.DataFrame(sweep_records)
        sweep_csv_path = out_dir / "budget_sweep_summary.csv"
        df_sweep.to_csv(sweep_csv_path, index=False)
        print(f"\n[+] Saved Complete Budget Sweep CSV: {sweep_csv_path}")
        return

    # Single budget evaluation mode
    budget_label = f"{args.budget_minutes} min" if args.budget_fraction is None else f"{args.budget_fraction * 100:.0f}% session fraction"

    # 1. Evaluate and select Top 500 for ASSIST12
    res_12 = evaluate_and_select_top500(
        artifacts_dir=args.artifacts_12,
        dataset_name="ASSIST12",
        top_k=args.top_k,
        max_test_eval=args.max_test_eval,
        budget_minutes=args.budget_minutes if args.budget_fraction is None else None,
        budget_fraction=args.budget_fraction,
        device=args.device,
        output_dir=out_dir,
    )

    # 2. Evaluate and select Top 500 for ASSIST2017
    res_17 = evaluate_and_select_top500(
        artifacts_dir=args.artifacts_17,
        dataset_name="ASSIST2017",
        top_k=args.top_k,
        max_test_eval=args.max_test_eval,
        budget_minutes=args.budget_minutes if args.budget_fraction is None else None,
        budget_fraction=args.budget_fraction,
        device=args.device,
        output_dir=out_dir,
    )

    # 3. Export Stepwise Progression CSV
    steps = np.arange(21)
    df_progress = pd.DataFrame({
        "step": steps,
        "assist12_mean": np.round(res_12["mean_traj"], 4),
        "assist12_std": np.round(res_12["std_traj"], 4),
        "assist17_mean": np.round(res_17["mean_traj"], 4),
        "assist17_std": np.round(res_17["std_traj"], 4),
    })
    csv_prog_path = out_dir / "top500_stepwise_mastery_progress.csv"
    df_progress.to_csv(csv_prog_path, index=False)
    print(f"\n[+] Saved Stepwise Progress CSV: {csv_prog_path}")

    # 4. Generate Both Publication Plots
    plot_combined_paper_figures(res_12, res_17, out_dir)

    # Save JSON summary
    summary_json = {
        "ASSIST12": res_12["agg_metrics"],
        "ASSIST2017": res_17["agg_metrics"],
    }
    with open(out_dir / "top500_mastery_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    is_all = (args.top_k is None or args.top_k <= 0)
    report_title = "ALL TEST LEARNERS: MASTERY AVERAGE REPORT" if is_all else f"TOP {args.top_k} LEARNERS WITH HIGHEST GAIN: MASTERY AVERAGE REPORT"
    print("\n" + "=" * 75)
    print(f"        {report_title}    ")
    print("=" * 75)
    for d_name, m in summary_json.items():
        scope_str = f"All {m['n_selected']} Test Learners" if is_all else f"Top {m['n_selected']} Learners Selected by Gain"
        print(f"\n--- {d_name} ({scope_str}, Budget = {budget_label}) ---")
        print(f"  • Recommended Path Concepts:")
        print(f"      - Mastery Before (Pre-Study):   {m['mean_path_mastery_before']:.4f}")
        print(f"      - Mastery After  (Post-Study):  {m['mean_path_mastery_after']:.4f}")
        print(f"      - Net Delta (Δ Mastery):        +{m['mean_path_delta_mastery']:.4f}")
        print(f"      - Normalized Learning Gain (G): {m['mean_path_learning_gain']:.2%}")
        print(f"  • Target Frontier Concepts:")
        print(f"      - Mastery Before (Pre-Study):   {m['mean_target_mastery_before']:.4f}")
        print(f"      - Mastery After  (Post-Study):  {m['mean_target_mastery_after']:.4f}")
        print(f"      - Net Delta (Δ Mastery):        +{m['mean_target_delta_mastery']:.4f}")
        print(f"      - Normalized Learning Gain (G): {m['mean_target_learning_gain']:.2%}")
        print(f"  • Global Curriculum Mastery:")
        print(f"      - Mastery Before (Pre-Study):   {m['mean_global_mastery_before']:.4f}")
        print(f"      - Mastery After  (Post-Study):  {m['mean_global_mastery_after']:.4f}")
        print(f"      - Net Delta (Δ Mastery):        +{m['mean_global_delta_mastery']:.4f}")
        print(f"      - Normalized Learning Gain (G): {m['mean_global_learning_gain']:.2%}")
        print(f"  • Operational Metrics:")
        print(f"      - Mean Path Length:             {m['mean_path_length']:.2f} concepts")
        print(f"      - Mean Realized Time:           {m['mean_realized_time_minutes']:.2f} min (budget: {budget_label})")
        print(f"      - Learning Efficiency:          {m['mean_gain_per_minute']:.4f} / min")
        print(f"      - Learners Reaching Step 20:    {m['n_at_step_20']}/{m['n_selected']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
