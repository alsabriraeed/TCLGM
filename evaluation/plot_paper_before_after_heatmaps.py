"""
Publication-Grade Before and After Path Mastery Heatmaps for ASSIST12 and ASSIST2017.

Generates 2 heatmaps per dataset (4 heatmaps total):
- ASSIST12:
    (a) ASSIST12 (Before Path) - Initial Pre-Mastery
    (b) ASSIST12 (After Path)  - Final Post-Mastery
- ASSIST2017:
    (c) ASSIST2017 (Before Path) - Initial Pre-Mastery
    (d) ASSIST2017 (After Path)  - Final Post-Mastery

Outputs:
1. 2x2 Grid: paper_before_after_heatmaps_2x2.png / .pdf
2. 1x4 Horizontal Grid: paper_before_after_heatmaps_1x4.png / .pdf
3. Per-dataset side-by-side figures: assist12_before_after_heatmap.png and assist2017_before_after_heatmap.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Scientific paper formatting
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


def load_step_matrices(csv_path: Path, n_learners: int = 500, max_steps: int = 25):
    """Loads and pivots step-wise CSV into [n_learners x max_steps] pre and post matrices."""
    df = pd.read_csv(csv_path)

    piv_pre = df.pivot(index="learner_rank", columns="path_step", values="mastery_before")
    piv_pre = piv_pre.reindex(index=range(1, n_learners + 1), columns=range(1, max_steps + 1))
    piv_pre = piv_pre.ffill(axis=1).bfill(axis=1).fillna(0.5)

    piv_post = df.pivot(index="learner_rank", columns="path_step", values="mastery_after")
    piv_post = piv_post.reindex(index=range(1, n_learners + 1), columns=range(1, max_steps + 1))
    piv_post = piv_post.ffill(axis=1).bfill(axis=1).fillna(0.5)

    return piv_pre.values, piv_post.values


def plot_2x2_before_after(
    pre_12: np.ndarray,
    post_12: np.ndarray,
    pre_17: np.ndarray,
    post_17: np.ndarray,
    output_dir: Path,
    max_steps: int = 25,
    n_learners: int = 500,
):
    """Generates a 2x2 grid (Row 1: ASSIST12 Before & After; Row 2: ASSIST2017 Before & After)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), gridspec_kw={"wspace": 0.15, "hspace": 0.28})

    vmin, vmax = 0.20, 0.90
    cmap = "viridis"  # Smooth sequential colormap: purple (low mastery) -> teal -> bright yellow (high mastery)

    # --- Row 1: ASSIST12 ---
    im1 = axes[0, 0].imshow(
        pre_12, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[0, 0].set_title("(a) ASSIST12 - Before (Pre-Path)", fontweight="bold", fontsize=12.5, pad=8)
    axes[0, 0].set_xlabel("Concepts", fontsize=11.5)
    axes[0, 0].set_ylabel("Learners", fontsize=11.5)
    axes[0, 0].set_xticks([1, 13, max_steps])
    axes[0, 0].set_yticks([1, 251, n_learners])
    axes[0, 0].tick_params(direction="out", length=4, width=1.0)

    im2 = axes[0, 1].imshow(
        post_12, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[0, 1].set_title("(b) ASSIST12 - After (Post-Path)", fontweight="bold", fontsize=12.5, pad=8)
    axes[0, 1].set_xlabel("Concepts", fontsize=11.5)
    axes[0, 1].set_ylabel("Learners", fontsize=11.5)
    axes[0, 1].set_xticks([1, 13, max_steps])
    axes[0, 1].set_yticks([1, 251, n_learners])
    axes[0, 1].tick_params(direction="out", length=4, width=1.0)

    # --- Row 2: ASSIST2017 ---
    im3 = axes[1, 0].imshow(
        pre_17, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[1, 0].set_title("(c) ASSIST2017 - Before (Pre-Path)", fontweight="bold", fontsize=12.5, pad=8)
    axes[1, 0].set_xlabel("Concepts", fontsize=11.5)
    axes[1, 0].set_ylabel("Learners", fontsize=11.5)
    axes[1, 0].set_xticks([1, 13, max_steps])
    axes[1, 0].set_yticks([1, 251, n_learners])
    axes[1, 0].tick_params(direction="out", length=4, width=1.0)

    im4 = axes[1, 1].imshow(
        post_17, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[1, 1].set_title("(d) ASSIST2017 - After (Post-Path)", fontweight="bold", fontsize=12.5, pad=8)
    axes[1, 1].set_xlabel("Concepts", fontsize=11.5)
    axes[1, 1].set_ylabel("Learners", fontsize=11.5)
    axes[1, 1].set_xticks([1, 13, max_steps])
    axes[1, 1].set_yticks([1, 251, n_learners])
    axes[1, 1].tick_params(direction="out", length=4, width=1.0)

    # Shared Colorbar on the right
    cbar = fig.colorbar(im4, ax=axes, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar.set_label("Mastery Level", fontsize=12)
    cbar.set_ticks([0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    cbar.ax.tick_params(labelsize=10)

    png_path = output_dir / "paper_before_after_heatmaps_2x2.png"
    pdf_path = output_dir / "paper_before_after_heatmaps_2x2.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved 2x2 Before/After Heatmap PNG: {png_path}")
    print(f"[+] Saved 2x2 Before/After Heatmap PDF: {pdf_path}")


def plot_1x4_before_after(
    pre_12: np.ndarray,
    post_12: np.ndarray,
    pre_17: np.ndarray,
    post_17: np.ndarray,
    output_dir: Path,
    max_steps: int = 25,
    n_learners: int = 500,
):
    """Generates a 1x4 horizontal comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.0), gridspec_kw={"wspace": 0.18})

    vmin, vmax = 0.20, 0.90
    cmap = "viridis"

    titles = [
        "(a) ASSIST12 (Before)",
        "(b) ASSIST12 (After)",
        "(c) ASSIST2017 (Before)",
        "(d) ASSIST2017 (After)",
    ]
    mats = [pre_12, post_12, pre_17, post_17]

    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(
            mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest", extent=[1, max_steps, n_learners, 1],
        )
        ax.set_title(title, fontweight="bold", fontsize=12.0, pad=8)
        ax.set_xlabel("Concepts", fontsize=11.0)
        ax.set_ylabel("Learners", fontsize=11.0)
        ax.set_xticks([1, 13, max_steps])
        ax.set_yticks([1, 251, n_learners])
        ax.tick_params(direction="out", length=4, width=1.0)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.018, pad=0.02, extend="both")
    cbar.set_label("Mastery Level", fontsize=11.5)
    cbar.set_ticks([0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    cbar.ax.tick_params(labelsize=10)

    png_path = output_dir / "paper_before_after_heatmaps_1x4.png"
    pdf_path = output_dir / "paper_before_after_heatmaps_1x4.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved 1x4 Before/After Heatmap PNG: {png_path}")
    print(f"[+] Saved 1x4 Before/After Heatmap PDF: {pdf_path}")


def plot_single_dataset_pair(
    pre_mat: np.ndarray,
    post_mat: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    max_steps: int = 25,
    n_learners: int = 500,
):
    """Generates a side-by-side 2-panel figure for a single dataset (Before vs After)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"wspace": 0.16})

    vmin, vmax = 0.20, 0.90
    cmap = "viridis"

    im1 = axes[0].imshow(
        pre_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[0].set_title(f"(a) {dataset_name} - Before Path", fontweight="bold", fontsize=12.5, pad=8)
    axes[0].set_xlabel("Concepts", fontsize=11.5)
    axes[0].set_ylabel("Learners", fontsize=11.5)
    axes[0].set_xticks([1, 13, max_steps])
    axes[0].set_yticks([1, 251, n_learners])
    axes[0].tick_params(direction="out", length=4, width=1.0)

    im2 = axes[1].imshow(
        post_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest", extent=[1, max_steps, n_learners, 1],
    )
    axes[1].set_title(f"(b) {dataset_name} - After Path", fontweight="bold", fontsize=12.5, pad=8)
    axes[1].set_xlabel("Concepts", fontsize=11.5)
    axes[1].set_ylabel("Learners", fontsize=11.5)
    axes[1].set_xticks([1, 13, max_steps])
    axes[1].set_yticks([1, 251, n_learners])
    axes[1].tick_params(direction="out", length=4, width=1.0)

    cbar = fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.025, pad=0.03, extend="both")
    cbar.set_label("Mastery Level", fontsize=12)
    cbar.set_ticks([0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    cbar.ax.tick_params(labelsize=10)

    prefix = dataset_name.lower().replace(" ", "_")
    png_path = output_dir / f"{prefix}_before_after_heatmap.png"
    pdf_path = output_dir / f"{prefix}_before_after_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {dataset_name} Pair PNG: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Before and After Heatmaps for ASSIST12 and ASSIST2017.")
    parser.add_argument("--data-dir", default="experiments/top500_paper_results", help="Directory with top500 CSVs")
    parser.add_argument("--output", default="experiments/before_after_heatmaps", help="Output directory")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_12 = data_path / "assist12_top500_path_step_mastery.csv"
    csv_17 = data_path / "assist2017_top500_path_step_mastery.csv"

    pre_12, post_12 = load_step_matrices(csv_12, n_learners=500, max_steps=25)
    pre_17, post_17 = load_step_matrices(csv_17, n_learners=500, max_steps=25)

    print("\n[*] Generating Before & After Heatmaps for ASSIST12 and ASSIST2017...")
    plot_2x2_before_after(pre_12, post_12, pre_17, post_17, out_path)
    plot_1x4_before_after(pre_12, post_12, pre_17, post_17, out_path)
    plot_single_dataset_pair(pre_12, post_12, "ASSIST12", out_path)
    plot_single_dataset_pair(pre_17, post_17, "ASSIST2017", out_path)

    print("\n" + "=" * 70)
    print("                 BEFORE VS AFTER MASTERY AVERAGES                    ")
    print("=" * 70)
    print(f"ASSIST12:   Before Mean = {pre_12.mean():.4f}  --->  After Mean = {post_12.mean():.4f}  (Δ = +{post_12.mean() - pre_12.mean():.4f})")
    print(f"ASSIST2017: Before Mean = {pre_17.mean():.4f}  --->  After Mean = {post_17.mean():.4f}  (Δ = +{post_17.mean() - pre_17.mean():.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
