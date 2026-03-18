"""Lobotomy scan visualisation — heatmaps and skyline plots.

Reads sweep results from CSV and produces:
  - Per-probe heatmaps (math, EQ, multilingual, combined)
  - Skyline plots (row/column marginal averages)
  - Optimal configuration annotation
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)


def load_sweep_results(csv_path: str | Path) -> list[dict]:
    """Load sweep results from CSV.

    Expected columns: i, j, math_score, eq_score, combined_score
    (plus optional multilingual_score).
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "i": int(row["i"]),
                    "j": int(row["j"]),
                    "math_score": float(row["math_score"]),
                    "eq_score": float(row["eq_score"]),
                    "combined_score": float(row["combined_score"]),
                    "multilingual_score": float(row.get("multilingual_score", 0.0)),
                }
            )
    return rows


def _get_baseline(rows: list[dict], key: str) -> float:
    for r in rows:
        if r["i"] == 0 and r["j"] == 0:
            return r[key]
    return 0.0


def _build_delta_matrix(
    rows: list[dict],
    key: str,
    n_layers: int,
) -> np.ndarray:
    """Build an n × n matrix of score deltas vs baseline.

    Rows = start layer (i), columns = end layer (j).
    The baseline (0, 0) is excluded (set to NaN) so it does not bias
    marginal averages.  Only valid configs with i < j are filled.
    """
    baseline = _get_baseline(rows, key)
    mat = np.full((n_layers, n_layers), np.nan)
    for r in rows:
        i, j = r["i"], r["j"]
        if i == 0 and j == 0:
            continue  # baseline excluded — not a perturbation
        if 0 <= i < n_layers and 0 < j <= n_layers:
            mat[i, j - 1] = r[key] - baseline
    return mat


def _symmetric_clim(mat: np.ndarray) -> tuple[float, float]:
    """Return symmetric (vmin, vmax) so white sits exactly at zero."""
    valid = mat[~np.isnan(mat)]
    if len(valid) == 0:
        return -1.0, 1.0
    abs_max = max(abs(valid.min()), abs(valid.max()))
    if abs_max == 0:
        abs_max = 1.0
    return -abs_max, abs_max


def plot_lobotomy_heatmap(
    rows: list[dict],
    key: str,
    title: str,
    output_path: str | Path,
    n_layers: int,
    *,
    annotate_best: bool = True,
) -> None:
    """Generate and save a single heatmap.

    Colour convention: red = better than baseline, blue = worse.
    The colour scale is always symmetric around zero so white = no change.
    """
    mat = _build_delta_matrix(rows, key, n_layers)
    baseline = _get_baseline(rows, key)
    vmin, vmax = _symmetric_clim(mat)

    # Build a symmetric diverging colourmap: blue → white → red
    cmap = plt.cm.RdBu_r  # low=blue, high=red

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        mat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Δ vs baseline  (baseline {key} = {baseline:.4f})", fontsize=11)
    cbar.ax.axhline(0, color="black", linewidth=1.0, linestyle="--")

    # Tick marks every 5 layers
    tick_positions = list(range(0, n_layers, 5))
    tick_labels = [str(t) for t in tick_positions]
    # x-axis: j goes from 1..n_layers, stored in columns 0..n_layers-1
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t + 1) for t in tick_positions], fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    if annotate_best:
        valid_deltas = [
            (r["i"], r["j"], r[key])
            for r in rows
            if not (r["i"] == 0 and r["j"] == 0)
        ]
        if valid_deltas:
            best = max(valid_deltas, key=lambda x: x[2])
            besti, bestj = best[0], best[1]
            # matrix coords: row=i, col=j-1
            ax.plot(
                bestj - 1, besti,
                "o", color="lime", markersize=14,
                markeredgecolor="black", markeredgewidth=2,
                zorder=5,
            )
            delta_val = best[2] - baseline
            ax.annotate(
                f"best ({besti},{bestj})\nΔ={delta_val:+.4f}",
                xy=(bestj - 1, besti),
                xytext=(bestj + 2, besti - 4),
                fontsize=9,
                color="black",
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", fc="lime", alpha=0.9),
                zorder=6,
            )

    ax.set_xlabel("End layer (j)", fontsize=12)
    ax.set_ylabel("Start layer (i)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)


def plot_skyline(
    rows: list[dict],
    key: str,
    title: str,
    output_path: str | Path,
    n_layers: int,
) -> None:
    """Skyline: marginal averages of Δ vs baseline over i and j.

    Left panel: for each start layer i, mean delta over all valid j.
    Right panel: for each end layer j, mean delta over all valid i.

    Bars are red when the mean is positive (improvement) and blue when
    negative (degradation).  The baseline row/column is excluded so it
    does not artificially pull means toward zero.
    """
    mat = _build_delta_matrix(rows, key, n_layers)

    # axis=1 → collapse j → one value per i (start layer)
    row_means = np.nanmean(mat, axis=1)
    # axis=0 → collapse i → one value per j-slot (end layer j=col+1)
    col_means = np.nanmean(mat, axis=0)

    def _bar_colors(values: np.ndarray) -> list[str]:
        colors = []
        for v in values:
            if np.isnan(v):
                colors.append("none")
            elif v >= 0:
                colors.append("#d62728")   # red — improvement
            else:
                colors.append("#1f77b4")   # blue — degradation
        return colors

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---------- left: by start layer i ----------
    x_i = np.arange(n_layers)
    mask_i = ~np.isnan(row_means)
    axes[0].bar(
        x_i[mask_i], row_means[mask_i],
        color=[c for c, m in zip(_bar_colors(row_means), mask_i) if m],
        alpha=0.85, width=0.8,
    )
    axes[0].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlim(-1, n_layers)
    axes[0].set_xlabel("Start layer (i)", fontsize=11)
    axes[0].set_ylabel(f"Mean {key} Δ vs baseline", fontsize=11)
    axes[0].set_title("By start layer — best i to begin duplication", fontsize=11)
    if mask_i.any():
        yabs = np.nanmax(np.abs(row_means[mask_i]))
        axes[0].set_ylim(-yabs * 1.2 - 1e-9, yabs * 1.2 + 1e-9)
    axes[0].tick_params(axis="x", labelsize=8)
    _add_best_label(axes[0], x_i[mask_i], row_means[mask_i], "i")

    # ---------- right: by end layer j ----------
    # col index k corresponds to end layer j = k+1
    x_j = np.arange(1, n_layers + 1)
    mask_j = ~np.isnan(col_means)
    axes[1].bar(
        x_j[mask_j], col_means[mask_j],
        color=[c for c, m in zip(_bar_colors(col_means), mask_j) if m],
        alpha=0.85, width=0.8,
    )
    axes[1].axhline(0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlim(0, n_layers + 1)
    axes[1].set_xlabel("End layer (j)", fontsize=11)
    axes[1].set_ylabel(f"Mean {key} Δ vs baseline", fontsize=11)
    axes[1].set_title("By end layer — best j to stop duplication", fontsize=11)
    if mask_j.any():
        yabs = np.nanmax(np.abs(col_means[mask_j]))
        axes[1].set_ylim(-yabs * 1.2 - 1e-9, yabs * 1.2 + 1e-9)
    axes[1].tick_params(axis="x", labelsize=8)
    _add_best_label(axes[1], x_j[mask_j], col_means[mask_j], "j")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.85, label="Improvement vs baseline"),
        Patch(facecolor="#1f77b4", alpha=0.85, label="Degradation vs baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved skyline: %s", output_path)


def _add_best_label(ax: plt.Axes, x: np.ndarray, y: np.ndarray, axis_name: str) -> None:
    """Annotate the bar with the highest mean value."""
    if len(y) == 0:
        return
    best_idx = np.argmax(y)
    best_x, best_y = x[best_idx], y[best_idx]
    ax.annotate(
        f"best {axis_name}={int(best_x)}\n({best_y:+.4f})",
        xy=(best_x, best_y),
        xytext=(best_x, best_y + (0.02 if best_y >= 0 else -0.02)),
        ha="center",
        fontsize=8,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9),
    )


def generate_all_plots(
    csv_path: str | Path,
    output_dir: str | Path,
    n_layers: int,
) -> None:
    """Generate all heatmaps and skyline plots from a sweep CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sweep_results(csv_path)
    logger.info("Loaded %d sweep results", len(rows))

    for key, label in [
        ("math_score", "Math"),
        ("eq_score", "EQ"),
        ("combined_score", "Combined"),
    ]:
        plot_lobotomy_heatmap(
            rows, key,
            f"Lobotomy Scan — {label} Δ vs baseline",
            output_dir / f"heatmap_{key}.png",
            n_layers,
        )
        plot_skyline(
            rows, key,
            f"Lobotomy Skyline — {label}",
            output_dir / f"skyline_{key}.png",
            n_layers,
        )

    has_multilingual = any(r["multilingual_score"] != 0.0 for r in rows)
    if has_multilingual:
        plot_lobotomy_heatmap(
            rows, "multilingual_score",
            "Lobotomy Scan — Multilingual Δ vs baseline",
            output_dir / "heatmap_multilingual.png",
            n_layers,
        )
        plot_skyline(
            rows, "multilingual_score",
            "Lobotomy Skyline — Multilingual",
            output_dir / "skyline_multilingual.png",
            n_layers,
        )


def main():
    parser = argparse.ArgumentParser(description="Generate Lobotomy heatmaps")
    parser.add_argument("--input", required=True, help="Path to sweep CSV")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    parser.add_argument("--n-layers", type=int, required=True, help="Number of layers in the model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_all_plots(args.input, args.output, args.n_layers)


if __name__ == "__main__":
    main()
