#!/usr/bin/env python3
"""Merge per-task CSV results from parallel sweep and generate heatmaps.

Run after all array jobs or multi-GPU jobs have completed.

Usage:
    python slurm/merge_results.py --input-dir results/scores --output results --n-layers 54
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobotomy.heatmap import generate_all_plots

logger = logging.getLogger("lobotomy.merge")


def merge_csvs(input_dir: Path, output_path: Path) -> int:
    """Merge all sweep_*.csv files into a single CSV, deduplicating."""
    csv_files = sorted(input_dir.glob("sweep_*.csv"))
    if not csv_files:
        logger.error("No sweep_*.csv files found in %s", input_dir)
        return 0

    logger.info("Found %d CSV files to merge", len(csv_files))

    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    fieldnames = None

    for csv_path in csv_files:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                key = (int(row["i"]), int(row["j"]))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    rows.sort(key=lambda r: (int(r["i"]), int(r["j"])))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Merged %d unique configs into %s", len(rows), output_path)
    return len(rows)


def print_best(csv_path: Path) -> None:
    """Print the best configuration from merged results."""
    best = None
    best_score = -float("inf")
    baseline_score = 0.0

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["combined_score"])
            if row["i"] == "0" and row["j"] == "0":
                baseline_score = score
            elif score > best_score:
                best_score = score
                best = row

    if best is None:
        return

    print("\n" + "=" * 60)
    print("BEST LOBOTOMY CONFIGURATION")
    print(f"  Config:     ({best['i']}, {best['j']})")
    print(f"  Dup layers: {best['n_dup_layers']}")
    print(f"  Combined:   {best_score:.4f} (baseline: {baseline_score:.4f}, Δ: {best_score - baseline_score:+.4f})")
    print(f"  Math:       {best['math_score']}")
    print(f"  EQ:         {best['eq_score']}")
    if float(best.get("multilingual_score", 0)):
        print(f"  Multilingual: {best['multilingual_score']}")
    print("=" * 60)
    print(f"\nTo apply: python run_lobotomy.py apply --model MODEL --config {best['i']},{best['j']} --output OUTPUT_DIR")


def main():
    parser = argparse.ArgumentParser(description="Merge parallel sweep results")
    parser.add_argument("--input-dir", required=True, help="Directory with per-task CSVs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-layers", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"

    merged_csv = output_dir / "sweep_merged.csv"
    n_configs = merge_csvs(input_dir, merged_csv)

    if n_configs == 0:
        return

    print_best(merged_csv)

    logger.info("Generating heatmaps...")
    generate_all_plots(merged_csv, heatmap_dir, args.n_layers)
    logger.info("Heatmaps saved to %s", heatmap_dir)


if __name__ == "__main__":
    main()
