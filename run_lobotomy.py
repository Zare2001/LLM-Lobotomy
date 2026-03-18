#!/usr/bin/env python3
"""LLM Lobotomy — main CLI entry point.

Usage:
    # Full exhaustive sweep
    python run_lobotomy.py sweep --model utter-project/EuroLLM-22B-Instruct-2512

    # Focused sweep on a sub-region
    python run_lobotomy.py sweep --model ... --i-range 15,40 --j-range 25,50

    # Bayesian optimization (faster)
    python run_lobotomy.py bayesian --model ... --n-calls 200

    # Generate heatmaps from results
    python run_lobotomy.py heatmap --input results/scores/sweep.csv --n-layers 54

    # Apply optimal config and save the model
    python run_lobotomy.py apply --model ... --config 25,32 --output ./EuroLLM-Lobotomy
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("lobotomy")


# ======================================================================
# SWEEP
# ======================================================================

def cmd_sweep(args: argparse.Namespace) -> None:
    """Run an exhaustive sweep over all (or a range of) configurations."""
    from lobotomy.probes import EQProbe, MathProbe, MultilingualProbe
    from lobotomy.scanner import LobotomyScanner, iter_configs

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sweep.csv"

    scanner = LobotomyScanner(
        args.model,
        load_in_4bit=args.load_4bit,
        torch_dtype=args.dtype,
    )
    n_layers = scanner.n_layers

    math_probe = MathProbe(f"{args.data_dir}/math_probes.json")
    eq_probe = EQProbe(f"{args.data_dir}/eq_probes.json")
    ml_probe = (
        MultilingualProbe(f"{args.data_dir}/multilingual_probes.json")
        if args.multilingual
        else None
    )

    # Parse optional range restrictions
    i_range = None
    j_range = None
    if args.i_range:
        parts = args.i_range.split(",")
        i_range = (int(parts[0]), int(parts[1]))
    if args.j_range:
        parts = args.j_range.split(",")
        j_range = (int(parts[0]), int(parts[1]))

    # Load already-completed configs for resume support
    completed: set[tuple[int, int]] = set()
    fieldnames = [
        "i", "j", "math_score", "eq_score", "multilingual_score",
        "combined_score", "n_dup_layers", "path_length", "total_params",
        "elapsed_sec",
    ]
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((int(row["i"]), int(row["j"])))
        logger.info("Resuming — %d configs already completed", len(completed))
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # ------------------------------------------------------------------
    # Baseline evaluation: run the unmodified model through ALL probes
    # exactly the same way each (i, j) perturbation is evaluated.
    # This catches tokenization / generation bugs before burning GPU hours.
    # ------------------------------------------------------------------
    if (0, 0) not in completed:
        logger.info("=" * 60)
        logger.info("BASELINE EVALUATION — testing unmodified model with all probes")
        logger.info("=" * 60)
        try:
            t0 = time.time()
            baseline_math = math_probe.run(scanner)
            baseline_eq = eq_probe.run(scanner)
            baseline_ml = ml_probe.run(scanner) if ml_probe else None
            elapsed = time.time() - t0

            bl_math = baseline_math.mean_score
            bl_eq = baseline_eq.mean_score
            bl_ml = baseline_ml.mean_score if baseline_ml else 0.0
            bl_combined = bl_math + bl_eq + bl_ml

            writer.writerow({
                "i": 0, "j": 0,
                "math_score": f"{bl_math:.6f}",
                "eq_score": f"{bl_eq:.6f}",
                "multilingual_score": f"{bl_ml:.6f}",
                "combined_score": f"{bl_combined:.6f}",
                "n_dup_layers": 0,
                "path_length": n_layers,
                "total_params": scanner.effective_params(n_layers),
                "elapsed_sec": f"{elapsed:.1f}",
            })
            csv_file.flush()
            completed.add((0, 0))

            logger.info("=" * 60)
            logger.info("BASELINE RESULTS (unmodified model)")
            logger.info("  Layers:       %d (path length: %d)", n_layers, n_layers)
            logger.info("  Parameters:   %s", f"{scanner.effective_params(n_layers):,}")
            logger.info("  Math:         %.4f", bl_math)
            logger.info("  EQ:           %.4f", bl_eq)
            if baseline_ml:
                logger.info("  Multilingual: %.4f", bl_ml)
            logger.info("  Combined:     %.4f", bl_combined)
            logger.info("  Elapsed:      %.1fs", elapsed)
            logger.info("=" * 60)
        except Exception:
            import traceback
            logger.error(
                "BASELINE FAILED — probes do not work on the unmodified model:\n%s",
                traceback.format_exc(),
            )
            logger.error("Fix the error above before running the sweep.")
            csv_file.close()
            return
    else:
        logger.info("Baseline (0, 0) already completed — skipping")

    configs = list(iter_configs(n_layers, i_range, j_range))
    total = len(configs)
    done = 0

    logger.info(
        "Starting Lobotomy sweep: %d configurations (%d already done)",
        total, len(completed),
    )

    try:
        for i, j in configs:
            if (i, j) in completed:
                done += 1
                continue

            n_dup = j - i if not (i == 0 and j == 0) else 0
            logger.info(
                "[%d/%d] Config (%d, %d) — %d duplicated layers",
                done + 1, total, i, j, n_dup,
            )

            t0 = time.time()
            with scanner.config(i, j):
                math_report = math_probe.run(scanner)
                eq_report = eq_probe.run(scanner)
                ml_report = ml_probe.run(scanner) if ml_probe else None
            elapsed = time.time() - t0

            math_score = math_report.mean_score
            eq_score = eq_report.mean_score
            ml_score = ml_report.mean_score if ml_report else 0.0
            combined = math_score + eq_score + ml_score

            path_len = n_layers + n_dup
            eff_params = scanner.effective_params(path_len)
            writer.writerow({
                "i": i,
                "j": j,
                "math_score": f"{math_score:.6f}",
                "eq_score": f"{eq_score:.6f}",
                "multilingual_score": f"{ml_score:.6f}",
                "combined_score": f"{combined:.6f}",
                "n_dup_layers": n_dup,
                "path_length": path_len,
                "total_params": eff_params,
                "elapsed_sec": f"{elapsed:.1f}",
            })
            csv_file.flush()

            done += 1
            logger.info(
                "  → math=%.4f, eq=%.4f, ml=%.4f, combined=%.4f "
                "| layers=%d, params=%s (%.1fs)",
                math_score, eq_score, ml_score, combined,
                path_len, f"{eff_params:,}", elapsed,
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted — progress saved to %s", csv_path)
    finally:
        csv_file.close()

    logger.info("Sweep complete. Results saved to %s", csv_path)
    _print_best(csv_path)


# ======================================================================
# BAYESIAN
# ======================================================================

def cmd_bayesian(args: argparse.Namespace) -> None:
    """Run Bayesian optimization to find the optimal configuration."""
    from lobotomy.optimize import bayesian_optimize
    from lobotomy.scanner import LobotomyScanner

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scanner = LobotomyScanner(
        args.model,
        load_in_4bit=args.load_4bit,
        torch_dtype=args.dtype,
    )

    result = bayesian_optimize(
        scanner,
        n_calls=args.n_calls,
        include_multilingual=args.multilingual,
        data_dir=args.data_dir,
    )

    # Save evaluations to CSV
    csv_path = output_dir / "bayesian.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "combined_delta"])
        for i, j, score in result.all_evaluations:
            writer.writerow([i, j, f"{score:.6f}"])

    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION RESULT")
    logger.info("  Best config:  (%d, %d)", result.best_i, result.best_j)
    logger.info("  Best score:   %.4f", result.best_score)
    logger.info("  Dup layers:   %d", result.best_j - result.best_i)
    logger.info("  Evaluations:  %d", result.n_calls)
    logger.info("  Results:      %s", csv_path)
    logger.info("=" * 60)
    logger.info(
        "Recommended next step:\n"
        "  python run_lobotomy.py sweep --model %s "
        "--i-range %d,%d --j-range %d,%d --output %s",
        args.model,
        max(0, result.best_i - 10), min(scanner.n_layers, result.best_i + 10),
        max(1, result.best_j - 10), min(scanner.n_layers + 1, result.best_j + 10),
        args.output,
    )


# ======================================================================
# HEATMAP
# ======================================================================

def cmd_heatmap(args: argparse.Namespace) -> None:
    """Generate heatmaps and skyline plots from sweep results."""
    from lobotomy.heatmap import generate_all_plots

    generate_all_plots(args.input, args.output, args.n_layers)
    logger.info("Plots saved to %s", args.output)


# ======================================================================
# APPLY
# ======================================================================

def cmd_apply(args: argparse.Namespace) -> None:
    """Apply the optimal Lobotomy configuration and save the model."""
    from lobotomy.surgeon import save_lobotomized_model

    i_str, j_str = args.config.split(",")
    i, j = int(i_str.strip()), int(j_str.strip())

    save_lobotomized_model(
        args.model, i, j, args.output,
        load_in_4bit=args.load_4bit,
        torch_dtype=args.dtype,
    )
    logger.info("Lobotomized model saved to %s", args.output)


# ======================================================================
# HELPERS
# ======================================================================

def _print_best(csv_path: str | Path) -> None:
    """Print the best configuration from a sweep CSV."""
    best = None
    best_score = -float("inf")
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["combined_score"])
            if score > best_score and not (row["i"] == "0" and row["j"] == "0"):
                best_score = score
                best = row
    if best is None:
        return

    baseline_score = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["i"] == "0" and row["j"] == "0":
                baseline_score = float(row["combined_score"])
                break

    logger.info("=" * 60)
    logger.info("BEST LOBOTOMY CONFIGURATION")
    logger.info("  Config:     (%s, %s)", best["i"], best["j"])
    logger.info("  Dup layers: %s", best["n_dup_layers"])
    logger.info("  Combined:   %.4f (baseline: %.4f, Δ: %+.4f)",
                best_score, baseline_score, best_score - baseline_score)
    logger.info("  Math:       %s", best["math_score"])
    logger.info("  EQ:         %s", best["eq_score"])
    if float(best.get("multilingual_score", 0)):
        logger.info("  Multilingual: %s", best["multilingual_score"])
    logger.info("=" * 60)
    logger.info(
        "To apply this configuration:\n"
        "  python run_lobotomy.py apply --model MODEL --config %s,%s --output OUTPUT_DIR",
        best["i"], best["j"],
    )


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_lobotomy",
        description="LLM Lobotomy — layer duplication scanner and surgeon",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- sweep --
    p_sweep = subparsers.add_parser("sweep", help="Exhaustive configuration sweep")
    p_sweep.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_sweep.add_argument("--output", default="results", help="Output directory")
    p_sweep.add_argument("--data-dir", default="data", help="Probe data directory")
    p_sweep.add_argument("--4bit", dest="load_4bit", action="store_true", help="Load model in 4-bit")
    p_sweep.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p_sweep.add_argument("--i-range", default=None, help="Restrict start range, e.g. '15,40'")
    p_sweep.add_argument("--j-range", default=None, help="Restrict end range, e.g. '25,50'")
    p_sweep.add_argument("--multilingual", action="store_true", help="Include multilingual probes")
    p_sweep.set_defaults(func=cmd_sweep)

    # -- bayesian --
    p_bayes = subparsers.add_parser("bayesian", help="Bayesian optimization sweep")
    p_bayes.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_bayes.add_argument("--output", default="results", help="Output directory")
    p_bayes.add_argument("--data-dir", default="data", help="Probe data directory")
    p_bayes.add_argument("--4bit", dest="load_4bit", action="store_true", help="Load model in 4-bit")
    p_bayes.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p_bayes.add_argument("--n-calls", type=int, default=200, help="Number of evaluations")
    p_bayes.add_argument("--multilingual", action="store_true", help="Include multilingual probes")
    p_bayes.set_defaults(func=cmd_bayesian)

    # -- heatmap --
    p_heat = subparsers.add_parser("heatmap", help="Generate heatmaps from sweep results")
    p_heat.add_argument("--input", required=True, help="Path to sweep CSV")
    p_heat.add_argument("--output", default="results/heatmaps", help="Output directory for plots")
    p_heat.add_argument("--n-layers", type=int, required=True, help="Number of layers in model")
    p_heat.set_defaults(func=cmd_heatmap)

    # -- apply --
    p_apply = subparsers.add_parser("apply", help="Apply Lobotomy and save model")
    p_apply.add_argument("--model", required=True, help="HuggingFace model name or path")
    p_apply.add_argument("--config", required=True, help="Optimal (i,j), e.g. '25,32'")
    p_apply.add_argument("--output", required=True, help="Output directory for saved model")
    p_apply.add_argument("--4bit", dest="load_4bit", action="store_true")
    p_apply.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p_apply.set_defaults(func=cmd_apply)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
