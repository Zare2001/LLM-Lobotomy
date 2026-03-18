#!/usr/bin/env python3
"""HPC batch runner — processes a subset of Lobotomy configurations.

Designed for Slurm array jobs: each task gets a unique (task_id, n_tasks)
pair and processes only its share of the 1486 configurations.

The model is loaded ONCE per task, then all assigned configs are evaluated
sequentially. Results are written to a per-task CSV file.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobotomy.probes import EQProbe, MathProbe, MultilingualProbe
from lobotomy.scanner import LobotomyScanner, iter_configs

logger = logging.getLogger("lobotomy.hpc")


def get_task_configs(
    n_layers: int, task_id: int, n_tasks: int
) -> list[tuple[int, int]]:
    """Split all configs across n_tasks and return this task's share."""
    all_configs = list(iter_configs(n_layers))
    chunk_size = len(all_configs) // n_tasks
    remainder = len(all_configs) % n_tasks

    start = task_id * chunk_size + min(task_id, remainder)
    end = start + chunk_size + (1 if task_id < remainder else 0)

    return all_configs[start:end]


def main():
    parser = argparse.ArgumentParser(description="HPC Lobotomy batch runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Output CSV path for this task")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--4bit", dest="load_4bit", action="store_true")
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    scanner = LobotomyScanner(
        args.model,
        load_in_4bit=args.load_4bit,
        torch_dtype=args.dtype,
    )

    math_probe = MathProbe(f"{args.data_dir}/math_probes.json")
    eq_probe = EQProbe(f"{args.data_dir}/eq_probes.json")
    ml_probe = (
        MultilingualProbe(f"{args.data_dir}/multilingual_probes.json")
        if args.multilingual
        else None
    )

    configs = get_task_configs(scanner.n_layers, args.task_id, args.n_tasks)
    logger.info(
        "Task %d/%d: processing %d configs (out of %d total)",
        args.task_id, args.n_tasks, len(configs),
        (scanner.n_layers * (scanner.n_layers + 1)) // 2 + 1,
    )

    fieldnames = [
        "i", "j", "math_score", "eq_score", "multilingual_score",
        "combined_score", "n_dup_layers", "elapsed_sec",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[int, int]] = set()
    if output_path.exists():
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add((int(row["i"]), int(row["j"])))
        logger.info("Resuming — %d configs already done", len(completed))
        csv_file = open(output_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    else:
        csv_file = open(output_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    done = 0
    try:
        for i, j in configs:
            if (i, j) in completed:
                done += 1
                continue

            n_dup = j - i if not (i == 0 and j == 0) else 0
            logger.info(
                "[%d/%d] Config (%d, %d) — %d dup layers",
                done + 1, len(configs), i, j, n_dup,
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

            writer.writerow({
                "i": i, "j": j,
                "math_score": f"{math_score:.6f}",
                "eq_score": f"{eq_score:.6f}",
                "multilingual_score": f"{ml_score:.6f}",
                "combined_score": f"{combined:.6f}",
                "n_dup_layers": n_dup,
                "elapsed_sec": f"{elapsed:.1f}",
            })
            csv_file.flush()
            done += 1

    except KeyboardInterrupt:
        logger.warning("Interrupted at config %d/%d", done, len(configs))
    finally:
        csv_file.close()

    logger.info("Task %d complete: %d configs processed", args.task_id, done)


if __name__ == "__main__":
    main()
