"""Bayesian optimization for finding optimal Lobotomy configuration.

Uses scikit-optimize's Gaussian-Process-based minimiser to find the
best (i, j) pair without exhaustive sweep.  Typically converges in
~100-200 evaluations instead of the full 1486 for a 54-layer model.

Based on:
    Snoek, J., Larochelle, H., & Adams, R. P. (2012).
    Practical Bayesian Optimization of Machine Learning Hyperparameters.
    Advances in Neural Information Processing Systems 25 (NeurIPS 2012).
    https://arxiv.org/abs/1206.2944
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from skopt import gp_minimize
from skopt.space import Integer

from .probes import EQProbe, MathProbe, MultilingualProbe
from .scanner import LobotomyScanner

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Result of Bayesian optimization."""

    best_i: int
    best_j: int
    best_score: float
    n_calls: int
    all_evaluations: list[tuple[int, int, float]]


def bayesian_optimize(
    scanner: LobotomyScanner,
    *,
    n_calls: int = 200,
    random_state: int = 42,
    include_multilingual: bool = False,
    data_dir: str = "data",
    math_weight: float = 1.0,
    eq_weight: float = 1.0,
    multilingual_weight: float = 0.5,
) -> BayesianResult:
    """Find the optimal (i, j) configuration via Bayesian optimization.

    Args:
        scanner:              Loaded LobotomyScanner.
        n_calls:              Total evaluations (including initial random).
        random_state:         RNG seed for reproducibility.
        include_multilingual: Whether to include multilingual probes.
        data_dir:             Path to probe data directory.
        math_weight:          Weight for math probe in combined score.
        eq_weight:            Weight for EQ probe in combined score.
        multilingual_weight:  Weight for multilingual probe in combined score.

    Returns:
        BayesianResult with the optimal configuration and trace.
    """
    n_layers = scanner.n_layers
    math_probe = MathProbe(f"{data_dir}/math_probes.json")
    eq_probe = EQProbe(f"{data_dir}/eq_probes.json")
    ml_probe = (
        MultilingualProbe(f"{data_dir}/multilingual_probes.json")
        if include_multilingual
        else None
    )

    all_evals: list[tuple[int, int, float]] = []

    # Cache the baseline scores
    logger.info("Evaluating baseline (0, 0)...")
    baseline_math = math_probe.run(scanner).mean_score
    baseline_eq = eq_probe.run(scanner).mean_score
    baseline_ml = ml_probe.run(scanner).mean_score if ml_probe else 0.0
    logger.info(
        "Baseline scores — math: %.4f, eq: %.4f, ml: %.4f",
        baseline_math, baseline_eq, baseline_ml,
    )

    eval_count = 0

    def objective(params: list[int]) -> float:
        nonlocal eval_count
        i, j = params

        if i >= j or j > n_layers or i < 0:
            return 0.0

        eval_count += 1
        logger.info(
            "[%d/%d] Evaluating config (%d, %d) — %d duplicated layers",
            eval_count, n_calls, i, j, j - i,
        )

        with scanner.config(i, j):
            math_score = math_probe.run(scanner).mean_score
            eq_score = eq_probe.run(scanner).mean_score
            ml_score = ml_probe.run(scanner).mean_score if ml_probe else 0.0

        math_delta = math_score - baseline_math
        eq_delta = eq_score - baseline_eq
        ml_delta = ml_score - baseline_ml

        combined = (
            math_weight * math_delta
            + eq_weight * eq_delta
            + multilingual_weight * ml_delta
        )

        all_evals.append((i, j, combined))
        logger.info(
            "  → math Δ=%.4f, eq Δ=%.4f, ml Δ=%.4f, combined=%.4f",
            math_delta, eq_delta, ml_delta, combined,
        )
        return -combined  # gp_minimize minimises, we want to maximise

    space = [
        Integer(0, n_layers - 1, name="i"),
        Integer(1, n_layers, name="j"),
    ]

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=min(30, n_calls // 3),
        random_state=random_state,
        verbose=False,
    )

    best_i, best_j = result.x
    best_score = -result.fun

    logger.info(
        "Bayesian optimization complete. Best config: (%d, %d), score: %.4f",
        best_i, best_j, best_score,
    )

    return BayesianResult(
        best_i=best_i,
        best_j=best_j,
        best_score=best_score,
        n_calls=eval_count,
        all_evaluations=all_evals,
    )
