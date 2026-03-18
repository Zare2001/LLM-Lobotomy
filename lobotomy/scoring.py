"""Scoring functions for Lobotomy probes.

Partial-credit math scoring and logit-based EQ scoring, adapted from
the RYS methodology by David Noel Ng.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F


def calculate_math_score(actual: int | float, estimate: int | float) -> float:
    """Partial-credit scoring for numerical answers.

    Handles common LLM arithmetic quirks: dropped digits, transpositions,
    truncated outputs.  Pads the shorter string representation and penalises
    proportionally via a correction factor.

    Returns a float in [0, 1].
    """
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError, TypeError):
        return 0.0

    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")

    padding_size = max_length - min(len(actual_str), len(estimate_str))
    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)

    if max(actual_int, estimate_int) == 0:
        return 0.0

    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))


def calculate_eq_score(expected: list[float], predicted: list[float]) -> float:
    """EQ-Bench style scoring: normalised inverse MAE.

    Computes 1 - (MAE / 100) so that a perfect prediction scores 1.0 and
    maximum error scores 0.0.  Both lists must have the same length.
    """
    if len(expected) != len(predicted) or len(expected) == 0:
        return 0.0

    mae = sum(abs(e - p) for e, p in zip(expected, predicted)) / len(expected)
    return max(0.0, 1.0 - mae / 100.0)


def logit_expected_value(
    logits: torch.Tensor,
    digit_token_ids: list[int],
) -> float:
    """Compute the expected value of a score distribution over digit tokens.

    Given the raw logits at a generation position, restrict to the 10 digit
    tokens (0-9), softmax, and return Σ k·p(k).  This produces a smooth
    score (e.g. 5.4) instead of a noisy sampled integer.
    """
    digit_logits = logits[digit_token_ids]
    probs = F.softmax(digit_logits.float(), dim=0)
    values = torch.arange(len(digit_token_ids), dtype=probs.dtype, device=probs.device)
    return (probs * values).sum().item()


def logit_score_variance(
    logits: torch.Tensor,
    digit_token_ids: list[int],
) -> float:
    """Variance of the restricted digit distribution (uncertainty estimate)."""
    digit_logits = logits[digit_token_ids]
    probs = F.softmax(digit_logits.float(), dim=0)
    values = torch.arange(len(digit_token_ids), dtype=probs.dtype, device=probs.device)
    mean = (probs * values).sum()
    return (probs * (values - mean) ** 2).sum().item()


def parse_number_from_text(text: str) -> int | None:
    """Extract the first integer from a model's text output.

    Handles commas, spaces, leading/trailing junk.
    Returns None if no number is found.
    """
    cleaned = text.strip().replace(",", "").replace(" ", "")
    match = re.search(r"-?\d+", cleaned)
    if match is None:
        return None
    try:
        return int(match.group())
    except (ValueError, OverflowError):
        return None


def parse_eq_scores_from_text(text: str, n_expected: int) -> list[float] | None:
    """Parse comma-separated or newline-separated numbers from EQ probe output.

    Returns a list of floats, or None if parsing fails or the count doesn't
    match n_expected.
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < n_expected:
        return None
    try:
        return [float(x) for x in numbers[:n_expected]]
    except ValueError:
        return None
