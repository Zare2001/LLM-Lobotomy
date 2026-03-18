"""Evaluation probes for the Lobotomy scanner.

Three probe types:
  - MathProbe:         hard math guessing (no chain-of-thought)
  - EQProbe:           emotional quotient scenarios (EQ-Bench style)
  - MultilingualProbe: math probes translated into multiple languages
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .scanner import LobotomyScanner
from .scoring import (
    calculate_eq_score,
    calculate_math_score,
    parse_eq_scores_from_text,
    parse_number_from_text,
)

logger = logging.getLogger(__name__)

MATH_SYSTEM_PROMPT = (
    "You are a calculator. Answer with ONLY the number, no explanation, "
    "no units, no punctuation. Just the integer."
)

EQ_SYSTEM_PROMPT = (
    "You are an emotional intelligence expert. Read the scenario and rate "
    "how strongly the described person would feel each listed emotion on a "
    "scale of 0 to 100. Respond with ONLY the numbers separated by commas, "
    "in the exact order the emotions are listed. No words, no explanation."
)

MULTILINGUAL_SYSTEM_PROMPTS = {
    "fr": (
        "Tu es une calculatrice. Réponds UNIQUEMENT avec le nombre, "
        "sans explication. Juste le nombre entier."
    ),
    "de": (
        "Du bist ein Taschenrechner. Antworte NUR mit der Zahl, "
        "ohne Erklärung. Nur die ganze Zahl."
    ),
    "es": (
        "Eres una calculadora. Responde SOLO con el número, "
        "sin explicación. Solo el número entero."
    ),
    "pt": (
        "Você é uma calculadora. Responda APENAS com o número, "
        "sem explicação. Apenas o número inteiro."
    ),
    "nl": (
        "Je bent een rekenmachine. Antwoord ALLEEN met het getal, "
        "zonder uitleg. Alleen het gehele getal."
    ),
}


@dataclass
class ProbeResult:
    """Result from running a single probe item."""

    question: str
    expected: float | list[float]
    raw_output: str
    parsed_output: float | list[float] | None
    score: float


@dataclass
class ProbeReport:
    """Aggregated report from running all items in a probe."""

    probe_type: str
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


# --------------------------------------------------------------------------
# Math Probe
# --------------------------------------------------------------------------

class MathProbe:
    """Hard math questions scored with partial credit."""

    def __init__(self, data_path: str | Path = "data/math_probes.json"):
        with open(data_path) as f:
            self.items: list[dict] = json.load(f)
        logger.info("MathProbe loaded %d questions", len(self.items))

    def run(self, scanner: LobotomyScanner) -> ProbeReport:
        report = ProbeReport(probe_type="math")
        for item in self.items:
            messages = [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            try:
                raw = scanner.generate(messages, max_new_tokens=48)
            except Exception:
                logger.error("Generation failed for: %s\n%s", item["question"], traceback.format_exc())
                raw = ""

            parsed = parse_number_from_text(raw)
            score = calculate_math_score(item["answer"], parsed) if parsed is not None else 0.0

            report.results.append(
                ProbeResult(
                    question=item["question"],
                    expected=item["answer"],
                    raw_output=raw,
                    parsed_output=parsed,
                    score=score,
                )
            )
        return report


# --------------------------------------------------------------------------
# EQ Probe
# --------------------------------------------------------------------------

class EQProbe:
    """Emotional quotient scenarios scored by inverse MAE."""

    def __init__(self, data_path: str | Path = "data/eq_probes.json"):
        with open(data_path) as f:
            self.items: list[dict] = json.load(f)
        logger.info("EQProbe loaded %d scenarios", len(self.items))

    def run(self, scanner: LobotomyScanner) -> ProbeReport:
        report = ProbeReport(probe_type="eq")
        for item in self.items:
            emotion_list = ", ".join(item["emotions"])
            user_msg = (
                f"Scenario: {item['scenario']}\n\n"
                f"Rate these emotions (0-100): {emotion_list}"
            )
            messages = [
                {"role": "system", "content": EQ_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            try:
                raw = scanner.generate(messages, max_new_tokens=48)
            except Exception:
                logger.error("Generation failed for EQ scenario\n%s", traceback.format_exc())
                raw = ""

            parsed = parse_eq_scores_from_text(raw, len(item["emotions"]))
            score = calculate_eq_score(item["expected"], parsed) if parsed is not None else 0.0

            report.results.append(
                ProbeResult(
                    question=item["scenario"][:80] + "...",
                    expected=item["expected"],
                    raw_output=raw,
                    parsed_output=parsed,
                    score=score,
                )
            )
        return report


# --------------------------------------------------------------------------
# Multilingual Probe
# --------------------------------------------------------------------------

class MultilingualProbe:
    """Math probes in multiple languages for multilingual validation."""

    def __init__(self, data_path: str | Path = "data/multilingual_probes.json"):
        with open(data_path) as f:
            data = json.load(f)
        self.language_sets: list[dict] = data["probes"]
        total = sum(len(ls["questions"]) for ls in self.language_sets)
        logger.info(
            "MultilingualProbe loaded %d questions across %d languages",
            total,
            len(self.language_sets),
        )

    def run(self, scanner: LobotomyScanner) -> ProbeReport:
        report = ProbeReport(probe_type="multilingual")
        for lang_set in self.language_sets:
            lang = lang_set["language"]
            sys_prompt = MULTILINGUAL_SYSTEM_PROMPTS.get(lang, MATH_SYSTEM_PROMPT)
            for item in lang_set["questions"]:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": item["question"]},
                ]
                try:
                    raw = scanner.generate(messages, max_new_tokens=48)
                except Exception:
                    logger.error(
                        "Generation failed for [%s]: %s\n%s", lang, item["question"], traceback.format_exc()
                    )
                    raw = ""

                parsed = parse_number_from_text(raw)
                score = (
                    calculate_math_score(item["answer"], parsed)
                    if parsed is not None
                    else 0.0
                )

                report.results.append(
                    ProbeResult(
                        question=f"[{lang}] {item['question']}",
                        expected=item["answer"],
                        raw_output=raw,
                        parsed_output=parsed,
                        score=score,
                    )
                )
        return report


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def run_all_probes(
    scanner: LobotomyScanner,
    *,
    include_multilingual: bool = False,
    data_dir: str | Path = "data",
) -> dict[str, ProbeReport]:
    """Run all probes and return a dict of reports keyed by probe type."""
    data_dir = Path(data_dir)
    reports: dict[str, ProbeReport] = {}

    reports["math"] = MathProbe(data_dir / "math_probes.json").run(scanner)
    reports["eq"] = EQProbe(data_dir / "eq_probes.json").run(scanner)

    if include_multilingual:
        reports["multilingual"] = MultilingualProbe(
            data_dir / "multilingual_probes.json"
        ).run(scanner)

    return reports
