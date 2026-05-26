# SPDX-License-Identifier: Apache-2.0
"""Multiple-choice answer extraction strategies.

Consolidates the repetitive _extract_mc_answer and prompt-building logic
found across arc.py, mathqa.py, mmlu.py, and other benchmark modules into
a single ``MultipleChoiceEvaluator`` class that subclasses can compose.
"""

from __future__ import annotations

import re
from typing import Literal


class MultipleChoiceEvaluator:
    """Unified multiple-choice answer extraction.

    Subclasses can override ``VALID_LETTERS`` or provide a custom
    ``answer_pattern`` to tailor extraction to their dataset format.
    """

    # Default valid letters for multiple-choice questions
    VALID_LETTERS = ("A", "B", "C", "D")

    # Regex patterns for explicit answer extraction
    ANSWER_PATTERNS = (
        r"answer\s+(?:is|:)\s*",
        r"^\s*answer\s+",
    )

    @classmethod
    def extract_answer(
        cls,
        response: str,
        valid_letters: tuple[str, ...] | list[str] = VALID_LETTERS,
    ) -> str:
        """Extract multiple choice answer from model response.

        Strategy:
        1. Look for explicit "answer is X" / "answer: X" patterns (last match)
        2. Fall back to last valid letter in response
        3. Case-insensitive

        Args:
            response: Model response text.
            valid_letters: Valid answer letters (e.g. ("A", "B", "C", "D")).

        Returns:
            Extracted answer letter, or empty string if no match.
        """
        response_upper = response.strip().upper()
        pattern_letters = "".join(valid_letters)

        # 1. Look for "answer is X", "answer: X", "answer X" patterns
        answer_patterns = re.findall(
            r"(?:answer\s*(?:is|:)\s*)([" + pattern_letters + r"])\b",
            response_upper,
        )
        if answer_patterns:
            return answer_patterns[-1]

        # 2. Fall back to last valid letter with word boundary
        all_matches = re.findall(
            r"\b([" + pattern_letters + r"])\b",
            response_upper,
        )
        if all_matches:
            return all_matches[-1]

        # 3. Check first character
        if response.strip() and response.strip()[0].upper() in valid_letters:
            return response.strip()[0].upper()

        return ""

    @classmethod
    def format_choices(
        cls,
        labels: tuple[str, ...] | list[str],
        choices: tuple[str, ...] | list[str],
    ) -> str:
        """Format multiple-choice options as a readable string.

        Args:
            labels: Option labels (e.g. ("A", "B", "C", "D")).
            choices: Option text content.

        Returns:
            Formatted choices string (e.g. "A. Option 1\\nB. Option 2\\n...").
        """
        parts: list[str] = []
        for label, choice in zip(labels, choices):
            parts.append(f"{label}. {choice}")
        return "\n".join(parts)

    @classmethod
    def build_mc_prompt(
        cls,
        question: str,
        labels: tuple[str, ...] | list[str],
        choices: tuple[str, ...] | list[str],
        instruction: str = "Answer the following question. Answer with just the letter.",
    ) -> str:
        """Build a complete multiple-choice prompt.

        Args:
            question: The question text.
            labels: Option labels (A, B, C, D, ...).
            choices: Option text content.
            instruction: Instruction prefix.

        Returns:
            Complete prompt string ready for model input.
        """
        parts: list[str] = [
            f"{instruction}\n",
            f"Question: {question}\n",
        ]
        parts.append(cls.format_choices(labels, choices))
        parts.append("\nAnswer:")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Benchmark-specific evaluators (one per module)
# ---------------------------------------------------------------------------


class ARCChallengeEvaluator(MultipleChoiceEvaluator):
    """ARC-Challenge multiple-choice evaluator."""

    VALID_LETTERS = ("A", "B", "C", "D", "E")


class MathQAEvaluator(MultipleChoiceEvaluator):
    """MathQA multiple-choice evaluator (5 options)."""

    VALID_LETTERS = ("A", "B", "C", "D", "E")


class MMLUEvaluator(MultipleChoiceEvaluator):
    """MMLU multiple-choice evaluator (4 options)."""

    VALID_LETTERS = ("A", "B", "C", "D")


class WinograndeEvaluator(MultipleChoiceEvaluator):
    """Winogrande multiple-choice evaluator (2 options)."""

    VALID_LETTERS = ("A", "B")


class GSM8KEvaluator(MultipleChoiceEvaluator):
    """GSM8K evaluator — numeric answers, not multiple choice."""

    VALID_LETTERS = ()  # No letters — expects numeric extraction


class BBQEvaluator(MultipleChoiceEvaluator):
    """BBQ bias multiple-choice evaluator."""

    VALID_LETTERS = ("A", "B", "C", "D")


class HellaSwagEvaluator(MultipleChoiceEvaluator):
    """HellaSwag multiple-choice evaluator."""

    VALID_LETTERS = ("A", "B", "C", "D")
