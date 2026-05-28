# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for the evaluation framework.

This module consolidates common patterns found across benchmark implementations:
- Multiple-choice answer extraction from model responses
- Code block extraction from model responses
- Numeric answer extraction for math benchmarks
- Index-to-letter conversion for MCQs
- Common prompt formatting for MCQs
- Common data normalization for benchmark datasets
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def extract_mc_answer(response: str, valid_letters: list[str]) -> str:
    """Extract multiple choice answer from model response.

    Strategy:
    1. Look for explicit "answer is X" / "answer: X" patterns (last match)
    2. Fall back to last valid letter in response
    3. Case-insensitive

    Args:
        response: Model response text to extract answer from.
        valid_letters: List of valid letter options (e.g., ["A", "B", "C", "D"]).

    Returns:
        Extracted answer letter, or empty string if no match found.
    """
    response_upper = response.strip().upper()
    pattern_letters = "".join(valid_letters)

    # 1. Look for "answer is X", "answer: X" patterns — use LAST match
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


def extract_last_code_block(response: str) -> str:
    """Extract the LAST code block from model response.

    Uses last match to avoid picking up drafts/examples.
    Falls back to line-by-line detection if no code blocks found.

    Args:
        response: Model response text containing code.

    Returns:
        Extracted code block, or the full response if no code found.
    """
    response = response.strip()

    # Find ALL python code blocks, use LAST
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Generic code blocks
    blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Line-by-line fallback
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if not in_code and (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("import ")
            or line.startswith("from ")
            or line.startswith("#")
        ):
            in_code = True
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else response


def extract_numeric_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K-style response.

    Looks for #### pattern first, then falls back to the last number.

    Args:
        text: Response text that may contain a numeric answer.

    Returns:
        Extracted numeric answer as string, or empty string if not found.
    """
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def index_to_letter(idx: int) -> str:
    """Convert 0-based index to letter (A, B, C, ...).

    Args:
        idx: Zero-based index (0 → "A", 1 → "B", etc.).

    Returns:
        Corresponding uppercase letter.
    """
    return chr(ord("A") + idx)


def format_mc_prompt(
    instruction: str,
    question: str,
    choices: list[str],
    labels: list[str] | None = None,
) -> list[dict[str, str]]:
    """Format a multiple-choice question as a chat prompt.

    Common pattern used across ARC, TruthfulQA, SafetyBench, etc.

    Args:
        instruction: Prompt instruction (e.g., "Answer with just the letter.").
        question: The question text.
        choices: List of choice texts.
        labels: Optional list of label letters (A, B, C, D...).
            Defaults to A, B, C, D, E, ... based on choices length.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if labels is None:
        labels = [index_to_letter(i) for i in range(len(choices))]

    parts: list[str] = [
        f"{instruction}\n",
        f"{question}\n",
    ]
    for label, choice in zip(labels, choices):
        parts.append(f"{label}. {choice}")

    parts.append("\nAnswer:")

    return [{"role": "user", "content": "\n".join(parts)}]


def format_mc_prompt_with_context(
    instruction: str,
    context: str,
    question: str,
    choices: list[str],
    labels: list[str] | None = None,
) -> list[dict[str, str]]:
    """Format a multiple-choice question with context as a chat prompt.

    Common pattern used across BBQ and similar benchmarks.

    Args:
        instruction: Prompt instruction.
        context: Context text preceding the question.
        question: The question text.
        choices: List of choice texts.
        labels: Optional list of label letters.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    if labels is None:
        labels = [index_to_letter(i) for i in range(len(choices))]

    parts: list[str] = [
        f"{instruction}\n",
        f"{context}\n",
        f"{question}\n",
    ]
    for label, choice in zip(labels, choices):
        parts.append(f"{label}. {choice}")

    parts.append("\nAnswer:")

    return [{"role": "user", "content": "\n".join(parts)}]


def normalize_mc_item(
    item: dict,
    id_key: str = "id",
    answer_key: str = "answer",
    question_key: str = "question",
    choices_key: str = "choices",
    labels_key: str = "labels",
) -> dict:
    """Normalize a raw dataset item into a standard MCQ format.

    Common pattern used across ARC, TruthfulQA, SafetyBench, etc.

    Args:
        item: Raw dataset item with choices/labels/answer fields.
        id_key: Key for the item ID (default: "id").
        answer_key: Key for the answer value (default: "answer").
        question_key: Key for the question text (default: "question").
        choices_key: Key for choices list (default: "choices").
        labels_key: Key for labels list (default: "labels").

    Returns:
        Normalized dict with consistent keys, or None if invalid.
    """
    choices = item.get(choices_key, [])
    labels = item.get(labels_key, [])
    if not choices or not labels:
        return None

    normalized: dict[str, Any] = {
        "id": item.get(id_key, ""),
        "question": item.get(question_key, ""),
        "choices": choices,
        "labels": labels,
        "answer": item.get(answer_key, ""),
    }
    return normalized


def normalize_indexed_item(
    item: dict,
    index: int,
    id_key: str = "id",
    answer_key: str = "answer",
    question_key: str = "question",
    choices_key: str = "choices",
    labels_key: str = "labels",
) -> dict:
    """Normalize a raw dataset item with auto-generated index-based ID.

    Common pattern used across GSM8K, MMLU, and similar benchmarks.

    Args:
        item: Raw dataset item.
        index: Index used for auto-generated ID when id_key is missing.
        id_key: Key for the item ID (default: "id").
        answer_key: Key for the answer value (default: "answer").
        question_key: Key for the question text (default: "question").
        choices_key: Key for choices list (default: "choices").
        labels_key: Key for labels list (default: "labels").

    Returns:
        Normalized dict with consistent keys.
    """
    return {
        "id": item.get(id_key, str(index)),
        "question": item.get(question_key, ""),
        "choices": item.get(choices_key, []),
        "labels": item.get(labels_key, []),
        "answer": item.get(answer_key, ""),
    }


def normalize_few_shot_item(
    item: dict,
    answer_key: str = "answer",
    question_key: str = "question",
    choices_key: str = "choices",
) -> dict:
    """Normalize a few-shot example item.

    Common pattern used across MMLU, CMMLU, KMMLU benchmarks.

    Args:
        item: Raw dataset item.
        answer_key: Key for the answer value (default: "answer").
        question_key: Key for the question text (default: "question").
        choices_key: Key for choices list (default: "choices").

    Returns:
            Normalized dict with consistent keys.
    """
    return {
        "question": item.get(question_key, ""),
        "choices": item.get(choices_key, []),
        "answer": item.get(answer_key, "A"),
    }
