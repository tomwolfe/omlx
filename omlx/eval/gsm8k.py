# SPDX-License-Identifier: Apache-2.0
"""GSM8K benchmark.

Tests mathematical reasoning using grade school math word problems.
5-shot chain-of-thought prompting, answer extraction from "#### N" pattern.
Dataset bundled from openai/gsm8k on HuggingFace.
"""

import json
import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl
from .utils import extract_numeric_answer

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Standard 5-shot examples for GSM8K
FEW_SHOT_EXAMPLES = json.loads((DATA_DIR / "gsm8k_few_shot.json").read_text())


def _normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K: 5-shot chain-of-thought math reasoning."""

    name = "gsm8k"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load GSM8K from bundled data."""
        items = load_jsonl(DATA_DIR / "gsm8k_test.jsonl")

        normalized = []
        for i, item in enumerate(items):
            answer_text = item.get("answer", "")
            numeric = extract_numeric_answer(answer_text)
            normalized.append(
                {
                    "id": str(i),
                    "question": item.get("question", ""),
                    "answer_text": answer_text,
                    "answer": numeric,
                }
            )

        logger.info(f"GSM8K: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 512

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format with 5-shot chain-of-thought examples."""
        parts = [
            "Solve the following math problem step by step. "
            "End your answer with #### followed by the final numeric answer.\n"
        ]

        for ex in FEW_SHOT_EXAMPLES:
            parts.append(f"Question: {ex['question']}")
            parts.append(f"Answer: {ex['answer']}\n")

        parts.append(f"Question: {item['question']}")
        parts.append("Answer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return extract_numeric_answer(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        if not predicted:
            return False
        return _normalize_number(predicted) == _normalize_number(item["answer"])

    def get_category(self, item: dict) -> str | None:
        return None
