# SPDX-License-Identifier: Apache-2.0
"""MathQA benchmark.

Tests quantitative reasoning with 5-choice math problems
sourced from standardized tests (GRE, GMAT, etc.).
Dataset bundled from math_qa on HuggingFace.
"""

import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl
from .utils import format_mc_prompt, normalize_mc_item

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class MathQABenchmark(BaseBenchmark):
    """MathQA: 0-shot quantitative reasoning with 5 choices."""

    name = "mathqa"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MathQA from bundled data."""
        items = load_jsonl(DATA_DIR / "mathqa_test.jsonl")

        normalized: list[dict] = []
        for item in items:
            normalized_item = normalize_mc_item(item)
            if normalized_item is not None:
                normalized_item["category"] = item.get("category", "general")
                normalized.append(normalized_item)

        logger.info(f"MathQA: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options."""
        return format_mc_prompt(
            instruction="Solve the following math problem. Answer with just the letter.",
            question=item["question"],
            choices=item["choices"],
            labels=item.get("labels"),
        )

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(
            response, item.get("labels", ["A", "B", "C", "D", "E"])
        )

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> str | None:
        return item.get("category")
