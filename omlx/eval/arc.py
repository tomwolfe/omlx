# SPDX-License-Identifier: Apache-2.0
"""ARC-Challenge benchmark.

Tests science reasoning with multiple choice questions.
Dataset bundled from allenai/ai2_arc (Challenge split) on HuggingFace.
1,172 questions requiring scientific knowledge and deduction.
"""

import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl
from .utils import format_mc_prompt, normalize_mc_item

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class ARCChallengeBenchmark(BaseBenchmark):
    """ARC-Challenge: 0-shot science reasoning multiple choice."""

    name = "arc_challenge"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load ARC-Challenge from bundled data."""
        items = load_jsonl(DATA_DIR / "arc_challenge.jsonl")

        normalized = []
        for item in items:
            normalized_item = normalize_mc_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)

        logger.info(f"ARC-Challenge: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options."""
        return format_mc_prompt(
            instruction="Answer the following science question. Answer with just the letter.",
            question=item["question"],
            choices=item["choices"],
            labels=item.get("labels"),
        )

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(
            response, item.get("labels", ["A", "B", "C", "D"])
        )

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> str | None:
        return None
