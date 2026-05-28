# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro benchmark.

A harder version of MMLU with 10 answer choices instead of 4,
requiring deeper reasoning. Covers 14 academic subjects.
Dataset bundled from TIGER-Lab/MMLU-Pro on HuggingFace.
"""

import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import load_jsonl, stratified_sample
from .utils import format_mc_prompt, normalize_mc_item

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class MMLUProBenchmark(BaseBenchmark):
    """MMLU-Pro: 0-shot hard knowledge MC with 10 choices."""

    name = "mmlu_pro"
    quick_size = 300

    def get_max_tokens(self) -> int:
        return 2048

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MMLU-Pro from bundled data."""
        items = load_jsonl(DATA_DIR / "mmlu_pro_test.jsonl")

        normalized: list[dict] = []
        for item in items:
            normalized_item = normalize_mc_item(item)
            if normalized_item is not None:
                normalized_item["subject"] = item.get("subject", "general")
                normalized.append(normalized_item)

        logger.info(f"MMLU-Pro: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return stratified_sample(normalized, sample_size, "subject")

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as multiple choice with lettered options (A-J)."""
        return format_mc_prompt(
            instruction="Answer the following question. Answer with just the letter.",
            question=item["question"],
            choices=item["choices"],
            labels=item.get("labels"),
        )

    def extract_answer(self, response: str, item: dict) -> str:
        return self._extract_mc_answer(
            response,
            item.get("labels", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]),
        )

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_category(self, item: dict) -> str | None:
        return item.get("subject")
