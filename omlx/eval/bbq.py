# SPDX-License-Identifier: Apache-2.0
"""BBQ (Bias Benchmark for QA) benchmark.

Tests social bias across 11 categories including age, gender,
race, religion, disability, nationality, and more.
3-choice multiple choice format.
Dataset bundled from lighteval/bbq_helm on HuggingFace.
"""

import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl
from .utils import extract_mc_answer, format_mc_prompt_with_context

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class BBQBenchmark(BaseBenchmark):
    """BBQ: 0-shot bias detection with 3 choices."""

    name = "bbq"
    quick_size = 300

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load BBQ from bundled data."""
        items = load_jsonl(DATA_DIR / "bbq_test.jsonl")

        normalized: list[dict] = []
        for item in items:
            choices = item.get("choices", [])
            labels = item.get("labels", [])
            if not choices or not labels:
                continue
            normalized.append(
                {
                    "id": item.get("id", ""),
                    "context": item["context"],
                    "question": item["question"],
                    "choices": choices,
                    "labels": labels,
                    "answer": item["answer"],
                    "category": item.get("category", "unknown"),
                }
            )

        logger.info(f"BBQ: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as context + question with 3 choices."""
        return format_mc_prompt_with_context(
            instruction="Read the context and answer the question. Answer with just the letter.",
            context=item["context"],
            question=item["question"],
            choices=item["choices"],
            labels=item.get("labels"),
        )

    def extract_answer(self, response: str, item: dict) -> str:
        return extract_mc_answer(response, item.get("labels", ["A", "B", "C"]))

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item["answer"]

    def get_question_text(self, item: dict) -> str:
        return f"{item.get('context', '')} {item.get('question', '')}"

    def get_category(self, item: dict) -> str | None:
        return item.get("category")
