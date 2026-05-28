# SPDX-License-Identifier: Apache-2.0
"""HellaSwag benchmark.

Tests commonsense reasoning by choosing the most plausible
continuation of a scenario. 0-shot multiple choice.
Dataset bundled from Rowan/hellaswag on HuggingFace.
"""

import logging
from pathlib import Path

from .base import BaseBenchmark
from .datasets import deterministic_sample, load_jsonl
from .utils import extract_mc_answer, index_to_letter

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class HellaSwagBenchmark(BaseBenchmark):
    """HellaSwag: 0-shot commonsense reasoning with 4 choices."""

    name = "hellaswag"
    quick_size = 200

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HellaSwag from bundled data."""
        items = load_jsonl(DATA_DIR / "hellaswag_val.jsonl")

        normalized = []
        for item in items:
            normalized.append(
                {
                    "id": item.get("ind", ""),
                    "context": item.get("ctx", ""),
                    "endings": item.get("endings", []),
                    "answer": int(item.get("label", "0")),
                    "activity_label": item.get("activity_label", ""),
                }
            )

        logger.info(f"HellaSwag: loaded {len(normalized)} questions")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format context + 4 endings as multiple choice."""
        context = item["context"]
        endings = item["endings"]

        parts = [
            "Choose the most plausible continuation. "
            "Answer with just the letter (A, B, C, or D).\n",
            f"Context: {context}\n",
        ]
        for i, ending in enumerate(endings[:4]):
            parts.append(f"{index_to_letter(i)}. {ending}")

        parts.append("\nAnswer:")

        return [{"role": "user", "content": "\n".join(parts)}]

    def extract_answer(self, response: str, item: dict) -> str:
        return extract_mc_answer(response, ["A", "B", "C", "D"])

    def check_answer(self, predicted: str, item: dict) -> bool:
        expected_letter = index_to_letter(item["answer"])
        return predicted == expected_letter

    def get_category(self, item: dict) -> str | None:
        return item.get("activity_label")
