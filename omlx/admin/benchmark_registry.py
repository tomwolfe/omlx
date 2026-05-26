# SPDX-License-Identifier: Apache-2.0
"""Benchmark registry — single source of truth for benchmark definitions.

This module provides:
- `BenchmarkRegistry`: A Pydantic model that defines available benchmarks,
  their sample sizes, and default configurations.
- `get_benchmark_registry()`: Returns the registry as a dict for API consumption.
- `validate_benchmarks()`: Validates benchmark requests against the registry.

The registry is the authoritative source for both backend validation and
frontend UI generation.  Clients fetch `/v1/benchmarks/registry` to
populate their UI dynamically rather than hardcoding benchmark names
and sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Definition of a single benchmark.

    Attributes:
        name: Unique benchmark name (e.g. "mmlu", "gsm8k").
        display_name: Human-readable name for UI display.
        default_sample_size: Default number of samples when sample_size=0.
        max_samples: Maximum allowed sample size for this benchmark.
        supports_batching: Whether the benchmark supports batched evaluation.
        category: Category for grouping (e.g. "knowledge", "reasoning", "coding").
        description: Brief description shown in UI.
    """

    name: str
    display_name: str
    default_sample_size: int = 100
    max_samples: int = 16384
    supports_batching: bool = True
    category: str = "general"
    description: str = ""


# ---------------------------------------------------------------------------
# Valid prompt lengths and batch sizes
# ---------------------------------------------------------------------------

VALID_PROMPT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]
VALID_BATCH_SIZES = [2, 4, 8]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VALID_BENCHMARKS: list[BenchmarkDefinition] = [
    BenchmarkDefinition(
        name="mmlu",
        display_name="MMLU",
        default_sample_size=100,
        max_samples=16384,
        category="knowledge",
        description="Massive Multilingual Understanding — general knowledge benchmark",
    ),
    BenchmarkDefinition(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        default_sample_size=100,
        max_samples=16384,
        category="knowledge",
        description="MMLU-Pro — extended MMLU with reasoning tasks",
    ),
    BenchmarkDefinition(
        name="kmmlu",
        display_name="K-MMLU",
        default_sample_size=50,
        max_samples=16384,
        category="knowledge",
        description="Korean MMLU — Korean language knowledge",
    ),
    BenchmarkDefinition(
        name="cmmlu",
        display_name="C-MMLU",
        default_sample_size=50,
        max_samples=16384,
        category="knowledge",
        description="Chinese MMLU — Chinese language knowledge",
    ),
    BenchmarkDefinition(
        name="jmmlu",
        display_name="J-MMLU",
        default_sample_size=50,
        max_samples=16384,
        category="knowledge",
        description="Japanese MMLU — Japanese language knowledge",
    ),
    BenchmarkDefinition(
        name="hellaswag",
        display_name="HellaSwag",
        default_sample_size=100,
        max_samples=10000,
        category="reasoning",
        description="HellaSwag — action understanding benchmark",
    ),
    BenchmarkDefinition(
        name="truthfulqa",
        display_name="TruthfulQA",
        default_sample_size=100,
        max_samples=8177,
        category="truthfulness",
        description="TruthfulQA — truthfulness evaluation",
    ),
    BenchmarkDefinition(
        name="arc_challenge",
        display_name="ARC Challenge",
        default_sample_size=50,
        max_samples=8000,
        category="reasoning",
        description="ARC Challenge — science question answering",
    ),
    BenchmarkDefinition(
        name="winogrande",
        display_name="Winogrande",
        default_sample_size=100,
        max_samples=10000,
        category="reasoning",
        description="Winogrande — coreference resolution",
    ),
    BenchmarkDefinition(
        name="gsm8k",
        display_name="GSM8K",
        default_sample_size=100,
        max_samples=1300,
        category="reasoning",
        description="GSM8K — grade-school math word problems",
    ),
    BenchmarkDefinition(
        name="mathqa",
        display_name="MathQA",
        default_sample_size=100,
        max_samples=10000,
        category="reasoning",
        description="MathQA — mathematical reasoning benchmark",
    ),
    BenchmarkDefinition(
        name="humaneval",
        display_name="HumanEval",
        default_sample_size=100,
        max_samples=164,
        category="coding",
        description="HumanEval — Python code generation (function completion)",
    ),
    BenchmarkDefinition(
        name="mbpp",
        display_name="MBPP",
        default_sample_size=200,
        max_samples=500,
        category="coding",
        description="MBPP — Mostly Basic Python Problems",
    ),
    BenchmarkDefinition(
        name="livecodebench",
        display_name="LiveCodeBench",
        default_sample_size=50,
        max_samples=500,
        category="coding",
        description="LiveCodeBench — competitive programming benchmark",
    ),
    BenchmarkDefinition(
        name="bbq",
        display_name="BBQ",
        default_sample_size=200,
        max_samples=2000,
        category="safety",
        description="BBQ — bias benchmark for QA",
    ),
    BenchmarkDefinition(
        name="safetybench",
        display_name="SafetyBench",
        default_sample_size=100,
        max_samples=5000,
        category="safety",
        description="SafetyBench — safety evaluation benchmark",
    ),
]


def get_benchmark_registry() -> dict[str, dict[str, Any]]:
    """Return the benchmark registry as a serialisable dict.

    Returns:
        Dict mapping benchmark names to their definition dicts.
    """
    return {
        b.name: {
            "display_name": b.display_name,
            "default_sample_size": b.default_sample_size,
            "max_samples": b.max_samples,
            "supports_batching": b.supports_batching,
            "category": b.category,
            "description": b.description,
        }
        for b in VALID_BENCHMARKS
    }


def get_valid_benchmark_names() -> list[str]:
    """Return a list of valid benchmark names."""
    return [b.name for b in VALID_BENCHMARKS]


def validate_benchmarks(
    benchmarks: dict[str, int],
) -> list[str]:
    """Validate benchmark names and sample sizes against the registry.

    Returns:
        List of error messages.  Empty list means validation passed.
    """
    errors: list[str] = []
    registry = get_benchmark_registry()

    for name, size in benchmarks.items():
        if name not in registry:
            errors.append(f"Unknown benchmark '{name}'")
        else:
            defn = registry[name]
            if size < 0:
                errors.append(
                    f"Sample size for '{name}' must be >= 0, got {size}"
                )
            if size > defn.max_samples:
                errors.append(
                    f"Sample size for '{name}' exceeds max ({defn.max_samples})"
                )

    return errors
