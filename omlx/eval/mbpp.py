# SPDX-License-Identifier: Apache-2.0
"""MBPP (Mostly Basic Python Problems) benchmark.

Tests code generation with natural language descriptions and assertion tests.
Dataset bundled from google-research-datasets/mbpp (full test) on HuggingFace.
500 problems with assert-based test cases.

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
"""

import contextlib
import logging
import os
import resource
import subprocess
import tempfile
from pathlib import Path

from .base import BaseBenchmark
from .constants import EXEC_MEMORY_LIMIT_BYTES, EXEC_TIMEOUT_SECONDS
from .datasets import deterministic_sample, load_jsonl
from .utils import extract_last_code_block

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def _set_resource_limits():
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES)
        )
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5)
        )


def _execute_with_tests(
    code: str, test_list: list[str], setup_code: str = ""
) -> tuple[bool, str]:
    """Execute generated code with assertion-based test cases."""
    test_code = "\n".join(test_list)
    script = f"{setup_code}\n{code}\n{test_code}\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            preexec_fn=_set_resource_limits,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": "en_US.UTF-8",
            },
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


class MBPPBenchmark(BaseBenchmark):
    """MBPP: code generation with assertion-based test verification."""

    name = "mbpp"
    quick_size = 200

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load MBPP from bundled data."""
        items = load_jsonl(DATA_DIR / "mbpp.jsonl")

        normalized = []
        for item in items:
            test_list = item.get("test_list", [])
            if not test_list:
                continue
            normalized.append(
                {
                    "id": str(item["task_id"]),
                    "prompt": item["prompt"],
                    "test_list": test_list,
                    "test_setup_code": item.get("test_setup_code", ""),
                    "question": item["prompt"],
                }
            )

        logger.info(f"MBPP: loaded {len(normalized)} problems")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 2048

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a code generation prompt with test cases for function name."""
        prompt = item["prompt"]
        tests = item.get("test_list", [])
        test_str = "\n".join(tests[:3])
        content = (
            "Write a Python function to solve the following problem. "
            "Provide only the complete function implementation, no explanations.\n\n"
            f"Problem: {prompt}\n\n"
            f"Test cases:\n{test_str}\n\n"
            "Solution:"
        )
        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, item: dict) -> str:
        return extract_last_code_block(response)

    def check_answer(self, predicted: str, item: dict) -> bool:
        if not predicted.strip():
            return False

        passed, error = _execute_with_tests(
            predicted,
            item["test_list"],
            item.get("test_setup_code", ""),
        )
        return passed

    async def runCode(self, predicted_code: str, item: dict) -> bool:
        """Execute code and verify against assertion-based test cases."""
        return self.check_answer(predicted_code, item)
