# SPDX-License-Identifier: Apache-2.0
"""HumanEval benchmark.

Tests code generation ability using function completion problems.
Model receives a function signature + docstring and must complete the body.
Verification: generated code + unit tests run in sandboxed subprocess.
Dataset bundled from openai/openai_humaneval on HuggingFace (164 problems).

SECURITY NOTE: This benchmark executes model-generated code on the local
machine. Mitigations: subprocess with timeout, memory limits, temp file cleanup.
"""

import contextlib
import logging
import os
import re
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


def _get_imports(prompt: str) -> str:
    """Extract import lines from the prompt."""
    lines: list[str] = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
    return "\n".join(lines)


def _extract_code(response: str, prompt: str) -> str:
    """Extract the function body from model response.

    The model may return the full function (including signature) or just the body.
    We need to combine it with the original prompt to form a complete function.
    Always prepends imports from the prompt to avoid NameError.
    """
    response = response.strip()
    imports = _get_imports(prompt)

    # If response contains a code block, extract it
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            if imports and not any(
                line.strip().startswith(("import ", "from "))
                for line in code.split("\n")
            ):
                return imports + "\n\n" + code
            return code
        return prompt + code

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "def " in code:
            if imports and not any(
                line.strip().startswith(("import ", "from "))
                for line in code.split("\n")
            ):
                return imports + "\n\n" + code
            return code
        return prompt + code

    # No code block — response is the continuation of the prompt
    if response.startswith("def "):
        if imports:
            return imports + "\n\n" + response
        return response
    if response.startswith("from ") or response.startswith("import "):
        return response

    return prompt + response


def _set_resource_limits():
    """Set resource limits for subprocess."""
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES)
        )
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(
            resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5)
        )


def _execute_with_tests(
    code: str, test_code: str, entry_point: str
) -> tuple[bool, str]:
    """Execute generated code with test cases.

    Combines the generated function with test assertions and runs in subprocess.

    Returns:
        (passed, error_message)
    """
    # Build the complete test script
    script = f"""{code}

{test_code}

check({entry_point})
"""
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


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval: function completion with unit test verification."""

    name = "humaneval"
    quick_size = 100

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        """Load HumanEval from bundled data."""
        items = load_jsonl(DATA_DIR / "humaneval.jsonl")

        normalized = []
        for item in items:
            normalized.append(
                {
                    "id": item["task_id"],
                    "prompt": item["prompt"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                    "question": item["prompt"],  # for get_question_text
                }
            )

        logger.info(f"HumanEval: loaded {len(normalized)} problems")

        if sample_size == 0:
            return normalized

        return deterministic_sample(normalized, sample_size)

    def get_max_tokens(self) -> int:
        return 2048

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        """Format as a function completion prompt."""
        prompt = item["prompt"]
        content = (
            "Complete the following Python function. "
            "Provide only the complete function implementation, no explanations.\n\n"
            f"{prompt}"
        )
        return [{"role": "user", "content": content}]

    def extract_answer(self, response: str, item: dict) -> str:
        """Extract the complete function from model response."""
        code = extract_last_code_block(response)
        imports = _get_imports(item["prompt"])

        if "def " in code and imports and not any(
            line.strip().startswith(("import ", "from "))
            for line in code.split("\n")
        ):
            return imports + "\n\n" + code

        if "def " not in code:
            return item["prompt"] + code

        return code

    def check_answer(self, predicted: str, item: dict) -> bool:
        """Execute the generated code with test cases."""
        if not predicted.strip():
            return False

        passed, error = _execute_with_tests(
            predicted, item["test"], item["entry_point"]
        )
        return passed

    async def runCode(self, predicted_code: str, item: dict) -> bool:
        """Execute code and verify against unit test cases."""
        return self.check_answer(predicted_code, item)
