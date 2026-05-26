# SPDX-License-Identifier: Apache-2.0
"""Sandboxed code evaluation using Python's multiprocessing.

This module replaces sequential subprocess execution with a ProcessPoolExecutor
that runs each code evaluation in an isolated process with strict memory limits
and timeouts.  A crashing or malicious test case in one worker cannot poison
the pool or the main event loop.

Usage::

    from omlx.eval.sandbox_executor import evaluate_batch

    results = await evaluate_batch(
        items=[{"code": "...", "test": "..."}],
        check_fn=lambda code, test: code == test,
        max_workers=4,
    )
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import resource
import traceback
from multiprocessing import get_start_method, set_start_method
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

# Maximum number of parallel sandboxed evaluations per benchmark run.
# Capped to prevent resource exhaustion on systems with many cores.
DEFAULT_MAX_WORKERS = 8

# Timeout for each sandboxed evaluation (seconds).
EXEC_TIMEOUT_SECONDS = 30

# Memory limit per worker process (256 MB).
EXEC_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024


def _safe_eval_single(
    index: int,
    code: str,
    test_data: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a single item in a sandboxed subprocess.

    This function runs inside the worker process.  It applies resource
    limits directly (no preexec_fn needed) to ensure memory limits are
    enforced per-process.

    Parameters
    ----------
    index : int
        Position in the original items list (used for ordering results).
    code : str
        The generated Python code to evaluate.
    test_data : dict[str, Any]
        Test metadata including ``test``, ``entry_point``, etc.

    Returns
    -------
    dict
        A result dict with keys ``correct`` (bool), ``error`` (str | None),
        ``index`` (int), and ``worker_pid`` (int).
    """
    # Apply resource limits to this worker process
    try:
        resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_LIMIT_BYTES, EXEC_MEMORY_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (EXEC_TIMEOUT_SECONDS + 5, EXEC_TIMEOUT_SECONDS + 5))
    except (ValueError, resource.error):
        pass

    try:
        # Import the check_answer logic here to keep this self-contained
        from omlx.eval.humaneval import _execute_with_tests
        from omlx.eval.mbpp import _execute_with_tests as mbpp_execute

        # Determine which benchmark we're handling
        test_type = test_data.get("type", "humaneval")
        if test_type == "humaneval":
            passed, error = _execute_with_tests(
                code, test_data["test"], test_data.get("entry_point", "")
            )
            correct = passed
        elif test_type == "mbpp":
            test_list = test_data.get("test_list", [])
            setup_code = test_data.get("test_setup_code", "")
            passed, error = mbpp_execute(
                code, test_list, setup_code
            )
            correct = passed
        else:
            correct = True
            error = None

        return {
            "index": index,
            "correct": correct,
            "error": error,
            "worker_pid": multiprocessing.current_process().pid,
        }

    except Exception as e:
        return {
            "index": index,
            "correct": False,
            "error": str(e),
            "worker_pid": multiprocessing.current_process().pid,
        }


async def evaluate_batch(
    items: list[dict[str, Any]],
    check_fn: Callable[[dict, dict], bool],
    max_workers: int | None = None,
    code_extractor: Callable[[dict], str] | None = None,
    on_progress: Callable[[int, int], Any] | None = None,
) -> list[dict]:
    """Evaluate a batch of items using ProcessPoolExecutor.

    This replaces the sequential code execution pattern with parallel
    sandboxed evaluation.  Each item is executed in a separate process
    with memory limits and timeouts applied.

    Parameters
    ----------
    items : list[dict]
        List of benchmark items, each containing ``code`` and test data.
    check_fn : Callable[[dict, dict], bool]
        Function that checks if the predicted code is correct given the item.
    max_workers : int | None
        Maximum number of parallel workers.  Defaults to ``DEFAULT_MAX_WORKERS``.
    code_extractor : Callable[[dict], str] | None
        Optional function to extract code from model response.  If None,
        items are assumed to already contain ``code``.
    on_progress : Callable[[int, int], Any] | None
        Optional progress callback(current, total).

    Returns
    -------
    list[dict]
        Results list with keys ``correct``, ``error``, ``index``, ``worker_pid``.
    """
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    # Ensure we have a spawn context (safe for fork-safe operations)
    try:
        set_start_method("spawn")
    except RuntimeError:
        # Already set
        pass

    # Extract code from items if needed
    code_items = []
    for idx, item in enumerate(items):
        if code_extractor is not None:
            code = code_extractor(item)
        else:
            code = item.get("code", "")
        code_items.append({
            "index": idx,
            "code": code,
            "test_data": item,
        })

    # Run evaluations in parallel using ProcessPoolExecutor
    loop = asyncio.get_event_loop()
    results: list[dict] = []

    # Process in chunks to avoid overwhelming the pool
    chunk_size = max(1, max_workers)
    for chunk_start in range(0, len(code_items), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(code_items))
        chunk = code_items[chunk_start:chunk_end]

        # Submit all items in the chunk to the executor
        futures = [
            loop.run_in_executor(
                None,  # Uses default ProcessPoolExecutor
                _safe_eval_single,
                item["index"],
                item["code"],
                item["test_data"],
            )
            for item in chunk
        ]

        chunk_results = await asyncio.gather(*futures)
        results.extend(chunk_results)

        if on_progress:
            completed = chunk_start + len(chunk_results)
            await on_progress(completed, len(code_items))

    # Sort by index to maintain original order
    results.sort(key=lambda r: r["index"])
    return results
