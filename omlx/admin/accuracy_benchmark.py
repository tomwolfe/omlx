# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark execution logic for oMLX admin panel.

Orchestrates MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluations with real-time progress reporting via SSE events.

Supports server-side queue and persistent result accumulation.
Results survive browser close and persist until explicitly reset.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from pydantic import BaseModel, field_validator

from .event_stream import BenchmarkEventStream
from .state_manager import StateManager

logger = logging.getLogger(__name__)

# Module-level state manager (singletons for the process lifetime)
_accuracy_runs: StateManager = StateManager()

# Accumulated results — persists until explicit reset
_accumulated_results: StateManager = StateManager()

# Server-side queue — all queue state is protected by _queue lock
_queue: StateManager = StateManager()

# Pre-initialized default values for queue state keys.
# These are set synchronously because they are simple values that
# don't need async protection — they are only written once at import time.
_queue_running: StateManager = StateManager()
_current_run_id: StateManager = StateManager()
_current_model: StateManager = StateManager()
_engine_pool_ref: StateManager = StateManager()


def _init_queue_defaults() -> None:
    """Initialize default values for queue-related state keys.

    Called once at import time to seed the StateManager stores with
    their initial values.  No async context is needed because the
    StateManager's internal lock is only needed for concurrent access.
    """
    _queue_running.init_value("running", False)
    _current_run_id.init_value("id", None)
    _current_model = None
    _engine_pool_ref = None


VALID_BENCHMARKS = [
    "mmlu",
    "mmlu_pro",
    "kmmlu",
    "cmmlu",
    "jmmlu",
    "hellaswag",
    "truthfulqa",
    "arc_challenge",
    "winogrande",
    "gsm8k",
    "mathqa",
    "humaneval",
    "mbpp",
    "livecodebench",
    "bbq",
    "safetybench",
]


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_id: str
    benchmarks: dict[str, int]  # name -> sample_size (0 = full dataset)
    batch_size: int = 1
    enable_thinking: bool = False

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8, 16, 32):
            raise ValueError("batch_size must be 1, 2, 4, 8, 16, or 32")
        return v

    @field_validator("benchmarks")
    @classmethod
    def validate_benchmarks(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("At least one benchmark is required")
        for name, size in v.items():
            if name not in VALID_BENCHMARKS:
                raise ValueError(
                    f"Invalid benchmark '{name}'. Must be one of {VALID_BENCHMARKS}"
                )
            if size < 0:
                raise ValueError(f"Sample size for '{name}' must be >= 0")
        return v


@dataclass
class AccuracyBenchmarkRun:
    """Tracks the state of a running accuracy benchmark.

    SSE delivery model mirrors `BenchmarkRun`: append-only `events`
    log + `cond` for live notification + `terminal` flag set on the
    final event. See benchmark.py for the rationale.

    Uses ``BenchmarkEventStream`` for unified SSE delivery.
    """

    bench_id: str
    request: AccuracyBenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    terminal: bool = False
    task: asyncio.Task | None = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    last_progress: dict | None = None  # last progress event for reconnect
    # Finer-grained lifecycle than `status` — surfaces the difference between
    # "still scoring questions" and "cleaning up after the last result was
    # emitted". The serialization gate (_queue_running) stays True across
    # both, but a UI rendering the running row wants to hide it once
    # phase=="unloading" so the user isn't told "still running" when the
    # result card has already appeared on screen. Transitions:
    #   pending → loading → evaluating → unloading → completed
    # (cancelled / error replace the terminal phase on those branches.)
    phase: str = "pending"

    def _make_event_stream(self) -> BenchmarkEventStream:
        """Return a shared event stream backed by this run's state."""
        stream = BenchmarkEventStream(bench_id=self.bench_id)
        for event in self.events:
            stream.events.append(event)
            terminal_types = {"done", "error"}
            if event.get("type") in terminal_types:
                stream.terminal = True
        return stream

    def send(self, event: dict) -> None:
        """Append an event to the run's log and wake any subscribers."""
        terminal_types = {"done", "error"}
        self.events.append(event)
        if event.get("type") in terminal_types:
            self.terminal = True


# Accuracy stream closes on `done` (run finished) or `error`. Unlike the
# throughput bench there's no separate upload phase to ride out.
_ACCURACY_TERMINAL_TYPES = frozenset({"done", "error"})


# --- Run management ---


async def get_run(bench_id: str) -> AccuracyBenchmarkRun | None:
    """Get an accuracy benchmark run by ID."""
    return await _accuracy_runs.get(bench_id)


async def create_run(request: AccuracyBenchmarkRequest) -> AccuracyBenchmarkRun:
    """Create a new accuracy benchmark run.

    Raises ValueError if a run with the same ID already exists.
    """
    bench_id = str(uuid.uuid4())[:8]
    run = AccuracyBenchmarkRun(bench_id=bench_id, request=request)
    await _accuracy_runs.create(bench_id, run)
    return run


async def cleanup_old_runs() -> None:
    """Remove completed/errored runs to prevent memory leaks."""
    items = await _accuracy_runs.items()
    to_remove = [
        bid for bid, run in items if run.status in ("completed", "cancelled", "error")
    ]
    for bid in to_remove:
        await _accuracy_runs.delete(bid)


# --- Accumulated results ---


async def get_accumulated_results() -> list[dict]:
    """Get all accumulated benchmark results."""
    items = await _accumulated_results.items()
    return [v for _, v in items]


async def reset_accumulated_results() -> None:
    """Clear all accumulated results."""
    await _accumulated_results.clear()


# --- Queue management ---


async def add_to_queue(request: AccuracyBenchmarkRequest) -> None:
    """Add a benchmark request to the queue."""
    await _queue.create(str(uuid.uuid4()), request)


async def get_queue_status() -> dict:
    """Get current queue status."""
    last_progress = None
    phase = None
    current_run_id = await _current_run_id.get("id")
    if current_run_id:
        run = await get_run(current_run_id)
        if run:
            last_progress = run.last_progress
            phase = run.phase
    queue_running = await _queue_running.get("running")
    current_model = await _current_model.get("model")
    items = await _queue.items()
    return {
        "running": queue_running,
        "current_model": current_model,
        "current_bench_id": current_run_id,
        "last_progress": last_progress,
        # Finer-grained than `running`: distinguishes "still scoring" from
        # "cleaning up after the last result emitted". Polling UIs hide
        # the running row once phase becomes "unloading" / "completed" so
        # the result card alone tells the story.
        "phase": phase,
        "queue": [
            {"model_id": r.model_id, "benchmarks": list(r.benchmarks.keys())}
            for _, r in items
        ],
    }


async def remove_from_queue(idx: int) -> bool:
    """Remove an item from the queue by index."""
    items = await _queue.items()
    if 0 <= idx < len(items):
        _, _ = list(items)[idx], None  # consume the entry
        await _queue.delete(list(dict(items).keys())[idx])
        return True
    return False


async def start_next_from_queue(engine_pool: Any) -> str | None:
    """Pop next item from queue, create run, start background task.

    Returns bench_id if a run was started, None if already running or queue empty.
    """
    queue_running = await _queue_running.get("running")

    if queue_running:
        return None

    items = await _queue.items()
    if not items:
        return None

    keys = list(dict(items).keys())
    request = await _queue.delete(keys[0])
    _queue_running.init_value("running", True)
    _current_model = request.model_id

    await cleanup_old_runs()
    run = await create_run(request)
    _current_run_id.init_value("id", run.bench_id)

    logger.info(
        f"Queue: starting {request.model_id} "
        f"benchmarks={list(request.benchmarks.keys())}"
    )

    async def _run_and_continue():
        try:
            await run_accuracy_benchmark(run, engine_pool)
        except Exception as e:
            logger.error(f"Queue: error running {request.model_id}: {e}")
        # Auto-continue with next in queue
        await _continue_queue(engine_pool)

    run.task = asyncio.create_task(_run_and_continue())
    return run.bench_id


async def _continue_queue(engine_pool: Any) -> None:
    """Continue processing the queue after a run completes."""
    queue_items = await _queue.items()
    if not queue_items:
        _queue_running.init_value("running", False)
        _current_run_id.init_value("id", None)
        _current_model = None
        return

    keys = list(dict(queue_items).keys())
    request = await _queue.delete(keys[0])
    _current_model = request.model_id

    await cleanup_old_runs()
    run = await create_run(request)
    _current_run_id.init_value("id", run.bench_id)

    logger.info(
        f"Queue: continuing with {request.model_id} "
        f"benchmarks={list(request.benchmarks.keys())}"
    )

    try:
        await run_accuracy_benchmark(run, engine_pool)
    except Exception as e:
        logger.error(f"Queue: error running {request.model_id}: {e}")

    await _continue_queue(engine_pool)


async def cancel_queue() -> None:
    """Cancel the current run and clear the queue."""
    await _queue.clear()

    current_run_id = await _current_run_id.get("id")
    if current_run_id:
        run = await get_run(current_run_id)
        if run and run.status == "running":
            run.status = "cancelled"
            if run.task and not run.task.done():
                run.task.cancel()

    _queue_running.init_value("running", False)
    _current_run_id.init_value("id", None)
    _current_model = None


# --- SSE ---


def _send_event(run: AccuracyBenchmarkRun, event: dict) -> None:
    """Append an event to the run's log and wake any subscribers.

    Updates `last_progress` (used by the REST `queue/status` endpoint
    for reconnect hints) and sets `run.terminal` on the final event.
    """
    if event.get("type") == "progress":
        run.last_progress = event
    run.send(event)


# --- Benchmark execution ---


async def run_accuracy_benchmark(run: AccuracyBenchmarkRun, engine_pool: Any) -> None:
    """Execute accuracy benchmark run.

    Phases:
    1. Unload all models
    2. Load target model
    3. For each selected benchmark: load data, evaluate, report
    4. Unload model
    5. Send done event
    """
    from ..eval import BENCHMARKS

    request = run.request

    # Suppress TTL auto-unload during benchmark
    engine_pool._suppress_ttl = True
    start_time = time.time()

    try:
        # Phase 1: Unload all models
        run.phase = "loading"
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "unload",
                    "model_id": request.model_id,
                    "benchmark": "",
                    "message": f"Unloading {len(loaded_ids)} model(s)...",
                    "current": 0,
                    "total": len(request.benchmarks),
                },
            )
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                except Exception as e:
                    logger.warning(f"Failed to unload {model_id}: {e}")

        # Phase 2: Load target model
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "load",
                "model_id": request.model_id,
                "benchmark": "",
                "message": f"Loading {request.model_id}...",
                "current": 0,
                "total": len(request.benchmarks),
            },
        )

        # Force LM engine for accuracy benchmarks — text-only tasks
        # don't need VLM and the VLM adapter can produce empty responses.
        engine = await engine_pool.get_engine(request.model_id, force_lm=True)

        # Load model sampling settings
        sampling_kwargs = {}
        if engine_pool._settings_manager is not None:
            ms = engine_pool._settings_manager.get_settings(request.model_id)
            if ms.top_p is not None:
                sampling_kwargs["top_p"] = ms.top_p
            if ms.top_k is not None:
                sampling_kwargs["top_k"] = ms.top_k
            if ms.min_p is not None:
                sampling_kwargs["min_p"] = ms.min_p
            if ms.repetition_penalty is not None:
                sampling_kwargs["repetition_penalty"] = ms.repetition_penalty
            if ms.presence_penalty is not None:
                sampling_kwargs["presence_penalty"] = ms.presence_penalty
            if ms.chat_template_kwargs:
                sampling_kwargs["chat_template_kwargs"] = ms.chat_template_kwargs

        # Phase 3: Run each benchmark
        run.phase = "evaluating"
        completed = 0
        for bench_name, sample_size in request.benchmarks.items():
            if run.status == "cancelled":
                break

            bench_cls = BENCHMARKS.get(bench_name)
            if bench_cls is None:
                logger.warning(f"Unknown benchmark: {bench_name}")
                continue

            evaluator = bench_cls()

            # Load dataset
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "download",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Loading {bench_name} dataset...",
                    "current": completed,
                    "total": len(request.benchmarks),
                },
            )

            try:
                items = await evaluator.load_dataset(sample_size=sample_size)
            except Exception as e:
                logger.error(f"Failed to load {bench_name} dataset: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Failed to load {bench_name} dataset: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            # Run evaluation with progress
            total_items = len(items)

            async def _on_progress_factory(
                run_obj,
                model_id,
                bench_name,
                completed,
                total_benchmarks,
                current=0,
                total=0,
            ):
                if run_obj.status == "cancelled":
                    raise asyncio.CancelledError()
                await _send_event(
                    run_obj,
                    {
                        "type": "progress",
                        "phase": "eval",
                        "model_id": model_id,
                        "benchmark": bench_name,
                        "message": f"Evaluating {bench_name} ({current}/{total})...",
                        "current": completed,
                        "total": total_benchmarks,
                        "bench_current": current,
                        "bench_total": total,
                    },
                )

            on_progress = partial(
                _on_progress_factory,
                run,
                request.model_id,
                bench_name,
                completed,
                len(request.benchmarks),
                0,
                total_items,
            )

            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "eval",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Evaluating {bench_name} (0/{total_items})...",
                    "current": completed,
                    "total": len(request.benchmarks),
                    "bench_current": 0,
                    "bench_total": total_items,
                },
            )

            try:
                result = await evaluator.run(
                    engine,
                    items,
                    on_progress,
                    batch_size=request.batch_size,
                    sampling_kwargs=sampling_kwargs,
                    enable_thinking=request.enable_thinking,
                )
            except asyncio.CancelledError:
                run.status = "cancelled"
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": "Benchmark cancelled",
                    },
                )
                return
            except Exception as e:
                logger.error(f"Error running {bench_name}: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Error running {bench_name}: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            # Build result
            result_data = {
                "model_id": request.model_id,
                "benchmark": result.benchmark_name,
                "accuracy": round(result.accuracy, 4),
                "thinking_used": result.thinking_used,
                "total": result.total_questions,
                "correct": result.correct_count,
                "time_s": round(result.time_seconds, 1),
                "question_results": [
                    {
                        "id": qr.question_id,
                        "correct": qr.correct,
                        "expected": qr.expected,
                        "predicted": qr.predicted,
                        "question": qr.question_text,
                        "raw_response": qr.raw_response,
                        "category": qr.category,
                        "time_s": round(qr.time_seconds, 3),
                    }
                    for qr in result.question_results
                ],
            }
            if result.category_scores:
                result_data["category_scores"] = {
                    k: round(v, 4) for k, v in result.category_scores.items()
                }

            # Accumulate persistently
            await _accumulated_results.append(result_data)

            run.results.append(result_data)
            completed += 1

            await _send_event(
                run,
                {
                    "type": "result",
                    "data": result_data,
                },
            )

        # Phase 4: Unload model. The result(s) are already emitted by now,
        # so flip phase so polling clients hide the running indicator
        # (the result card has already appeared on screen — telling the
        # user "still running" while we clean up reads as a bug).
        run.phase = "unloading"
        with contextlib.suppress(Exception):
            await engine_pool._unload_engine(request.model_id)

        # Phase 5: Done
        total_time = time.time() - start_time
        run.status = "completed"
        run.phase = "completed"

        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "total_time": round(total_time, 1),
                    "benchmarks_completed": completed,
                },
            },
        )

    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        await _send_event(
            run,
            {
                "type": "error",
                "message": "Benchmark cancelled",
            },
        )
    except Exception as e:
        logger.exception(f"Accuracy benchmark error: {e}")
        run.status = "error"
        run.phase = "error"
        run.error_message = str(e)
        await _send_event(
            run,
            {
                "type": "error",
                "message": str(e),
            },
        )
    finally:
        # Re-enable TTL auto-unload
        engine_pool._suppress_ttl = False
