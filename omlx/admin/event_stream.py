# SPDX-License-Identifier: Apache-2.0
"""Unified benchmark event delivery layer.

Both throughput and accuracy benchmarks share the same SSE delivery model:
append-only event log + asyncio.Condition for live notification + terminal
flag that tells subscribers when the stream has ended.

This module extracts that shared logic into ``BenchmarkEventStream`` so
that neither ``benchmark.py`` nor ``accuracy_benchmark.py`` needs to
re-implement the replay-then-attach loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class BenchmarkEventStream:
    """Shared delivery model for benchmark SSE events.

    Both ``BenchmarkRun`` (throughput) and ``AccuracyBenchmarkRun``
    (accuracy) use this pattern:

    1. **Replay**: a subscriber that connects after events were emitted
       still sees the entire history.
    2. **Multi-consumer**: multiple subscribers all see every event in order.
    3. **Terminal close**: the stream closes cleanly after a terminal
       event without blocking on a follow-up that will never arrive.

    Callers create an instance, append events via ``send()``, and
    subscribers use ``drain()`` to replay-then-attach.
    """

    bench_id: str
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False

    def send(self, event: dict) -> None:
        """Append an event and wake any waiting subscribers.

        Sets ``terminal`` when the event type is terminal (done/error).
        """
        terminal_types = {"upload_done", "error", "done"}
        with self.cond:
            self.events.append(event)
            if event.get("type") in terminal_types:
                self.terminal = True
            self.cond.notify_all()

    async def drain(
        self,
        *,
        max_events: int | None = None,
        timeout: float = 1.0,
    ) -> list[dict]:
        """Read the event log replay-then-attach style.

        Matches the SSE endpoint loop: snapshot events under the lock,
        release, yield, repeat. Returns once the run is terminal,
        ``max_events`` is reached, or the wait times out.
        """
        seen = 0
        out: list[dict] = []
        while True:
            async with self.cond:
                while seen >= len(self.events) and not self.terminal:
                    try:
                        await asyncio.wait_for(self.cond.wait(), timeout=timeout)
                    except TimeoutError:
                        break
                new = list(self.events[seen:])
                seen = len(self.events)
                done = self.terminal
            out.extend(new)
            if max_events is not None and len(out) >= max_events:
                break
            if done:
                break
            if not new:
                break
        return out
