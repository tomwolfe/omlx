# SPDX-License-Identifier: Apache-2.0
"""
Async-safe state management for oMLX.

This module provides a `StateManager` class that encapsulates mutable global
state behind asyncio.Lock-protected access, replacing raw module-level
dictionaries that were previously unprotected.

Usage:
    manager = StateManager()
    await manager.create(key, data)   # atomic create
    await manager.get(key)            # thread-safe read
    await manager.delete(key)         # atomic remove
    await manager.clear()             # bulk clear
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any


class StateManager:
    """Async-safe state manager with per-key locking.

    All mutations are guarded by an asyncio.Lock so that concurrent
    create/read/delete operations remain atomic and race-free.

    Usage::

        state = StateManager()
        await state.create("run_42", {"status": "running"})
        data = await state.get("run_42")
        assert data is not None
        await state.delete("run_42")
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def create(self, key: str, value: Any) -> None:
        """Atomically create a new entry. Raises ValueError if key exists."""
        async with self._lock:
            if key in self._store:
                raise ValueError(f"Key '{key}' already exists")
            self._store[key] = value

    async def get(self, key: str) -> Any | None:
        """Read a value by key. Returns None if the key does not exist."""
        async with self._lock:
            return self._store.get(key)

    async def update(self, key: str, value: Any) -> None:
        """Atomically update an existing entry. Raises KeyError if key missing."""
        async with self._lock:
            if key not in self._store:
                raise KeyError(key)
            self._store[key] = value

    async def delete(self, key: str) -> Any | None:
        """Remove a key and return its value. Returns None if key missing."""
        async with self._lock:
            return self._store.pop(key, None)

    async def clear(self) -> None:
        """Remove all entries."""
        async with self._lock:
            self._store.clear()

    def init_value(self, key: str, value: Any) -> None:
        """Set a value under a key without using the lock.

        Intended for one-time initialisation at import time where no
        concurrent access is possible.  For all other mutations use the
        async methods which are lock-protected.
        """
        self._store[key] = value

    async def keys(self) -> list[str]:
        """Return all keys."""
        async with self._lock:
            return list(self._store.keys())

    async def items(self) -> list[tuple[str, Any]]:
        """Return all (key, value) pairs."""
        async with self._lock:
            return list(self._store.items())

    async def append(self, value: Any) -> None:
        """Append a value to a list stored under a generated key.

        Returns the key that was used for the appended value.
        """
        key = str(uuid.uuid4())
        async with self._lock:
            if key not in self._store:
                self._store.setdefault(key, []).append(value)
            return key

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
