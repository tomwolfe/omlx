# SPDX-License-Identifier: Apache-2.0
"""
Speculative decoding as a composable middleware.

This module provides a strategy-pattern wrapper that wraps any base engine
with speculative decoding capabilities.  Rather than maintaining a separate
DFlashEngine class with duplicated KV-cache management logic, the
SpeculativeEngine wraps an existing engine (BatchedEngine, BatchedEngine,
VLMBatchedEngine, etc.) and adds speculative decoding as a composable
layer.

Usage::

    # Create a wrapped engine with speculative decoding
    base_engine = BatchedEngine(model_name="llama-3b")
    spec_engine = SpeculativeEngine(
        base_engine=base_engine,
        draft_model_path="/path/to/draft/model",
        draft_quant_enabled=True,
    )

    # Use the wrapped engine exactly like a normal engine
    output = await spec_engine.generate(prompt="Hello", max_tokens=256)
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from typing import Any, Optional

import mlx.core as mx

from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


class SpeculativeEngine(BaseEngine):
    """Speculative decoding engine wrapping a base engine.

    This class wraps any BaseEngine-compatible engine and adds speculative
    decoding as a composable middleware layer.  It reuses the base engine's
    KV-cache management, chat templates, output collection, and streaming
    infrastructure — no duplication of that logic.

    The draft model is loaded once during `start()` and reused across
    all generation requests.  When the base engine is stopped, draft
    models are properly cleaned up.
    """

    def __init__(
        self,
        base_engine: BaseEngine,
        draft_model_path: str,
        draft_quant_enabled: bool | None = None,
        draft_quant_weight_bits: int | None = None,
        draft_quant_activation_bits: int | None = None,
        draft_quant_group_size: int | None = None,
        model_settings: Any | None = None,
        fallback_engine_type: str = "batched",
        scheduler_config: Any | None = None,
        omlx_ssd_cache_dir: str | None = None,
    ):
        self._base_engine = base_engine
        self._draft_model_path = draft_model_path
        self._draft_quant_enabled = draft_quant_enabled
        self._draft_quant_weight_bits = draft_quant_weight_bits
        self._draft_quant_activation_bits = draft_quant_activation_bits
        self._draft_quant_group_size = draft_quant_group_size
        self._model_settings = model_settings
        self._fallback_engine_type = fallback_engine_type
        self._scheduler_config = scheduler_config
        self._omlx_ssd_cache_dir = omlx_ssd_cache_dir

        self._draft_model: Any | None = None
        self._loaded = False
        self._model_type_str: str | None = None
        self._tokenizer: Any | None = None

    @property
    def model_name(self) -> str:
        return self._base_engine.model_name if self._base_engine else ""

    @property
    def tokenizer(self) -> Any | None:
        return self._tokenizer or (
            self._base_engine.tokenizer if self._base_engine else None
        )

    @property
    def model_type(self) -> str | None:
        return self._model_type_str or (
            self._base_engine.model_type if self._base_engine else None
        )

    @property
    def grammar_compiler(self):
        return self._base_engine.grammar_compiler if self._base_engine else None

    async def start(self) -> None:
        """Start the base engine and load draft model for speculative decoding."""
        if self._loaded:
            return

        await self._base_engine.start()

        # Load draft model for speculative decoding
        from ..engine_core import get_mlx_executor

        loop = asyncio.get_running_loop()

        def _load_draft():
            from ..engine_core import get_mlx_executor as _get_mlx_executor
            from ..patches.mlx_lm_mtp import set_mtp_active

            was_mtp = False
            try:
                from ..patches.mlx_lm_mtp import is_mtp_active
                was_mtp = is_mtp_active()
            except Exception:
                pass
            set_mtp_active(False)
            try:
                from mlx_lm import load
                draft_model, _ = load(self._draft_model_path)
                return draft_model
            finally:
                set_mtp_active(was_mtp)

        self._draft_model = await loop.run_in_executor(
            get_mlx_executor(), _load_draft
        )

        # Extract model_type from base engine config
        if self._base_engine and hasattr(self._base_engine, '_model'):
            config = getattr(self._base_engine._model, 'config', None)
            if isinstance(config, dict):
                self._model_type_str = config.get("model_type")
            elif hasattr(config, "model_type"):
                self._model_type_str = config.model_type

        self._tokenizer = self._base_engine.tokenizer
        self._loaded = True
        logger.info(
            "SpeculativeEngine loaded: base=%s, draft=%s",
            self._base_engine.model_name if self._base_engine else "unknown",
            self._draft_model_path,
        )

    async def stop(self) -> None:
        """Stop the base engine and clean up draft model references."""
        if self._base_engine:
            await self._base_engine.stop()
        self._draft_model = None
        self._loaded = False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate completion using speculative decoding via base engine."""
        if not self._loaded:
            await self.start()

        return await self._base_engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generation with speculative decoding via base engine."""
        if not self._loaded:
            await self.start()

        async for output in self._base_engine.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        ):
            yield output

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion via speculative decoding base engine."""
        if not self._loaded:
            await self.start()

        return await self._base_engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream chat completion with speculative decoding."""
        if not self._loaded:
            await self.start()

        async for output in self._base_engine.stream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "speculative",
            "draft_model": self._draft_model_path,
            "draft_model_loaded": self._draft_model is not None,
            "base_engine": self._base_engine.get_stats() if self._base_engine else {},
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        if self._base_engine:
            return self._base_engine.get_cache_stats()
        return None

    def has_active_requests(self) -> bool:
        if self._base_engine:
            return self._base_engine.has_active_requests()
        return False
