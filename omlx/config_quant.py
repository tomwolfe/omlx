# SPDX-License-Identifier: Apache-2.0
"""Structured quantization configuration.

Replaces magic-dictionary keys (``_oq_sensitivity_map``, ``_oq_boost_map``,
``_oq_use_budget_plan``, ``_oq_non_quantizable``) with strongly-typed
dataclasses so that every access is type-checked and self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig:
    """Strongly-typed quantization configuration.

    Equivalent to the old ``config`` dict that used magic keys
    ``_oq_sensitivity_map``, ``_oq_boost_map``, ``_oq_use_budget_plan``,
    ``_oq_non_quantizable``, ``text_config``, ``num_hidden_layers``,
    ``num_local_experts``, ``num_experts``, ``hidden_size``.
    """

    sensitivity_map: dict[str, float] = field(default_factory=dict)
    boost_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    use_budget_plan: bool = False
    non_quantizable: set[str] = field(default_factory=set)
    text_config: "QuantizationConfig" | None = None
    num_hidden_layers: int = 32
    num_local_experts: int = 0
    num_experts: int = 0
    hidden_size: int = 0

    # -- class-factory helpers (backwards-compatible with plain dicts) -------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantizationConfig":
        """Build a QuantizationConfig from a raw config dict."""
        return cls(
            sensitivity_map=data.get("_oq_sensitivity_map", {}),
            boost_map=data.get("_oq_boost_map", {}),
            use_budget_plan=bool(data.get("_oq_use_budget_plan")),
            non_quantizable=set(data.get("_oq_non_quantizable") or ()),
            text_config=(
                cls.from_dict(data["text_config"])
                if isinstance(data.get("text_config"), dict)
                else None
            ),
            num_hidden_layers=int(data.get("num_hidden_layers", 32)),
            num_local_experts=int(data.get("num_local_experts", 0)),
            num_experts=int(data.get("num_experts", 0)),
            hidden_size=int(data.get("hidden_size", 0)),
        )

    # -- thin dict-like accessors (for code that still passes dicts) -------

    def get(self, key: str, default: Any = None) -> Any:
        """Minimal dict-like ``get`` for existing callers."""
        mapping = {
            "_oq_sensitivity_map": self.sensitivity_map,
            "_oq_boost_map": self.boost_map,
            "_oq_use_budget_plan": self.use_budget_plan,
            "_oq_non_quantizable": self.non_quantizable,
            "text_config": self.text_config,
            "num_hidden_layers": self.num_hidden_layers,
            "num_local_experts": self.num_local_experts,
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
        }
        if key in mapping:
            return mapping[key]
        return default

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in (
            "_oq_sensitivity_map",
            "_oq_boost_map",
            "_oq_use_budget_plan",
            "_oq_non_quantizable",
            "text_config",
            "num_hidden_layers",
            "num_local_experts",
            "num_experts",
            "hidden_size",
        )


@dataclass(frozen=True)
class OQTaskConfig:
    """Parameters for starting an oQ quantization task.

    Mirrors the ``OQStartRequest`` pydantic model in *routes.py* but uses
    plain dataclasses so that the quantization pipeline does not depend on
    pydantic at all.
    """

    model_path: str
    oq_level: float
    group_size: int = 64
    sensitivity_model_path: str = ""
    text_only: bool = False
    dtype: str = "bfloat16"
    preserve_mtp: bool = False
    auto_proxy_sensitivity: bool = True

    @property
    def output_name(self) -> str:
        """Model name with oQ suffix appended."""
        from omlx.oq import resolve_output_name

        model_name = (
            self.model_path.rsplit("/", 1)[-1]
            if "/" in self.model_path
            else self.model_path
        )
        return resolve_output_name(
            model_name,
            self.oq_level,
            self.dtype,
            preserve_mtp=self.preserve_mtp,
        )


@dataclass(frozen=True)
class QuantPlan:
    """Byte-budgeted mixed-precision plan for a single quantization run."""

    boost_map: dict[str, dict]
    effective_bpw: float
    target_bpw: float
    hard_cap_bpw: float

    @property
    def is_complete(self) -> bool:
        """Return ``True`` when the plan has non-trivial boosts."""
        return bool(self.boost_map)


@dataclass(frozen=True)
class BenchmarkRequest:
    """Request model for starting a benchmark.

    Mirrors the pydantic ``BenchmarkRequest`` in *benchmark.py* but uses
    plain dataclasses to avoid adding a pydantic import to the benchmark
    module.
    """

    model_id: str
    prompt_lengths: list[int]
    generation_length: int = 128
    batch_sizes: list[int] = field(default_factory=list)

    @staticmethod
    def is_valid_prompt_length(length: int) -> bool:
        """Check whether *length* is an accepted prompt length."""
        return length in {1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000}

    @staticmethod
    def is_valid_batch_size(size: int) -> bool:
        """Check whether *size* is an accepted batch size."""
        return size in {2, 4, 8}


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the engine core.

    Mirrors the old ``EngineConfig`` dataclass in *engine_core.py*.
    """

    max_num_seqs: int = 8
    completion_batch_size: int = 8
    stream_interval: int = 1
    max_context_window: int = 32768
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    force_sampling: bool = False
    max_tool_result_tokens: int = 0
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    enable_thinking: bool | None = None
    thinking_budget_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    mtp_enabled: bool = False
    # TurboQuant KV cache (mlx-vlm backend)
    turboquant_kv_enabled: bool = False
    turboquant_kv_bits: float = 4.0
    # SpecPrefill (experimental)
    specprefill_enabled: bool = False
    specprefill_draft_model: str | None = None
    specprefill_keep_pct: float = 0.0
    specprefill_threshold: int = 0
    # DFlash (block diffusion speculative decoding)
    dflash_enabled: bool = False
    dflash_draft_model: str | None = None
    dflash_draft_quant_enabled: bool = False
    dflash_draft_quant_weight_bits: int = 0
    dflash_draft_quant_activation_bits: int = 0
    dflash_draft_quant_group_size: int = 0
    dflash_max_ctx: int = 0
    dflash_in_memory_cache: bool = False
    dflash_in_memory_cache_max_entries: int = 0
    dflash_in_memory_cache_max_bytes: int = 0
    dflash_ssd_cache: bool = False
    dflash_ssd_cache_max_bytes: int = 0
    dflash_draft_window_size: int = 0
    dflash_draft_sink_size: int = 0
    dflash_verify_mode: str | None = None
    # VLM MTP speculative decoding via external assistant drafter (mlx-vlm 191d7c8+)
    vlm_mtp_enabled: bool = False
    vlm_mtp_draft_model: str | None = None
    vlm_mtp_draft_block_size: int = 0
    reasoning_parser: str | None = None
    is_pinned: bool = False
    is_default: bool = False
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        """Build an EngineConfig from a raw dict."""
        return cls(
            max_context_window=data.get("max_context_window", 32768),
            max_tokens=data.get("max_tokens", 32768),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 0),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            min_p=data.get("min_p", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            force_sampling=data.get("force_sampling", False),
            max_tool_result_tokens=data.get("max_tool_result_tokens", 0),
            chat_template_kwargs=data.get("chat_template_kwargs"),
            forced_ct_kwargs=data.get("forced_ct_kwargs"),
            enable_thinking=data.get("enable_thinking"),
            thinking_budget_enabled=data.get("thinking_budget_enabled"),
            thinking_budget_tokens=data.get("thinking_budget_tokens"),
            turboquant_kv_enabled=data.get("turboquant_kv_enabled", False),
            turboquant_kv_bits=data.get("turboquant_kv_bits", 4.0),
            specprefill_enabled=data.get("specprefill_enabled", False),
            specprefill_draft_model=data.get("specprefill_draft_model"),
            specprefill_keep_pct=data.get("specprefill_keep_pct", 0.0),
            specprefill_threshold=data.get("specprefill_threshold", 0),
            dflash_enabled=data.get("dflash_enabled", False),
            dflash_draft_model=data.get("dflash_draft_model"),
            dflash_draft_quant_enabled=data.get("dflash_draft_quant_enabled", False),
            dflash_draft_quant_weight_bits=data.get(
                "dflash_draft_quant_weight_bits", 0
            ),
            dflash_draft_quant_activation_bits=data.get(
                "dflash_draft_quant_activation_bits", 0
            ),
            dflash_draft_quant_group_size=data.get("dflash_draft_quant_group_size", 0),
            dflash_max_ctx=data.get("dflash_max_ctx", 0),
            dflash_in_memory_cache=data.get("dflash_in_memory_cache", False),
            dflash_in_memory_cache_max_entries=data.get(
                "dflash_in_memory_cache_max_entries", 0
            ),
            dflash_in_memory_cache_max_bytes=data.get(
                "dflash_in_memory_cache_max_bytes", 0
            ),
            dflash_ssd_cache=data.get("dflash_ssd_cache", False),
            dflash_ssd_cache_max_bytes=data.get("dflash_ssd_cache_max_bytes", 0),
            dflash_draft_window_size=data.get("dflash_draft_window_size", 0),
            dflash_draft_sink_size=data.get("dflash_draft_sink_sink", 0),
            dflash_verify_mode=data.get("dflash_verify_mode"),
            mtp_enabled=data.get("mtp_enabled", False),
            vlm_mtp_enabled=data.get("vlm_mtp_enabled", False),
            vlm_mtp_draft_model=data.get("vlm_mtp_draft_model"),
            vlm_mtp_draft_block_size=data.get("vlm_mtp_draft_block_size", 0),
            reasoning_parser=data.get("reasoning_parser"),
            is_pinned=data.get("is_pinned", False),
            is_default=data.get("is_default", False),
            trust_remote_code=data.get("trust_remote_code", False),
        )

    # Thin dict-like access for legacy callers.
    def get(self, key: str, default: Any = None) -> Any:
        import dataclasses

        mapping = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return mapping.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        import dataclasses

        return any(f.name == key for f in dataclasses.fields(self))
