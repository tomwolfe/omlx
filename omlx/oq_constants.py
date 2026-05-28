# SPDX-License-Identifier: Apache-2.0
"""Shared constants and helpers for quantization.

This module holds values and helpers that are needed by both ``oq.py``
and ``oq_policies.py`` so that neither module imports the other at
import-time (which would cause a circular-import error).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Any

# Quantization level configuration
OQ_LEVELS = {2, 3, 3.5, 4, 5, 6, 8}
OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")
_OQ_DEFAULT_GROUP_SIZE = 64
_MAX_MODEL_RAM_FRACTION = 0.8
_PROXY_QUANT_BITS = 4
_PROXY_QUANT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
_LEVEL_PROTECTION: dict[float, str] = {
    2: "full",
    3: "full",
    3.5: "full",
    4: "full",
    5: "full",
    6: "full",
    8: "full",
}
_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}

# --- helpers ---


def _mode_for_bits(bits: int) -> str:
    """Select quantization mode. Always affine to minimize kernel combos."""
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get group_size. Always default to minimize kernel combos."""
    return default_gs


def _bits_fn_factory(base_bits: int) -> Callable[[int], dict]:
    """Create a bits-calculating function that closes over base_bits."""

    def _bits(n: int) -> dict:
        effective = int(max(n, base_bits))
        return {
            "bits": effective,
            "group_size": _OQ_DEFAULT_GROUP_SIZE,
            "mode": _mode_for_bits(effective),
        }

    return _bits


# ---------------------------------------------------------------------------
# Structured tensor path parsing
# ---------------------------------------------------------------------------


class ProjectionType(IntFlag):
    """Quantisation projection type for a tensor path."""

    NONE = 0
    GATE = auto()
    DOWN = auto()
    UP = auto()
    ACT = auto()
    O = auto()
    V = auto()
    K = auto()
    Q = auto()
    WO = auto()
    MHA = auto()
    MLP = auto()
    # Aliases for common projections
    RoutedExpert = GATE | DOWN | UP  # switch_mlp paths


class ModuleType(IntFlag):
    """Module-level classification of a tensor path."""

    NONE = 0
    LAYER = auto()
    VISION = auto()
    AUDIO = auto()
    SSM = auto()
    SWITCH_MLP = auto()
    SHARED_EXPERT = auto()
    DENSE = auto()
    PADDING = auto()
    EMBEDDING = auto()
    EMBED_TOKENS = auto()
    LM_HEAD = auto()
    CLASSIFIER = auto()
    INPUT_LAYERNORM = auto()
    POST_LAYERNORM = auto()
    ROUTER = auto()
    GATE = auto()
    DOWN = auto()
    UP = auto()
    O = auto()
    V = auto()
    K = auto()
    Q = auto()
    WO = auto()
    MHA = auto()
    MLP = auto()
    # Aliases for common projections
    GATE_PROJ = GATE  # gate_proj
    DOWN_PROJ = DOWN  # down_proj
    UP_PROJ = UP  # up_proj


@dataclass(frozen=True, slots=True)
class TensorPath:
    """Structured representation of a tensor module path.

    Parses strings like ``"model.layers.0.mlp.experts.4.gate_proj.weight"``
    into typed attributes that replace brittle substring checks.
    """

    parts: tuple[str, ...]
    layer_idx: int | None
    module_type: ModuleType
    projection: ProjectionType
    expert_idx: int | None
    prefix: str
    _raw: str

    @classmethod
    def from_string(cls, path: str) -> "TensorPath":
        """Parse a raw module-path string into a structured TensorPath."""
        cleaned = (
            path.replace(".weight", "")
            .replace(".scales", "")
            .replace(".biases", "")
            .rstrip(".")
        )
        parts = tuple(cleaned.split("."))
        layer_idx: int | None = None
        module_type: ModuleType = ModuleType.NONE
        projection: ProjectionType = ProjectionType.NONE
        expert_idx: int | None = None

        # Layer index detection
        m = re.search(r"layers\.(\d+)", path)
        if m:
            layer_idx = int(m.group(1))

        # Module type detection
        parts_lower = [p.lower() for p in parts]
        joined = ".".join(parts_lower)

        if "switch_mlp" in joined:
            module_type = ModuleType.SWITCH_MLP
        elif "shared_expert" in joined:
            module_type = ModuleType.SHARED_EXPERT
        elif "router" in joined:
            module_type = ModuleType.ROUTER
        elif "visual." in joined or "vision_" in joined:
            module_type = ModuleType.VISION
        elif "audio_tower" in joined:
            module_type = ModuleType.AUDIO
        elif "ssm" in joined and not any(
            s in joined for s in ("alpha", "beta", "a_log", "time_decay", "time_faaaa")
        ):
            module_type = ModuleType.SSM
        elif "layers." in joined and len(parts) >= 3:
            module_type = ModuleType.LAYER
        elif "embed_tokens" in joined or "wte" in joined:
            module_type = ModuleType.EMBED_TOKENS
        elif "lm_head" in joined:
            module_type = ModuleType.LM_HEAD
        elif "classifier" in joined:
            module_type = ModuleType.CLASSIFIER
        elif "input_layernorm" in joined:
            module_type = ModuleType.INPUT_LAYERNORM
        elif "post_layernorm" in joined:
            module_type = ModuleType.POST_LAYERNORM
        elif "padd" in joined:
            module_type = ModuleType.PADDING
        elif "embed_" in joined:
            module_type = ModuleType.EMBEDDING
        elif "dela" in joined:
            module_type = ModuleType.DENSE

        # Projection type detection
        if "gate_proj" in joined:
            projection = ProjectionType.GATE
        elif "down_proj" in joined:
            projection = ProjectionType.DOWN
        elif "up_proj" in joined:
            projection = ProjectionType.UP
        elif "act" in joined and "switch_mlp" not in joined:
            projection = ProjectionType.ACT
        elif "o_proj" in joined:
            projection = ProjectionType.O
        elif "v_proj" in joined:
            projection = ProjectionType.V
        elif "k_proj" in joined:
            projection = ProjectionType.K
        elif "q_proj" in joined:
            projection = ProjectionType.Q
        elif "wo" in joined:
            projection = ProjectionType.WO
        elif "qkv_proj" in joined or "attn_qkv" in joined:
            projection = ProjectionType.MHA
        elif "mlp" in joined:
            projection = ProjectionType.MLP

        # Expert index detection
        if "experts" in joined:
            expert_match = re.search(r"experts\.(\d+)", path)
            if expert_match:
                expert_idx = int(expert_match.group(1))

        return cls(
            parts=parts,
            layer_idx=layer_idx,
            module_type=module_type,
            projection=projection,
            expert_idx=expert_idx,
            prefix=".".join(parts) if parts else "",
            _raw=path,
        )

    @property
    def is_layer(self) -> bool:
        return self.module_type == ModuleType.LAYER

    @property
    def is_vision(self) -> bool:
        return self.module_type == ModuleType.VISION

    @property
    def is_audio(self) -> bool:
        return self.module_type == ModuleType.AUDIO

    @property
    def is_ssm(self) -> bool:
        return self.module_type == ModuleType.SSM

    @property
    def is_switch_mlp(self) -> bool:
        return self.module_type == ModuleType.SWITCH_MLP

    @property
    def is_shared_expert(self) -> bool:
        return self.module_type == ModuleType.SHARED_EXPERT

    @property
    def is_router(self) -> bool:
        return self.module_type == ModuleType.ROUTER

    @property
    def is_gate_proj(self) -> bool:
        return self.projection == ProjectionType.GATE

    @property
    def is_down_proj(self) -> bool:
        return self.projection == ProjectionType.DOWN

    @property
    def is_up_proj(self) -> bool:
        return self.projection == ProjectionType.UP

    @property
    def is_o_proj(self) -> bool:
        return self.projection == ProjectionType.O

    @property
    def is_v_proj(self) -> bool:
        return self.projection == ProjectionType.V

    @property
    def is_k_proj(self) -> bool:
        return self.projection == ProjectionType.K

    @property
    def is_q_proj(self) -> bool:
        return self.projection == ProjectionType.Q

    @property
    def is_mha(self) -> bool:
        return self.projection == ProjectionType.MHA

    @property
    def is_mlp(self) -> bool:
        return self.projection == ProjectionType.MLP

    @property
    def is_routed_expert(self) -> bool:
        return bool(
            self.module_type == ModuleType.SWITCH_MLP
            or (
                self.module_type == ModuleType.NONE
                and "experts" in ".".join(self.parts)
                and "shared_expert" not in ".".join(self.parts).lower()
            )
        )

    def __str__(self) -> str:
        return self._raw

    def __repr__(self) -> str:
        return f"TensorPath({self._raw!r})"


def _extract_layer_index(path: str) -> int:
    """Extract transformer layer index from module path. Returns -1 if absent."""
    m = re.search(r"layers\.(\d+)", path)
    return int(m.group(1)) if m else -1


def _mode_for_bits(bits: int) -> str:
    """Select quantization mode. Always affine to minimize kernel combos."""
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get group_size. Always default to minimize kernel combos."""
    return default_gs


def _bits_fn_factory(base_bits: int) -> Callable[[int], dict]:
    """Create a bits-calculating function that closes over base_bits."""

    def _bits(n: int) -> dict:
        effective = int(max(n, base_bits))
        return {
            "bits": effective,
            "group_size": _OQ_DEFAULT_GROUP_SIZE,
            "mode": _mode_for_bits(effective),
        }

    return _bits
