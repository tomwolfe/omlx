# SPDX-License-Identifier: Apache-2.0
"""
Quantization Rule Engine — Strategy/Rule Pattern.

This module provides a chain-of-responsibility pattern for quantization
decision-making.  Each ``QuantizationRule`` encapsulates a single
concern (MoE router protection, SSM state protection, vision/audio
tensors, layer sensitivity, etc.) and the ``RuleRegistry`` class
iterates through them in priority order, returning the first match.

This replaces the monolithic ``universal_quant_predicate`` function
with composable, testable rule classes.
"""

from __future__ import annotations

import re

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from .oq_constants import TensorPath

# ---------------------------------------------------------------------------
# Protocol / interface
# ---------------------------------------------------------------------------


class QuantizationRule(Protocol):
    """Protocol for a quantization rule."""

    @property
    def priority(self) -> int: ...

    @abstractmethod
    def evaluate(
        self, path: str, config: dict, oq_level: float
    ) -> bool | dict | None: ...


# ---------------------------------------------------------------------------
# Concrete rule implementations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NonQuantizableRule:
    """Rule: skip paths listed in ``_oq_non_quantizable``."""

    priority: int = -100

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        non_quantizable = config.get("_oq_non_quantizable", set())
        if path in non_quantizable:
            return False
        return None  # no match — continue chain


@dataclass(slots=True)
class MoERouterRule:
    """Rule: MoE router / gate layers stay fp16."""

    priority: int = -90

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        tp = TensorPath.from_string(path)
        if tp.is_router or tp.is_gate_proj:
            return False
        return None


@dataclass(slots=True)
class SharedExpertRule:
    """Rule: shared expert (non-gate) stays 8-bit affine."""

    priority: int = -80

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        tp = TensorPath.from_string(path)
        if tp.is_shared_expert:
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        return None


@dataclass(slots=True)
class SSMStateRule:
    """Rule: SSM-sensitive parameters (alpha, beta, dt_bias, …) stay fp16."""

    priority: int = -70

    _SSM_SENSITIVE = (
        "ssm_alpha",
        "ssm_beta",
        "a_log",
        "time_decay",
        "time_faaaa",
    )

    _dt_bias_pattern = re.compile(r"\bdt_bias\b")

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        path_lower = path.lower()
        if any(p in path_lower for p in self._SSM_SENSITIVE):
            return False
        if self._dt_bias_pattern.search(path_lower):
            return False
        if path.endswith(".D"):
            return False
        return None


@dataclass(slots=True)
class Qwen35HybridRule:
    """Rule: Qwen3.5 hybrid dt_bias drives discretization step (fp16)."""

    priority: int = -65

    _dt_bias_pattern = re.compile(r"\bdt_bias\b")

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        path_lower = path.lower()
        if self._dt_bias_pattern.search(path_lower):
            return False
        if "conv1d" in path_lower and "linear_attn" in path_lower:
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        if "linear_attn.out_proj" in path_lower:
            return {"bits": 5, "group_size": 64, "mode": "affine"}
        return None


@dataclass(slots=True)
class VisionTensorRule:
    """Rule: vision encoder / projector tensors stay fp16."""

    priority: int = -50

    _VISION_PATTERNS = (
        "visual.",
        "vision_",
        "patch_embed",
        "pos_embed",
        "image_newline",
        "multi_modal_projector",
        "visual.merger",
        "image_norm",
        "temporal_embed",
    )

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        if any(p in path for p in self._VISION_PATTERNS):
            return False
        return None


@dataclass(slots=True)
class AudioTensorRule:
    """Rule: audio_tower tensors stay fp16."""

    priority: int = -45

    _AUDIO_PATTERNS = ("audio_tower",)

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        if any(p in path for p in self._AUDIO_PATTERNS):
            return False
        return None


@dataclass(slots=True)
class BoostMapRule:
    """Rule: honour explicit boost-map overrides from config."""

    priority: int = -40

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        boost_map = config.get("_oq_boost_map")
        if boost_map is not None and path in boost_map:
            return dict(boost_map[path])
        return None


@dataclass(slots=True)
class BudgetPlanRule:
    """Rule: budget-plan overrides (ssm_output → 8-bit, lora.2 → 8-bit)."""

    priority: int = -35

    _SSM_OUTPUT_PATTERNS = ("ssm_output", "ssm_out")

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        if not config.get("_oq_use_budget_plan"):
            return None
        # Check boost_map first - if path is in boost_map, delegate to BoostMapRule
        boost_map = config.get("_oq_boost_map")
        if boost_map is not None and path in boost_map:
            return None  # Let BoostMapRule handle this
        if any(p in path for p in self._SSM_OUTPUT_PATTERNS):
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        if "lora.2" in path:
            return {"bits": 8, "group_size": 64, "mode": "affine"}
        return True  # budget plan active — use default bits


@dataclass(slots=True)
class PolicyRule:
    """Rule: delegate to the policy registry for quantization decisions."""

    priority: int = -999  # lowest priority — last resort

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        # Import here to avoid circular dependency at import time
        from .oq_policies import get_policy_for_model

        model_type = config.get("model_type")
        policy = get_policy_for_model(model_type, config)
        return policy.evaluate(path, config, oq_level, None)


# ---------------------------------------------------------------------------
# Rule Registry
# ---------------------------------------------------------------------------


@dataclass
class RuleRegistry:
    """Priority-ordered chain-of-responsibility for quantization rules.

    Rules are evaluated in priority order (highest first).  The first rule
    that returns a non-None value short-circuits the chain.
    """

    _rules: list[QuantizationRule] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add(self, rule: QuantizationRule) -> RuleRegistry:
        """Append a rule and re-sort."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        return self

    def evaluate(self, path: str, config: dict, oq_level: float) -> bool | dict | None:
        """Iterate rules in priority order; return first non-None result."""
        for rule in self._rules:
            result = rule.evaluate(path, config, oq_level)
            if result is not None:
                return result
        return None

    def __iter__(self) -> Iterator[QuantizationRule]:
        return iter(self._rules)


# ---------------------------------------------------------------------------
# Default registry (pre-built with all built-in rules)
# ---------------------------------------------------------------------------

_default_registry: RuleRegistry | None = None


def _get_default_registry() -> RuleRegistry:
    """Return the global default registry, building it once on first access."""
    global _default_registry
    if _default_registry is None:
        reg = RuleRegistry()
        # Build-in rules in registration order (priority is per-rule)
        reg.add(NonQuantizableRule())
        reg.add(MoERouterRule())
        reg.add(SharedExpertRule())
        reg.add(SSMStateRule())
        reg.add(Qwen35HybridRule())
        reg.add(VisionTensorRule())
        reg.add(AudioTensorRule())
        reg.add(BudgetPlanRule())
        reg.add(BoostMapRule())
        reg.add(PolicyRule())
        _default_registry = reg
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (useful for testing)."""
    global _default_registry
    _default_registry = None


# ---------------------------------------------------------------------------
# Public predicate — drop-in replacement for the old universal_quant_predicate
# ---------------------------------------------------------------------------


def universal_quant_predicate(
    path: str,
    module: Any,
    config: dict,
    oq_level: int = 4,
) -> bool | dict:
    """Per-tensor quantization decision using the rule registry.

    This is the public API and **maintains the exact same contract** as the
    legacy ``universal_quant_predicate`` so that callers do **not** change.
    """
    # Normalize the path before delegating to the rule registry.
    if path.endswith(".weight") or path.endswith(".scales") or path.endswith(".biases"):
        path = path[:-7]

    result = _get_default_registry().evaluate(path, config, oq_level)
    if result is None:
        # Fallback to policy evaluation (legacy behaviour)
        from .oq_policies import get_policy_for_model

        model_type = config.get("model_type")
        policy = get_policy_for_model(model_type, config)
        return policy.evaluate(path, config, oq_level, None)
    return result  # type: ignore[return-value]  # bool | dict from rule


# ---------------------------------------------------------------------------
# Re-export for consumers
# ---------------------------------------------------------------------------

__all__ = [
    "QuantizationRule",
    "RuleRegistry",
    "NonQuantizableRule",
    "MoERouterRule",
    "SharedExpertRule",
    "SSMStateRule",
    "Qwen35HybridRule",
    "VisionTensorRule",
    "AudioTensorRule",
    "BoostMapRule",
    "BudgetPlanRule",
    "PolicyRule",
    "universal_quant_predicate",
    "reset_default_registry",
]
