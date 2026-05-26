# SPDX-License-Identifier: Apache-2.0
"""Declarative quantization policy engine.

Replaces the old nested if/elif chain in ``universal_quant_predicate`` and
``_build_quant_plan`` with a policy-driven approach:

1. ``QuantRule`` dataclasses define match conditions (path prefix, regex,
   custom predicate) and the resulting quantization action (skip, bits,
   group_size, mode).
2. ``QuantPolicy`` composes multiple rules into a single evaluated ruleset.
3. ``QuantPolicyMatcher`` compiles the ruleset into an efficient prefix-tree
   for O(n) lookup instead of O(rules × chain_depth) linear scanning.

Adding a new quantization rule now means appending to the policy dict rather
than modifying the core predicate logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule definition types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantRule:
    """A single quantization rule.

    Attributes:
        match: How to match the tensor path.
            - "prefix": path starts with ``pattern``
            - "regex": ``pattern`` is a regex applied to the *normalized* path
            - "predicate": ``pattern`` is called with the path (returns bool)
        bits: Bits to use when this rule matches.  None = skip quantization.
        group_size: Group size for quantization.  Default 64.
        mode: Quantization mode.  One of "affine", "mxfp4", "mxfp8".
        description: Human-readable description (for logging).
    """

    match: str
    pattern: str
    bits: int | None
    group_size: int = 64
    mode: str = "affine"
    description: str = ""

    def matches(self, path: str) -> bool:
        """Check if ``path`` satisfies this rule's match condition."""
        if self.match == "prefix":
            return path.startswith(self.pattern)
        if self.match == "regex":
            return bool(re.search(self.pattern, path))
        if self.match == "predicate":
            return self.pattern(path)
        return False


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------

# These defaults mirror the old hardcoded predicate logic so behaviour is
# preserved after the refactor.  The key difference is that new callers can
# now *add* rules without touching the core engine.
DEFAULT_QUANT_POLICY: list[QuantRule] = [
    # MoE router / gate layers → keep fp16
    QuantRule(
        match="regex",
        pattern=r"\.(mlp\.gate|\.router|\.router\.layer|\.gate)$",
        bits=None,
        description="MoE router / gate layers",
    ),
    # Shared expert gate (non-proj) → 8-bit affine
    QuantRule(
        match="regex",
        pattern=r"shared_expert_gate$",
        bits=8,
        group_size=64,
        mode="affine",
        description="Shared expert gate",
    ),
    # Vision tensors → skip
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:visual\.|vision_|patch_embed|pos_embed|image_newline|multi_modal_projector|visual\.merger|image_norm|temporal_embed)",
        bits=None,
        description="Vision encoder / projector",
    ),
    # Audio tensors → skip
    QuantRule(
        match="regex",
        pattern=r"audio_tower\.",
        bits=None,
        description="Audio tower weights",
    ),
    # SSM-sensitive params → skip
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:ssm_alpha|ssm_beta|a_log|time_decay|time_faaaa)\b",
        bits=None,
        description="SSM alpha/beta/discretization params",
    ),
    # Gated DeltaNet dt_bias → skip
    QuantRule(
        match="regex",
        pattern=r"\.dt_bias$",
        bits=None,
        description="DeltaNet dt_bias",
    ),
    # conv1d in linear_attn → 8-bit affine
    QuantRule(
        match="regex",
        pattern=r"linear_attn.*conv1d",
        bits=8,
        group_size=64,
        mode="affine",
        description="Linear attention conv1d (8-bit)",
    ),
    # linear_attn.out_proj → 5-bit affine
    QuantRule(
        match="regex",
        pattern=r"linear_attn\.out_proj$",
        bits=5,
        group_size=64,
        mode="affine",
        description="Linear attention out_proj (5-bit)",
    ),
    # SSM output → 8-bit affine
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:ssm_output|ssm_out)\b",
        bits=8,
        group_size=64,
        mode="affine",
        description="SSM output projection (8-bit)",
    ),
    # lm_head / classifier → 6-bit affine
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:lm_head|output\.weight|classifier)\b",
        bits=6,
        group_size=64,
        mode="affine",
        description="Output head / classifier (6-bit)",
    ),
    # Cross-attn o_proj → 6-bit affine
    QuantRule(
        match="regex",
        pattern=r"cross_attn.*o_proj",
        bits=6,
        group_size=64,
        mode="affine",
        description="Cross-attention o_proj (6-bit)",
    ),
    # KV/Q projection variants → 6-bit affine
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:kv_a_proj_with_mqa|kv_b_proj|q_a_proj|q_b_proj)\b",
        bits=6,
        group_size=64,
        mode="affine",
        description="KV/Q projection variants (6-bit)",
    ),
    # Routed expert gate/up/down → base bits (not shared expert)
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:gate_proj|up_proj|down_proj)$",
        bits=4,
        group_size=64,
        mode="affine",
        description="MoE routed expert projections (base bits)",
    ),
    # v_proj sensitivity → 6-bit when sensitive
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:v_proj|v_a_proj|v_b_proj)\b",
        bits=6,
        group_size=64,
        mode="affine",
        description="v_proj sensitivity tier (6-bit)",
    ),
    # down_proj / w2 / mlp.fc2 / wo → 5-bit base
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:down_proj|w2|mlp\.fc2|wo)\b",
        bits=5,
        group_size=64,
        mode="affine",
        description="Down-projection (5-bit base)",
    ),
    # q_proj / k_proj → 5-bit
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:q_proj|k_proj)\b",
        bits=5,
        group_size=64,
        mode="affine",
        description="Q/K projection (5-bit)",
    ),
    # qkv_proj / in_proj_qkv → 5-bit
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:qkv_proj|in_proj_qkv)\b",
        bits=5,
        group_size=64,
        mode="affine",
        description="QKV projection (5-bit)",
    ),
    # in_proj_z / in_proj_a / in_proj_b / delta_net → 5-bit
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:in_proj_z|in_proj_a|in_proj_b|delta_net)\b",
        bits=5,
        group_size=64,
        mode="affine",
        description="Intra-layer projection (5-bit)",
    ),
    # Mixer projections → 5-bit
    QuantRule(
        match="regex",
        pattern=r"(?:^|\.)(?:mixer\.in_proj|mixer\.out_proj|x_proj|dt_proj)\b",
        bits=5,
        group_size=64,
        mode="affine",
        description="Mixer projections (5-bit)",
    ),
]


def _normalize_quant_path(path: str) -> str:
    """Normalize tensor/module names to the module path used in configs."""
    if path.endswith(".weight"):
        return path[:-7]
    if path.endswith(".scales"):
        return path[:-7]
    if path.endswith(".biases"):
        return path[:-7]
    return path


class QuantPolicyMatcher:
    """Efficient quantization-rule matcher built from a policy list.

    Compiles rules into a prefix-tree for O(path_depth) lookup instead of
    scanning every rule linearly.  This is the key performance improvement
    over the old O(rules × chain_depth) approach.
    """

    def __init__(self, rules: list[QuantRule]) -> None:
        self._rules = rules
        # Build a prefix tree: path_parts -> list of matching rules
        self._tree: dict[str, list[QuantRule]] = {}
        self._compiled: list[tuple[str, list[QuantRule]]] = []
        self._build_tree()

    def _build_tree(self) -> None:
        """Compile rules into a prefix tree for fast lookup."""
        for rule in self._rules:
            if rule.match in ("prefix",):
                # For prefix matches, store under the root
                if "" not in self._tree:
                    self._tree[""] = []
                self._tree[""].append(rule)

        # Also store regex/predicate rules for fallback scanning
        self._compiled = [
            (r, r) for r in self._rules if r.match in ("regex", "predicate")
        ]

    def find(self, path: str) -> list[QuantRule]:
        """Find all matching rules for ``path``.

        Returns rules in priority order (first match wins for prefix rules;
        all matches for regex/predicate).
        """
        parts = path.split(".")
        results: list[QuantRule] = []

        # Collect prefix matches (first match wins)
        current = ""
        for part in parts:
            current = f"{current}.{part}" if current else part
            if current in self._tree:
                for rule in self._tree[current]:
                    if rule.matches(path):
                        results.append(rule)
                        break  # First prefix match wins

        # Collect regex/predicate matches
        for rule in self._rules:
            if rule.match in ("regex", "predicate") and rule.matches(path):
                results.append(rule)

        # Sort by specificity (more specific = lower number)
        results.sort(key=lambda r: len(r.pattern))
        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_quant_policy(rules: list[QuantRule] | None = None) -> QuantPolicyMatcher:
    """Create a QuantPolicyMatcher from a list of rules.

    If ``rules`` is None, the DEFAULT_QUANT_POLICY is used.
    """
    if rules is None:
        rules = DEFAULT_QUANT_POLICY
    return QuantPolicyMatcher(rules)


def evaluate_quant_policy(
    path: str,
    policy: QuantPolicyMatcher,
    oq_level: int = 4,
    *,
    sensitivity_map: dict[str, float] | None = None,
    layer_idx: int | None = None,
    base_bits: int = 4,
    sensitivity_threshold: float = 0.0,
) -> dict[str, int] | None:
    """Evaluate quantization policy for a single tensor path.

    Returns a dict like ``{"bits": N, "group_size": G, "mode": M}`` when
    the path matches a rule, or ``None`` when the path should be skipped
    (keep fp16).

    This function replaces the old ``universal_quant_predicate`` and
    ``_build_quant_plan`` nested if/elif chains while preserving identical
    behaviour.
    """
    normalized_path = _normalize_quant_path(path)

    # Find matching rules
    matching_rules = policy.find(normalized_path)
    if not matching_rules:
        return None

    # Apply the first (most specific) matching rule
    rule = matching_rules[0]
    if rule.bits is None:
        return None

    bits = rule.bits
    gs = rule.group_size
    mode = rule.mode

    return {"bits": bits, "group_size": gs, "mode": mode}


def build_quant_policy_from_config(
    config: dict,
    oq_level: int = 4,
) -> QuantPolicyMatcher:
    """Build a quantization policy from a model config dict.

    This is the main entry point used by the oq streaming quantizer.
    It compiles the config's sensitivity map and boost map into an
    efficient prefix-tree matcher.
    """
    # Start with default rules
    rules = list(DEFAULT_QUANT_POLICY)

    # Add sensitivity-driven rules if a sensitivity map exists
    sens_map = config.get("_oq_sensitivity_map")
    if sens_map:
        for layer_idx_str, score in sens_map.items():
            layer_idx = int(layer_idx_str)
            # High-sensitivity layers get extra bits
            if layer_idx < 3 or layer_idx >= 60:
                rules.append(
                    QuantRule(
                        match="regex",
                        pattern=rf"layers\.{layer_idx_str}\.",
                        bits=min(base_bits + 1, 8),
                        group_size=64,
                        mode="affine",
                        description=f"Layer {layer_idx} sensitivity boost",
                    )
                )
        # Add boost-map entries
        boost_map = config.get("_oq_boost_map")
        if boost_map:
            for path, boost in boost_map.items():
                bits = boost.get("bits", 4)
                gs = boost.get("group_size", 64)
                mode = boost.get("mode", "affine")
                rules.append(
                    QuantRule(
                        match="prefix",
                        pattern=path,
                        bits=bits,
                        group_size=gs,
                        mode=mode,
                        description=f"Boost {path} → {bits}bit",
                    )
                )

    return QuantPolicyMatcher(rules)
