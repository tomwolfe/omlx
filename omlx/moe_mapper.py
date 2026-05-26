# SPDX-License-Identifier: Apache-2.0
"""MoE Tensor Mapper: abstracted tensor manipulation utilities.

Eliminates duplicated iteration over ("gate_proj", "up_proj", "down_proj")
and ("weight", "scales", "biases") across multiple patches.

Usage:
    mapper = MoETensorMapper()
    for tensor in mapper.iterate(expert_weights):
        # unified processing
"""

from __future__ import annotations

from typing import Protocol

# Projections that belong to routed MoE experts (not shared experts).
MOE_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Standard weight tensor name patterns.
WEIGHT_PATTERNS = ("weight", "scales", "biases")

# Expert iteration patterns — the order matters for stacked tensors.
EXPERT_LAYERS = ("switch_mlp", "block_sparse_moe")


class TensorMapper(Protocol):
    """Protocol for tensor mapping operations."""

    def map(self, data: dict[str, object]) -> dict[str, object]: ...


class MoETensorMapper:
    """Unified MoE tensor manipulation utility.

    Instead of repeating loops over expert projections across multiple
    files, use this mapper to abstract the common patterns:

    - Iterate over expert layers (switch_mlp, block_sparse_moe)
    - Map weight/scales/biases to their normalized forms
    - Filter out shared_expert tensors
    """

    def __init__(
        self,
        expert_projections: tuple[str, ...] = MOE_EXPERT_PROJECTIONS,
        weight_patterns: tuple[str, ...] = WEIGHT_PATTERNS,
        expert_layers: tuple[str, ...] = EXPERT_LAYERS,
    ):
        self._expert_projections = expert_projections
        self._weight_patterns = weight_patterns
        self._expert_layers = expert_layers

    def is_expert_tensor(self, path: str) -> bool:
        """Check if path belongs to a routed MoE expert."""
        return any(
            layer in path or path.endswith(f".{layer}")
            for layer in self._expert_layers
        ) and "shared_expert" not in path

    def is_shared_expert(self, path: str) -> bool:
        """Check if path belongs to shared expert (not routed)."""
        return "shared_expert" in path

    def iterate_experts(
        self,
        weights: dict[str, object],
        *,
        skip_shared: bool = True,
    ) -> dict[str, object]:
        """Yield only routed expert weights, optionally skipping shared experts.

        Args:
            weights: Raw weight dict.
            skip_shared: If True, exclude shared_expert paths.

        Returns:
            Filtered weight dict containing only expert tensors.
        """
        result = {}
        for path, value in weights.items():
            if skip_shared and self.is_shared_expert(path):
                continue
            if not self.is_expert_tensor(path):
                continue
            result[path] = value
        return result

    def map_weights(self, weights: dict[str, object]) -> dict[str, object]:
        """Normalize weight names: strip .weight/.scales/.biases suffixes.

        Returns a dict with normalized keys (e.g. "layers.0.mlp.gate"
        instead of "layers.0.mlp.gate.weight").
        """
        result = {}
        for name, value in weights.items():
            norm = self._normalize(name)
            result[norm] = value
        return result

    def _normalize(self, name: str) -> str:
        """Strip common suffixes for normalization."""
        for suffix in (".weight", ".scales", ".biases"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def filter_by_patterns(
        self,
        weights: dict[str, object],
        patterns: tuple[str, ...],
    ) -> dict[str, object]:
        """Filter weights by name patterns (e.g. gate_proj, up_proj, down_proj).

        Returns weights whose keys contain any of the provided patterns.
        """
        return {
            k: v
            for k, v in weights.items()
            if any(p in k for p in patterns)
        }

    def filter_by_layers(
        self,
        weights: dict[str, object],
        layer_indices: tuple[int, ...],
    ) -> dict[str, object]:
        """Filter weights by layer indices (e.g. layers.0, layers.1, ...).

        Returns weights whose keys match any of the specified layer indices.
        """
        result = {}
        for path, value in weights.items():
            for idx in layer_indices:
                if f"layers.{idx}." in path:
                    result[path] = value
                    break
        return result
