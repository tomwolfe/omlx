# SPDX-License-Identifier: Apache-2.0
"""Architecture Registry: declarative, strategy-based model detection.

Replaces the old if/elif/hasattr heuristic chain with a registry pattern
where each architecture's detection strategy is a standalone, composable
strategy object.  Adding a new model now means registering a new strategy
instead of modifying the core detection logic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol

# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class QueryExtractor(Protocol):
    """Protocol for query-extractor functions.

    Each extractor receives (attn, x, cache, **kwargs) and returns
    an mx.array of shape (B, L, n_heads, head_dim).
    """

    def __call__(
        self,
        attn: object,
        x: object,
        cache: object | None = None,
        /,
        **kwargs: object,
    ) -> object: ...


# ---------------------------------------------------------------------------
# Built-in extractors (one per architecture family)
# ---------------------------------------------------------------------------


def _qwen35_extract_queries(attn, x, cache=None, **kwargs):
    """Qwen3.5: gate split + q_norm + RoPE."""
    B, L, D = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _qwen36_extract_queries(attn, x, cache=None, **kwargs):
    """Qwen3.6 MoE / non-gated q_norm models: q_proj + q_norm + RoPE."""
    B, L, _ = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "num_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _llama_extract_queries(attn, x, cache=None, **kwargs):
    """Standard transformer: q_proj + reshape + RoPE."""
    B, L, D = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _gemma4_extract_queries(attn, x, cache=None, offset=None, **kwargs):
    """Gemma 4: q_proj + per-head q_norm + RoPE with shared-KV offset support."""
    B, L, D = x.shape
    n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    rope_offset = (
        offset if offset is not None else (cache.offset if cache is not None else 0)
    )
    queries = attn.rope(queries, offset=rope_offset)
    return queries


def _nemotron_h_extract_queries(attn, x, cache=None, **kwargs):
    """Nemotron-H: q_proj only, no RoPE (content-based attention)."""
    B, L, D = x.shape
    queries = attn.q_proj(x).reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    return queries


# ---------------------------------------------------------------------------
# Detection predicates (one per architecture family)
# ---------------------------------------------------------------------------


def _uses_gated_q_proj(attn_obj) -> bool:
    """Detect Qwen-style gated q_proj layout from projection dimensions."""
    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    head_dim = getattr(attn_obj, "head_dim", None)
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if None in (n_heads, head_dim, q_out):
        return False
    return q_out == 2 * n_heads * head_dim


def _uses_non_gated_q_norm(attn_obj) -> bool:
    """Detect Qwen3.6-style attention: per-head RMSNorm on q, no gate split."""
    q_norm = getattr(attn_obj, "q_norm", None)
    if q_norm is None:
        return False
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_norm_weight is None:
        return False
    shape = getattr(q_norm_weight, "shape", None)
    if not shape:
        return False
    q_norm_dim = shape[0]

    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if not n_heads or q_out is None:
        return False
    return q_out == n_heads * q_norm_dim


def _is_gemma4_attention(attn_obj) -> bool:
    """Detect Gemma 4 attention by its shared-KV / offset call contract."""
    if not hasattr(attn_obj, "q_norm") or not hasattr(attn_obj, "rope"):
        return False
    call = getattr(attn_obj, "__call__", None)
    if call is None:
        return False
    try:
        params = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False
    return "shared_kv" in params and "offset" in params


def _accepts_extractor_kwargs(extractor, kwargs) -> bool:
    """Whether a query extractor accepts the supplied keyword arguments."""
    try:
        params = inspect.signature(extractor).parameters.values()
    except (TypeError, ValueError):
        return True

    accepted = {
        param.name
        for param in params
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return set(kwargs).issubset(accepted)


def _linear_output_dims(linear) -> int | None:
    """Best-effort output dimension lookup for Linear / QuantizedLinear layers."""
    if linear is None:
        return None
    if hasattr(linear, "out_features"):
        return getattr(linear, "out_features")
    if hasattr(linear, "output_dims"):
        return getattr(linear, "output_dims")
    weight = getattr(linear, "weight", None)
    if weight is not None and getattr(weight, "shape", None):
        return weight.shape[0]
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping: model-family tag -> (detection_pred, query_extractor_fn)
# Adding a new architecture only requires appending to this dict.
ARCHITECTURE_REGISTRY: dict[str, tuple[Callable, QueryExtractor]] = {
    "qwen35": (
        lambda obj: _uses_gated_q_proj(obj),
        _qwen35_extract_queries,
    ),
    "qwen36_moe": (
        lambda obj: _uses_non_gated_q_norm(obj),
        _qwen36_extract_queries,
    ),
    "gemma4": (
        _is_gemma4_attention,
        _gemma4_extract_queries,
    ),
    "nemotron_h": (
        lambda obj: _linear_output_dims(getattr(attn_obj := None, "num_heads", None)) is not None and False,  # noqa: E731  # always-false sentinel
        _nemotron_h_extract_queries,
    ),
    "llama": (
        lambda obj: True,  # default fallback
        _llama_extract_queries,
    ),
}


def detect_query_extractor(attn_obj) -> QueryExtractor:
    """Auto-detect the appropriate query extractor for the model architecture.

    Walks the registry in insertion order and returns the first extractor
    whose detection predicate matches ``attn_obj``.
    """
    for _pred, extractor in ARCHITECTURE_REGISTRY.values():
        try:
            if _pred(attn_obj):
                return extractor
        except Exception:
            # Silently skip broken predicates; fall through to next.
            continue

    # Last resort: return the default (llama) extractor.
    return _llama_extract_queries
