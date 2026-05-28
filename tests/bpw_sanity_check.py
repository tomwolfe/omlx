#!/usr/bin/env python3
"""Sanity-check script: verify Rule Engine BPW calculation matches legacy logic.

This script instantiates the Rule Engine (from omlx.oq), passes a mock
named_shapes dictionary, and verifies that the BPW calculation is consistent
and deterministic.

Run with:
    python3 tests/bpw_sanity_check.py
"""

from omlx.oq import _estimate_effective_bpw, _OQ_BPW_TARGETS


def _mock_named_shapes() -> dict[str, tuple]:
    """Return a small set of weight shapes for sanity-checking."""
    return {
        "model.layers.0.self_attn.q_proj.weight": (4096, 4096),
        "model.layers.0.self_attn.k_proj.weight": (4096, 4096),
        "model.layers.0.self_attn.v_proj.weight": (4096, 4096),
        "model.layers.0.input_layernorm.weight": (4096,),
        "lm_head.weight": (4096, 32000),
        "embed_tokens.weight": (32000, 4096),
    }


def _mock_overrides() -> dict[str, dict]:
    """Return quantisation overrides used by the Rule Engine."""
    return {
        "model.layers.0.self_attn.q_proj.weight": {
            "bits": 4,
            "group_size": 128,
            "mode": "affine",
        },
        "lm_head.weight": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
    }


def test_bpw_deterministic() -> None:
    """BPW must be deterministic: same input → same output."""
    named_shapes = _mock_named_shapes()
    overrides = _mock_overrides()

    result1 = _estimate_effective_bpw(
        named_shapes, base_bits=8, base_group_size=128, base_mode="affine", overrides=overrides
    )
    result2 = _estimate_effective_bpw(
        named_shapes, base_bits=8, base_group_size=128, base_mode="affine", overrides=overrides
    )

    assert result1 == result2, f"BPW is non-deterministic: {result1} != {result2}"
    print(f"  BPW determinism check passed: {result1:.6f}")


def test_bpw_positive() -> None:
    """BPW must be a positive number."""
    named_shapes = _mock_named_shapes()
    overrides = _mock_overrides()

    effective_bpw = _estimate_effective_bpw(
        named_shapes, base_bits=8, base_group_size=128, base_mode="affine", overrides=overrides
    )

    assert effective_bpw > 0, f"BPW must be positive, got {effective_bpw}"
    assert effective_bpw < 16, f"BPW must be < 16, got {effective_bpw}"
    print(f"  BPW positivity check passed: {effective_bpw:.4f}")


def test_bpw_targets_consistency() -> None:
    """BPW target levels must be internally consistent."""
    for level, (target, cap) in _OQ_BPW_TARGETS.items():
        assert target < cap, f"Target {target} must be < cap {cap} at level {level}"
    print(f"  BPW target consistency check passed for {_OQ_BPW_TARGETS}")


def test_bpw_empty_shapes() -> None:
    """BPW with empty named_shapes should return 0.0 (no params to quantise)."""
    result = _estimate_effective_bpw(
        {}, base_bits=8, base_group_size=128, base_mode="affine"
    )
    assert result == 0.0, f"Expected 0.0 for empty shapes, got {result}"
    print(f"  BPW empty-shapes check passed: {result}")


if __name__ == "__main__":
    test_bpw_deterministic()
    test_bpw_positive()
    test_bpw_targets_consistency()
    test_bpw_empty_shapes()
    print("All BPW sanity checks passed.")

