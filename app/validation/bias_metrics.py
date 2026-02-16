"""Bias metrics helpers."""

from __future__ import annotations


def compare_execution_modes(close_return: float, next_return: float) -> dict:
    if close_return == 0:
        drop = 0.0
    else:
        drop = (close_return - next_return) / abs(close_return)
    denom = max(1e-9, abs(close_return))
    relative_gap = abs(close_return - next_return) / denom
    alignment_ok = relative_gap <= 0.25
    alignment_error = relative_gap > 0.5
    return {
        "close_return": close_return,
        "next_open_return": next_return,
        "relative_drop": drop,
        "relative_gap": relative_gap,
        "alignment_ok": alignment_ok,
        "alignment_error": alignment_error,
        "lookahead_suspected": drop > 0.2,
    }


def compute_leakage_delta(original_confidence: float, strict_confidence: float) -> dict:
    if original_confidence is None or strict_confidence is None:
        return {"delta": None, "leakage_suspected": False}
    delta = float(original_confidence) - float(strict_confidence)
    return {"delta": delta, "leakage_suspected": delta > 0.05}
