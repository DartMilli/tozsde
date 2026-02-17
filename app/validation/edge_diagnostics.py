"""Edge filter diagnostics helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, percentile))


def compute_fold_edge_stats(
    fold_id: int,
    raw_signal_count: int,
    expected_edges: Iterable[float],
    thresholds: Iterable[float],
) -> dict:
    edges = [float(v) for v in expected_edges if v is not None]
    threshold_vals = [float(v) for v in thresholds if v is not None and v > 0]

    pairs = list(zip(edges, threshold_vals))
    distance_ratios = [edge / thr for edge, thr in pairs if thr > 0]

    def _count_above(level: float) -> int:
        return sum(1 for edge, thr in pairs if thr > 0 and edge >= thr * level)

    signals_above = _count_above(1.0)
    within_10 = sum(1 for r in distance_ratios if 0.9 <= r < 1.0)
    within_20 = sum(1 for r in distance_ratios if 0.8 <= r < 1.0)

    sensitivity_levels = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    threshold_sensitivity = {
        f"{level:.2f}": _count_above(level) for level in sensitivity_levels
    }

    return {
        "fold_id": int(fold_id),
        "raw_signal_count": int(raw_signal_count),
        "edge_threshold": float(np.mean(threshold_vals)) if threshold_vals else None,
        "expected_edge_values": edges,
        "mean_expected_edge": float(np.mean(edges)) if edges else None,
        "std_expected_edge": float(np.std(edges)) if edges else None,
        "min_expected_edge": float(min(edges)) if edges else None,
        "max_expected_edge": float(max(edges)) if edges else None,
        "p25": _percentile(edges, 25),
        "p50": _percentile(edges, 50),
        "p75": _percentile(edges, 75),
        "p90": _percentile(edges, 90),
        "signals_above_threshold": int(signals_above),
        "signals_within_10pct_of_threshold": int(within_10),
        "signals_within_20pct_of_threshold": int(within_20),
        "threshold_sensitivity": threshold_sensitivity,
        "distance_ratios": distance_ratios,
    }


def compute_global_summary(fold_stats: Iterable[dict]) -> dict:
    folds = [f for f in fold_stats if isinstance(f, dict)]
    total_raw = sum(int(f.get("raw_signal_count", 0)) for f in folds)
    total_above = sum(int(f.get("signals_above_threshold", 0)) for f in folds)

    thresholds = [
        f.get("edge_threshold") for f in folds if f.get("edge_threshold") is not None
    ]
    edges = []
    ratios = []
    sensitivity_agg = {
        "1.00": 0,
        "0.95": 0,
        "0.90": 0,
        "0.85": 0,
        "0.80": 0,
        "0.75": 0,
        "0.70": 0,
    }

    for fold in folds:
        edges.extend(fold.get("expected_edge_values", []))
        ratios.extend(fold.get("distance_ratios", []))
        sensitivity = fold.get("threshold_sensitivity", {})
        for key in sensitivity_agg:
            if key in sensitivity:
                sensitivity_agg[key] += int(sensitivity[key])

    global_mean_edge = float(np.mean(edges)) if edges else None
    global_std_edge = float(np.std(edges)) if edges else None
    global_p90_ratio = float(np.percentile(ratios, 90)) if ratios else None
    global_max_ratio = float(max(ratios)) if ratios else None

    return {
        "total_raw_signals": int(total_raw),
        "total_signals_above_threshold": int(total_above),
        "mean_edge_threshold": float(np.mean(thresholds)) if thresholds else None,
        "global_mean_expected_edge": global_mean_edge,
        "global_std_expected_edge": global_std_edge,
        "global_p90_distance_ratio": global_p90_ratio,
        "global_max_distance_ratio": global_max_ratio,
        "threshold_sensitivity_aggregate": sensitivity_agg,
    }


def classify_collapse_reason(summary: dict) -> str | None:
    max_ratio = summary.get("global_max_distance_ratio")
    p90_ratio = summary.get("global_p90_distance_ratio")

    if not isinstance(max_ratio, (int, float)) or not isinstance(
        p90_ratio, (int, float)
    ):
        return None
    if max_ratio < 0.7:
        return "no_edge_present"
    if p90_ratio < 0.9:
        return "weak_edge_distribution"
    # No collapse reason detected for strong edge distributions
    return None
