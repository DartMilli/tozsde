"""Stability metrics helpers."""

from __future__ import annotations

import numpy as np


def compute_oos_stability(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "std": None}
    return {"count": len(values), "std": float(np.std(values))}


def compute_parameter_variance(param_runs: list[dict]) -> dict:
    if not param_runs:
        return {}
    keys = param_runs[0].keys()
    variance = {}
    for key in keys:
        vals = [p.get(key) for p in param_runs if p.get(key) is not None]
        if vals:
            variance[key] = float(np.var(vals))
    return variance


def compute_seed_variance(seed_scores: list[float]) -> dict:
    if not seed_scores:
        return {"count": 0, "std": None}
    return {"count": len(seed_scores), "std": float(np.std(seed_scores))}


def compute_dispersion_ratio(values: list[float]) -> float | None:
    if not values:
        return None
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if mean_val == 0:
        return None
    return float(std_val / abs(mean_val))
