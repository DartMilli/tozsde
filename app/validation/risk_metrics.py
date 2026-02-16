"""Risk metrics helpers."""

from __future__ import annotations

import math


def compute_allocation_violations(
    allocations: list[float], max_pct: float = 1.0
) -> dict:
    if not allocations:
        return {"violations": 0}
    violations = sum(1 for a in allocations if a > max_pct)
    return {"violations": violations}


def compute_crash_drawdown(equity_curve: list[float]) -> dict:
    if not equity_curve:
        return {"max_drawdown": 0.0}
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
    return {"max_drawdown": float(max_dd)}


def compute_volatility_spike_response(
    returns: list[float], spike_threshold: float = 2.0
) -> dict:
    if not returns:
        return {"spike_count": 0, "avg_spike": 0.0}
    mean_val = sum(returns) / len(returns)
    var = sum((r - mean_val) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return {"spike_count": 0, "avg_spike": 0.0}
    spikes = [r for r in returns if abs(r - mean_val) > spike_threshold * std]
    avg_spike = sum(spikes) / len(spikes) if spikes else 0.0
    return {"spike_count": len(spikes), "avg_spike": float(avg_spike)}
