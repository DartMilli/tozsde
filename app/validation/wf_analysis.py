"""Walk-forward stability analysis adapter."""

from __future__ import annotations

import json
from typing import List

import numpy as np

from app.data_access.data_manager import DataManager
from app.validation.utils import get_validation_ticker


def run_walk_forward_analysis() -> dict:
    ticker = get_validation_ticker()
    dm = DataManager()

    query = """
        SELECT result_json
        FROM walk_forward_results
        WHERE ticker = ?
        ORDER BY computed_at ASC
    """
    with dm.connection() as conn:
        rows = conn.execute(query, (ticker,)).fetchall()

    if not rows:
        return {"status": "no_data", "ticker": ticker}

    results: List[dict] = []
    for (raw,) in rows:
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    if not results:
        return {"status": "no_data", "ticker": ticker}

    raw_fitness = []
    sharpe_proxy = []
    oos_returns = []
    execution_gap_means = []
    execution_gap_maxes = []
    sharpe_stds = []
    worst_case_sharpes = []
    latest_execution_stress = None
    for item in results:
        wf_summary = item.get("wf_summary") or {}
        avg_return = wf_summary.get("avg_return")
        avg_drawdown = wf_summary.get("avg_drawdown")
        if isinstance(avg_return, (int, float)) and isinstance(
            avg_drawdown, (int, float)
        ):
            denom = abs(avg_drawdown) if avg_drawdown != 0 else 1.0
            sharpe_proxy.append(float(avg_return) / denom)
            oos_returns.append(float(avg_return))
        score = item.get("raw_fitness")
        if score is None:
            score = item.get("wf_fitness")
        if isinstance(score, (int, float)):
            raw_fitness.append(float(score))

        mean_gap = wf_summary.get("mean_execution_gap")
        max_gap = wf_summary.get("max_execution_gap")
        if isinstance(mean_gap, (int, float)):
            execution_gap_means.append(float(mean_gap))
        if isinstance(max_gap, (int, float)):
            execution_gap_maxes.append(float(max_gap))

        sharpe_std_val = wf_summary.get("mean_sharpe_std")
        if isinstance(sharpe_std_val, (int, float)):
            sharpe_stds.append(float(sharpe_std_val))
        worst_case_val = wf_summary.get("worst_case_sharpe_global")
        if isinstance(worst_case_val, (int, float)):
            worst_case_sharpes.append(float(worst_case_val))

        if isinstance(item.get("execution_stress"), dict):
            latest_execution_stress = item.get("execution_stress")

    mean_oos_sharpe = float(np.mean(sharpe_proxy)) if sharpe_proxy else 0.0
    sharpe_std = float(np.std(sharpe_proxy)) if sharpe_proxy else None
    return_std = float(np.std(oos_returns)) if oos_returns else None
    fitness_std = float(np.std(raw_fitness)) if raw_fitness else None
    stability = None
    if sharpe_std and sharpe_std > 0:
        stability = float(mean_oos_sharpe / sharpe_std)

    mean_execution_gap = (
        float(np.mean(execution_gap_means)) if execution_gap_means else None
    )
    max_execution_gap = (
        float(np.max(execution_gap_maxes)) if execution_gap_maxes else None
    )
    gap_std = float(np.std(execution_gap_means)) if execution_gap_means else None
    execution_flag = None
    if isinstance(mean_execution_gap, (int, float)) and mean_execution_gap > 0.7:
        execution_flag = "HIGH_TIMING_DEPENDENCY"

    mean_sharpe_std = float(np.mean(sharpe_stds)) if sharpe_stds else None
    worst_case_sharpe_global = min(worst_case_sharpes) if worst_case_sharpes else None

    return {
        "status": "ok",
        "ticker": ticker,
        "runs": len(results),
        "mean_oos_sharpe": round(mean_oos_sharpe, 4),
        "sharpe_std": round(sharpe_std, 6) if sharpe_std is not None else None,
        "return_std": round(return_std, 6) if return_std is not None else None,
        "fitness_std": round(fitness_std, 6) if fitness_std is not None else None,
        "stability": round(stability, 6) if stability is not None else None,
        "mean_execution_gap": (
            round(mean_execution_gap, 6) if mean_execution_gap is not None else None
        ),
        "max_execution_gap": (
            round(max_execution_gap, 6) if max_execution_gap is not None else None
        ),
        "gap_std": round(gap_std, 6) if gap_std is not None else None,
        "execution_gap_flag": execution_flag,
        "mean_sharpe_std": (
            round(mean_sharpe_std, 6) if mean_sharpe_std is not None else None
        ),
        "worst_case_sharpe_global": (
            round(worst_case_sharpe_global, 6)
            if worst_case_sharpe_global is not None
            else None
        ),
        "execution_stress": latest_execution_stress,
    }
