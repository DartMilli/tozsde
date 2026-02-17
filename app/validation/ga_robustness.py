"""GA robustness adapter using stored walk-forward results."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from app.data_access.data_manager import DataManager
from app.validation.utils import get_validation_ticker
from app.config.config import Config


def _collect_param_variance(results: List[Dict]) -> Dict[str, float]:
    params_list = []
    for item in results:
        params = item.get("best_params") or {}
        if params:
            params_list.append(params)

    if not params_list:
        return {}

    variance = {}
    keys = params_list[0].keys()
    for key in keys:
        values = [p.get(key) for p in params_list if p.get(key) is not None]
        if values:
            variance[key] = float(np.var(values))
    return variance


def _collect_param_cv(results: List[Dict]) -> Dict[str, float]:
    params_list = []
    for item in results:
        params = item.get("best_params") or {}
        if params:
            params_list.append(params)

    if not params_list:
        return {}

    cv = {}
    keys = params_list[0].keys()
    for key in keys:
        values = [p.get(key) for p in params_list if p.get(key) is not None]
        if not values:
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        if mean_val == 0:
            cv[key] = None
        else:
            cv[key] = float(std_val / abs(mean_val))
    return cv


def run_ga_robustness_tests() -> dict:
    ticker = get_validation_ticker()
    dm = DataManager()

    query = """
        SELECT result_json, computed_at
        FROM walk_forward_results
        WHERE ticker = ?
        ORDER BY computed_at ASC
    """
    with dm.connection() as conn:
        rows = conn.execute(query, (ticker,)).fetchall()

    if not rows:
        return {"status": "no_data", "ticker": ticker}

    results: List[dict] = []
    for raw, computed_at in rows:
        try:
            payload = json.loads(raw)
            payload["computed_at"] = computed_at
            results.append(payload)
        except json.JSONDecodeError:
            continue

    if not results:
        return {"status": "no_data", "ticker": ticker}

    if Config.AGGREGATION_MODE == "latest_only":
        if results:
            latest = max(results, key=lambda r: r.get("computed_at") or "")
            run_id = latest.get("wf_run_id")
            if run_id:
                results = [r for r in results if r.get("wf_run_id") == run_id]
            else:
                results = [latest]

    fitness_vals = []
    seed_fitness = {}
    robustness_vals = []
    production_scores = []
    stress_scores = []
    stress_sharpe_stds = []
    stress_worst_cases = []
    stress_passed = []
    for item in results:
        score = item.get("raw_fitness")
        if score is None:
            score = item.get("wf_fitness")
        if isinstance(score, (int, float)):
            fitness_vals.append(float(score))
            seed_value = item.get("ga_seed") or item.get("seed")
            if seed_value is not None:
                seed_fitness.setdefault(seed_value, []).append(float(score))

        robustness = item.get("robustness_factor")
        if isinstance(robustness, (int, float)):
            robustness_vals.append(float(robustness))

        prod_score = item.get("production_score")
        if isinstance(prod_score, (int, float)):
            production_scores.append(float(prod_score))

        stress = item.get("execution_stress") or {}
        stress_score = stress.get("robustness_score")
        if isinstance(stress_score, (int, float)):
            stress_scores.append(float(stress_score))
        stress_std = stress.get("sharpe_std")
        if isinstance(stress_std, (int, float)):
            stress_sharpe_stds.append(float(stress_std))
        stress_worst = stress.get("worst_case_sharpe")
        if isinstance(stress_worst, (int, float)):
            stress_worst_cases.append(float(stress_worst))
        stress_passed.append(bool(stress.get("constraint_passed")) if stress else False)

    fitness_std = float(np.std(fitness_vals)) if fitness_vals else None
    param_variance = _collect_param_variance(results)
    param_cv = _collect_param_cv(results)
    cv_flags = []
    for key, value in (param_cv or {}).items():
        if isinstance(value, (int, float)) and value > 0.4:
            cv_flags.append(key)

    seed_means = [float(np.mean(vals)) for vals in seed_fitness.values() if vals]
    seed_std = float(np.std(seed_means)) if seed_means else None

    return {
        "status": "ok",
        "ticker": ticker,
        "runs": len(results),
        "fitness_std": round(fitness_std, 6) if fitness_std is not None else None,
        "param_variance": param_variance,
        "param_cv": param_cv,
        "param_cv_flags": cv_flags,
        "seed_runs": len(seed_means),
        "seed_fitness_std": round(seed_std, 6) if seed_std is not None else None,
        "robustness_mean": (
            round(float(np.mean(robustness_vals)), 6) if robustness_vals else None
        ),
        "production_score_mean": (
            round(float(np.mean(production_scores)), 6) if production_scores else None
        ),
        "stress_robustness_mean": (
            round(float(np.mean(stress_scores)), 6) if stress_scores else None
        ),
        "stress_sharpe_std_mean": (
            round(float(np.mean(stress_sharpe_stds)), 6) if stress_sharpe_stds else None
        ),
        "stress_worst_case_min": (
            round(float(min(stress_worst_cases)), 6) if stress_worst_cases else None
        ),
        "stress_pass_rate": (
            round(float(np.mean(stress_passed)), 6) if stress_passed else None
        ),
    }
