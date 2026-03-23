"""
scoring.py
Quant Stability Score szamitas.

Helye:
app/validation/scoring.py
"""

import math

from app.optimization.ga_wf_normalizer import normalize_drawdown, normalize_sharpe


def normalize(value, min_val, max_val):
    if value <= min_val:
        return 0
    if value >= max_val:
        return 1
    return (value - min_val) / (max_val - min_val)


def compute_quant_score(results: dict) -> dict:
    score = 0
    engine_integrity = results.get("engine_integrity", {})
    engine_valid = engine_integrity.get("status") == "ENGINE_VALID"
    engine_score = 1 if engine_valid else 0

    # --- Bias safety ---
    bias = results.get("bias", {})
    drop = bias.get("relative_drop", 1)
    bias_score = 1 - min(drop, 1)
    score += bias_score * 15

    # --- Walk-forward ---
    wf = results.get("walk_forward", {})
    sharpe = wf.get("mean_oos_sharpe", 0)
    sharpe_std = wf.get("sharpe_std")
    return_std = wf.get("return_std")

    wf_score = normalize(sharpe, 0.5, 1.5)
    if isinstance(sharpe_std, (int, float)):
        wf_score *= 1 - normalize(sharpe_std, 0.1, 0.6)
    if isinstance(return_std, (int, float)):
        wf_score *= 1 - normalize(return_std, 0.05, 0.3)
    score += wf_score * 20

    # --- RL Stability ---
    rl = results.get("rl_stability", {})
    if rl.get("status") == "NO_MODEL":
        rl_score = 0
    else:
        rl_std = rl.get("sharpe_std", 1)
        rl_score = 1 - normalize(rl_std, 0.1, 0.5)
    score += rl_score * 10

    # --- GA Robustness ---
    ga = results.get("ga_robustness", {})
    ga_var = ga.get("fitness_std", 1)
    ga_cv = ga.get("param_cv", {})
    ga_score = 1 - normalize(ga_var, 0.05, 0.3)
    if isinstance(ga_cv, dict) and ga_cv:
        cv_vals = [v for v in ga_cv.values() if isinstance(v, (int, float))]
        if cv_vals:
            cv_penalty = normalize(sum(cv_vals) / len(cv_vals), 0.1, 0.6)
            ga_score *= 1 - cv_penalty
    score += ga_score * 10

    # --- Shadow consistency ---
    shadow = results.get("shadow", {})
    diff = shadow.get("equity_diff", 1)
    shadow_score = 1 - normalize(diff, 0.01, 0.1)
    match_rate = shadow.get("action_match_rate")
    if isinstance(match_rate, (int, float)):
        shadow_score *= max(0.0, min(1.0, match_rate))
    score += shadow_score * 10

    # --- Risk ---
    risk = results.get("risk", {})
    max_dd = abs(risk.get("crash_drawdown", 1))
    risk_score = 1 - normalize(max_dd, 0.2, 0.5)

    sizing = risk.get("position_sizing", {})
    violations = sizing.get("violations", 0)
    if isinstance(violations, int) and violations > 0:
        risk_score *= 0.5

    bear_windows = risk.get("bear_windows", {})
    if isinstance(bear_windows, dict) and bear_windows:
        worst_dd = None
        for window in bear_windows.values():
            if isinstance(window, dict) and window.get("status") == "ok":
                dd = window.get("max_drawdown")
                if isinstance(dd, (int, float)):
                    worst_dd = dd if worst_dd is None else min(worst_dd, dd)
        if worst_dd is not None and abs(worst_dd) > 0.4:
            risk_score *= 0.7
    score += risk_score * 5

    execution_penalty = 0.0
    relative_gap = bias.get("relative_gap")
    if isinstance(relative_gap, (int, float)):
        execution_penalty = min(abs(relative_gap), 1.0)
        score = max(0.0, score - execution_penalty * 10)

    robustness_factor = math.exp(-2 * execution_penalty)

    final_score = round(score, 2)

    if final_score >= 75:
        status = "LIVE_READY"
    elif final_score >= 55:
        status = "PAPER_READY"
    else:
        status = "RESEARCH_ONLY"

    normalized_oos_sharpe = 0.0
    if isinstance(sharpe, (int, float)):
        normalized_oos_sharpe = normalize_sharpe(sharpe)

    stability_raw = wf.get("stability")
    stability_score = (
        normalize(stability_raw, 0.5, 2.0)
        if isinstance(stability_raw, (int, float))
        else 0.0
    )

    max_dd = risk.get("crash_drawdown")
    max_drawdown_norm = (
        normalize_drawdown(max_dd) if isinstance(max_dd, (int, float)) else 0.0
    )

    production_score = (
        0.4 * normalized_oos_sharpe
        + 0.2 * stability_score
        + 0.2 * robustness_factor
        + 0.2 * (1 - max_drawdown_norm)
    )
    production_score = round(max(0.0, min(1.0, production_score)), 4)
    deployable = production_score > 0.6

    return {
        "quant_score": final_score,
        "status": status,
        "engine_score": engine_score,
        "execution_penalty": round(execution_penalty, 4),
        "robustness_factor": round(robustness_factor, 4),
        "production_score": production_score,
        "deployable": deployable,
        "production_policy": "NEXT_OPEN",
    }
