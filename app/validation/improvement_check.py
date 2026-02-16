"""Post-validation improvement checks for deployment readiness."""

from __future__ import annotations

from app.validation.errors import DeploymentBlockedException


def evaluate_results(results: dict) -> dict:
    failures = []

    wf = results.get("walk_forward", {})
    final_score = results.get("final_score", {})
    bias = results.get("bias", {})

    mean_oos_sharpe = wf.get("mean_oos_sharpe")
    production_score = final_score.get("production_score")
    relative_gap = bias.get("relative_gap")
    sharpe_std = wf.get("sharpe_std")
    execution_flag = wf.get("execution_gap_flag")

    if not isinstance(mean_oos_sharpe, (int, float)) or mean_oos_sharpe < 0.4:
        failures.append("mean_oos_sharpe")
    if not isinstance(production_score, (int, float)) or production_score < 0.5:
        failures.append("production_score")
    if not isinstance(relative_gap, (int, float)) or relative_gap > 0.3:
        failures.append("relative_gap")
    if not isinstance(sharpe_std, (int, float)) or sharpe_std > 0.15:
        failures.append("sharpe_std")
    if execution_flag == "HIGH_TIMING_DEPENDENCY":
        failures.append("execution_gap_flag")

    status = "ok" if not failures else "failed"
    payload = {
        "status": status,
        "failures": failures,
    }

    if failures:
        raise DeploymentBlockedException(f"Deployment blocked: {', '.join(failures)}")

    return payload
