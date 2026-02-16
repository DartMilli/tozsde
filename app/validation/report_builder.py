"""Markdown report builder for validation results."""

import json


def _section(title: str, payload: dict) -> str:
    return "\n".join(
        [
            f"## {title}",
            "```json",
            json.dumps(payload, indent=2, default=str),
            "```",
            "",
        ]
    )


def build_markdown_report(results: dict) -> str:
    header = [
        "# Quant Validation Report",
        "",
    ]

    final_score = results.get("final_score", {})
    shadow = results.get("shadow", {})
    match_rate = shadow.get("action_match_rate")
    match_text = "n/a"
    if isinstance(match_rate, (int, float)):
        match_text = f"{match_rate:.2%}"

    summary = [
        "## Executive Summary",
        f"- Quant Score: {final_score.get('quant_score', 'n/a')}",
        f"- Status: {final_score.get('status', 'unknown')}",
        f"- Production Score: {final_score.get('production_score', 'n/a')}",
        f"- Deployable: {final_score.get('deployable', 'n/a')}",
        f"- Shadow Action Match: {match_text}",
        "",
    ]

    recommendation = "RESEARCH_ONLY"
    status = final_score.get("status")
    if status == "LIVE_READY":
        recommendation = "GO_LIVE"
    elif status == "PAPER_READY":
        recommendation = "PAPER_ONLY"

    execution_stress = {}
    wf = results.get("walk_forward", {})
    if isinstance(wf, dict):
        execution_stress = wf.get("execution_stress") or {}

    sections = [
        _section("Engine Integrity", results.get("engine_integrity", {})),
        _section("Bias Audit", results.get("bias", {})),
        _section("Execution Sensitivity", results.get("execution_sensitivity", {})),
        _section(
            "Execution Stress",
            {
                "robustness_score": execution_stress.get("robustness_score"),
                "sharpe_std": execution_stress.get("sharpe_std"),
                "worst_case_sharpe": execution_stress.get("worst_case_sharpe"),
                "constraint_passed": execution_stress.get("constraint_passed"),
                "stress_tested": execution_stress.get("stress_tested"),
            },
        ),
        _section("Walk Forward", results.get("walk_forward", {})),
        _section("RL Stability", results.get("rl_stability", {})),
        _section("GA Robustness", results.get("ga_robustness", {})),
        _section("Shadow Consistency", results.get("shadow", {})),
        _section("Risk Stress", results.get("risk", {})),
        _section(
            "Production Readiness",
            {
                "production_policy": final_score.get("production_policy"),
                "robustness_factor": final_score.get("robustness_factor"),
                "production_score": final_score.get("production_score"),
                "deployable": final_score.get("deployable"),
            },
        ),
        _section("Final Score", final_score),
        _section("Validation Status", {"status": results.get("validation_status")}),
        _section("Recommendation", {"recommendation": recommendation}),
    ]

    return "\n".join(header + summary + sections)
