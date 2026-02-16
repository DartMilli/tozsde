"""Post-upgrade diagnostics for stability framework."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from app.data_access.data_manager import DataManager
from app.validation.trade_quality_analysis import analyze_trade_quality
from app.validation.utils import get_validation_ticker


def _load_validation_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_report_path() -> Optional[str]:
    reports_dir = "validation_reports"
    if not os.path.isdir(reports_dir):
        return None
    candidates = [
        f
        for f in os.listdir(reports_dir)
        if f.startswith("validation_") and f.endswith(".json")
    ]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(reports_dir, candidates[-1])


def _load_latest_walk_forward_result(ticker: str) -> Optional[dict]:
    dm = DataManager()
    query = """
        SELECT result_json
        FROM walk_forward_results
        WHERE ticker = ?
        ORDER BY computed_at DESC
        LIMIT 1
    """
    with dm.connection() as conn:
        row = conn.execute(query, (ticker,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def _param_cv_mean(report: dict) -> Optional[float]:
    cv = report.get("ga_robustness", {}).get("param_cv")
    if not isinstance(cv, dict) or not cv:
        return None
    values = [float(v) for v in cv.values() if isinstance(v, (int, float))]
    return float(np.mean(values)) if values else None


def _rolling_slope(values: list[float]) -> Optional[float]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    y = np.array(clean, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _extract_metrics(
    report: dict, wf_result: Optional[dict], compute_trade_quality: bool = True
) -> dict:
    walk_forward = (
        report.get("walk_forward", {})
        if isinstance(report.get("walk_forward"), dict)
        else {}
    )
    bias = report.get("bias", {}) if isinstance(report.get("bias"), dict) else {}
    final_score = (
        report.get("final_score", {})
        if isinstance(report.get("final_score"), dict)
        else {}
    )
    execution_sensitivity = (
        report.get("execution_sensitivity", {})
        if isinstance(report.get("execution_sensitivity"), dict)
        else {}
    )

    wf_summary = wf_result.get("wf_summary", {}) if isinstance(wf_result, dict) else {}

    execution_stress = (
        walk_forward.get("execution_stress")
        or (wf_result.get("execution_stress") if wf_result else None)
        or {}
    )

    stress_pass_rate = report.get("ga_robustness", {}).get("stress_pass_rate")
    if not isinstance(stress_pass_rate, (int, float)):
        constraint_passed = execution_stress.get("constraint_passed")
        if isinstance(constraint_passed, bool):
            stress_pass_rate = 1.0 if constraint_passed else 0.0

    rolling_trend = wf_summary.get("rolling_sharpe_trend") or []
    rolling_slope = _rolling_slope(rolling_trend)

    trade_quality = None
    trade_count = None
    if compute_trade_quality:
        params = wf_result.get("best_params") if isinstance(wf_result, dict) else None
        trade_quality = analyze_trade_quality(params=params)
        if isinstance(trade_quality, dict):
            trade_count = trade_quality.get("trade_count")

    if trade_count is None:
        diagnostics = bias.get("diagnostics", {}) if isinstance(bias, dict) else {}
        trade_count = diagnostics.get("trade_count")

    return {
        "mean_oos_sharpe": walk_forward.get("mean_oos_sharpe"),
        "sharpe_std": walk_forward.get("sharpe_std"),
        "return_std": walk_forward.get("return_std"),
        "production_score": final_score.get("production_score"),
        "relative_gap": bias.get("relative_gap"),
        "trade_count": trade_count,
        "compounding_amplification": execution_sensitivity.get(
            "compounding_amplification"
        ),
        "fold_discard_ratio": wf_summary.get("discarded_ratio"),
        "rolling_sharpe_trend": rolling_trend,
        "rolling_sharpe_slope": rolling_slope,
        "execution_gap_flag": walk_forward.get("execution_gap_flag")
        or wf_summary.get("execution_gap_flag"),
        "avg_return": wf_summary.get("avg_return"),
        "param_cv_mean": _param_cv_mean(report),
        "stress_pass_rate": stress_pass_rate,
        "trade_quality": trade_quality,
    }


def _classify(pre: dict, post: dict) -> dict:
    status = "UNCLASSIFIED"
    action = None

    pre_sharpe = pre.get("mean_oos_sharpe")
    post_sharpe = post.get("mean_oos_sharpe")
    pre_std = pre.get("sharpe_std")
    post_std = post.get("sharpe_std")
    pre_cv = pre.get("param_cv_mean")
    post_cv = post.get("param_cv_mean")

    if (
        post.get("execution_gap_flag") == "HIGH_TIMING_DEPENDENCY"
        and isinstance(post_std, (int, float))
        and post_std > 0.2
    ):
        status = "STRUCTURAL_INSTABILITY"
        action = "revise WF window structure"
    elif (
        post.get("trade_count", 0) < 25
        and isinstance(pre_sharpe, (int, float))
        and isinstance(post_sharpe, (int, float))
        and post_sharpe > pre_sharpe
    ):
        pre_avg = pre.get("avg_return")
        post_avg = post.get("avg_return")
        if (
            isinstance(pre_avg, (int, float))
            and isinstance(post_avg, (int, float))
            and post_avg < pre_avg * 0.5
        ):
            status = "OVERFILTERED"
            action = "loosen edge threshold by 10%"
    elif (
        isinstance(post.get("relative_gap"), (int, float))
        and post.get("relative_gap") <= 0.3
        and isinstance(post_sharpe, (int, float))
        and post_sharpe < 0.3
        and isinstance(post_std, (int, float))
        and post_std <= 0.15
    ):
        status = "EDGE_WEAK"
        action = "feature redesign required"
    elif (
        isinstance(post_sharpe, (int, float))
        and post_sharpe >= 0.4
        and isinstance(pre_std, (int, float))
        and isinstance(post_std, (int, float))
        and post_std < pre_std
        and isinstance(pre_cv, (int, float))
        and isinstance(post_cv, (int, float))
        and post_cv < pre_cv
        and post.get("trade_count", 0) >= 40
    ):
        status = "STABLE_EDGE_FOUND"

    if status == "UNCLASSIFIED":
        if (
            isinstance(pre_std, (int, float))
            and isinstance(post_std, (int, float))
            and post_std < pre_std
        ):
            status = "EDGE_WEAK"
            action = "feature redesign required"

    return {"status": status, "action": action}


def build_summary(pre_report: dict, post_report: dict) -> dict:
    ticker = get_validation_ticker()
    wf_result = _load_latest_walk_forward_result(ticker)

    pre_metrics = _extract_metrics(pre_report, None, compute_trade_quality=False)
    post_metrics = _extract_metrics(post_report, wf_result, compute_trade_quality=True)

    classification = _classify(pre_metrics, post_metrics)

    delta_sharpe = None
    delta_production = None
    trade_delta = None
    param_cv_delta = None

    if isinstance(pre_metrics.get("mean_oos_sharpe"), (int, float)) and isinstance(
        post_metrics.get("mean_oos_sharpe"), (int, float)
    ):
        delta_sharpe = post_metrics["mean_oos_sharpe"] - pre_metrics["mean_oos_sharpe"]
    if isinstance(pre_metrics.get("production_score"), (int, float)) and isinstance(
        post_metrics.get("production_score"), (int, float)
    ):
        delta_production = (
            post_metrics["production_score"] - pre_metrics["production_score"]
        )
    if isinstance(pre_metrics.get("trade_count"), (int, float)) and isinstance(
        post_metrics.get("trade_count"), (int, float)
    ):
        trade_delta = post_metrics["trade_count"] - pre_metrics["trade_count"]
    if isinstance(pre_metrics.get("param_cv_mean"), (int, float)) and isinstance(
        post_metrics.get("param_cv_mean"), (int, float)
    ):
        param_cv_delta = post_metrics["param_cv_mean"] - pre_metrics["param_cv_mean"]

    summary = {
        "pre_upgrade_metrics": pre_metrics,
        "post_upgrade_metrics": post_metrics,
        "delta_sharpe": delta_sharpe,
        "delta_production_score": delta_production,
        "trade_count_change": trade_delta,
        "param_cv_change": param_cv_delta,
        "final_classification": classification,
        "deployable": classification.get("status") == "STABLE_EDGE_FOUND",
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Post-upgrade stability diagnostics")
    parser.add_argument(
        "--pre", required=True, help="Pre-upgrade validation report path"
    )
    parser.add_argument(
        "--post", help="Post-upgrade validation report path (defaults to latest)"
    )
    parser.add_argument(
        "--output", default="validation_reports/stability_upgrade_summary.json"
    )
    args = parser.parse_args()

    pre_report = _load_validation_report(args.pre)
    post_path = args.post or _latest_report_path()
    if not post_path:
        raise RuntimeError("No validation reports found for post-upgrade metrics")
    post_report = _load_validation_report(post_path)

    summary = build_summary(pre_report, post_report)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
