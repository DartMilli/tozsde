"""Self-validation checks for execution stress robustness."""

from __future__ import annotations

from dataclasses import dataclass

from app.analysis.analyzer import get_params
from app.data_access.data_loader import load_data
from app.validation.execution_stress import (
    ExecutionScenario,
    evaluate_constraints,
    evaluate_execution_stress,
)
from app.validation.utils import get_validation_ticker, get_validation_window
from app.backtesting.backtester import Backtester


@dataclass(frozen=True)
class SelfCheckResult:
    name: str
    passed: bool
    details: dict


def _test_constraint_enforcement() -> SelfCheckResult:
    passed, reasons = evaluate_constraints(
        mean_oos_sharpe=1.0,
        relative_gap_baseline=0.8,
        worst_case_sharpe=0.5,
        min_sharpe=0.4,
        max_relative_gap=0.4,
    )
    return SelfCheckResult(
        name="constraint_enforcement",
        passed=not passed and "relative_gap" in reasons,
        details={"reasons": reasons},
    )


def _test_stability_enforcement() -> SelfCheckResult:
    sharpe_values = [0.8, 0.1, -0.2]
    worst_case = min(sharpe_values)
    passed, reasons = evaluate_constraints(
        mean_oos_sharpe=sum(sharpe_values) / len(sharpe_values),
        relative_gap_baseline=0.1,
        worst_case_sharpe=worst_case,
        min_sharpe=0.4,
        max_relative_gap=0.4,
    )
    return SelfCheckResult(
        name="stability_enforcement",
        passed=not passed and "worst_case_sharpe" in reasons,
        details={"worst_case": worst_case, "reasons": reasons},
    )


def _test_engine_integrity() -> SelfCheckResult:
    ticker = get_validation_ticker()
    start, end = get_validation_window()
    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    if df is None or df.empty:
        return SelfCheckResult(
            name="engine_integrity",
            passed=False,
            details={"error": "no_data"},
        )

    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        return SelfCheckResult(
            name="engine_integrity",
            passed=False,
            details={"error": "no_data"},
        )

    params = get_params(ticker)
    backtester = Backtester(df, ticker)
    baseline_report = backtester.run(
        params,
        execution_policy="next_open",
        fixed_position_pct=0.1,
    )
    baseline_equity = baseline_report.diagnostics.get("equity_curve")
    if hasattr(baseline_equity, "tolist"):
        baseline_equity = baseline_equity.tolist()

    stress = evaluate_execution_stress(
        df,
        ticker,
        params,
        position_size_pct=0.1,
        enable_stress=True,
    )
    scenario = stress.get("scenarios", {}).get(ExecutionScenario.BASELINE_NEXT_OPEN)
    stress_equity = scenario.get("equity_curve") if scenario else None

    if not baseline_equity or not stress_equity:
        return SelfCheckResult(
            name="engine_integrity",
            passed=False,
            details={"error": "missing_equity"},
        )

    scale = float(baseline_equity[0]) if baseline_equity else 1.0
    if scale == 0:
        scale = 1.0
    normalized_baseline = [float(val) / scale for val in baseline_equity]
    stress_scale = float(stress_equity[0]) if stress_equity else 1.0
    if stress_scale == 0:
        stress_scale = 1.0
    normalized_stress = [float(val) / stress_scale for val in stress_equity]

    if len(normalized_baseline) != len(normalized_stress):
        return SelfCheckResult(
            name="engine_integrity",
            passed=False,
            details={
                "length_mismatch": (len(normalized_baseline), len(normalized_stress))
            },
        )

    for left, right in zip(normalized_baseline, normalized_stress):
        if abs(float(left) - float(right)) > 1e-6:
            return SelfCheckResult(
                name="engine_integrity",
                passed=False,
                details={"value_mismatch": (left, right)},
            )

    return SelfCheckResult(
        name="engine_integrity",
        passed=True,
        details={"status": "ok"},
    )


def run_self_checks() -> dict:
    checks = [
        _test_constraint_enforcement(),
        _test_stability_enforcement(),
        _test_engine_integrity(),
    ]
    return {
        "status": "ok" if all(c.passed for c in checks) else "failed",
        "checks": [c.__dict__ for c in checks],
    }


if __name__ == "__main__":
    print(run_self_checks())
