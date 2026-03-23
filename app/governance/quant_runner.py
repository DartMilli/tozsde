"""Unified Quant Validation & Governance Runner."""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from app.governance import get_settings
from app.bootstrap.build_settings import build_settings

settings = build_settings()
from app.reporting.report_builder import prepare_report_dir, write_report_bundle
from app.reporting.report_schema import SummaryReport, now_timestamp
from app.validation.scoring import compute_quant_score
from app.validation.improvement_check import evaluate_results
from app.validation.bias_tests import run_bias_tests
from app.validation.execution_sensitivity import run_execution_sensitivity
from app.validation.wf_analysis import run_walk_forward_analysis
from app.validation.ga_robustness import run_ga_robustness_tests
from app.validation.rl_stress_tests import run_rl_stress_tests
from app.validation.risk_stress import run_risk_stress_tests
from app.validation.shadow_compare import run_shadow_comparison
from app.validation.pipeline_audit import run_pipeline_audit
from app.validation.sanity_strategy import run_sanity_backtest
from app.validation.data_integrity_check import run_data_integrity_checks
from app.validation.utils import get_validation_ticker, get_validation_window
from app.data_access.data_loader import load_data
from app.backtesting.walk_forward import run_walk_forward
from app.backtesting.execution_utils import seed_deterministic
from app.governance.checklist_runner import evaluate_checklist
from app.validation.edge_diagnostics import classify_collapse_reason


def _git_commit() -> Optional[str]:
    cfg = get_settings()
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(getattr(cfg, "REPORTS_DIR", Path(".")).parent),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.strip()
    except Exception:
        return None


def _configure_logging(
    log_path: Path, run_id: str, mode: str, commit: Optional[str]
) -> logging.Logger:
    logger = logging.getLogger("quant_runner")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | run_id=%(run_id)s | mode=%(mode)s | commit=%(commit)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger = logging.LoggerAdapter(
        logger,
        {"run_id": run_id, "mode": mode, "commit": commit or "unknown"},
    )
    return logger


def _run_tests(logger: logging.Logger) -> dict:
    logger.info("Running pytest suite")
    try:
        import pytest
    except ImportError:
        return {"status": "error", "error": "pytest_not_available"}

    coverage_percent = None

    class _CountPlugin:
        def __init__(self) -> None:
            self.passed = 0
            self.failed = 0
            self.skipped = 0
            self.xfailed = 0
            self.xpassed = 0

        def pytest_runtest_logreport(self, report):
            if report.when != "call":
                return
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1
            if getattr(report, "wasxfail", False):
                if report.skipped:
                    self.xfailed += 1
                elif report.passed:
                    self.xpassed += 1

    plugin = _CountPlugin()
    args = ["-q", "--disable-warnings"]

    try:
        import coverage
        from io import StringIO

        cov = coverage.Coverage(source=["app"])
        cov.start()
        exit_code = pytest.main(args, plugins=[plugin])  # type: ignore[arg-type]
        cov.stop()
        cov.save()
        coverage_stream = StringIO()
        coverage_percent = cov.report(show_missing=False, file=coverage_stream)
    except ImportError:
        exit_code = pytest.main(args, plugins=[plugin])  # type: ignore[arg-type]

    return {
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "passed": plugin.passed,
        "failed": plugin.failed,
        "skipped": plugin.skipped,
        "xfailed": plugin.xfailed,
        "xpassed": plugin.xpassed,
        "total": plugin.passed
        + plugin.failed
        + plugin.skipped
        + plugin.xfailed
        + plugin.xpassed,
        "coverage_percent": coverage_percent,
    }


def _run_diagnostics(logger: logging.Logger) -> dict:
    logger.info("Running diagnostics pipeline")
    cfg = get_settings()
    settings.pipeline_audit_mode = True
    settings.edge_diagnostics_mode = True

    ticker = get_validation_ticker()
    start, end = get_validation_window()
    df = load_data(ticker, start=start.isoformat(), end=end.isoformat())
    df = df[(df.index.date >= start) & (df.index.date <= end)]

    diagnostics = {
        "pipeline_audit": run_pipeline_audit(ticker=ticker),
        "data_integrity": run_data_integrity_checks(df, lookback=0),
        "sanity_strategy": run_sanity_backtest(df),
    }

    return diagnostics


def _run_validation(
    logger: logging.Logger, include_shadow: bool, include_risk: bool, include_rl: bool
) -> dict:
    logger.info("Running validation pipeline")
    results: dict = {}

    results["bias"] = run_bias_tests()
    results["execution_sensitivity"] = run_execution_sensitivity()
    results["walk_forward"] = run_walk_forward_analysis()
    results["ga_robustness"] = run_ga_robustness_tests()

    if include_rl:
        results["rl_stability"] = run_rl_stress_tests()
    if include_risk:
        results["risk"] = run_risk_stress_tests()
    if include_shadow:
        results["shadow"] = run_shadow_comparison()

    results["final_score"] = compute_quant_score(results)

    try:
        results["improvement_check"] = evaluate_results(results)
    except Exception as exc:
        results["deployment_blocked"] = str(exc)

    return results


def _build_summary(
    mode: str,
    ticker: str,
    results: dict,
    diagnostics: dict,
    tests: dict,
    checklist: dict,
) -> SummaryReport:
    wf = results.get("walk_forward", {})
    final_score = results.get("final_score", {})

    status = "pass"
    if checklist.get("deployment_allowed") is False:
        status = "blocked"
    elif tests.get("failed", 0) > 0:
        status = "warning"

    return SummaryReport(
        status=status,
        mode=mode,
        timestamp=now_timestamp(),
        ticker=ticker,
        mean_oos_sharpe=wf.get("mean_oos_sharpe"),
        production_score=final_score.get("production_score"),
        collapse_stage=results.get("collapse_stage"),
        collapse_reason=results.get("collapse_reason"),
        stress_pass_rate=results.get("ga_robustness", {}).get("stress_pass_rate"),
        fold_discard_ratio=wf.get("discarded_ratio")
        or (wf.get("wf_summary") or {}).get("discarded_ratio"),
        tests_passed=int(tests.get("passed", 0)),
        tests_failed=int(tests.get("failed", 0)),
        checklist_passed=bool(checklist.get("checklist_passed")),
        deployment_allowed=bool(checklist.get("deployment_allowed")),
    )


def _apply_collapse_stage(results: dict, diagnostics: dict) -> None:
    trade_count = results.get("bias", {}).get("diagnostics", {}).get("trade_count")
    if not isinstance(trade_count, int) or trade_count != 0:
        return

    audit = None
    if isinstance(diagnostics, dict):
        audit = diagnostics.get("pipeline_audit") or diagnostics

    if not isinstance(audit, dict):
        return

    folds = audit.get("folds") or []
    raw_signal_count = sum(f.get("raw_signal_count", 0) for f in folds)
    post_dropout_signal_count = sum(
        f.get("post_dropout_signal_count", 0) for f in folds
    )
    post_edge_signal_count = sum(
        f.get("post_edge_filter_signal_count", 0) for f in folds
    )
    position_attempts = sum(f.get("position_attempts", 0) for f in folds)
    orders_created = sum(f.get("orders_created", 0) for f in folds)
    executed_trades = sum(f.get("executed_trades", 0) for f in folds)

    if raw_signal_count == 0:
        collapse_stage = "signal_generation"
    elif post_dropout_signal_count == 0:
        collapse_stage = "feature_dropout"
    elif post_edge_signal_count == 0:
        collapse_stage = "edge_filter"
    elif position_attempts == 0:
        collapse_stage = "position_sizing"
    elif orders_created == 0:
        collapse_stage = "order_creation"
    elif executed_trades == 0:
        collapse_stage = "execution_engine"
    else:
        collapse_stage = "unknown"

    results["collapse_stage"] = collapse_stage

    if collapse_stage == "edge_filter":
        edge_summary = audit.get("edge_diagnostics_summary")
        if isinstance(edge_summary, dict):
            results["collapse_reason"] = classify_collapse_reason(edge_summary)


def _exit_code(summary: SummaryReport) -> int:
    if summary.status == "blocked":
        return 2
    if summary.status == "warning":
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant Validation & Governance Runner")
    parser.add_argument(
        "--mode",
        choices=["research", "validation", "diagnostics", "predeploy", "full", "tests"],
        default="validation",
    )
    args = parser.parse_args()

    run_id = str(uuid.uuid4())
    timestamp = now_timestamp().replace(":", "").replace("-", "")
    report_dir = prepare_report_dir(timestamp)
    commit = _git_commit()
    logger = _configure_logging(report_dir / "run.log", run_id, args.mode, commit)
    logger.info("Starting quant runner")

    cfg = get_settings()
    settings.edge_diagnostics_mode = False
    settings.pipeline_audit_mode = False

    if args.mode != "tests":
        os.environ["VALIDATION_MODE"] = args.mode
        os.environ["VALIDATION_DISABLE_SAFETY"] = "true"
        os.environ["VALIDATION_DISABLE_POLICY"] = "true"
        # DataManager should be injected from DI root
        # dm.initialize_tables()
        seed_deterministic(42)
        random.seed(42)

    diagnostics = {}
    validation = {}
    tests = {}

    if args.mode == "research":
        settings.edge_diagnostics_mode = True
        settings.pipeline_audit_mode = True
        run_walk_forward(get_validation_ticker())
        diagnostics = _run_diagnostics(logger)
        validation = _run_validation(
            logger, include_shadow=False, include_risk=False, include_rl=False
        )

    elif args.mode == "diagnostics":
        diagnostics = _run_diagnostics(logger)

    elif args.mode == "validation":
        validation = _run_validation(
            logger, include_shadow=False, include_risk=False, include_rl=False
        )

    elif args.mode == "predeploy":
        diagnostics = _run_diagnostics(logger)
        validation = _run_validation(
            logger, include_shadow=True, include_risk=True, include_rl=True
        )

    elif args.mode == "tests":
        tests = _run_tests(logger)

    elif args.mode == "full":
        tests = _run_tests(logger)
        diagnostics = _run_diagnostics(logger)
        validation = _run_validation(
            logger, include_shadow=True, include_risk=True, include_rl=True
        )

    ticker = get_validation_ticker()

    _apply_collapse_stage(validation, diagnostics)

    checklist = evaluate_checklist(
        results=validation,
        diagnostics=diagnostics.get("pipeline_audit", diagnostics),
        tests=tests,
        report_files_present=True,
    )

    summary = _build_summary(
        args.mode, ticker, validation, diagnostics, tests, checklist
    )

    write_report_bundle(report_dir, summary, validation, diagnostics, tests, checklist)

    logger.info("Completed quant runner")
    return _exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
