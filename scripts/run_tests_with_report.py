#!/usr/bin/env python
import argparse
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.decision_quality_analyzer import DecisionQualityAnalyzer
from app.analysis.confidence_calibrator import ConfidenceCalibrator
from app.analysis.wf_stability_analyzer import WalkForwardStabilityAnalyzer
from app.analysis.safety_stress_tester import SafetyStressTester
from app.analysis.validation_report_builder import ValidationReportBuilder
from app.analysis.decision_effectiveness import DecisionEffectivenessAnalyzer
from app.analysis.phase6_validator import Phase6Validator
from app.data_access.data_manager import DataManager
from app.validation.validation_runner import ValidationRunner
from app.validation.report_builder import build_markdown_report

REPORT_PATH = Path("docs/testing/TEST_STATUS_REPORT.md")
VALIDATION_START = "<!-- VALIDATION_START -->"
VALIDATION_END = "<!-- VALIDATION_END -->"
QUANT_VALIDATION_START = "<!-- QUANT_VALIDATION_START -->"
QUANT_VALIDATION_END = "<!-- QUANT_VALIDATION_END -->"


def run_pytest(pytest_args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    return subprocess.call(cmd)


def update_report_with_validation(markdown: str) -> None:
    if not REPORT_PATH.exists():
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("# Test & Code Quality Status Report\n\n")

    content = REPORT_PATH.read_text(encoding="utf-8")

    block = f"{VALIDATION_START}\n{markdown}\n{VALIDATION_END}"
    if VALIDATION_START in content and VALIDATION_END in content:
        pre = content.split(VALIDATION_START)[0]
        post = content.split(VALIDATION_END)[1]
        new_content = pre + block + post
    else:
        new_content = content.rstrip() + "\n\n" + block + "\n"

    REPORT_PATH.write_text(new_content, encoding="utf-8")


def update_report_with_quant_validation(markdown: str) -> None:
    if not REPORT_PATH.exists():
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("# Test & Code Quality Status Report\n\n")

    content = REPORT_PATH.read_text(encoding="utf-8")

    block = f"{QUANT_VALIDATION_START}\n{markdown}\n{QUANT_VALIDATION_END}"
    if QUANT_VALIDATION_START in content and QUANT_VALIDATION_END in content:
        pre = content.split(QUANT_VALIDATION_START)[0]
        post = content.split(QUANT_VALIDATION_END)[1]
        new_content = pre + block + post
    else:
        new_content = content.rstrip() + "\n\n" + block + "\n"

    REPORT_PATH.write_text(new_content, encoding="utf-8")


def run_validation(args) -> None:
    DataManager().initialize_tables()

    DecisionQualityAnalyzer().analyze(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if not args.no_calibration:
        ConfidenceCalibrator().compute(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if args.ticker:
        WalkForwardStabilityAnalyzer().analyze(ticker=args.ticker)

        if args.start_date and args.end_date:
            SafetyStressTester().run(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                scenario=args.scenario,
            )

        DecisionEffectivenessAnalyzer().compute_for_range(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    report = ValidationReportBuilder().build()
    markdown = build_validation_markdown(args, report)
    phase6 = Phase6Validator().run(args.ticker or "ALL")
    markdown += "\n## Phase 6 Validation Checklist\n```json\n"
    markdown += json.dumps(phase6, indent=2, default=str)
    markdown += "\n```\n"
    update_report_with_validation(markdown)


def run_quant_validation(args) -> None:
    mode = args.quant_validation_mode
    runner = ValidationRunner(mode=mode)
    runner.execute()

    markdown = build_markdown_report(runner.results)
    update_report_with_quant_validation(markdown)


def build_validation_markdown(args, report) -> str:
    def _get_status(section, key="metrics"):
        try:
            return report.get(section, {}).get(key, {}).get("status", "ok")
        except Exception:
            return "unknown"

    summary = [
        "## Validation Snapshot Summary",
        f"- Decision Quality: {_get_status('decision_quality')}",
        f"- Decision Effectiveness: {_get_status('decision_effectiveness')}",
        f"- Confidence Calibration: {_get_status('confidence_calibration')}",
        f"- WF Stability: {_get_status('wf_stability')}",
        f"- Safety Stress: {_get_status('safety_stress', key='results')}",
        "",
    ]

    header = [
        "# Validation Report",
        "",
        "## Validation Snapshot",
        f"- **Ticker:** {args.ticker or 'ALL'}",
        f"- **Date range:** {args.start_date or 'N/A'} -> {args.end_date or 'N/A'}",
        f"- **Scenario:** {args.scenario}",
        "",
    ]
    body = ValidationReportBuilder().to_markdown(report)
    return "\n".join(header + summary) + body


def main():
    parser = argparse.ArgumentParser(
        description="Run tests and optionally append validation report"
    )
    parser.add_argument(
        "--with-validation", action="store_true", help="Run Phase 5 validation"
    )
    parser.add_argument(
        "--with-quant-validation",
        action="store_true",
        help="Run quant validation runner",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip pytest execution"
    )
    parser.add_argument("--ticker", type=str, help="Ticker for validation")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--scenario", type=str, default="elevated_volatility")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--quant-validation-mode",
        choices=["quick", "full", "shadow"],
        default="quick",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="Args passed to pytest"
    )
    args = parser.parse_args()

    exit_code = 0
    if not args.skip_tests:
        exit_code = run_pytest(args.pytest_args)

    if args.with_validation:
        run_validation(args)

    if args.with_quant_validation:
        run_quant_validation(args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
