import tempfile
from pathlib import Path

from app.config.config import Config
from app.governance.checklist_runner import evaluate_checklist
from app.reporting.report_builder import prepare_report_dir, write_report_bundle
from app.reporting.report_schema import SummaryReport


def _make_docs(
    root: Path, include_manual: bool = True, include_todo: bool = True
) -> None:
    docs_dir = root / "docs" / "testing"
    docs_dir.mkdir(parents=True, exist_ok=True)
    go_live = []
    if include_manual:
        go_live.append("Manual step: review results")
    (docs_dir / "go_live_checklist.md").write_text("\n".join(go_live), encoding="utf-8")

    plan = []
    if include_todo:
        plan.append("TODO: update validation plan")
    (docs_dir / "quant_validation_plan.md").write_text(
        "\n".join(plan), encoding="utf-8"
    )


def test_checklist_passes_with_required_metrics():
    original_reports_dir = Config.REPORTS_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            Config.REPORTS_DIR = temp_root / "reports"
            _make_docs(temp_root)

            results = {
                "walk_forward": {"mean_oos_sharpe": 0.8, "sharpe_std": 0.1},
                "bias": {"diagnostics": {"trade_count": 50}},
                "ga_robustness": {"stress_pass_rate": 0.9},
                "collapse_stage": None,
            }
            diagnostics = {
                "data_integrity": {
                    "duplicate_index_count": 0,
                    "nan_ohlc_rows": 0,
                    "monotonic_increasing": True,
                    "warnings": [],
                }
            }
            tests = {"failed": 0}

            checklist = evaluate_checklist(
                results=results,
                diagnostics=diagnostics,
                tests=tests,
                report_files_present=True,
            )

            assert checklist["deployment_allowed"] is True
            manual_items = [
                item
                for item in checklist["items"]
                if item["key"].startswith("manual_step_")
            ]
            todo_items = [
                item
                for item in checklist["items"]
                if item["key"].startswith("todo_item_")
            ]
            assert manual_items
            assert todo_items
    finally:
        Config.REPORTS_DIR = original_reports_dir


def test_checklist_blocks_on_improvement_gate():
    original_reports_dir = Config.REPORTS_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            Config.REPORTS_DIR = temp_root / "reports"
            _make_docs(temp_root, include_manual=False, include_todo=False)

            results = {
                "walk_forward": {"mean_oos_sharpe": 0.8, "sharpe_std": 0.1},
                "bias": {"diagnostics": {"trade_count": 50}},
                "ga_robustness": {"stress_pass_rate": 0.9},
                "collapse_stage": None,
                "deployment_blocked": "improvement gate failed",
            }
            diagnostics = {
                "data_integrity": {
                    "duplicate_index_count": 0,
                    "nan_ohlc_rows": 0,
                    "monotonic_increasing": True,
                    "warnings": [],
                }
            }
            tests = {"failed": 0}

            checklist = evaluate_checklist(
                results=results,
                diagnostics=diagnostics,
                tests=tests,
                report_files_present=True,
            )

            assert checklist["deployment_allowed"] is False
            improvement = next(
                item for item in checklist["items"] if item["key"] == "improvement_gate"
            )
            assert improvement["passed"] is False
    finally:
        Config.REPORTS_DIR = original_reports_dir


def test_report_bundle_writes_json():
    original_reports_dir = Config.REPORTS_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            Config.REPORTS_DIR = temp_root / "reports"
            report_dir = prepare_report_dir("20260217T000000")
            summary = SummaryReport(
                status="pass",
                mode="validation",
                timestamp="2026-02-17T00:00:00Z",
                ticker="TEST",
            )
            write_report_bundle(
                report_dir,
                summary,
                {"ok": True},
                {"diag": True},
                {"passed": 1},
                {"check": True},
            )

            assert (report_dir / "summary.json").exists()
            assert (report_dir / "validation.json").exists()
            assert (report_dir / "diagnostics.json").exists()
            assert (report_dir / "tests.json").exists()
            assert (report_dir / "checklist.json").exists()
    finally:
        Config.REPORTS_DIR = original_reports_dir
