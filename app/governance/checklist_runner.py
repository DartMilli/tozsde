"""Go-live checklist enforcement and quant validation plan scanning."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from app.config.config import Config


@dataclass
class ChecklistItem:
    key: str
    title: str
    required: bool
    passed: bool
    evidence: dict
    manual_review_required: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _scan_manual_items(text: str) -> list[str]:
    manual = []
    for line in text.splitlines():
        if "Manual step" in line or "Manualis lepes" in line:
            manual.append(line.strip())
    return manual


def _scan_todo_items(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "TODO" in line]


def _value_or_none(value):
    return value if isinstance(value, (int, float)) else None


def evaluate_checklist(
    results: dict,
    diagnostics: dict,
    tests: dict,
    report_files_present: bool,
) -> dict:
    items: list[ChecklistItem] = []

    mean_oos_sharpe = _value_or_none(
        results.get("walk_forward", {}).get("mean_oos_sharpe")
    )
    sharpe_std = _value_or_none(results.get("walk_forward", {}).get("sharpe_std"))
    trade_count = _value_or_none(
        results.get("bias", {}).get("diagnostics", {}).get("trade_count")
    )
    stress_pass_rate = _value_or_none(
        results.get("ga_robustness", {}).get("stress_pass_rate")
    )
    collapse_stage = results.get("collapse_stage")
    deployment_blocked = results.get("deployment_blocked")

    warnings = []
    data_integrity = (
        diagnostics.get("data_integrity") if isinstance(diagnostics, dict) else {}
    )
    if isinstance(data_integrity, dict):
        warnings = data_integrity.get("warnings") or []

    tests_failed = int(tests.get("failed", 0)) if isinstance(tests, dict) else 0

    items.append(
        ChecklistItem(
            key="min_oos_sharpe",
            title="Minimum OOS Sharpe",
            required=True,
            passed=mean_oos_sharpe is not None
            and mean_oos_sharpe >= Config.MIN_OOS_SHARPE,
            evidence={
                "mean_oos_sharpe": mean_oos_sharpe,
                "threshold": Config.MIN_OOS_SHARPE,
            },
        )
    )
    items.append(
        ChecklistItem(
            key="max_sharpe_variance",
            title="Maximum Sharpe variance",
            required=True,
            passed=sharpe_std is not None and sharpe_std <= 0.15,
            evidence={"sharpe_std": sharpe_std, "threshold": 0.15},
        )
    )
    items.append(
        ChecklistItem(
            key="min_trade_count",
            title="Minimum trade count",
            required=True,
            passed=trade_count is not None and trade_count >= 40,
            evidence={"trade_count": trade_count, "threshold": 40},
        )
    )
    items.append(
        ChecklistItem(
            key="stress_pass_rate",
            title="Stress pass rate threshold",
            required=True,
            passed=stress_pass_rate is not None and stress_pass_rate >= 0.6,
            evidence={"stress_pass_rate": stress_pass_rate, "threshold": 0.6},
        )
    )
    items.append(
        ChecklistItem(
            key="collapse_stage",
            title="No collapse stage",
            required=True,
            passed=collapse_stage in (None, "none"),
            evidence={"collapse_stage": collapse_stage},
        )
    )
    items.append(
        ChecklistItem(
            key="improvement_gate",
            title="Improvement gate passed",
            required=True,
            passed=deployment_blocked is None,
            evidence={"deployment_blocked": deployment_blocked},
        )
    )
    items.append(
        ChecklistItem(
            key="unresolved_warnings",
            title="No unresolved warnings",
            required=True,
            passed=not warnings,
            evidence={"warnings": warnings},
        )
    )
    items.append(
        ChecklistItem(
            key="failing_tests",
            title="No failing tests",
            required=True,
            passed=tests_failed == 0,
            evidence={"tests_failed": tests_failed},
        )
    )
    items.append(
        ChecklistItem(
            key="database_integrity",
            title="No database corruption",
            required=True,
            passed=bool(data_integrity)
            and data_integrity.get("duplicate_index_count", 0) == 0
            and data_integrity.get("nan_ohlc_rows", 0) == 0
            and data_integrity.get("monotonic_increasing", True) is True,
            evidence={
                "duplicate_index_count": data_integrity.get("duplicate_index_count"),
                "nan_ohlc_rows": data_integrity.get("nan_ohlc_rows"),
                "monotonic_increasing": data_integrity.get("monotonic_increasing"),
            },
        )
    )
    items.append(
        ChecklistItem(
            key="reports_present",
            title="No missing reports",
            required=True,
            passed=bool(report_files_present),
            evidence={"reports_present": report_files_present},
        )
    )

    docs_dir = Config.REPORTS_DIR.parent / "docs" / "testing"
    go_live_text = _load_text(docs_dir / "go_live_checklist.md")
    plan_text = _load_text(docs_dir / "quant_validation_plan.md")
    manual_lines = _scan_manual_items(go_live_text)
    todo_lines = _scan_todo_items(plan_text)

    for idx, line in enumerate(manual_lines):
        items.append(
            ChecklistItem(
                key=f"manual_step_{idx}",
                title="Manual checklist step",
                required=False,
                passed=False,
                evidence={"line": line},
                manual_review_required=True,
            )
        )

    for idx, line in enumerate(todo_lines):
        items.append(
            ChecklistItem(
                key=f"todo_item_{idx}",
                title="Validation plan TODO",
                required=False,
                passed=False,
                evidence={"line": line},
                manual_review_required=True,
            )
        )

    checklist_passed = all(item.passed for item in items if item.required)

    return {
        "status": "ok",
        "checklist_passed": checklist_passed,
        "deployment_allowed": checklist_passed,
        "items": [item.to_dict() for item in items],
    }
