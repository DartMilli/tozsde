"""Centralized report writer for governance runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config.config import Config
from app.reporting.report_schema import SummaryReport


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_report_dir(timestamp: str) -> Path:
    root = Config.REPORTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_report_bundle(
    output_dir: Path,
    summary: SummaryReport,
    validation: Optional[dict],
    diagnostics: Optional[dict],
    tests: Optional[dict],
    checklist: Optional[dict],
) -> None:
    _write_json(output_dir / "summary.json", summary.to_dict())
    _write_json(output_dir / "validation.json", validation or {})
    _write_json(output_dir / "diagnostics.json", diagnostics or {})
    _write_json(output_dir / "tests.json", tests or {})
    _write_json(output_dir / "checklist.json", checklist or {})
