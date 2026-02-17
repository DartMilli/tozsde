"""Report schema helpers for quant governance runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass
class SummaryReport:
    status: str
    mode: str
    timestamp: str
    ticker: str
    mean_oos_sharpe: Optional[float] = None
    production_score: Optional[float] = None
    collapse_stage: Optional[str] = None
    collapse_reason: Optional[str] = None
    stress_pass_rate: Optional[float] = None
    fold_discard_ratio: Optional[float] = None
    tests_passed: int = 0
    tests_failed: int = 0
    checklist_passed: bool = False
    deployment_allowed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def now_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
