import json
from datetime import datetime
from typing import Dict

from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class ValidationReportBuilder:
    """
    Aggregates validation metrics into a single snapshot report.
    """

    def __init__(self):
        self.dm = DataManager()

    def build(self) -> Dict:
        report = {
            "decision_quality": self._latest("decision_quality_metrics"),
            "confidence_calibration": self._latest("confidence_calibration"),
            "wf_stability": self._latest("wf_stability_metrics"),
            "safety_stress": self._latest("safety_stress_results"),
            "decision_effectiveness": self._latest("decision_effectiveness_rolling"),
        }

        self.dm.save_validation_report(json.dumps(report))
        return report

    def fetch_latest(self) -> Dict:
        return self.dm.fetch_latest_validation_report()

    def export(self, report: Dict, path: str, fmt: str = "json") -> None:
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        elif fmt == "md":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown(report))
        else:
            raise ValueError("Unsupported export format")

    def to_markdown(self, report: Dict) -> str:
        lines = []

        def section(title, data):
            lines.append(f"## {title}")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2, default=str))
            lines.append("```")
            lines.append("")

        section("Decision Quality", report.get("decision_quality"))
        section("Decision Effectiveness", report.get("decision_effectiveness"))
        section("Confidence Calibration", report.get("confidence_calibration"))
        section("WF Stability", report.get("wf_stability"))
        section("Safety Stress", report.get("safety_stress"))

        return "\n".join(lines)

    def _latest(self, table: str):
        query = f"SELECT * FROM {table} ORDER BY computed_at DESC LIMIT 1"
        with self.dm.connection() as conn:
            row = conn.execute(query).fetchone()

        if not row:
            return None

        # For JSON payload columns, detect by suffix
        as_dict = {"raw": row}
        if table == "decision_quality_metrics":
            as_dict = {"metrics": json.loads(row[5])}
        elif table == "confidence_calibration":
            as_dict = {"params": json.loads(row[6]), "metrics": json.loads(row[7])}
        elif table == "wf_stability_metrics":
            as_dict = {"metrics": json.loads(row[3])}
        elif table == "safety_stress_results":
            as_dict = {"results": json.loads(row[6])}
        elif table == "decision_effectiveness_rolling":
            as_dict = {"metrics": json.loads(row[4])}

        return as_dict
