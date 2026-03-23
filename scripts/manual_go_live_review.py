import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_conf
from app.analysis.explainability_linter import lint_explanation
from app.notifications.email_formatter import format_email_detail, format_email_summary
from app.reporting.audit_builder import build_audit_summary


def _load_rows(query: str, params: List[Any]) -> List[Dict]:
    rows = []
    with _connection() as conn:
        cur = conn.execute(query, params)
        colnames = [c[0] for c in cur.description]
        for raw in cur.fetchall():
            rows.append({colnames[i]: raw[i] for i in range(len(colnames))})
    return rows


def _connection():
    import sqlite3

    cfg = get_conf(None)
    return sqlite3.connect(str(getattr(cfg, "DB_PATH")))


def _json_load(raw: Any) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _build_explainability_section(rows: List[Dict]) -> str:
    lines = []
    lines.append("## Explainability Sample (5 random decisions)")
    lines.append("")
    lines.append("| timestamp | ticker | action | source | lint_ok | issues |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for row in rows:
        decision = _json_load(row.get("decision_blob"))
        explanation = _json_load(row.get("explanation_json"))
        position_sizing = _json_load(row.get("position_sizing_json"))
        lint = lint_explanation(explanation, decision, position_sizing)
        action = decision.get("action") or decision.get("action_label") or ""
        issues = ",".join(lint.get("issues", []))
        lines.append(
            f"| {row.get('timestamp')} | {row.get('ticker')} | {action} | "
            f"{row.get('decision_source')} | {lint.get('ok')} | {issues} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_email_section(rows: List[Dict]) -> str:
    lines = []
    lines.append("## Email Usability Sample (5 latest decisions)")
    lines.append("")

    for row in rows:
        decision = _json_load(row.get("decision_blob"))
        audit = _json_load(row.get("audit_blob"))
        explanation = _json_load(row.get("explanation_json"))
        model_votes = _json_load(row.get("model_votes_json"))
        position_sizing = _json_load(row.get("position_sizing_json"))
        payload = {"model_votes": model_votes}

        audit_summary = build_audit_summary(audit, payload, decision)
        summary = format_email_summary(
            ticker=row.get("ticker"),
            decision=decision,
            audit=audit_summary,
            position_sizing=position_sizing,
            lang="en",
        )
        detail = format_email_detail(
            explanation=explanation,
            audit=audit_summary,
            lang="en",
        ).replace("\u2192", "->")

        lines.append(f"### {row.get('timestamp')} {row.get('ticker')}")
        lines.append("```")
        lines.append("Summary:")
        lines.append(summary)
        lines.append("")
        lines.append("Details:")
        lines.append(detail)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    explain_rows = _load_rows(
        """
        SELECT timestamp, ticker, decision_source, decision_blob, explanation_json, position_sizing_json
        FROM decision_history
        ORDER BY RANDOM()
        LIMIT 5
        """,
        [],
    )

    email_rows = _load_rows(
        """
        SELECT timestamp, ticker, decision_blob, audit_blob, explanation_json, model_votes_json, position_sizing_json
        FROM decision_history
        ORDER BY timestamp DESC
        LIMIT 5
        """,
        [],
    )

    now = datetime.now(timezone.utc)
    report_lines = [
        "# Manual Go-Live Review",
        "",
        f"Generated: {now.isoformat()}",
        "",
        _build_explainability_section(explain_rows),
        _build_email_section(email_rows),
    ]

    report = "\n".join(report_lines)
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "testing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"manual_go_live_review_{now.date().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
