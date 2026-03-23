from datetime import date
from typing import Dict, Tuple

from app.analysis.explainability_linter import lint_explanation
from app.notifications.alerter import ErrorAlerter
from app.notifications.email_formatter import format_email_detail, format_email_summary
from app.reporting.audit_builder import build_audit_summary


class NotificationCoordinator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _augment_explanation_with_sizing(
        self,
        explanation: Dict,
        decision: Dict,
        position_sizing,
        allocation_amount,
        allocation_pct,
    ) -> Dict:
        if not explanation or decision.get("action_code") != 1:
            return explanation

        final_size = None
        if position_sizing and position_sizing.get("final_size") is not None:
            final_size = float(position_sizing.get("final_size"))
        elif allocation_amount is not None:
            final_size = float(allocation_amount)

        if final_size is None:
            return explanation

        pct_text = ""
        if allocation_pct is not None:
            pct_text = f" ({float(allocation_pct) * 100:.2f}% equity)"

        size_line_en = f"Size: ${final_size:,.2f}{pct_text}"
        size_line_hu = f"Meret: ${final_size:,.2f}{pct_text}"

        def _inject(text: str, marker: str, size_line: str) -> str:
            if size_line in text or "Size:" in text or "Meret:" in text:
                return text
            if marker in text:
                return text.replace(marker, f"{size_line}\n{marker}")
            return f"{text}\n{size_line}"

        explanation["en"] = _inject(
            explanation.get("en", ""), "Rationale:", size_line_en
        )
        explanation["hu"] = _inject(
            explanation.get("hu", ""), "Indoklas:", size_line_hu
        )
        explanation.setdefault("meta", {})["sizing_note"] = {
            "final_size": round(final_size, 2),
            "allocation_pct": allocation_pct,
        }
        return explanation

    def prepare_item(self, item: Dict, persist: bool) -> Tuple[str, str]:
        ticker_symbol = item["ticker"]
        decision = item["decision"]
        payload = item["payload"]
        audit = item["audit"]
        explanation = item["explanation"]

        explanation = self._augment_explanation_with_sizing(
            explanation=explanation,
            decision=decision,
            position_sizing=item.get("position_sizing"),
            allocation_amount=item.get("allocation_amount"),
            allocation_pct=item.get("allocation_pct"),
        )

        lint_result = lint_explanation(
            explanation=explanation,
            decision=decision,
            position_sizing=item.get("position_sizing"),
        )
        audit["explainability"] = lint_result
        if not lint_result["ok"]:
            self.pipeline.logger.warning(
                "EXPLAINABILITY_LINT %s: %s",
                ticker_symbol,
                ", ".join(lint_result["issues"]),
            )

        if persist:
            decision_id = self.pipeline.persist_decision(
                payload=payload,
                decision=decision,
                explanation=explanation,
                audit=audit,
                position_sizing=item.get("position_sizing"),
                decision_source=item.get("decision_source")
                or payload.get("decision_source"),
            )
            item["decision_id"] = decision_id

        audit_summary = build_audit_summary(audit, payload, decision)
        summary_line = format_email_summary(
            ticker=ticker_symbol,
            decision=decision,
            audit=audit_summary,
            position_sizing=item.get("position_sizing"),
        )
        detail_line = format_email_detail(
            explanation=explanation,
            audit=audit_summary,
        )
        return summary_line, detail_line

    def send_daily_email(self, summary_lines, detail_lines, dry_run: bool) -> None:
        if not summary_lines and not detail_lines:
            return

        cfg = self.pipeline._get_settings()
        subject = f"Napi ajanlasok ({date.today().isoformat()})"
        sections = []
        if summary_lines:
            sections.append("Summary:\n" + "\n".join(summary_lines))
        if detail_lines:
            limited_details = detail_lines[: getattr(cfg, "EMAIL_MAX_DETAIL_LINES")]
            sections.append("Details:\n" + "\n\n".join(limited_details))

        body = "\n\n".join(sections)
        max_chars = getattr(cfg, "EMAIL_MAX_BODY_CHARS")
        if max_chars and len(body) > max_chars:
            body = body[: max_chars - 12] + "\n[truncated]"
            self.pipeline.logger.warning(
                "EMAIL_TRUNCATED max_chars=%s detail_lines=%s",
                max_chars,
                len(detail_lines),
            )

        if dry_run:
            self.pipeline.logger.info("[DRY-RUN] Email would be sent: %s", subject)
            self.pipeline.logger.debug("Email body:\n%s", body)
            return

        notify_email = getattr(cfg, "NOTIFY_EMAIL")
        try:
            self.pipeline.send_notifications(subject, body, notify_email)
            self.pipeline.logger.info("OK Email sent to %s", notify_email)
        except Exception as exc:
            self.pipeline.logger.error("ERR Failed to send email: %s", exc)
            ErrorAlerter.alert(
                error_code="AUTHENTICATION_FAILED",
                message=f"Failed to send notification email: {exc}",
                details={"recipient": notify_email},
                severity="auto",
            )
