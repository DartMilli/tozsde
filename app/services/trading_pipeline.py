import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

from app.backtesting.history_store import HistoryStore
from app.config.config import Config
from app.data_access.data_cleaner import prepare_df
from app.decision.allocation import allocate_capital
from app.decision.decision_policy import apply_decision_policy
from app.decision.recommendation_builder import build_explanation, build_recommendation
from app.decision.volatility import compute_normalized_volatility
from app.decision.recommender import generate_daily_recommendation_payload
from app.notifications.alerter import ErrorAlerter
from app.reporting.audit_builder import build_audit_metadata, build_audit_summary
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.analysis.explainability_linter import lint_explanation
from app.analysis.go_live_metrics import compute_drawdown_summary, compute_loss_streak
from app.notifications.email_formatter import format_email_detail, format_email_summary
from app.decision.position_sizer import apply_position_sizing
from app.data_access.data_manager import DataManager


class TradingPipelineService:
    """
    Central orchestration layer for daily trading pipeline steps.
    """

    def __init__(
        self,
        history_store: HistoryStore,
        logger=None,
        data_fetcher: MarketDataFetcher = None,
        model_runner: ModelEnsembleRunner = None,
        email_notifier: EmailNotifier = None,
        execution_engine=None,
    ):
        if history_store is None:
            raise ValueError("history_store is required")
        if data_fetcher is None:
            raise ValueError("data_fetcher is required")
        if model_runner is None:
            raise ValueError("model_runner is required")
        if email_notifier is None:
            raise ValueError("email_notifier is required")
        if execution_engine is None:
            raise ValueError("execution_engine is required")

        self.history_store = history_store
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        self.model_runner = model_runner
        self.email_notifier = email_notifier
        self.execution_engine = execution_engine

    # --- Step primitives (kept thin for now) ---

    def load_market_data(self, ticker: str, start: str, end: Optional[str] = None):
        return self.data_fetcher.load_data(ticker, start=start, end=end)

    def compute_signals(self, df, ticker: str):
        return prepare_df(df.copy(), ticker)

    def run_decision_models(
        self,
        ticker: str,
        top_n: int = 3,
        debug: bool = True,
        as_of_date=None,
    ):
        return generate_daily_recommendation_payload(
            ticker=ticker,
            history_store=self.history_store,
            top_n=top_n,
            debug=debug,
            data_fetcher=self.data_fetcher,
            model_runner=self.model_runner,
            as_of_date=as_of_date,
        )

    def apply_safety_rules(self, decision: Dict, audit: Dict) -> Dict:
        return apply_decision_policy(decision, audit)

    def allocate_capital(self, candidates: List[Dict]) -> List[Dict]:
        return allocate_capital(candidates)

    def persist_decision(
        self,
        payload: Dict,
        decision: Dict,
        explanation: Dict,
        audit: Dict,
        position_sizing: Dict = None,
        decision_source: str = None,
    ) -> int:
        return self.history_store.save_decision(
            payload=payload,
            decision=decision,
            explanation=explanation,
            audit=audit,
            model_votes=payload.get("model_votes", []),
            safety_overrides={
                "safety_override": decision.get("safety_override"),
                "no_trade_reason": decision.get("no_trade_reason"),
                "reasons": decision.get("reasons", []),
                "warnings": decision.get("warnings", []),
            },
            position_sizing=position_sizing,
            decision_source=decision_source,
            model_id=payload.get("model_id"),
            timestamp=payload.get("timestamp"),
        )

    def send_notifications(self, subject: str, body: str, recipient: str) -> None:
        self.email_notifier.send(subject, body, recipient)

    def execute_trades(self, decisions: List[Dict], as_of: date) -> None:
        self.execution_engine.execute(decisions=decisions, as_of=as_of)

    def _log_go_live_metrics(self, ticker: str) -> None:
        drawdown = compute_drawdown_summary(ticker)
        streak = compute_loss_streak(ticker)

        if drawdown["status"] == "ok":
            self.logger.info(
                "GO_LIVE_METRICS %s outcomes=%s max_drawdown=%.4f current_drawdown=%.4f loss_streak=%s",
                ticker,
                drawdown.get("rows"),
                drawdown.get("max_drawdown", 0.0),
                drawdown.get("current_drawdown", 0.0),
                streak.get("loss_streak", 0),
            )
        else:
            self.logger.info(
                "GO_LIVE_METRICS %s outcomes=0 loss_streak=%s status=no_data",
                ticker,
                streak.get("loss_streak", 0),
            )

    def _augment_explanation_with_sizing(
        self,
        explanation: Dict,
        decision: Dict,
        position_sizing: Optional[Dict],
        allocation_amount: Optional[float],
        allocation_pct: Optional[float],
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
            explanation.get("hu", ""), "Indoklás:", size_line_hu
        )
        explanation.setdefault("meta", {})["sizing_note"] = {
            "final_size": round(final_size, 2),
            "allocation_pct": allocation_pct,
        }
        return explanation

    # --- Daily pipeline helper ---

    def build_daily_candidate(
        self,
        ticker: str,
        debug: bool = True,
        as_of_date=None,
    ) -> Dict:
        payload = self.run_decision_models(
            ticker=ticker,
            debug=debug,
            as_of_date=as_of_date,
        )

        if payload.get("error") == "NO_MODELS" and Config.ALLOW_NO_MODEL_FALLBACK:
            self.logger.warning(
                "NO_MODELS fallback enabled; generating HOLD decision for %s",
                ticker,
            )
            return self._build_fallback_candidate_no_models(ticker, as_of_date)

        if payload.get("error"):
            raise ValueError(payload["error"])

        if "decision" in payload:
            decision = payload["decision"]
            explanation = payload.get("explanation")
        else:
            decision = build_recommendation(payload)
            explanation = build_explanation(payload, decision)

        audit = build_audit_metadata(payload, decision)

        decision = self.apply_safety_rules(decision, audit)
        if explanation is None:
            explanation = build_explanation(payload, decision)

        return {
            "ticker": ticker,
            "payload": payload,
            "decision": decision,
            "explanation": explanation,
            "audit": audit,
        }

    def _build_fallback_candidate_no_models(
        self,
        ticker: str,
        as_of_date=None,
    ) -> Dict:
        today = as_of_date or date.today()
        start = today - timedelta(days=180)
        df_full = self.data_fetcher.load_data(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
        )
        if df_full.empty:
            raise ValueError("NO_DATA")

        df = prepare_df(df_full.copy(), ticker)
        latest_price = float(df["Close"].iloc[-1]) if not df.empty else None
        volatility = compute_normalized_volatility(df) if not df.empty else 0.0

        payload = {
            "ticker": ticker,
            "timestamp": today.isoformat(),
            "latest_price": latest_price,
            "volatility": volatility,
            "model_votes": [],
            "avg_confidence": 0.0,
            "avg_wf_score": 0.0,
            "ensemble_quality": 0.0,
            "action_code": 0,
            "decision_source": "fallback",
        }

        decision = build_recommendation(payload)
        decision.update(
            {
                "action_code": 0,
                "action": Config.ACTION_LABELS[Config.LANG][0],
                "strength": "NO_TRADE",
                "no_trade": True,
                "no_trade_reason": "FALLBACK_NO_MODELS",
            }
        )

        audit = build_audit_metadata(payload, decision)
        explanation = build_explanation(payload, decision)

        return {
            "ticker": ticker,
            "payload": payload,
            "decision": decision,
            "explanation": explanation,
            "audit": audit,
            "decision_source": "fallback",
        }

    def get_tickers_to_process(self, ticker: Optional[str] = None) -> List[str]:
        return [ticker] if ticker else Config.get_supported_tickers()

    def run_daily(self, dry_run: bool = False, ticker: Optional[str] = None) -> None:
        """
        End-to-end daily pipeline execution.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"DAILY pipeline started (dry_run={dry_run})")
        if ticker:
            self.logger.info(f"DEV mode: Analyzing {ticker} only")
        self.logger.info("=" * 80)

        dm = DataManager()
        today = date.today()

        tickers_to_process = self.get_tickers_to_process(ticker)
        daily_candidates = []

        for ticker_symbol in tickers_to_process:
            try:
                if dm.has_decision_for_date(ticker=ticker_symbol, as_of_date=today):
                    self.logger.info(
                        "SKIP %s: decision already exists for %s",
                        ticker_symbol,
                        today.isoformat(),
                    )
                    continue
                candidate = self.build_daily_candidate(ticker_symbol)
                daily_candidates.append(candidate)

                decision = candidate["decision"]
                self.logger.info(f"OK Analyzed {ticker_symbol}: {decision['action']}")

            except Exception as e:
                self.logger.error(
                    f"ERR DAILY analysis failed for {ticker_symbol}: {e}",
                    exc_info=True,
                )
                if not dry_run:
                    ErrorAlerter.alert(
                        error_code="MISSING_TICKER_DATA",
                        message=f"Daily analysis failed for {ticker_symbol}: {e}",
                        details={"ticker": ticker_symbol},
                        severity="auto",
                    )

        if not daily_candidates:
            self.logger.warning("No candidates generated")
            for ticker_symbol in tickers_to_process:
                self._log_go_live_metrics(ticker_symbol)
            return

        no_trade_candidates = [
            c for c in daily_candidates if c["decision"].get("no_trade", False)
        ]
        allocatable = [
            c for c in daily_candidates if not c["decision"].get("no_trade", False)
        ]

        self.logger.info(
            f"Allocatable candidates: {len(allocatable)}/{len(daily_candidates)}"
        )

        finalized_decisions = self.allocate_capital(allocatable)

        if Config.ENABLE_POSITION_SIZING and finalized_decisions:
            dm = DataManager()
            latest = dm.fetch_latest_portfolio_state(source="paper")
            equity = latest.get("equity", Config.INITIAL_CAPITAL)
            finalized_decisions = [
                apply_position_sizing(item, equity=equity)
                for item in finalized_decisions
            ]

        summary_lines = []
        detail_lines = []

        for item in no_trade_candidates:
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
                self.logger.warning(
                    "EXPLAINABILITY_LINT %s: %s",
                    ticker_symbol,
                    ", ".join(lint_result["issues"]),
                )

            decision_source = item.get("decision_source") or payload.get(
                "decision_source"
            )
            if decision_source == "fallback":
                decision_id = self.persist_decision(
                    payload=payload,
                    decision=decision,
                    explanation=explanation,
                    audit=audit,
                    position_sizing=item.get("position_sizing"),
                    decision_source=decision_source,
                )
                item["decision_id"] = decision_id

            audit_summary = build_audit_summary(audit, payload, decision)
            summary_lines.append(
                format_email_summary(
                    ticker=ticker_symbol,
                    decision=decision,
                    audit=audit_summary,
                    position_sizing=item.get("position_sizing"),
                )
            )
            detail_lines.append(
                format_email_detail(
                    explanation=explanation,
                    audit=audit_summary,
                )
            )

        for item in finalized_decisions:
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
                self.logger.warning(
                    "EXPLAINABILITY_LINT %s: %s",
                    ticker_symbol,
                    ", ".join(lint_result["issues"]),
                )

            if decision.get("action_code") == 1:
                amount = item.get("allocation_amount", 0)
                self.logger.info(f"  ALLOC {ticker_symbol}: ${amount:,.2f} allocated")

            decision_id = self.persist_decision(
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
            summary_lines.append(
                format_email_summary(
                    ticker=ticker_symbol,
                    decision=decision,
                    audit=audit_summary,
                    position_sizing=item.get("position_sizing"),
                )
            )
            detail_lines.append(
                format_email_detail(
                    explanation=explanation,
                    audit=audit_summary,
                )
            )

        if summary_lines or detail_lines:
            subject = f"Napi ajánlások ({date.today().isoformat()})"
            sections = []
            if summary_lines:
                sections.append("Summary:\n" + "\n".join(summary_lines))
            if detail_lines:
                limited_details = detail_lines[: Config.EMAIL_MAX_DETAIL_LINES]
                sections.append("Details:\n" + "\n\n".join(limited_details))

            body = "\n\n".join(sections)
            max_chars = Config.EMAIL_MAX_BODY_CHARS
            if max_chars and len(body) > max_chars:
                body = body[: max_chars - 12] + "\n[truncated]"
                self.logger.warning(
                    "EMAIL_TRUNCATED max_chars=%s detail_lines=%s",
                    max_chars,
                    len(detail_lines),
                )

            if dry_run:
                self.logger.info(f"[DRY-RUN] Email would be sent: {subject}")
                self.logger.debug(f"Email body:\n{body}")
            else:
                try:
                    self.send_notifications(subject, body, Config.NOTIFY_EMAIL)
                    self.logger.info(f"OK Email sent to {Config.NOTIFY_EMAIL}")
                except Exception as e:
                    self.logger.error(f"ERR Failed to send email: {e}")
                    ErrorAlerter.alert(
                        error_code="AUTHENTICATION_FAILED",
                        message=f"Failed to send notification email: {e}",
                        details={"recipient": Config.NOTIFY_EMAIL},
                        severity="auto",
                    )

        if finalized_decisions:
            as_of = date.fromisoformat(finalized_decisions[0]["payload"]["timestamp"])
            self.execute_trades(finalized_decisions, as_of=as_of)

        for ticker_symbol in tickers_to_process:
            self._log_go_live_metrics(ticker_symbol)

        self.logger.info("=" * 80)
        self.logger.info("DAILY pipeline completed")
        self.logger.info("=" * 80)
