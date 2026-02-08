import logging
from datetime import date
from typing import Dict, List, Optional

from app.backtesting.history_store import HistoryStore
from app.config.config import Config
from app.data_access.data_cleaner import prepare_df
from app.decision.allocation import allocate_capital
from app.decision.decision_policy import apply_decision_policy
from app.decision.recommendation_builder import build_explanation, build_recommendation
from app.decision.recommender import generate_daily_recommendation_payload
from app.notifications.alerter import ErrorAlerter
from app.reporting.audit_builder import build_audit_metadata, build_audit_summary
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.notifications.email_formatter import format_email_line


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
            model_id=payload.get("model_id"),
            timestamp=payload.get("timestamp"),
        )

    def send_notifications(self, subject: str, body: str, recipient: str) -> None:
        self.email_notifier.send(subject, body, recipient)

    def execute_trades(self, decisions: List[Dict], as_of: date) -> None:
        self.execution_engine.execute(decisions=decisions, as_of=as_of)

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

        tickers_to_process = self.get_tickers_to_process(ticker)
        daily_candidates = []

        for ticker_symbol in tickers_to_process:
            try:
                candidate = self.build_daily_candidate(ticker_symbol)
                daily_candidates.append(candidate)

                decision = candidate["decision"]
                self.logger.info(f"✓ Analyzed {ticker_symbol}: {decision['action']}")

            except Exception as e:
                self.logger.error(
                    f"✗ DAILY analysis failed for {ticker_symbol}: {e}",
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
            return

        allocatable = [
            c for c in daily_candidates if not c["decision"].get("no_trade", False)
        ]

        self.logger.info(
            f"Allocatable candidates: {len(allocatable)}/{len(daily_candidates)}"
        )

        finalized_decisions = self.allocate_capital(allocatable)

        email_lines = []

        for item in finalized_decisions:
            ticker_symbol = item["ticker"]
            decision = item["decision"]
            payload = item["payload"]
            audit = item["audit"]
            explanation = item["explanation"]

            if decision.get("action_code") == 1:
                amount = item.get("allocation_amount", 0)
                self.logger.info(f"  💰 {ticker_symbol}: ${amount:,.2f} allocated")

            decision_id = self.persist_decision(
                payload=payload,
                decision=decision,
                explanation=explanation,
                audit=audit,
            )
            item["decision_id"] = decision_id

            email_lines.append(
                format_email_line(
                    explanation=explanation,
                    decision=decision,
                    audit=build_audit_summary(audit, payload, decision),
                )
            )

        if email_lines:
            subject = f"Napi ajánlások ({date.today().isoformat()})"
            body = "\n".join(email_lines)

            if dry_run:
                self.logger.info(f"[DRY-RUN] Email would be sent: {subject}")
                self.logger.debug(f"Email body:\n{body}")
            else:
                try:
                    self.send_notifications(subject, body, Config.NOTIFY_EMAIL)
                    self.logger.info(f"✓ Email sent to {Config.NOTIFY_EMAIL}")
                except Exception as e:
                    self.logger.error(f"✗ Failed to send email: {e}")
                    ErrorAlerter.alert(
                        error_code="AUTHENTICATION_FAILED",
                        message=f"Failed to send notification email: {e}",
                        details={"recipient": Config.NOTIFY_EMAIL},
                        severity="auto",
                    )

        if finalized_decisions:
            as_of = date.fromisoformat(finalized_decisions[0]["payload"]["timestamp"])
            self.execute_trades(finalized_decisions, as_of=as_of)

        self.logger.info("=" * 80)
        self.logger.info("DAILY pipeline completed")
        self.logger.info("=" * 80)
