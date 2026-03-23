from datetime import date
from typing import Dict, List, Optional

from app.application.use_cases.execution_coordinator import ExecutionCoordinator
from app.application.use_cases.notification_coordinator import NotificationCoordinator
from app.notifications.alerter import ErrorAlerter


class DailyPipelineUseCase:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.execution = ExecutionCoordinator(pipeline)
        self.notifications = NotificationCoordinator(pipeline)

    def _collect_candidates(
        self,
        tickers_to_process: List[str],
        today: date,
        dry_run: bool,
    ) -> List[Dict]:
        daily_candidates: List[Dict] = []

        for ticker_symbol in tickers_to_process:
            try:
                if self.pipeline.state_repo.has_decision_for_date(
                    ticker=ticker_symbol,
                    as_of_date=today,
                ):
                    self.pipeline.logger.info(
                        "SKIP %s: decision already exists for %s",
                        ticker_symbol,
                        today.isoformat(),
                    )
                    continue

                candidate = self.pipeline.build_daily_candidate(
                    ticker_symbol,
                    as_of_date=today,
                )
                daily_candidates.append(candidate)
                self.pipeline.logger.info(
                    "OK Analyzed %s: %s",
                    ticker_symbol,
                    candidate["decision"]["action"],
                )
            except Exception as exc:
                self.pipeline.logger.error(
                    "ERR DAILY analysis failed for %s: %s",
                    ticker_symbol,
                    exc,
                    exc_info=True,
                )
                if not dry_run:
                    ErrorAlerter.alert(
                        error_code="MISSING_TICKER_DATA",
                        message=f"Daily analysis failed for {ticker_symbol}: {exc}",
                        details={"ticker": ticker_symbol},
                        severity="auto",
                    )

        return daily_candidates

    def run(self, dry_run: bool = False, ticker: Optional[str] = None) -> Dict:
        self.pipeline.logger.info("=" * 80)
        self.pipeline.logger.info("DAILY pipeline started (dry_run=%s)", dry_run)
        if ticker:
            self.pipeline.logger.info("DEV mode: Analyzing %s only", ticker)
        self.pipeline.logger.info("=" * 80)

        today = date.today()
        tickers_to_process = self.pipeline.get_tickers_to_process(ticker)
        daily_candidates = self._collect_candidates(tickers_to_process, today, dry_run)

        if not daily_candidates:
            self.pipeline.logger.warning("No candidates generated")
            for ticker_symbol in tickers_to_process:
                self.pipeline._log_go_live_metrics(ticker_symbol)
            return {"completed": True, "processed": 0}

        no_trade_candidates, finalized_decisions = self.execution.split_and_finalize(
            daily_candidates
        )

        summary_lines = []
        detail_lines = []

        for item in no_trade_candidates:
            decision_source = item.get("decision_source") or item["payload"].get(
                "decision_source"
            )
            persist = decision_source == "fallback"
            summary, detail = self.notifications.prepare_item(item, persist=persist)
            summary_lines.append(summary)
            detail_lines.append(detail)

        for item in finalized_decisions:
            if item["decision"].get("action_code") == 1:
                amount = item.get("allocation_amount", 0)
                self.pipeline.logger.info(
                    "  ALLOC %s: $%s allocated",
                    item["ticker"],
                    f"{amount:,.2f}",
                )
            summary, detail = self.notifications.prepare_item(item, persist=True)
            summary_lines.append(summary)
            detail_lines.append(detail)

        self.notifications.send_daily_email(
            summary_lines, detail_lines, dry_run=dry_run
        )
        self.execution.execute_finalized(finalized_decisions)

        for ticker_symbol in tickers_to_process:
            self.pipeline._log_go_live_metrics(ticker_symbol)

        self.pipeline.logger.info("=" * 80)
        self.pipeline.logger.info("DAILY pipeline completed")
        self.pipeline.logger.info("=" * 80)

        return {
            "completed": True,
            "processed": len(daily_candidates),
            "executed": len(finalized_decisions),
            "no_trade": len(no_trade_candidates),
        }
