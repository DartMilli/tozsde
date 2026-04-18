from datetime import date
from typing import Dict, List, Optional

from app.application.use_cases.execution_coordinator import ExecutionCoordinator
from app.application.use_cases.notification_coordinator import NotificationCoordinator
from app.application.use_cases.result import UseCaseResult, ok
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

    def _run_rebalancer_check(self, daily_candidates: List[Dict]) -> None:
        """Check portfolio drift against today's target allocation.

        Only active when ENABLE_REBALANCER=true. No-op otherwise.
        Safe to call even when daily_candidates is empty.
        """
        cfg = self.pipeline._get_settings()
        if not getattr(cfg, "ENABLE_REBALANCER", False):
            return

        try:
            if not daily_candidates:
                return

            today = date.fromisoformat(
                daily_candidates[0]["payload"].get(
                    "timestamp", date.today().isoformat()
                )
            )
            state = self.pipeline.state_repo.fetch_latest_portfolio_state(
                source="paper"
            )
            if not state:
                return

            equity = state.get("equity", 0.0)
            if equity <= 0:
                return

            raw_positions = state.get(
                "positions", {}
            )  # {ticker: {"qty":..., "entry_price":...}}

            # Build prices and target_allocation from today's candidates
            prices: Dict[str, float] = {
                item["ticker"]: item["payload"].get(
                    "current_price",
                    item["payload"].get("latest_price", 1.0),
                )
                for item in daily_candidates
            }
            target_allocation: Dict[str, float] = {
                item["ticker"]: item.get("allocation_pct", 0.0)
                for item in daily_candidates
            }

            # Compute current position weights using today's prices where available
            current_weights: Dict[str, float] = {}
            for ticker, pos in raw_positions.items():
                qty = pos.get("qty", 0)
                price = prices.get(ticker, pos.get("entry_price", 1.0))
                current_weights[ticker] = (qty * price) / equity

            # Only check tickers present in today's candidates
            filtered_weights = {
                t: current_weights.get(t, 0.0) for t in target_allocation
            }

            result = self.pipeline.check_portfolio_drift(
                current_positions=filtered_weights,
                target_allocation=target_allocation,
                prices=prices,
                total_value=equity,
            )

            if result.get("should_rebalance"):
                trades = result.get("trades", [])
                estimated_cost = result.get("estimated_cost", 0.0)
                benefit = self._estimate_rebalance_benefit(result, equity)
                self.pipeline.logger.info(
                    "Rebalancer: %d suggested trade(s), estimated cost $%.2f, benefit $%.2f",
                    len(trades),
                    estimated_cost,
                    benefit,
                )

                execute_enabled = getattr(cfg, "REBALANCE_EXECUTE", False)
                cost_multiplier = getattr(cfg, "REBALANCE_COST_MULTIPLIER", 2.0)
                if (
                    execute_enabled
                    and trades
                    and benefit > estimated_cost * cost_multiplier
                ):
                    rebalance_decisions = self._convert_rebalance_trades_to_decisions(
                        trades=trades,
                        today=today,
                    )
                    self.execution.execute_finalized(rebalance_decisions)
                    self.pipeline.logger.info(
                        "Rebalancer executed %d trade(s)",
                        len(rebalance_decisions),
                    )
                elif execute_enabled and trades:
                    self.pipeline.logger.info(
                        "Rebalancer skipped execution: benefit %.2f <= cost threshold %.2f",
                        benefit,
                        estimated_cost * cost_multiplier,
                    )
        except Exception as exc:
            self.pipeline.logger.warning("Rebalancer check skipped: %s", exc)

    def _estimate_rebalance_benefit(self, result: Dict, equity: float) -> float:
        cfg = self.pipeline._get_settings()
        drift_avg = result.get("drift_info", {}).get("drift_avg", 0.0)
        impact_factor = getattr(cfg, "DRIFT_ANNUAL_IMPACT_FACTOR", 0.5)
        return float(equity) * float(drift_avg) * float(impact_factor)

    def _convert_rebalance_trades_to_decisions(
        self,
        trades: List[Dict],
        today: date,
    ) -> List[Dict]:
        converted = []
        for trade in trades:
            action = trade.get("action", "BUY")
            price = float(trade.get("price", 0.0))
            qty = float(trade.get("qty", 0.0))
            converted.append(
                {
                    "ticker": trade["ticker"],
                    "decision": {
                        "action_code": 1 if action == "BUY" else 2,
                        "action": action,
                        "confidence": 1.0,
                        "wf_score": 1.0,
                    },
                    "payload": {
                        "timestamp": today.isoformat(),
                        "latest_price": price,
                    },
                    "allocation_amount": round(price * qty, 2),
                }
            )
        return converted

    def _run_shadow_evaluation(
        self,
        daily_candidates: List[Dict],
        today: date,
        dry_run: bool,
    ) -> None:
        cfg = self.pipeline._get_settings()
        if not getattr(cfg, "ENABLE_SHADOW_EVAL", False) or dry_run:
            return

        try:
            from app.models.shadow_evaluator import ShadowEvaluator

            evaluator = ShadowEvaluator(
                settings=cfg,
                history_store=self.pipeline.history_store,
                data_fetcher=self.pipeline.data_fetcher,
                email_notifier=self.pipeline.email_notifier,
            )
            for candidate in daily_candidates:
                results = evaluator.evaluate_daily_shadows(
                    champion_candidate=candidate,
                    as_of_date=today,
                )
                if results:
                    promotions = sum(1 for item in results if item.get("promote"))
                    self.pipeline.logger.info(
                        "Shadow evaluation %s: %d challenger(s), %d promotion(s)",
                        candidate["ticker"],
                        len(results),
                        promotions,
                    )
        except Exception as exc:
            self.pipeline.logger.warning("Shadow evaluation skipped: %s", exc)

    def run(self, dry_run: bool = False, ticker: Optional[str] = None) -> UseCaseResult:
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
            return ok(
                "daily_pipeline",
                data={"completed": True, "processed": 0},
                dry_run=dry_run,
            )

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
        self._run_rebalancer_check(daily_candidates)
        self._run_shadow_evaluation(daily_candidates, today, dry_run)

        for ticker_symbol in tickers_to_process:
            self.pipeline._log_go_live_metrics(ticker_symbol)

        self.pipeline.logger.info("=" * 80)
        self.pipeline.logger.info("DAILY pipeline completed")
        self.pipeline.logger.info("=" * 80)

        return ok(
            "daily_pipeline",
            data={
                "completed": True,
                "processed": len(daily_candidates),
                "executed": len(finalized_decisions),
                "no_trade": len(no_trade_candidates),
            },
            dry_run=dry_run,
        )
