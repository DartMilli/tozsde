import logging
from datetime import date
from typing import Optional

import pandas as pd

from app.backtesting.history_store import HistoryStore
from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.decision.position_sizer import apply_position_sizing
from app.decision.volatility import compute_normalized_volatility
from app.data_access.data_cleaner import prepare_df
from app.reporting.audit_builder import build_audit_metadata
from app.models.model_trainer import TradingEnv
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.services.paper_execution import PaperExecutionEngine
from app.services.trading_pipeline import TradingPipelineService


class HistoricalPaperRunner:
    """
    Run deterministic, historical paper trading using TradingPipelineService.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.dm = DataManager()

    def run(self, ticker: str, start_date: str, end_date: str) -> None:
        self.dm.initialize_tables()

        fallback_enabled = self._should_use_fallback()
        if fallback_enabled:
            self.logger.warning(
                "Historical runner fallback enabled (no RL model files found)."
            )

        pipeline = TradingPipelineService(
            history_store=HistoryStore(),
            logger=self.logger,
            data_fetcher=MarketDataFetcher(),
            model_runner=ModelEnsembleRunner(
                model_dir=Config.MODEL_DIR, env_class=TradingEnv
            ),
            email_notifier=EmailNotifier(),
            execution_engine=PaperExecutionEngine(self.dm, self.logger),
        )

        days = pd.bdate_range(start=start_date, end=end_date)
        if days.empty:
            self.logger.warning("No business days in range")
            return

        for day in days:
            as_of = day.date()
            if self.dm.has_decision_for_date(ticker=ticker, as_of_date=as_of):
                self.logger.info(
                    f"[SKIP] {ticker} {as_of.isoformat()} decision already exists"
                )
                continue

            try:
                if fallback_enabled:
                    candidate = self._build_fallback_candidate(
                        pipeline=pipeline,
                        ticker=ticker,
                        as_of_date=as_of,
                    )
                else:
                    candidate = pipeline.build_daily_candidate(
                        ticker=ticker,
                        as_of_date=as_of,
                    )
            except Exception as exc:
                self.logger.error(
                    f"[ERROR] {ticker} {as_of.isoformat()} build failed: {exc}",
                    exc_info=True,
                )
                pipeline.execute_trades([], as_of=as_of)
                continue

            daily_candidates = [candidate]
            allocatable = [
                c for c in daily_candidates if not c["decision"].get("no_trade", False)
            ]

            finalized = pipeline.allocate_capital(allocatable)

            if Config.ENABLE_POSITION_SIZING and finalized:
                latest = self.dm.fetch_latest_portfolio_state(source="paper")
                equity = latest.get("equity", Config.INITIAL_CAPITAL)
                finalized = [
                    apply_position_sizing(item, equity=equity) for item in finalized
                ]

            for item in finalized:
                decision = item["decision"]
                decision_id = pipeline.persist_decision(
                    payload=item["payload"],
                    decision=decision,
                    explanation=item["explanation"],
                    audit=item["audit"],
                    position_sizing=item.get("position_sizing"),
                    decision_source=item.get("decision_source"),
                )
                item["decision_id"] = decision_id

                amount = item.get("allocation_amount", 0.0)
                self.logger.info(
                    f"{as_of.isoformat()} {ticker} {decision.get('action')} size={amount:.2f}"
                )

            pipeline.execute_trades(finalized, as_of=as_of)

    def _should_use_fallback(self) -> bool:
        model_dir = Config.MODEL_DIR
        if not model_dir.exists():
            return True
        return not any(model_dir.glob("*.zip"))

    def _build_fallback_candidate(
        self,
        pipeline: TradingPipelineService,
        ticker: str,
        as_of_date: date,
    ) -> dict:
        start = as_of_date - pd.Timedelta(days=180)
        df_full = pipeline.data_fetcher.load_data(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=as_of_date.strftime("%Y-%m-%d"),
        )
        if df_full.empty:
            raise ValueError("NO_DATA")

        df = prepare_df(df_full.copy(), ticker)
        latest_price = float(df["Close"].iloc[-1]) if not df.empty else None
        volatility = compute_normalized_volatility(df) if not df.empty else 0.0

        payload = {
            "ticker": ticker,
            "timestamp": as_of_date.isoformat(),
            "latest_price": latest_price,
            "volatility": volatility,
            "model_votes": [],
            "wf_scores": [],
            "ensemble_quality": 0.0,
        }

        decision = {
            "action_code": 0,
            "action": "HOLD",
            "confidence": 0.0,
            "wf_score": 0.0,
            "strength": "NO_TRADE",
            "ensemble_quality": 0.0,
            "quality_score": 0.0,
            "no_trade": True,
            "no_trade_reason": "FALLBACK_NO_MODELS",
        }

        audit = build_audit_metadata(payload, decision)
        explanation = {
            "hu": "Fallback: nincs RL modell, nincs kereskedes.",
            "en": "Fallback: no RL models available; no trade executed.",
        }

        return {
            "ticker": ticker,
            "payload": payload,
            "decision": decision,
            "explanation": explanation,
            "audit": audit,
            "decision_source": "fallback",
        }
