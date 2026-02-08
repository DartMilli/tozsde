from datetime import date, timedelta
import json
from typing import Optional

import pandas as pd

from app.backtesting.outcome_evaluator import OutcomeEvaluator
from app.services.trading_pipeline import TradingPipelineService


class PipelineBacktester:
    """
    Backtest runner that reuses TradingPipelineService for decisions
    and persists records using the shared schema.
    """

    def __init__(self, pipeline: TradingPipelineService):
        self.pipeline = pipeline

    def run(
        self,
        ticker: str,
        start: date,
        end: date,
        debug: bool = False,
        evaluate_outcomes: bool = True,
    ) -> int:
        if start > end:
            return 0

        df = self.pipeline.load_market_data(
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if df is None or df.empty:
            return 0

        df = df[(df.index.date >= start) & (df.index.date <= end)]
        dates = pd.to_datetime(df.index).date

        persisted = 0
        for as_of in dates:
            candidate = self.pipeline.build_daily_candidate(
                ticker=ticker,
                debug=debug,
                as_of_date=as_of,
            )
            self.pipeline.persist_decision(
                payload=candidate["payload"],
                decision=candidate["decision"],
                explanation=candidate["explanation"],
                audit=candidate["audit"],
            )
            self.pipeline.history_store.dm.save_portfolio_state(
                timestamp=as_of.isoformat(),
                cash=None,
                equity=None,
                pnl_pct=None,
                positions_json=json.dumps(
                    {
                        "ticker": ticker,
                        "action": candidate["decision"].get("action"),
                        "action_code": candidate["decision"].get("action_code"),
                    }
                ),
                source="pipeline_backtest",
            )
            persisted += 1

        if evaluate_outcomes:
            lookback_days = (end - start).days + 1
            OutcomeEvaluator().evaluate_past_decisions(lookback_days=lookback_days)

        return persisted
