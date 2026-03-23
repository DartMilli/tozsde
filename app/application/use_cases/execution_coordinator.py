from datetime import date
from typing import Dict, List, Tuple

from app.core.decision.position_sizer import apply_position_sizing


class ExecutionCoordinator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def split_and_finalize(
        self, daily_candidates: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        no_trade_candidates = [
            c for c in daily_candidates if c["decision"].get("no_trade", False)
        ]
        allocatable = [
            c for c in daily_candidates if not c["decision"].get("no_trade", False)
        ]

        self.pipeline.logger.info(
            "Allocatable candidates: %s/%s",
            len(allocatable),
            len(daily_candidates),
        )

        finalized_decisions = self.pipeline.allocate_capital(allocatable)

        cfg = self.pipeline._get_settings()
        if getattr(cfg, "ENABLE_POSITION_SIZING") and finalized_decisions:
            latest = self.pipeline.state_repo.fetch_latest_portfolio_state(
                source="paper"
            )
            equity = latest.get("equity", getattr(cfg, "INITIAL_CAPITAL"))
            finalized_decisions = [
                apply_position_sizing(
                    item, equity=equity, settings=self.pipeline.settings
                )
                for item in finalized_decisions
            ]

        return no_trade_candidates, finalized_decisions

    def execute_finalized(self, finalized_decisions: List[Dict]) -> None:
        if not finalized_decisions:
            return
        as_of = date.fromisoformat(finalized_decisions[0]["payload"]["timestamp"])
        self.pipeline.execute_trades(finalized_decisions, as_of=as_of)
