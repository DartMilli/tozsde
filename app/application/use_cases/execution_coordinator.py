from datetime import date
from typing import Dict, List, Tuple

from app.core.decision.implementation_shortfall import ImplementationShortfallEstimator
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

        if getattr(cfg, "ENABLE_COST_GATE", False) and finalized_decisions:
            estimator = ImplementationShortfallEstimator(cfg)
            finalized_decisions = [
                item
                for item in finalized_decisions
                if self._passes_cost_gate(item, estimator)
            ]

        return no_trade_candidates, finalized_decisions

    def _passes_cost_gate(
        self,
        item: Dict,
        estimator: ImplementationShortfallEstimator,
    ) -> bool:
        notional = float(item.get("allocation_amount", 0.0))
        if notional <= 0:
            return False

        payload = item.get("payload", {})
        decision = item.get("decision", {})
        expectancy = payload.get("expectancy", {}) or {}
        expected_edge = expectancy.get("expected_net_pnl")
        if expected_edge is None:
            return True

        latest_price = float(payload.get("latest_price", 0.0) or 0.0)
        atr = payload.get("atr")
        atr_pct = None
        if atr and latest_price > 0:
            atr_pct = float(atr) / latest_price

        estimate = estimator.estimate(notional=notional, atr_pct=atr_pct)
        if float(expected_edge) < estimate.min_edge_required:
            decision["no_trade"] = True
            decision["no_trade_reason"] = "COST_GATE_INSUFFICIENT_EDGE"
            decision["action_code"] = 0
            self.pipeline.logger.info(
                "Cost gate blocked %s: expected_edge=%.5f required=%.5f",
                item.get("ticker"),
                float(expected_edge),
                estimate.min_edge_required,
            )
            return False

        payload["implementation_shortfall"] = {
            "commission_pct": estimate.commission_pct,
            "slippage_pct": estimate.slippage_pct,
            "spread_pct": estimate.spread_pct,
            "total_cost_pct": estimate.total_cost_pct,
            "min_edge_required": estimate.min_edge_required,
        }
        return True

    def execute_finalized(self, finalized_decisions: List[Dict]) -> None:
        if not finalized_decisions:
            return
        as_of = date.fromisoformat(finalized_decisions[0]["payload"]["timestamp"])
        self.pipeline.execute_trades(finalized_decisions, as_of=as_of)
