import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List

from app.config.config import Config
from app.data_access.data_manager import DataManager


@dataclass
class PaperPosition:
    ticker: str
    qty: float
    entry_price: float
    entry_timestamp: str
    decision_id: int


class PaperExecutionEngine:
    """
    Simulated execution for paper trading mode.
    """

    def __init__(self, dm: DataManager, logger):
        self.dm = dm
        self.logger = logger

    def execute(self, decisions: List[Dict], as_of: date) -> None:
        state = self._load_latest_state()
        positions = state.get("positions", {})
        cash = state.get("cash", Config.INITIAL_CAPITAL)

        for item in decisions:
            decision = item["decision"]
            payload = item["payload"]
            ticker = item["ticker"]
            action_code = decision.get("action_code")
            price = payload.get("latest_price")

            if price is None:
                self.logger.warning(f"Paper mode: missing price for {ticker}")
                continue

            if action_code == 1:
                allocation_amount = item.get("allocation_amount", 0.0)
                if allocation_amount <= 0:
                    continue
                qty = allocation_amount / price
                cash -= allocation_amount
                positions[ticker] = PaperPosition(
                    ticker=ticker,
                    qty=qty,
                    entry_price=price,
                    entry_timestamp=payload.get("timestamp") or as_of.isoformat(),
                    decision_id=item.get("decision_id"),
                )

            elif action_code == 2 and ticker in positions:
                pos = positions[ticker]
                proceeds = pos.qty * price
                cash += proceeds

                pnl_pct = (price - pos.entry_price) / pos.entry_price
                entry_date = datetime.fromisoformat(pos.entry_timestamp).date()
                horizon_days = (as_of - entry_date).days

                outcome = {
                    "pnl_pct": round(pnl_pct, 4),
                    "evaluated_at": datetime.now().isoformat(),
                    "exit_reason": "PAPER_SELL",
                    "horizon_days": horizon_days,
                    "exit_price": price,
                }

                if pos.decision_id is not None:
                    self.dm.save_outcome(
                        decision_id=pos.decision_id,
                        ticker=ticker,
                        decision_timestamp=pos.entry_timestamp,
                        pnl_pct=outcome["pnl_pct"],
                        success=outcome["pnl_pct"] > 0,
                        future_return=outcome["pnl_pct"],
                        exit_reason=outcome["exit_reason"],
                        horizon_days=outcome["horizon_days"],
                        outcome_json=json.dumps(outcome),
                    )
                else:
                    self.logger.warning(
                        f"Paper mode: missing decision_id for outcome ({ticker})"
                    )

                positions.pop(ticker, None)

        equity = cash + sum(
            pos.qty * decisions_by_ticker(decisions).get(pos.ticker, pos.entry_price)
            for pos in positions.values()
        )
        pnl_pct = (equity - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL

        self.dm.save_portfolio_state(
            timestamp=as_of.isoformat(),
            cash=cash,
            equity=equity,
            pnl_pct=pnl_pct,
            positions_json=json.dumps(
                {
                    t: {
                        "qty": p.qty,
                        "entry_price": p.entry_price,
                        "entry_timestamp": p.entry_timestamp,
                        "decision_id": p.decision_id,
                    }
                    for t, p in positions.items()
                }
            ),
            source="paper",
        )

    def _load_latest_state(self) -> Dict:
        latest = self.dm.fetch_latest_portfolio_state(source="paper")
        if not latest:
            return {"cash": Config.INITIAL_CAPITAL, "positions": {}}
        return {
            "cash": latest.get("cash", Config.INITIAL_CAPITAL),
            "positions": latest.get("positions", {}),
        }


def decisions_by_ticker(decisions: List[Dict]) -> Dict[str, float]:
    prices = {}
    for item in decisions:
        price = item.get("payload", {}).get("latest_price")
        if price is not None:
            prices[item["ticker"]] = price
    return prices
