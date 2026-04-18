import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List

from app.config.build_settings import build_settings


@dataclass
class PaperPosition:
    ticker: str
    qty: float
    entry_price: float
    entry_timestamp: str
    decision_id: int
    strategy_name: str = "RL_ENSEMBLE"


class PaperExecutionEngine:
    """
    Simulated execution for paper trading mode.
    """

    def __init__(self, dm, logger, settings=None):
        self.dm = dm
        self.logger = logger
        self.settings = settings

    def execute(self, decisions: List[Dict], as_of: date) -> None:
        state = self._load_latest_state()
        positions = state.get("positions", {})
        cfg = self.settings or build_settings()
        cash = state.get("cash", getattr(cfg, "INITIAL_CAPITAL"))

        # S14.2 Drawdown circuit breaker: suppress BUY orders when portfolio
        # losses exceed DRAWDOWN_HALT_PCT (e.g. 10 %).
        drawdown_halt = getattr(cfg, "DRAWDOWN_HALT_PCT", 0.10)
        initial_capital = getattr(cfg, "INITIAL_CAPITAL", 10_000.0)
        portfolio_value = state.get("equity", cash)
        drawdown_pct = (
            (initial_capital - portfolio_value) / initial_capital
            if initial_capital
            else 0.0
        )
        if drawdown_pct >= drawdown_halt:
            self.logger.warning(
                "Circuit breaker active: drawdown %.1f%% >= halt threshold %.1f%%. "
                "Only SELL orders will be executed.",
                drawdown_pct * 100,
                drawdown_halt * 100,
            )
            decisions = [d for d in decisions if d["decision"].get("action_code") == 2]
        execution_policy = getattr(cfg, "EXECUTION_POLICY", "next_open")
        execution_policy = (execution_policy or "next_open").lower()
        if execution_policy not in {"close_to_close", "next_open"}:
            execution_policy = "next_open"

        exec_prices = {}
        exec_dates = []
        fee_pct = getattr(cfg, "TRANSACTION_FEE_PCT", 0.0)

        for item in decisions:
            decision = item["decision"]
            payload = item["payload"]
            ticker = item["ticker"]
            action_code = decision.get("action_code")
            price, exec_date = self._resolve_execution_price(
                ticker=ticker,
                payload=payload,
                as_of=as_of,
                execution_policy=execution_policy,
            )

            if price is None:
                self.logger.warning(
                    f"Paper mode: missing execution price for {ticker} ({execution_policy})"
                )
                continue

            payload["execution_price"] = price
            payload["execution_date"] = exec_date.isoformat()
            exec_prices[ticker] = price
            exec_dates.append(exec_date)

            if action_code == 1:
                allocation_amount = item.get("allocation_amount", 0.0)
                if allocation_amount <= 0:
                    continue
                # S14.1 ATR-based position sizing: if ATR is present in the
                # payload use it to cap shares to a risk-per-share budget.
                # Fall back to the pre-calculated allocation_amount otherwise.
                atr = payload.get("atr") if payload else None
                atr_multiplier = getattr(cfg, "ATR_MULTIPLIER", 2.0)
                risk_fraction = getattr(cfg, "RISK", 0.02)
                if atr and atr > 0:
                    risk_budget = cash * risk_fraction
                    risk_per_share = atr * atr_multiplier
                    atr_qty = math.floor(risk_budget / risk_per_share)
                    allocation_amount = min(allocation_amount, atr_qty * price)
                commission = allocation_amount * fee_pct
                total_cost = allocation_amount + commission
                if total_cost > cash:
                    self.logger.warning(
                        "Paper mode: insufficient cash for %s (need=%.2f incl. fee, have=%.2f)",
                        ticker,
                        total_cost,
                        cash,
                    )
                    continue
                qty = allocation_amount / price
                cash -= total_cost
                positions[ticker] = PaperPosition(
                    ticker=ticker,
                    qty=qty,
                    entry_price=price,
                    entry_timestamp=payload.get("execution_date")
                    or payload.get("timestamp")
                    or as_of.isoformat(),
                    decision_id=item.get("decision_id"),
                    strategy_name=decision.get("strategy_name", "RL_ENSEMBLE"),
                )

            elif action_code == 2 and ticker in positions:
                pos = positions[ticker]
                proceeds = pos.qty * price
                commission = proceeds * fee_pct
                cash += proceeds - commission

                pnl_pct = (price - pos.entry_price) / pos.entry_price
                entry_date = datetime.fromisoformat(pos.entry_timestamp).date()
                horizon_days = (exec_date - entry_date).days

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

                if getattr(cfg, "ENABLE_ADAPTIVE_STRATEGY", False):
                    from app.core.decision.adaptive_strategy_selector import (
                        AdaptiveStrategySelector,
                    )

                    selector = AdaptiveStrategySelector(
                        settings=cfg, data_manager=self.dm
                    )
                    selector.update_strategy(
                        pos.strategy_name or "RL_ENSEMBLE",
                        success=outcome["pnl_pct"] > 0,
                    )

                positions.pop(ticker, None)

        equity = cash + sum(
            pos.qty * exec_prices.get(pos.ticker, pos.entry_price)
            for pos in positions.values()
        )
        initial_cap = getattr(cfg, "INITIAL_CAPITAL")
        pnl_pct = (equity - initial_cap) / initial_cap

        portfolio_date = max(exec_dates) if exec_dates else as_of

        self.dm.save_portfolio_state(
            timestamp=portfolio_date.isoformat(),
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
                        "strategy_name": p.strategy_name,
                    }
                    for t, p in positions.items()
                }
            ),
            source="paper",
        )

    def _resolve_execution_price(
        self,
        ticker: str,
        payload: Dict,
        as_of: date,
        execution_policy: str,
    ) -> tuple[float | None, date]:
        if execution_policy == "close_to_close":
            return payload.get("latest_price"), as_of

        start_date = (as_of + timedelta(days=1)).isoformat()
        df = self.dm.load_ohlcv(ticker=ticker, start_date=start_date)
        if df is None or df.empty:
            return None, as_of

        df = df[df.index.date > as_of]
        if df.empty:
            return None, as_of

        row = df.iloc[0]
        exec_date = df.index[0].date()
        open_price = float(
            row.get("open", row.get("close", row.get("Open", row.get("Close"))))
        )
        return open_price, exec_date

    def _load_latest_state(self) -> Dict:
        cfg = self.settings or build_settings()
        latest = self.dm.fetch_latest_portfolio_state(source="paper")
        if not latest:
            return {"cash": getattr(cfg, "INITIAL_CAPITAL"), "positions": {}}
        raw_positions = latest.get("positions", {})
        positions = {
            ticker: (
                PaperPosition(ticker=ticker, **v)
                if isinstance(v, dict) and "ticker" not in v
                else PaperPosition(**v) if isinstance(v, dict) else v
            )
            for ticker, v in raw_positions.items()
        }
        return {
            "cash": latest.get("cash", getattr(cfg, "INITIAL_CAPITAL")),
            "positions": positions,
        }


def decisions_by_ticker(decisions: List[Dict]) -> Dict[str, float]:
    prices = {}
    for item in decisions:
        payload = item.get("payload", {})
        price = payload.get("execution_price")
        if price is None:
            price = payload.get("latest_price")
        if price is not None:
            prices[item["ticker"]] = price
    return prices
