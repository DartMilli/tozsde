"""Portfolio rebalancing core logic."""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class PortfolioRebalancer:
    """Manages portfolio rebalancing to maintain target allocations."""

    def __init__(
        self,
        rebalance_threshold: float = 0.20,
        max_rebalance_trades: int = 5,
        settings=None,
    ):
        self.rebalance_threshold = rebalance_threshold
        self.max_rebalance_trades = max_rebalance_trades
        self.settings = settings

    def should_rebalance(
        self, current_weights: Dict[str, float], target_weights: Dict[str, float]
    ) -> Dict:
        drifts = {}
        for ticker in target_weights:
            target = target_weights.get(ticker, 0)
            current = current_weights.get(ticker, 0)

            if target > 0:
                drift = abs(current - target) / target
            else:
                drift = 0.0

            drifts[ticker] = drift

        avg_drift = np.mean(list(drifts.values())) if drifts else 0.0
        max_drift = max(drifts.values()) if drifts else 0.0

        should_rebal = avg_drift > self.rebalance_threshold

        if should_rebal:
            logger.info(
                f"Rebalancing triggered: avg_drift={avg_drift:.1%} "
                f"> threshold={self.rebalance_threshold:.1%}"
            )

        return {
            "should_rebalance": should_rebal,
            "drift_per_ticker": drifts,
            "drift_avg": avg_drift,
            "drift_max": max_drift,
            "message": (
                f"Portfolio drift {avg_drift:.1%}"
                if should_rebal
                else "Portfolio within threshold"
            ),
        }

    def generate_rebalancing_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        total_portfolio_value: float,
    ) -> List[Dict]:
        trades = []

        for ticker in target_weights:
            target_pct = target_weights[ticker]
            current_pct = current_weights.get(ticker, 0)

            target_amt = total_portfolio_value * target_pct
            current_amt = total_portfolio_value * current_pct

            diff_amt = target_amt - current_amt
            price = current_prices.get(ticker, 1.0)
            qty = int(diff_amt / price)

            if qty != 0:
                trade = {
                    "ticker": ticker,
                    "action": "BUY" if qty > 0 else "SELL",
                    "qty": abs(qty),
                    "price": price,
                    "reason": f"Rebalance: {current_pct:.1%} -> {target_pct:.1%}",
                    "timestamp": datetime.now().isoformat(),
                }
                trades.append(trade)

        trades = sorted(trades, key=lambda t: t["qty"], reverse=True)
        trades = trades[: self.max_rebalance_trades]

        logger.info(f"Generated {len(trades)} rebalancing trades")

        return trades

    def compute_rebalancing_cost(self, trades: List[Dict]) -> float:
        total_cost = 0.0

        cfg = self.settings or build_settings()
        for trade in trades:
            notional = trade["qty"] * trade["price"]
            cost = notional * (
                getattr(cfg, "TRANSACTION_FEE_PCT")
                + getattr(cfg, "MIN_SLIPPAGE_PCT")
                + getattr(cfg, "SPREAD_PCT")
            )
            total_cost += cost

        return total_cost

    def minimize_rebalancing_costs(
        self,
        trades: List[Dict],
        min_trade_size: float = 100.0,
        allow_partial: bool = True,
    ) -> List[Dict]:
        optimized_trades = []

        for trade in trades:
            notional = trade["qty"] * trade["price"]

            if allow_partial and notional < min_trade_size:
                logger.debug(
                    f"Skipping small trade: {trade['ticker']} "
                    f"notional=${notional:.2f} < ${min_trade_size}"
                )
                continue

            optimized_trades.append(trade)

        original_cost = self.compute_rebalancing_cost(trades)
        optimized_cost = self.compute_rebalancing_cost(optimized_trades)
        savings = original_cost - optimized_cost

        logger.info(
            f"Cost optimization: {len(trades)} -> {len(optimized_trades)} trades, "
            f"saved ${savings:.2f} ({savings/original_cost*100:.1f}% reduction)"
        )

        return optimized_trades

    def apply_tax_efficiency(
        self,
        trades: List[Dict],
        holdings_age: Dict[str, int],
        unrealized_gains: Dict[str, float],
    ) -> List[Dict]:
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        buy_trades = [t for t in trades if t["action"] == "BUY"]

        scored_sells = []

        for trade in sell_trades:
            ticker = trade["ticker"]
            age = holdings_age.get(ticker, 0)
            gain_pct = unrealized_gains.get(ticker, 0)

            if gain_pct < 0:
                tax_score = 0
            elif age >= 365:
                tax_score = 1
            else:
                tax_score = 2

            scored_sells.append((trade, tax_score, age, gain_pct))

        scored_sells.sort(key=lambda x: (x[1], -x[2]))

        tax_efficient_trades = [t[0] for t in scored_sells] + buy_trades

        logger.info(
            f"Tax optimization: prioritized {len([s for s in scored_sells if s[1] == 0])} "
            f"loss harvests, deferred {len([s for s in scored_sells if s[1] == 2])} "
            f"short-term gains"
        )

        return tax_efficient_trades

    def rebalance_multi_asset(
        self,
        current_portfolio: Dict[str, Dict],
        target_allocation: Dict[str, Dict],
        prices: Dict[str, float],
        total_value: float,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Dict]:
        trades = []

        all_tickers = set(current_portfolio.keys()) | set(target_allocation.keys())

        for ticker in all_tickers:
            current_data = current_portfolio.get(ticker, {"weight": 0})
            target_data = target_allocation.get(ticker, {"weight": 0})

            current_weight = current_data.get("weight", 0)
            target_weight = target_data.get("weight", 0)

            current_amt = total_value * current_weight
            target_amt = total_value * target_weight

            diff_amt = target_amt - current_amt
            price = prices.get(ticker, 1.0)
            qty = int(diff_amt / price)

            if qty != 0:
                asset_type = target_data.get("asset_type", "STOCK")
                trade = {
                    "ticker": ticker,
                    "action": "BUY" if qty > 0 else "SELL",
                    "qty": abs(qty),
                    "price": price,
                    "asset_type": asset_type,
                    "reason": f"Multi-asset rebalance: {asset_type}",
                    "timestamp": datetime.now().isoformat(),
                }
                trades.append(trade)

        if correlation_matrix:
            trades = self._filter_correlated_trades(trades, correlation_matrix)

        asset_type_order = {"ETF": 0, "STOCK": 1, "BOND": 2}
        trades.sort(
            key=lambda t: (
                asset_type_order.get(t.get("asset_type", "STOCK"), 3),
                -t["qty"],
            )
        )

        trades = trades[: self.max_rebalance_trades]

        logger.info(
            f"Multi-asset rebalancing: {len(trades)} trades across asset classes"
        )

        return trades

    def _filter_correlated_trades(
        self,
        trades: List[Dict],
        correlation_matrix: Dict[str, Dict[str, float]],
        max_correlation: float = 0.8,
    ) -> List[Dict]:
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]

        filtered_buys = []

        for trade in buy_trades:
            ticker = trade["ticker"]

            is_redundant = False
            for selected in filtered_buys:
                selected_ticker = selected["ticker"]
                corr = correlation_matrix.get(ticker, {}).get(selected_ticker, 0)

                if abs(corr) > max_correlation:
                    logger.debug(
                        f"Skipping {ticker} (corr={corr:.2f} with {selected_ticker})"
                    )
                    is_redundant = True
                    break

            if not is_redundant:
                filtered_buys.append(trade)

        return filtered_buys + sell_trades


def check_and_rebalance(
    current_positions: Dict[str, float],
    target_allocation: Dict[str, float],
    prices: Dict[str, float],
    total_value: float,
) -> Dict:
    rebalancer = PortfolioRebalancer()

    drift_info = rebalancer.should_rebalance(current_positions, target_allocation)

    if drift_info["should_rebalance"]:
        trades = rebalancer.generate_rebalancing_trades(
            current_positions, target_allocation, prices, total_value
        )
        cost = rebalancer.compute_rebalancing_cost(trades)
    else:
        trades = []
        cost = 0.0

    return {
        "should_rebalance": drift_info["should_rebalance"],
        "drift_info": drift_info,
        "trades": trades,
        "estimated_cost": cost,
    }
