"""
Portfolio Rebalancing Module (P7 — Portfolio Optimization)

Responsibility:
    - Track portfolio drift from target allocation
    - Trigger rebalancing when drift exceeds threshold
    - Execute rebalancing trades to restore target weights
    - Audit trail of rebalancing actions

Features:
    - Current weight computation from positions
    - Target weight specification per ticker
    - Drift detection and threshold-based triggering
    - Rebalancing trade generation

Usage:
    rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)
    should_rebal = rebalancer.should_rebalance(current_weights, target_weights)
    if should_rebal["should_rebalance"]:
        trades = rebalancer.generate_rebalancing_trades(...)
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class PortfolioRebalancer:
    """
    Manages portfolio rebalancing to maintain target allocations.

    Attributes:
        rebalance_threshold: Drift % to trigger rebalancing (e.g., 0.20 = 20%)
        max_rebalance_trades: Maximum number of trades per rebalancing event
    """

    def __init__(
        self, rebalance_threshold: float = 0.20, max_rebalance_trades: int = 5
    ):
        """
        Initialize rebalancer.

        Args:
            rebalance_threshold: If avg drift > this %, trigger rebalancing
            max_rebalance_trades: Maximum trades to prevent overtrading
        """
        self.rebalance_threshold = rebalance_threshold
        self.max_rebalance_trades = max_rebalance_trades

    def should_rebalance(
        self, current_weights: Dict[str, float], target_weights: Dict[str, float]
    ) -> Dict:
        """
        Compare current vs. target allocation and detect drift.

        Args:
            current_weights: {ticker: current_pct}  (sum=1.0)
            target_weights: {ticker: target_pct}    (sum=1.0)

        Returns:
            {
                "should_rebalance": bool,
                "drift_per_ticker": {ticker: drift_pct},
                "drift_avg": float,
                "drift_max": float,
                "message": str
            }
        """
        # TODO: Implement
        # 1. Compute drift for each ticker: abs(current - target) / target
        # 2. Calculate average drift
        # 3. Find maximum drift
        # 4. Check if average drift exceeds threshold
        # 5. Log drift details

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
        """
        Generate rebalancing trades to restore target allocation.

        Args:
            current_weights: {ticker: current_pct}
            target_weights: {ticker: target_pct}
            current_prices: {ticker: price}
            total_portfolio_value: Portfolio market value

        Returns:
            [{ticker, action (BUY|SELL), qty, reason}, ...]
        """
        # TODO: Implement
        # 1. For each ticker, compute target vs. current dollar amount
        # 2. Compute needed trade: qty = (target_amt - current_amt) / price
        # 3. Sort by absolute trade size (largest first)
        # 4. Limit to max_rebalance_trades
        # 5. Generate trade instructions

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
                    "reason": f"Rebalance: {current_pct:.1%} → {target_pct:.1%}",
                    "timestamp": datetime.now().isoformat(),
                }
                trades.append(trade)

        # Sort by absolute size and limit
        trades = sorted(trades, key=lambda t: t["qty"], reverse=True)
        trades = trades[: self.max_rebalance_trades]

        logger.info(f"Generated {len(trades)} rebalancing trades")

        return trades

    def compute_rebalancing_cost(self, trades: List[Dict]) -> float:
        """
        Estimate transaction cost of rebalancing.

        Args:
            trades: Output from generate_rebalancing_trades()

        Returns:
            Total estimated transaction cost in dollars
        """
        # TODO: Implement
        total_cost = 0.0

        for trade in trades:
            notional = trade["qty"] * trade["price"]
            # Commission + slippage + spread
            cost = notional * (
                Config.TRANSACTION_FEE_PCT + Config.MIN_SLIPPAGE_PCT + Config.SPREAD_PCT
            )
            total_cost += cost

        return total_cost


def check_and_rebalance(
    current_positions: Dict[str, float],
    target_allocation: Dict[str, float],
    prices: Dict[str, float],
    total_value: float,
) -> Dict:
    """
    Convenience function: check if rebalancing needed and generate trades.

    Returns:
        {
            "should_rebalance": bool,
            "drift_info": {...},
            "trades": [...],
            "estimated_cost": float
        }
    """
    # TODO: Implement
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
