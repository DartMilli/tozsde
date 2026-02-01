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


    def minimize_rebalancing_costs(
        self,
        trades: List[Dict],
        min_trade_size: float = 100.0,
        allow_partial: bool = True
    ) -> List[Dict]:
        """
        Minimize transaction costs by filtering/aggregating trades.
        
        Strategies:
        - Skip trades below min_trade_size (notional value)
        - Aggregate trades in same direction (if exchange allows)
        - Prioritize larger trades (lower relative cost)
        
        Args:
            trades: Original trade list
            min_trade_size: Minimum trade notional value in dollars
            allow_partial: Allow partial rebalancing (skip small trades)
            
        Returns:
            Filtered/optimized trade list
        """
        optimized_trades = []
        
        for trade in trades:
            notional = trade["qty"] * trade["price"]
            
            # Skip small trades
            if allow_partial and notional < min_trade_size:
                logger.debug(
                    f"Skipping small trade: {trade['ticker']} "
                    f"notional=${notional:.2f} < ${min_trade_size}"
                )
                continue
            
            optimized_trades.append(trade)
        
        # Log savings
        original_cost = self.compute_rebalancing_cost(trades)
        optimized_cost = self.compute_rebalancing_cost(optimized_trades)
        savings = original_cost - optimized_cost
        
        logger.info(
            f"Cost optimization: {len(trades)} → {len(optimized_trades)} trades, "
            f"saved ${savings:.2f} ({savings/original_cost*100:.1f}% reduction)"
        )
        
        return optimized_trades
    
    def apply_tax_efficiency(
        self,
        trades: List[Dict],
        holdings_age: Dict[str, int],  # {ticker: days_held}
        unrealized_gains: Dict[str, float]  # {ticker: gain_pct}
    ) -> List[Dict]:
        """
        Apply tax-efficient rebalancing rules.
        
        Rules:
        1. Prefer selling long-term holdings (> 365 days) → lower capital gains tax
        2. Defer short-term capital gains (< 365 days) unless necessary
        3. Prioritize tax-loss harvesting (sell losers first)
        
        Args:
            trades: Original trade list
            holdings_age: Days each position has been held
            unrealized_gains: Unrealized gain/loss percentage per ticker
            
        Returns:
            Tax-optimized trade list
        """
        # Separate SELL trades
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        
        # Score each SELL trade (lower score = better tax efficiency)
        scored_sells = []
        
        for trade in sell_trades:
            ticker = trade["ticker"]
            age = holdings_age.get(ticker, 0)
            gain_pct = unrealized_gains.get(ticker, 0)
            
            # Tax score calculation
            if gain_pct < 0:
                # Loss → harvest immediately (score = 0)
                tax_score = 0
            elif age >= 365:
                # Long-term gain → acceptable (score = 1)
                tax_score = 1
            else:
                # Short-term gain → defer if possible (score = 2)
                tax_score = 2
            
            scored_sells.append((trade, tax_score, age, gain_pct))
        
        # Sort by tax efficiency (best first)
        scored_sells.sort(key=lambda x: (x[1], -x[2]))  # tax_score ASC, age DESC
        
        # Reconstruct trade list (tax-efficient sells + all buys)
        tax_efficient_trades = [t[0] for t in scored_sells] + buy_trades
        
        logger.info(
            f"Tax optimization: prioritized {len([s for s in scored_sells if s[1] == 0])} "
            f"loss harvests, deferred {len([s for s in scored_sells if s[1] == 2])} "
            f"short-term gains"
        )
        
        return tax_efficient_trades
    
    def rebalance_multi_asset(
        self,
        current_portfolio: Dict[str, Dict],  # {ticker: {weight, asset_type, sector}}
        target_allocation: Dict[str, Dict],  # {ticker: {weight, asset_type, sector}}
        prices: Dict[str, float],
        total_value: float,
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[Dict]:
        """
        Rebalance multi-asset portfolio (ETF + stocks + bonds).
        
        Features:
        - Cross-asset rebalancing (sell ETF, buy stocks or vice versa)
        - Correlation-aware rebalancing (avoid concentrating in correlated assets)
        - Asset-class level constraints
        
        Args:
            current_portfolio: {ticker: {weight, asset_type, sector}}
            target_allocation: {ticker: {weight, asset_type, sector}}
            prices: {ticker: price}
            total_value: Portfolio total value
            correlation_matrix: Optional correlation data for optimization
            
        Returns:
            Rebalancing trades
        """
        trades = []
        
        # Compute target vs. current per ticker
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
                    "timestamp": datetime.now().isoformat()
                }
                trades.append(trade)
        
        # Apply correlation-aware filtering (optional)
        if correlation_matrix:
            trades = self._filter_correlated_trades(trades, correlation_matrix)
        
        # Sort by asset type (ETF first, then stocks, then bonds)
        asset_type_order = {"ETF": 0, "STOCK": 1, "BOND": 2}
        trades.sort(
            key=lambda t: (asset_type_order.get(t.get("asset_type", "STOCK"), 3), -t["qty"])
        )
        
        # Limit total trades
        trades = trades[:self.max_rebalance_trades]
        
        logger.info(
            f"Multi-asset rebalancing: {len(trades)} trades across asset classes"
        )
        
        return trades
    
    def _filter_correlated_trades(
        self,
        trades: List[Dict],
        correlation_matrix: Dict[str, Dict[str, float]],
        max_correlation: float = 0.8
    ) -> List[Dict]:
        """
        Filter trades to avoid concentrating in highly correlated assets.
        
        If adding two BUY trades with correlation > 0.8, keep only one.
        """
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        
        filtered_buys = []
        
        for trade in buy_trades:
            ticker = trade["ticker"]
            
            # Check if highly correlated with already selected buys
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
