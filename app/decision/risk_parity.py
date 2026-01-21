"""
Risk Parity Allocation Module (P7 — Portfolio Optimization)

Responsibility:
    - Allocate capital using inverse volatility (risk parity)
    - Ensure each position contributes equally to portfolio risk
    - Replace basic equal-weight allocation with dynamic risk-based sizing

Features:
    - Volatility computation per ticker
    - Inverse-volatility weighting
    - Capital allocation with risk parity
    - Integration with decision builder

Usage:
    allocator = RiskParityAllocator(lookback_days=60)
    decisions = allocator.allocate(decisions, price_history)
    # Each position now sized inversely to its volatility
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class RiskParityAllocator:
    """
    Allocates capital using inverse volatility (risk parity principle).

    Principle:
        - Assets with higher volatility get smaller positions
        - Assets with lower volatility get larger positions
        - Each position contributes equally to portfolio risk

    Attributes:
        lookback_days: Days of price history to use for volatility calculation
    """

    def __init__(self, lookback_days: int = 60):
        """
        Initialize risk parity allocator.

        Args:
            lookback_days: Number of past days for volatility estimation
        """
        self.lookback_days = lookback_days

    def allocate(
        self, decisions: List[Dict], price_history: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Allocate capital using risk parity.

        Args:
            decisions: [{ticker, signal, confidence, ...}]
            price_history: {ticker: array of closing prices}

        Returns:
            decisions with updated allocation_amount and allocation_pct
        """
        # TODO: Implement
        # 1. Filter tradeable positions (signal != HOLD)
        # 2. Compute volatility for each ticker
        # 3. Compute inverse-volatility weights
        # 4. Normalize weights to sum to 1.0
        # 5. Allocate capital: amount = total_capital * weight
        # 6. Update decision dictionaries
        # 7. Log allocation details

        logger.info(f"Risk parity allocation for {len(decisions)} positions")

        return decisions

    def _compute_volatilities(
        self, price_history: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute volatility (annualized standard deviation of returns).

        Args:
            price_history: {ticker: array of prices}

        Returns:
            {ticker: annualized_volatility}
        """
        # TODO: Implement
        volatilities = {}

        for ticker, prices in price_history.items():
            if len(prices) < 2:
                volatilities[ticker] = 0.1  # Default high volatility
                continue

            # Compute daily returns
            returns = np.diff(prices) / prices[:-1]

            # Annualize (252 trading days per year)
            daily_std = np.std(returns)
            annualized_vol = daily_std * np.sqrt(252)

            volatilities[ticker] = max(annualized_vol, 0.01)  # Floor at 1%
            logger.debug(f"{ticker}: volatility = {annualized_vol:.2%}")

        return volatilities

    def _compute_inverse_volatility_weights(
        self, volatilities: Dict[str, float], tradeable_tickers: List[str]
    ) -> np.ndarray:
        """
        Compute inverse-volatility weights (normalized).

        Args:
            volatilities: {ticker: volatility}
            tradeable_tickers: Tickers to allocate to

        Returns:
            Array of normalized weights, one per tradeable ticker
        """
        # TODO: Implement
        # 1. Weight = 1 / volatility for each ticker
        # 2. Normalize: weight / sum(all weights)
        # 3. Handle zero/very small volatilities gracefully

        inv_vols = np.array(
            [1.0 / max(volatilities.get(t, 0.1), 0.01) for t in tradeable_tickers]
        )
        weights = inv_vols / inv_vols.sum()

        return weights

    def _apply_allocation(
        self, decisions: List[Dict], weights: np.ndarray, tradeable_indices: List[int]
    ) -> List[Dict]:
        """
        Apply computed weights to decision dictionaries.

        Args:
            decisions: Original decisions list
            weights: Normalized weight array
            tradeable_indices: Indices of tradeable positions in decisions

        Returns:
            decisions with allocation_amount and allocation_pct updated
        """
        # TODO: Implement
        capital = Config.INITIAL_CAPITAL * 0.95  # Reserve 5% cash buffer

        for idx, weight in zip(tradeable_indices, weights):
            allocation_amt = capital * weight
            decisions[idx]["allocation_amount"] = round(allocation_amt, 2)
            decisions[idx]["allocation_pct"] = round(weight, 4)

            logger.info(
                f"{decisions[idx]['ticker']}: {weight:.1%} allocation "
                f"(${allocation_amt:.2f})"
            )

        return decisions


def apply_risk_parity(
    decisions: List[Dict], price_history: Dict[str, np.ndarray]
) -> List[Dict]:
    """
    Convenience function: apply risk parity allocation to decisions.

    Args:
        decisions: List of decision dictionaries
        price_history: {ticker: price array}

    Returns:
        decisions with allocations updated
    """
    allocator = RiskParityAllocator(lookback_days=60)
    return allocator.allocate(decisions, price_history)
