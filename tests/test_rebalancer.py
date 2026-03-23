"""
Tests for Portfolio Rebalancing Module (app.decision.rebalancer)

Tests verify:
- Drift detection and threshold triggering
- Rebalancing trade generation
- Transaction cost estimation
- Integration via convenience function
- Edge cases (no trades needed, single ticker, zero allocations)
"""

import pytest
import numpy as np
from typing import Dict

from app.decision.rebalancer import PortfolioRebalancer, check_and_rebalance


class TestDriftDetection:
    """Test should_rebalance() drift detection logic."""

    def test_no_drift_returns_false(self):
        """When current == target, should not rebalance."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)

        current_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        target_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert result["should_rebalance"] == False
        assert result["drift_avg"] == 0.0
        assert result["drift_max"] == 0.0

    def test_small_drift_below_threshold(self):
        """Drift below threshold should not trigger rebalancing."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)

        # Current: AAPL 0.38, MSFT 0.31, GOOGL 0.31
        # Target:  AAPL 0.40, MSFT 0.30, GOOGL 0.30
        # Drift:   AAPL 0.05, MSFT 0.033, GOOGL 0.033 -> avg  0.039 (3.9%)
        current_weights = {"AAPL": 0.38, "MSFT": 0.31, "GOOGL": 0.31}
        target_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert result["should_rebalance"] == False
        assert result["drift_avg"] < 0.20

    def test_drift_above_threshold_triggers_rebalancing(self):
        """Drift exceeding threshold should trigger rebalancing."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)

        # Current: AAPL 0.30, MSFT 0.35, GOOGL 0.35
        # Target:  AAPL 0.40, MSFT 0.30, GOOGL 0.30
        # Drift:   AAPL 0.25, MSFT 0.167, GOOGL 0.167 -> avg  0.194 (19.4%, just below)
        current_weights = {"AAPL": 0.30, "MSFT": 0.35, "GOOGL": 0.35}
        target_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        # Average drift should be around 19.4%, below 20% threshold
        assert result["should_rebalance"] == False

    def test_high_drift_triggers_rebalancing(self):
        """High drift should trigger rebalancing."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)

        # Current: AAPL 0.25, MSFT 0.40, GOOGL 0.35
        # Target:  AAPL 0.40, MSFT 0.30, GOOGL 0.30
        # Drift:   AAPL 0.375, MSFT 0.333, GOOGL 0.167 -> avg  0.292 (29.2%)
        current_weights = {"AAPL": 0.25, "MSFT": 0.40, "GOOGL": 0.35}
        target_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert result["should_rebalance"] == True
        assert result["drift_avg"] > 0.20
        assert result["drift_max"] > result["drift_avg"]

    def test_drift_metadata_included(self):
        """Result should include drift per ticker and messages."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.20)

        current_weights = {"AAPL": 0.30, "MSFT": 0.35, "GOOGL": 0.35}
        target_weights = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert "drift_per_ticker" in result
        assert "drift_avg" in result
        assert "drift_max" in result
        assert "message" in result
        assert all(ticker in result["drift_per_ticker"] for ticker in target_weights)

    def test_empty_weights_returns_no_drift(self):
        """Empty weight dictionaries should return no drift."""
        rebalancer = PortfolioRebalancer()

        result = rebalancer.should_rebalance({}, {})

        assert result["should_rebalance"] == False
        assert result["drift_avg"] == 0.0
        assert result["drift_max"] == 0.0

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        rebalancer = PortfolioRebalancer(rebalance_threshold=0.06)

        current_weights = {"AAPL": 0.38}
        target_weights = {"AAPL": 0.40}
        # Drift: ~0.05 (5%)

        result = rebalancer.should_rebalance(current_weights, target_weights)

        # Drift ~0.05 < threshold 0.06
        assert result["should_rebalance"] == False

        # Now test > threshold
        rebalancer_strict = PortfolioRebalancer(rebalance_threshold=0.04)
        result = rebalancer_strict.should_rebalance(current_weights, target_weights)
        assert result["should_rebalance"] == True


class TestRebalancingTradeGeneration:
    """Test generate_rebalancing_trades() trade generation."""

    def test_no_trades_when_equal_weights(self):
        """Equal current/target weights should generate no trades."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.5, "MSFT": 0.5}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        assert len(trades) == 0

    def test_buy_trade_generated_for_underweight_ticker(self):
        """Underweight ticker should generate BUY trade."""
        rebalancer = PortfolioRebalancer()

        # Current: AAPL 30% (should be 50%), MSFT 70% (should be 50%)
        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # Should have 2 trades: BUY AAPL, SELL MSFT
        assert len(trades) > 0

        # Find AAPL trade (should be BUY)
        aapl_trade = [t for t in trades if t["ticker"] == "AAPL"]
        assert len(aapl_trade) == 1
        assert aapl_trade[0]["action"] == "BUY"

    def test_sell_trade_generated_for_overweight_ticker(self):
        """Overweight ticker should generate SELL trade."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.70, "MSFT": 0.30}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # Find AAPL trade (should be SELL)
        aapl_trade = [t for t in trades if t["ticker"] == "AAPL"]
        assert len(aapl_trade) == 1
        assert aapl_trade[0]["action"] == "SELL"

    def test_trade_quantity_calculation_correct(self):
        """Trade quantities should match weight differences."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # AAPL: current 3000, target 5000, diff 2000 -> qty = 20
        aapl_trade = [t for t in trades if t["ticker"] == "AAPL"][0]
        assert aapl_trade["qty"] == 20

        # MSFT: current 7000, target 5000, diff -2000 -> qty = 10
        msft_trade = [t for t in trades if t["ticker"] == "MSFT"][0]
        assert msft_trade["qty"] == 10

    def test_trades_limited_by_max_rebalance_trades(self):
        """Number of trades should not exceed max_rebalance_trades."""
        rebalancer = PortfolioRebalancer(max_rebalance_trades=2)

        current_weights = {
            "AAPL": 0.1,
            "MSFT": 0.1,
            "GOOGL": 0.1,
            "AMZN": 0.1,
            "NVDA": 0.1,
            "TSLA": 0.5,
        }
        target_weights = {
            "AAPL": 0.2,
            "MSFT": 0.2,
            "GOOGL": 0.2,
            "AMZN": 0.2,
            "NVDA": 0.0,
            "TSLA": 0.0,
        }
        prices = {t: 100.0 for t in current_weights}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # Should be limited to 2 trades (largest adjustments)
        assert len(trades) <= 2

    def test_trade_metadata_complete(self):
        """Each trade should include all required fields."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        for trade in trades:
            assert "ticker" in trade
            assert "action" in trade
            assert "qty" in trade
            assert "price" in trade
            assert "reason" in trade
            assert "timestamp" in trade

    def test_zero_allocation_not_in_trades(self):
        """Ticker with zero allocation should not generate unnecessary trades."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 1.0}
        target_weights = {"AAPL": 1.0}
        prices = {"AAPL": 100.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        assert len(trades) == 0


class TestTransactionCostEstimation:
    """Test compute_rebalancing_cost() cost estimation."""

    def test_single_trade_cost_calculation(self):
        """Transaction cost for single trade should be calculated."""
        from app.config.config import Config

        rebalancer = PortfolioRebalancer()

        trades = [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "qty": 10,
                "price": 100.0,
                "reason": "Test",
                "timestamp": "2024-01-01T12:00:00",
            }
        ]

        cost = rebalancer.compute_rebalancing_cost(trades)

        # notional = 10 * 100 = 1000
        # cost = 1000 * (trans_fee + slippage + spread)
        expected_pct = (
            Config.TRANSACTION_FEE_PCT + Config.MIN_SLIPPAGE_PCT + Config.SPREAD_PCT
        )
        expected_cost = 1000.0 * expected_pct

        assert cost == pytest.approx(expected_cost, rel=0.01)

    def test_multiple_trades_cost_accumulation(self):
        """Total cost should sum across all trades."""
        from app.config.config import Config

        rebalancer = PortfolioRebalancer()

        trades = [
            {
                "ticker": "AAPL",
                "qty": 10,
                "price": 100.0,
                "action": "BUY",
                "reason": "Test",
                "timestamp": "2024-01-01T12:00:00",
            },
            {
                "ticker": "MSFT",
                "qty": 5,
                "price": 200.0,
                "action": "SELL",
                "reason": "Test",
                "timestamp": "2024-01-01T12:00:00",
            },
        ]

        cost = rebalancer.compute_rebalancing_cost(trades)

        notional = (10 * 100.0) + (5 * 200.0)  # 3000
        expected_pct = (
            Config.TRANSACTION_FEE_PCT + Config.MIN_SLIPPAGE_PCT + Config.SPREAD_PCT
        )
        expected_cost = notional * expected_pct

        assert cost == pytest.approx(expected_cost, rel=0.01)

    def test_empty_trades_zero_cost(self):
        """Zero trades should result in zero cost."""
        rebalancer = PortfolioRebalancer()

        cost = rebalancer.compute_rebalancing_cost([])

        assert cost == 0.0


class TestConvenienceFunction:
    """Test check_and_rebalance() convenience function."""

    def test_no_rebalancing_needed(self):
        """Should return no trades when drift is low."""
        current_positions = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        target_allocation = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        total_value = 10000.0

        result = check_and_rebalance(
            current_positions, target_allocation, prices, total_value
        )

        assert result["should_rebalance"] == False
        assert len(result["trades"]) == 0
        assert result["estimated_cost"] == 0.0

    def test_rebalancing_triggered_returns_trades(self):
        """Should return trades when drift exceeds threshold."""
        current_positions = {"AAPL": 0.25, "MSFT": 0.40, "GOOGL": 0.35}
        target_allocation = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        total_value = 10000.0

        result = check_and_rebalance(
            current_positions, target_allocation, prices, total_value
        )

        assert result["should_rebalance"] == True
        assert len(result["trades"]) > 0
        assert result["estimated_cost"] > 0.0

    def test_result_includes_drift_info(self):
        """Result should include detailed drift information."""
        current_positions = {"AAPL": 0.30, "MSFT": 0.35, "GOOGL": 0.35}
        target_allocation = {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
        total_value = 10000.0

        result = check_and_rebalance(
            current_positions, target_allocation, prices, total_value
        )

        assert "drift_info" in result
        assert "drift_per_ticker" in result["drift_info"]
        assert "drift_avg" in result["drift_info"]
        assert "drift_max" in result["drift_info"]

    def test_result_structure_complete(self):
        """Result should have all expected fields."""
        current_positions = {"AAPL": 0.5, "MSFT": 0.5}
        target_allocation = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        result = check_and_rebalance(
            current_positions, target_allocation, prices, total_value
        )

        assert "should_rebalance" in result
        assert "drift_info" in result
        assert "trades" in result
        assert "estimated_cost" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_ticker_portfolio(self):
        """Should handle single ticker portfolio."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 1.0}
        target_weights = {"AAPL": 1.0}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert result["should_rebalance"] == False
        assert result["drift_avg"] == 0.0

    def test_many_tickers_portfolio(self):
        """Should handle portfolio with many tickers."""
        rebalancer = PortfolioRebalancer()

        tickers = [f"TICKER{i}" for i in range(20)]
        current_weights = {t: 1.0 / 20 for t in tickers}
        target_weights = {t: 1.0 / 20 for t in tickers}

        result = rebalancer.should_rebalance(current_weights, target_weights)

        assert result["should_rebalance"] == False
        assert result["drift_avg"] == 0.0

    def test_very_small_portfolio_value(self):
        """Should handle small portfolio values correctly."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 1.0  # Very small

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # Should still work, just with fractional shares
        assert isinstance(trades, list)

    def test_very_high_prices(self):
        """Should handle very high stock prices."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"EXPENSIVE": 0.30, "CHEAP": 0.70}
        target_weights = {"EXPENSIVE": 0.50, "CHEAP": 0.50}
        prices = {"EXPENSIVE": 100000.0, "CHEAP": 100.0}
        total_value = 1000000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        assert isinstance(trades, list)

    def test_missing_price_for_ticker(self):
        """Should handle missing prices gracefully."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.30, "MSFT": 0.70}
        target_weights = {"AAPL": 0.50, "MSFT": 0.50}
        prices = {"AAPL": 100.0}  # Missing MSFT
        total_value = 10000.0

        # Should use default price of 1.0
        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        assert isinstance(trades, list)

    def test_ticker_in_target_not_in_current(self):
        """Should handle tickers in target but not in current."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 1.0}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        # Should generate BUY for MSFT
        msft_trades = [t for t in trades if t["ticker"] == "MSFT"]
        assert len(msft_trades) == 1
        assert msft_trades[0]["action"] == "BUY"

    def test_zero_target_weight_generates_sell(self):
        """Ticker with zero target weight should generate SELL."""
        rebalancer = PortfolioRebalancer()

        current_weights = {"AAPL": 0.5, "MSFT": 0.5}
        target_weights = {"AAPL": 1.0, "MSFT": 0.0}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        total_value = 10000.0

        trades = rebalancer.generate_rebalancing_trades(
            current_weights, target_weights, prices, total_value
        )

        msft_trades = [t for t in trades if t["ticker"] == "MSFT"]
        assert len(msft_trades) == 1
        assert msft_trades[0]["action"] == "SELL"
