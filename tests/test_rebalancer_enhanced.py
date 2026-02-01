"""
Tests for Enhanced Rebalancer (Sprint 7)
"""

import pytest
from datetime import datetime
from app.decision.rebalancer import PortfolioRebalancer


class TestRebalancerEnhanced:
    
    def test_minimize_rebalancing_costs_filters_small_trades(self):
        """Test that small trades are filtered out."""
        rebalancer = PortfolioRebalancer()
        
        trades = [
            {"ticker": "AAPL", "action": "BUY", "qty": 10, "price": 150},  # $1500
            {"ticker": "MSFT", "action": "BUY", "qty": 1, "price": 50},    # $50 (small)
            {"ticker": "GOOGL", "action": "SELL", "qty": 5, "price": 100}, # $500
        ]
        
        optimized = rebalancer.minimize_rebalancing_costs(
            trades, min_trade_size=100.0, allow_partial=True
        )
        
        # Should keep AAPL and GOOGL, filter out MSFT
        assert len(optimized) == 2
        assert all(t["ticker"] != "MSFT" for t in optimized)
    
    def test_minimize_rebalancing_costs_no_filter_when_disabled(self):
        """Test that all trades kept when partial=False."""
        rebalancer = PortfolioRebalancer()
        
        trades = [
            {"ticker": "AAPL", "action": "BUY", "qty": 1, "price": 50},  # $50 (small)
        ]
        
        optimized = rebalancer.minimize_rebalancing_costs(
            trades, min_trade_size=100.0, allow_partial=False
        )
        
        # Should keep all trades
        assert len(optimized) == 1
    
    def test_apply_tax_efficiency_prioritizes_losses(self):
        """Test that losses are harvested first."""
        rebalancer = PortfolioRebalancer()
        
        trades = [
            {"ticker": "AAPL", "action": "SELL", "qty": 10, "price": 150},
            {"ticker": "MSFT", "action": "SELL", "qty": 5, "price": 300},
            {"ticker": "GOOGL", "action": "BUY", "qty": 3, "price": 100},
        ]
        
        holdings_age = {
            "AAPL": 300,  # Short-term
            "MSFT": 400,  # Long-term
        }
        
        unrealized_gains = {
            "AAPL": 0.10,   # +10% gain (short-term = bad)
            "MSFT": -0.05,  # -5% loss (tax-loss harvest = good)
        }
        
        optimized = rebalancer.apply_tax_efficiency(
            trades, holdings_age, unrealized_gains
        )
        
        # MSFT (loss) should come before AAPL (short-term gain)
        sell_order = [t["ticker"] for t in optimized if t["action"] == "SELL"]
        assert sell_order.index("MSFT") < sell_order.index("AAPL")
    
    def test_apply_tax_efficiency_prefers_long_term(self):
        """Test that long-term gains preferred over short-term."""
        rebalancer = PortfolioRebalancer()
        
        trades = [
            {"ticker": "AAPL", "action": "SELL", "qty": 10, "price": 150},
            {"ticker": "MSFT", "action": "SELL", "qty": 5, "price": 300},
        ]
        
        holdings_age = {
            "AAPL": 300,  # Short-term
            "MSFT": 400,  # Long-term
        }
        
        unrealized_gains = {
            "AAPL": 0.10,  # +10% gain (short-term)
            "MSFT": 0.15,  # +15% gain (long-term)
        }
        
        optimized = rebalancer.apply_tax_efficiency(
            trades, holdings_age, unrealized_gains
        )
        
        # MSFT (long-term) should come before AAPL (short-term)
        sell_order = [t["ticker"] for t in optimized if t["action"] == "SELL"]
        assert sell_order.index("MSFT") < sell_order.index("AAPL")
    
    def test_rebalance_multi_asset_basic(self):
        """Test multi-asset rebalancing."""
        rebalancer = PortfolioRebalancer()
        
        current_portfolio = {
            "VOO": {"weight": 0.4, "asset_type": "ETF", "sector": "Broad Market"},
            "AAPL": {"weight": 0.3, "asset_type": "STOCK", "sector": "Technology"},
            "JNJ": {"weight": 0.3, "asset_type": "STOCK", "sector": "Healthcare"},
        }
        
        target_allocation = {
            "VOO": {"weight": 0.5, "asset_type": "ETF", "sector": "Broad Market"},
            "AAPL": {"weight": 0.25, "asset_type": "STOCK", "sector": "Technology"},
            "JNJ": {"weight": 0.25, "asset_type": "STOCK", "sector": "Healthcare"},
        }
        
        prices = {"VOO": 400, "AAPL": 150, "JNJ": 160}
        total_value = 100000
        
        trades = rebalancer.rebalance_multi_asset(
            current_portfolio, target_allocation, prices, total_value
        )
        
        # Should have trades for VOO, AAPL, JNJ
        assert len(trades) > 0
        assert all("asset_type" in t for t in trades)
    
        def test_rebalance_multi_asset_with_correlation_filter(self):
            """Test multi-asset rebalancing with correlation awareness."""
        rebalancer = PortfolioRebalancer()
        
        current_portfolio = {
            "AAPL": {"weight": 0.0, "asset_type": "STOCK"},
            "MSFT": {"weight": 0.0, "asset_type": "STOCK"},
        }
        
        target_allocation = {
            "AAPL": {"weight": 0.5, "asset_type": "STOCK"},
            "MSFT": {"weight": 0.5, "asset_type": "STOCK"},
        }
        
        prices = {"AAPL": 150, "MSFT": 300}
        total_value = 100000
        
        # High correlation matrix
        correlation_matrix = {
            "AAPL": {"AAPL": 1.0, "MSFT": 0.95},
            "MSFT": {"AAPL": 0.95, "MSFT": 1.0},
        }
        
        trades = rebalancer.rebalance_multi_asset(
            current_portfolio,
            target_allocation,
            prices,
            total_value,
            correlation_matrix
        )
        
        # With high correlation, should filter to 1 BUY trade
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        assert len(buy_trades) == 1  # Only one of AAPL/MSFT
