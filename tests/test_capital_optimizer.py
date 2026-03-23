"""
Tests for CapitalUtilizationOptimizer module.
"""

import pytest
import math
from datetime import datetime
from app.decision.capital_optimizer import (
    CapitalUtilizationOptimizer,
    PositionSize,
    CapitalAllocation
)


@pytest.fixture
def optimizer():
    """Create a test optimizer."""
    return CapitalUtilizationOptimizer(
        total_capital=100000.0,
        max_position_pct=0.05,
        min_position_size=100.0
    )


class TestKellyCriterion:
    """Test Kelly criterion position sizing."""
    
    def test_kelly_with_winning_odds(self, optimizer):
        """Test Kelly fraction with winning trading odds."""
        # 60% win rate, avg win $100, avg loss $100
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=100.0
        )
        
        # Kelly % = (0.6 * 100 - 0.4 * 100) / 100 = 0.20
        assert kelly == pytest.approx(0.20, rel=0.01)
    
    def test_kelly_with_losing_odds(self, optimizer):
        """Test Kelly fraction with losing trading odds."""
        # 40% win rate, avg win $100, avg loss $100
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.40,
            avg_win=100.0,
            avg_loss=100.0
        )
        
        # Kelly % = (0.4 * 100 - 0.6 * 100) / 100 = -0.20 (clamped to 0.01)
        assert kelly == 0.01
    
    def test_kelly_with_different_payoffs(self, optimizer):
        """Test Kelly with different win/loss amounts."""
        # 50% win rate, avg win $150, avg loss $100
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.50,
            avg_win=150.0,
            avg_loss=100.0
        )
        
        # Kelly % = (0.5 * 150 - 0.5 * 100) / 150  0.167
        assert kelly == pytest.approx(0.167, rel=0.01)
    
    def test_kelly_clamped_to_max(self, optimizer):
        """Test Kelly is clamped to maximum of 0.50."""
        # Very high win rate
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.99,
            avg_win=1000.0,
            avg_loss=1.0
        )
        
        assert kelly <= 0.50
    
    def test_kelly_with_zero_avg_loss(self, optimizer):
        """Test Kelly with zero average loss returns 0."""
        kelly = optimizer.calculate_kelly_fraction(
            win_rate=0.50,
            avg_win=100.0,
            avg_loss=0.0
        )
        
        assert kelly == 0.0


class TestPositionSizing:
    """Test optimal position size calculation."""
    
    def test_optimal_position_basic(self, optimizer):
        """Test basic position sizing."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="AAPL",
            kelly_fraction=0.25,
            volatility=0.02
        )
        
        assert pos.ticker == "AAPL"
        assert pos.kelly_fraction == 0.25
        assert pos.volatility == 0.02
        assert pos.optimal_size > 0
        assert pos.risk_adjusted_size > 0
    
    def test_position_respects_max_limit(self, optimizer):
        """Test position sizing respects maximum limit."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.50,
            volatility=0.01
        )
        
        # Max position = 100000 * 0.05 = 5000
        assert pos.risk_adjusted_size <= 5000.0
    
    def test_position_with_high_volatility(self, optimizer):
        """Test position sizing reduced with high volatility."""
        pos_low_vol = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.05,
            volatility=0.01
        )
        
        pos_high_vol = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.05,
            volatility=0.20
        )
        
        assert pos_high_vol.risk_adjusted_size < pos_low_vol.risk_adjusted_size
    
    def test_position_respects_min_size(self, optimizer):
        """Test position sizing respects minimum size."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.01,
            volatility=0.20
        )
        
        assert pos.risk_adjusted_size >= 100.0
    
    def test_portfolio_weight_calculation(self, optimizer):
        """Test portfolio weight calculation."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.25,
            volatility=0.02
        )
        
        expected_weight = pos.risk_adjusted_size / optimizer.total_capital
        assert pos.portfolio_weight == pytest.approx(expected_weight, rel=0.01)


class TestCapitalAllocationOptimization:
    """Test capital allocation optimization."""
    
    def test_allocate_single_position(self, optimizer):
        """Test allocating capital to single position."""
        positions = {
            "AAPL": {"kelly": 0.25, "volatility": 0.02, "expected_return": 0.10}
        }
        
        allocation = optimizer.optimize_capital_allocation(positions)
        
        assert allocation.total_capital == 100000.0
        assert "AAPL" in allocation.positions
        assert allocation.positions["AAPL"] > 0
    
    def test_allocate_multiple_positions(self, optimizer):
        """Test allocating capital to multiple positions."""
        positions = {
            "AAPL": {"kelly": 0.25, "volatility": 0.02},
            "MSFT": {"kelly": 0.20, "volatility": 0.03},
            "GOOGL": {"kelly": 0.15, "volatility": 0.04}
        }
        
        allocation = optimizer.optimize_capital_allocation(positions)
        
        assert len(allocation.positions) == 3
        assert allocation.allocated_capital <= allocation.total_capital
        assert allocation.utilization_rate > 0
    
    def test_utilization_rate(self, optimizer):
        """Test utilization rate calculation."""
        positions = {
            "AAPL": {"kelly": 0.25, "volatility": 0.02}
        }
        
        allocation = optimizer.optimize_capital_allocation(positions)
        
        expected_rate = allocation.allocated_capital / allocation.total_capital
        assert allocation.utilization_rate == pytest.approx(expected_rate, rel=0.01)
    
    def test_diversification_score_single(self, optimizer):
        """Test diversification score for single position."""
        positions = {
            "AAPL": {"kelly": 0.25, "volatility": 0.02}
        }
        
        allocation = optimizer.optimize_capital_allocation(positions)
        
        # Single position = 1.0 (fully concentrated)
        assert allocation.diversification_score == pytest.approx(1.0, rel=0.01)
    
    def test_diversification_score_multiple(self, optimizer):
        """Test diversification score improves with multiple positions."""
        positions = {
            "AAPL": {"kelly": 0.20, "volatility": 0.02},
            "MSFT": {"kelly": 0.20, "volatility": 0.02},
            "GOOGL": {"kelly": 0.20, "volatility": 0.02}
        }
        
        allocation = optimizer.optimize_capital_allocation(positions)
        
        # Multiple equal positions = lower concentration
        assert allocation.diversification_score < 1.0


class TestDiversificationScoring:
    """Test diversification score calculation (Herfindahl index)."""
    
    def test_herfindahl_perfect_equal(self, optimizer):
        """Test Herfindahl for perfectly equal positions."""
        positions = {
            "AAPL": 10000.0,
            "MSFT": 10000.0,
            "GOOGL": 10000.0
        }
        
        score = optimizer._calculate_diversification_score(positions)
        
        # HHI = (1/3)^2 + (1/3)^2 + (1/3)^2  0.333
        assert score == pytest.approx(1.0/3, rel=0.01)
    
    def test_herfindahl_single_position(self, optimizer):
        """Test Herfindahl for single position."""
        positions = {"AAPL": 100000.0}
        
        score = optimizer._calculate_diversification_score(positions)
        
        assert score == pytest.approx(1.0, rel=0.01)
    
    def test_herfindahl_concentrated(self, optimizer):
        """Test Herfindahl for concentrated portfolio."""
        positions = {
            "AAPL": 80000.0,
            "MSFT": 20000.0
        }
        
        score = optimizer._calculate_diversification_score(positions)
        
        # HHI = 0.8^2 + 0.2^2 = 0.68
        assert score == pytest.approx(0.68, rel=0.01)


class TestMaxDrawdownEstimation:
    """Test maximum drawdown estimation."""
    
    def test_estimate_drawdown_basic(self, optimizer):
        """Test basic max drawdown estimation."""
        positions = {"AAPL": 50000.0}
        volatility = 0.02
        
        drawdown = optimizer.estimate_max_drawdown(positions, volatility)
        
        assert 0 <= drawdown <= 1.0
    
    def test_drawdown_capped_at_100(self, optimizer):
        """Test drawdown is capped at 100%."""
        positions = {"AAPL": 100000.0}
        volatility = 1.0  # Extremely high
        
        drawdown = optimizer.estimate_max_drawdown(positions, volatility)
        
        assert drawdown <= 1.0
    
    def test_drawdown_increases_with_volatility(self, optimizer):
        """Test drawdown increases with volatility."""
        positions = {"AAPL": 50000.0}
        
        dd_low = optimizer.estimate_max_drawdown(positions, 0.01)
        dd_high = optimizer.estimate_max_drawdown(positions, 0.05)
        
        assert dd_high > dd_low


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_positions(self, optimizer):
        """Test allocation with empty positions."""
        allocation = optimizer.optimize_capital_allocation({})
        
        assert allocation.allocated_capital == 0
        assert allocation.utilization_rate == 0
    
    def test_zero_kelly(self, optimizer):
        """Test position sizing with zero kelly fraction."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.0,
            volatility=0.02
        )
        
        # Should get minimum position size
        assert pos.risk_adjusted_size >= 100.0
    
    def test_very_low_volatility(self, optimizer):
        """Test position sizing with very low volatility."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.25,
            volatility=0.001
        )
        
        assert pos.risk_adjusted_size > 0
    
    def test_custom_max_position(self, optimizer):
        """Test position sizing with custom max position override."""
        pos = optimizer.calculate_optimal_position_size(
            ticker="TEST",
            kelly_fraction=0.50,
            volatility=0.01,
            max_position_override=2000.0
        )
        
        assert pos.risk_adjusted_size <= 2000.0
