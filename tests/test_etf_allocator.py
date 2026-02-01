"""
Tests for ETF Allocator (Sprint 7)
"""

import pytest
from app.decision.etf_allocator import (
    ETFAllocator,
    AssetType,
    classify_asset_type,
    get_low_cost_etf,
    estimate_portfolio_cost
)


class TestETFAllocator:
    
    def test_classify_asset_type_etf(self):
        """Test ETF classification."""
        assert classify_asset_type("SPY") == AssetType.ETF
        assert classify_asset_type("VOO") == AssetType.ETF
        assert classify_asset_type("QQQ") == AssetType.ETF
    
    def test_classify_asset_type_stock(self):
        """Test stock classification."""
        assert classify_asset_type("AAPL") == AssetType.STOCK
        assert classify_asset_type("MSFT") == AssetType.STOCK
        assert classify_asset_type("GOOGL") == AssetType.STOCK
    
    def test_classify_asset_type_unknown(self):
        """Test unknown ticker."""
        # Unknown tickers default to STOCK
        assert classify_asset_type("ZZZZ") == AssetType.STOCK
        assert classify_asset_type("") == AssetType.STOCK
    
    def test_analyze_sector_exposure(self):
        """Test sector exposure calculation."""
        allocator = ETFAllocator()
        
        portfolio = {
            "AAPL": 0.3,  # Technology
            "MSFT": 0.3,  # Technology
            "JPM": 0.2,   # Financials
            "JNJ": 0.2    # Healthcare
        }
        
        exposure = allocator.analyze_sector_exposure(portfolio)
        
        assert "Technology" in exposure
        assert exposure["Technology"] == pytest.approx(0.6, abs=0.01)
        assert exposure["Financials"] == pytest.approx(0.2, abs=0.01)
        assert exposure["Healthcare"] == pytest.approx(0.2, abs=0.01)
    
    def test_compare_etf_costs(self):
        """Test ETF cost comparison."""
        allocator = ETFAllocator()
        
        comparison = allocator.compare_etf_costs("SPY", "VOO")
        
        assert comparison.cheaper_option in ["SPY", "VOO"]
        assert comparison.annual_savings_per_10k >= 0
        assert 0 <= comparison.cost_difference <= 1000  # in basis points
    
    def test_calculate_optimal_mix_basic(self):
        """Test optimal mix calculation."""
        allocator = ETFAllocator()
        
        # Stocks only portfolio
        stock_portfolio = {
            "AAPL": 0.4,
            "MSFT": 0.3,
            "GOOGL": 0.3
        }
        
        # Use correct parameters based on actual implementation
        mix = allocator.calculate_optimal_mix(
            target_sectors={"Technology": 1.0},
            available_etfs=["QQQ"],
            available_stocks=["AAPL", "MSFT", "GOOGL"]
        )
        
        # Check structure
        assert isinstance(mix.etf_weights, dict)
        assert isinstance(mix.stock_weights, dict)
        assert mix.diversification_score >= 0
        assert mix.total_estimated_cost >= 0
    
    def test_get_low_cost_etf(self):
        """Test low-cost ETF recommendation."""
        low_cost = get_low_cost_etf()
        
        # Should return VOO (lowest cost S&P 500 ETF)
        assert low_cost in ["VOO", "SPLG", "IVV"]
    
    def test_estimate_portfolio_cost(self):
        """Test portfolio cost estimation."""
        portfolio = {
            "VOO": 0.5,   # 0.03% expense ratio
            "AAPL": 0.25,  # No expense ratio
            "MSFT": 0.25   # No expense ratio
        }
        
        total_cost = estimate_portfolio_cost(portfolio)
        
        # Should be roughly 0.5 * 3 = 1.5 bps
        assert 1.0 <= total_cost <= 2.0
