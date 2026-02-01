"""
Tests for Portfolio Correlation Manager (Sprint 7)
"""

import pytest
import pandas as pd
import numpy as np
from app.decision.portfolio_correlation_manager import (
    PortfolioCorrelationManager,
    check_portfolio_diversification,
    find_uncorrelated_assets
)


class TestPortfolioCorrelationManager:
    
    def test_compute_correlation_matrix_empty(self):
        """Test correlation matrix with no data."""
        manager = PortfolioCorrelationManager(lookback_days=90)
        
        # With real DataManager, will return empty if no tickers found
        corr_matrix = manager.compute_correlation_matrix([])
        
        # Should return DataFrame
        assert isinstance(corr_matrix, pd.DataFrame)
    
    def test_calculate_diversification_score_single_asset(self, test_db):
        """Test diversification score for single asset."""
        manager = PortfolioCorrelationManager()
        
        portfolio = {"AAPL": 1.0}
        
        score = manager.calculate_diversification_score(portfolio)
        
        # Single asset = zero diversification
        assert score == 0.0
    
    def test_calculate_diversification_score_multi_asset(self, test_db):
        """Test diversification score returns valid range."""
        manager = PortfolioCorrelationManager()
        
        portfolio = {"AAPL": 0.5, "JNJ": 0.5}
        
        score = manager.calculate_diversification_score(portfolio)
        
        # Should always be between 0 and 1 (even with real but missing data)
        assert 0 <= score <= 1.0
    
    def test_decompose_portfolio_risk(self, test_db):
        """Test risk decomposition returns valid structure."""
        manager = PortfolioCorrelationManager()
        
        portfolio = {"AAPL": 0.5, "MSFT": 0.5}
        
        risk_decomp = manager.decompose_portfolio_risk(portfolio)
        
        assert risk_decomp.total_risk >= 0
        assert risk_decomp.idiosyncratic_risk >= 0
        assert risk_decomp.systematic_risk >= 0
        assert len(risk_decomp.top_risk_contributors) > 0
    
    def test_optimize_for_low_correlation(self, test_db):
        """Test portfolio selection."""
        manager = PortfolioCorrelationManager()

        selected = manager.optimize_for_low_correlation(
            candidates=["AAPL", "JNJ", "XOM"],
            target_size=2,
            max_correlation=0.7
        )
        
        assert len(selected) <= 2
        assert all(ticker in ["AAPL", "JNJ", "XOM"] for ticker in selected)
    
    def test_get_highly_correlated_pairs(self, test_db):
        """Test high correlation pair detection."""
        manager = PortfolioCorrelationManager()

        pairs = manager.get_highly_correlated_pairs(
            tickers=["SPY", "VOO", "QQQ"],
            threshold=0.8
        )
        
        # Should return list (may be empty if no data)
        assert isinstance(pairs, list)


def test_check_portfolio_diversification(test_db):
    """Test diversification convenience function."""
    portfolio = {"AAPL": 0.5, "MSFT": 0.5}
    
    score = check_portfolio_diversification(portfolio)
    
    assert 0 <= score <= 1.0


def test_find_uncorrelated_assets(test_db):
    """Test uncorrelated asset finder returns list."""
    uncorrelated = find_uncorrelated_assets(
        base_ticker="AAPL",
        candidates=["JNJ", "MSFT"],
        max_correlation=0.5
    )
    
    # Should return list (may be empty without data)
    assert isinstance(uncorrelated, list)
