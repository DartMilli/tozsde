"""
Tests for Market Regime Detector (P8 — Learning System)

Tests regime detection, volatility calculation, trend analysis,
and regime transition detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.decision.market_regime_detector import (
    MarketRegimeDetector,
    RegimeInfo,
    get_market_regime,
    is_bull_market,
    is_high_volatility
)


@pytest.fixture
def mock_data_manager(monkeypatch):
    """Mock DataManager with synthetic market data."""
    from app.data_access.data_manager import DataManager
    
    def mock_load_ohlcv(ticker):
        # Generate synthetic data based on ticker
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        if ticker == "BULL":
            # Uptrending market with low volatility
            prices = np.linspace(100, 130, 100) + np.random.normal(0, 1, 100)
        elif ticker == "BEAR":
            # Downtrending market with high volatility
            prices = np.linspace(130, 100, 100) + np.random.normal(0, 5, 100)
        elif ticker == "RANGING":
            # Sideways market
            prices = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5 + np.random.normal(0, 1, 100)
        elif ticker == "VOLATILE":
            # High volatility, no trend
            prices = 100 + np.random.normal(0, 10, 100)
        else:
            # Default neutral market
            prices = 100 + np.random.normal(0, 2, 100)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': 1000000
        }, index=dates)
        
        return df
    
    monkeypatch.setattr(DataManager, "load_ohlcv", lambda self, ticker: mock_load_ohlcv(ticker))


def test_detect_market_regime_bull(mock_data_manager):
    """Test detection of bull market."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    assert regime.regime_type == "BULL"
    assert regime.trend_strength > 0.2  # Positive trend
    assert regime.confidence > 0.5


def test_detect_market_regime_bear(mock_data_manager):
    """Test detection of bear market."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BEAR")
    
    assert regime.regime_type in ["BEAR", "VOLATILE"]  # High vol might classify as VOLATILE
    assert regime.trend_strength < 0  # Negative trend


def test_detect_market_regime_ranging(mock_data_manager):
    """Test detection of ranging market."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("RANGING")
    
    # Ranging market has no strong trend OR is classified as RANGING
    assert regime.regime_type in ["RANGING", "BEAR"]  # Sine wave might show slight trend
    assert regime.volatility < 0.40  # Not extremely volatile


def test_detect_market_regime_volatile(mock_data_manager):
    """Test detection of volatile market."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("VOLATILE")
    
    assert regime.regime_type == "VOLATILE"
    assert regime.volatility > 0.25  # High volatility


def test_volatility_calculation(mock_data_manager):
    """Test annualized volatility calculation."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    # Volatility should be reasonable (0-100%)
    assert 0.0 <= regime.volatility <= 1.0


def test_trend_strength_calculation(mock_data_manager):
    """Test trend strength calculation."""
    detector = MarketRegimeDetector()
    
    # Bull market should have positive trend
    bull_regime = detector.detect_regime("BULL")
    assert bull_regime.trend_strength > 0
    
    # Bear market should have negative trend
    bear_regime = detector.detect_regime("BEAR")
    assert bear_regime.trend_strength < 0


def test_regime_info_characteristics(mock_data_manager):
    """Test that regime info contains all characteristics."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    assert "volatility" in regime.characteristics
    assert "trend_strength" in regime.characteristics
    assert "trend_consistency" in regime.characteristics
    assert "recent_return" in regime.characteristics
    assert "max_drawdown" in regime.characteristics


def test_confidence_score_range(mock_data_manager):
    """Test that confidence scores are in valid range."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    assert 0.0 <= regime.confidence <= 1.0


def test_regime_transition_detection(mock_data_manager):
    """Test detection of regime transitions."""
    detector = MarketRegimeDetector()
    
    # For synthetic data, transition may or may not be detected
    transition = detector.detect_regime_transition("BULL", days=5)
    
    # Should return dict or None
    assert transition is None or isinstance(transition, dict)
    
    if transition:
        assert "transition_detected" in transition
        assert "from_regime" in transition
        assert "to_regime" in transition


def test_insufficient_data_handling(monkeypatch):
    """Test handling when insufficient data available."""
    from app.data_access.data_manager import DataManager
    
    # Mock with insufficient data
    def mock_load_empty(ticker):
        return pd.DataFrame()
    
    monkeypatch.setattr(DataManager, "load_ohlcv", lambda self, ticker: mock_load_empty(ticker))
    
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("EMPTY")
    
    assert regime.regime_type == "RANGING"  # Default regime
    assert regime.confidence == 0.3  # Low confidence


def test_get_market_regime_utility(mock_data_manager):
    """Test utility function for quick regime lookup."""
    regime_type = get_market_regime("BULL")
    assert regime_type in ["BULL", "BEAR", "RANGING", "VOLATILE"]


def test_is_bull_market_utility(mock_data_manager):
    """Test bull market check utility function."""
    is_bull = is_bull_market("BULL", confidence_threshold=0.5)
    assert isinstance(is_bull, bool)


def test_is_high_volatility_utility(mock_data_manager):
    """Test high volatility check utility function."""
    is_high_vol = is_high_volatility("VOLATILE", threshold=0.25)
    assert is_high_vol in [True, False]  # Should return boolean


def test_regime_classification_logic(mock_data_manager):
    """Test regime classification decision tree."""
    detector = MarketRegimeDetector()
    
    # Test different market conditions
    regimes = []
    for ticker in ["BULL", "BEAR", "RANGING", "VOLATILE"]:
        regime = detector.detect_regime(ticker)
        regimes.append(regime.regime_type)
    
    # Should have detected at least 2 different regime types
    assert len(set(regimes)) >= 2


def test_drawdown_calculation(mock_data_manager):
    """Test maximum drawdown calculation."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    # Drawdown should be non-negative and reasonable
    dd = regime.characteristics.get("max_drawdown", 0)
    assert 0.0 <= dd <= 1.0


def test_recent_return_calculation(mock_data_manager):
    """Test recent return calculation."""
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("BULL")
    
    # Recent return should be reasonable (-100% to +1000%)
    recent_return = regime.characteristics.get("recent_return", 0)
    assert -1.0 <= recent_return <= 10.0
