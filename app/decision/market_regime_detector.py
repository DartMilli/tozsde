"""
Market Regime Detector (P8 — Learning System)

Responsibility:
    - Detect current market regime (BULL, BEAR, RANGING, VOLATILE)
    - Analyze market conditions across multiple timeframes
    - Provide context for adaptive strategy selection
    - Track regime transitions

Features:
    - Multiple regime classification methods
    - Volatility-based detection
    - Trend-based detection
    - Volume analysis
    - Regime transition tracking

Usage:
    detector = MarketRegimeDetector()
    regime = detector.detect_regime("SPY")
    print(f"Current regime: {regime.regime_type}")
    print(f"Confidence: {regime.confidence:.1%}")
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RegimeInfo:
    """Market regime detection result."""
    regime_type: str  # "BULL" | "BEAR" | "RANGING" | "VOLATILE"
    confidence: float  # 0.0 to 1.0
    volatility: float  # Annualized volatility
    trend_strength: float  # -1.0 (strong down) to +1.0 (strong up)
    detected_at: str  # ISO timestamp
    characteristics: Dict[str, float]  # Additional regime characteristics


class MarketRegimeDetector:
    """
    Detects current market regime based on price action and volatility.
    
    Classifies markets into four regimes:
    - BULL: Strong uptrend, low to medium volatility
    - BEAR: Strong downtrend, high volatility
    - RANGING: No clear trend, low to medium volatility
    - VOLATILE: High volatility regardless of trend
    """
    
    def __init__(self, lookback_days: int = 60):
        """
        Initialize regime detector.
        
        Args:
            lookback_days: Days of historical data to analyze
        """
        self.lookback_days = lookback_days
        self.dm = DataManager()
    
    def detect_regime(self, ticker: str = "SPY") -> RegimeInfo:
        """
        Detect current market regime for given ticker.
        
        Args:
            ticker: Ticker symbol (default SPY for broad market)
            
        Returns:
            RegimeInfo object with classification and characteristics
        """
        # Load historical data
        df = self.dm.load_ohlcv(ticker)
        
        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for regime detection: {ticker}")
            return self._default_regime()
        
        # Limit to lookback period
        df = df.tail(self.lookback_days)
        
        # Calculate regime indicators
        volatility = self._calculate_volatility(df)
        trend_strength = self._calculate_trend_strength(df)
        trend_consistency = self._calculate_trend_consistency(df)
        
        # Classify regime
        regime_type = self._classify_regime(volatility, trend_strength, trend_consistency)
        
        # Calculate confidence in classification
        confidence = self._calculate_confidence(volatility, trend_strength, trend_consistency)
        
        # Additional characteristics
        characteristics = {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "trend_consistency": trend_consistency,
            "recent_return": self._calculate_recent_return(df),
            "max_drawdown": self._calculate_drawdown(df)
        }
        
        return RegimeInfo(
            regime_type=regime_type,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            detected_at=datetime.now().isoformat(),
            characteristics=characteristics
        )
    
    def detect_regime_transition(
        self, ticker: str = "SPY", days: int = 5
    ) -> Optional[Dict]:
        """
        Detect if market regime has recently changed.
        
        Args:
            ticker: Ticker symbol
            days: Days to look back for transition
            
        Returns:
            Dict with transition info or None if no transition
        """
        current_regime = self.detect_regime(ticker)
        
        # Get regime from N days ago
        df = self.dm.load_ohlcv(ticker)
        if df is None or len(df) < self.lookback_days + days:
            return None
        
        # Historical regime detection
        historical_df = df[:-days].tail(self.lookback_days)
        historical_volatility = self._calculate_volatility(historical_df)
        historical_trend = self._calculate_trend_strength(historical_df)
        historical_consistency = self._calculate_trend_consistency(historical_df)
        historical_regime = self._classify_regime(
            historical_volatility, historical_trend, historical_consistency
        )
        
        # Check for transition
        if historical_regime != current_regime.regime_type:
            return {
                "transition_detected": True,
                "from_regime": historical_regime,
                "to_regime": current_regime.regime_type,
                "transition_date": (datetime.now() - timedelta(days=days)).isoformat(),
                "confidence": current_regime.confidence
            }
        
        return None
    
    # Calculation methods
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate annualized volatility from returns.
        
        Returns:
            Annualized volatility (e.g., 0.20 = 20%)
        """
        if len(df) < 2:
            return 0.0
        
        returns = df["Close"].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Annualized volatility (assuming 252 trading days)
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength using linear regression slope.
        
        Returns:
            Value between -1.0 (strong downtrend) and +1.0 (strong uptrend)
        """
        if len(df) < 10:
            return 0.0
        
        # Normalize prices to 0-1 range
        prices = df["Close"].values
        normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-6)
        
        # Linear regression
        x = np.arange(len(normalized))
        slope = np.polyfit(x, normalized, 1)[0]
        
        # Normalize slope to -1 to +1 range
        # Typical daily slope is ~0.001 to 0.01, so multiply by 100
        trend_strength = np.clip(slope * 100, -1.0, 1.0)
        
        return trend_strength
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """
        Calculate how consistent the trend is (R-squared of linear fit).
        
        Returns:
            Value between 0.0 (no trend) and 1.0 (perfect trend)
        """
        if len(df) < 10:
            return 0.0
        
        prices = df["Close"].values
        x = np.arange(len(prices))
        
        # Linear regression
        coeffs = np.polyfit(x, prices, 1)
        fitted = np.polyval(coeffs, x)
        
        # R-squared
        ss_res = np.sum((prices - fitted) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        
        if ss_tot < 1e-6:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        return max(0.0, min(1.0, r_squared))
    
    def _calculate_recent_return(self, df: pd.DataFrame, days: int = 30) -> float:
        """Calculate recent return over specified days."""
        if len(df) < days:
            days = len(df)
        
        if days < 2:
            return 0.0
        
        recent = df.tail(days)
        return_pct = (recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent["Close"].iloc[0]
        
        return return_pct
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown in period."""
        if len(df) < 2:
            return 0.0
        
        cumulative = (1 + df["Close"].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    # Classification methods
    
    def _classify_regime(
        self, volatility: float, trend_strength: float, trend_consistency: float
    ) -> str:
        """
        Classify market regime based on calculated metrics.
        
        Decision tree:
        1. If volatility > 0.30 (30%): VOLATILE
        2. If trend_strength > 0.3 and consistency > 0.5: BULL
        3. If trend_strength < -0.3 and consistency > 0.5: BEAR
        4. Otherwise: RANGING
        """
        # High volatility dominates
        if volatility > 0.30:
            return "VOLATILE"
        
        # Strong uptrend
        if trend_strength > 0.3 and trend_consistency > 0.5:
            return "BULL"
        
        # Strong downtrend
        if trend_strength < -0.3 and trend_consistency > 0.5:
            return "BEAR"
        
        # Default: ranging market
        return "RANGING"
    
    def _calculate_confidence(
        self, volatility: float, trend_strength: float, trend_consistency: float
    ) -> float:
        """
        Calculate confidence in regime classification.
        
        Higher confidence when:
        - Clear trend (high consistency)
        - Extreme values (very high/low trend strength or volatility)
        
        Returns:
            Confidence score 0.0 to 1.0
        """
        # Base confidence from trend consistency
        confidence = trend_consistency
        
        # Boost for extreme trends
        if abs(trend_strength) > 0.5:
            confidence = min(1.0, confidence + 0.2)
        
        # Boost for very high volatility (clear VOLATILE regime)
        if volatility > 0.40:
            confidence = min(1.0, confidence + 0.3)
        
        # Reduce for weak trends (less certain RANGING)
        if abs(trend_strength) < 0.1 and volatility < 0.15:
            confidence *= 0.7
        
        return max(0.3, min(1.0, confidence))  # Clamp to 0.3-1.0
    
    def _default_regime(self) -> RegimeInfo:
        """Return default regime when data is insufficient."""
        return RegimeInfo(
            regime_type="RANGING",
            confidence=0.3,
            volatility=0.15,
            trend_strength=0.0,
            detected_at=datetime.now().isoformat(),
            characteristics={}
        )


# Utility functions

def get_market_regime(ticker: str = "SPY") -> str:
    """
    Quick function to get current market regime.
    
    Args:
        ticker: Market ticker (default SPY)
        
    Returns:
        Regime type string
    """
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.regime_type


def is_bull_market(ticker: str = "SPY", confidence_threshold: float = 0.6) -> bool:
    """
    Check if current market is bullish with high confidence.
    
    Args:
        ticker: Market ticker
        confidence_threshold: Minimum confidence required
        
    Returns:
        True if bull market with sufficient confidence
    """
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.regime_type == "BULL" and regime.confidence >= confidence_threshold


def is_high_volatility(ticker: str = "SPY", threshold: float = 0.25) -> bool:
    """
    Check if current volatility is high.
    
    Args:
        ticker: Market ticker
        threshold: Volatility threshold (default 25%)
        
    Returns:
        True if volatility exceeds threshold
    """
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.volatility >= threshold
