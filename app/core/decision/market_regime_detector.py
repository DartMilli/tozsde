from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.bootstrap.build_settings import build_settings
from app.infrastructure.logger import setup_logger
from app.infrastructure.repositories import DataManagerRepository

logger = setup_logger(__name__)


@dataclass
class RegimeInfo:
    regime_type: str
    confidence: float
    volatility: float
    trend_strength: float
    detected_at: str
    characteristics: Dict[str, float]


class MarketRegimeDetector:
    def __init__(self, lookback_days: int = 60, settings=None):
        self.lookback_days = lookback_days
        self.settings = settings
        try:
            if getattr(self, "dm", None) is None:
                self.dm = self._create_data_repository()
        except Exception:
            self.dm = self._create_data_repository()

    def detect_regime(self, ticker: str = "SPY") -> RegimeInfo:
        df = self.dm.load_ohlcv(ticker)

        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for regime detection: {ticker}")
            return self._default_regime()

        df = df.tail(self.lookback_days)

        volatility = self._calculate_volatility(df)
        trend_strength = self._calculate_trend_strength(df)
        trend_consistency = self._calculate_trend_consistency(df)

        regime_type = self._classify_regime(
            volatility, trend_strength, trend_consistency
        )

        confidence = self._calculate_confidence(
            volatility, trend_strength, trend_consistency
        )

        characteristics = {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "trend_consistency": trend_consistency,
            "recent_return": self._calculate_recent_return(df),
            "max_drawdown": self._calculate_drawdown(df),
        }

        return RegimeInfo(
            regime_type=regime_type,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            detected_at=datetime.now().isoformat(),
            characteristics=characteristics,
        )

    def detect_regime_transition(
        self, ticker: str = "SPY", days: int = 5
    ) -> Optional[Dict]:
        current_regime = self.detect_regime(ticker)

        df = self.dm.load_ohlcv(ticker)
        if df is None or len(df) < self.lookback_days + days:
            return None

        historical_df = df[:-days].tail(self.lookback_days)
        historical_volatility = self._calculate_volatility(historical_df)
        historical_trend = self._calculate_trend_strength(historical_df)
        historical_consistency = self._calculate_trend_consistency(historical_df)
        historical_regime = self._classify_regime(
            historical_volatility, historical_trend, historical_consistency
        )

        if historical_regime != current_regime.regime_type:
            return {
                "transition_detected": True,
                "from_regime": historical_regime,
                "to_regime": current_regime.regime_type,
                "transition_date": (datetime.now() - timedelta(days=days)).isoformat(),
                "confidence": current_regime.confidence,
            }

        return None

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0

        returns = df["Close"].pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        return annualized_vol

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        if len(df) < 10:
            return 0.0

        prices = df["Close"].values
        normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-6)

        x = np.arange(len(normalized))
        slope = np.polyfit(x, normalized, 1)[0]

        trend_strength = np.clip(slope * 100, -1.0, 1.0)

        return trend_strength

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        if len(df) < 10:
            return 0.0

        prices = df["Close"].values
        x = np.arange(len(prices))

        coeffs = np.polyfit(x, prices, 1)
        fitted = np.polyval(coeffs, x)

        ss_res = np.sum((prices - fitted) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)

        if ss_tot < 1e-6:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)

        return max(0.0, min(1.0, r_squared))

    def _calculate_recent_return(self, df: pd.DataFrame, days: int = 30) -> float:
        if len(df) < days:
            days = len(df)

        if days < 2:
            return 0.0

        recent = df.tail(days)
        return_pct = (recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent[
            "Close"
        ].iloc[0]

        return return_pct

    def _calculate_drawdown(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0

        cumulative = (1 + df["Close"].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return abs(drawdown.min())

    def _classify_regime(
        self, volatility: float, trend_strength: float, trend_consistency: float
    ) -> str:
        if volatility > 0.30:
            return "VOLATILE"

        if trend_strength > 0.3 and trend_consistency > 0.5:
            return "BULL"

        if trend_strength < -0.3 and trend_consistency > 0.5:
            return "BEAR"

        return "RANGING"

    def _calculate_confidence(
        self, volatility: float, trend_strength: float, trend_consistency: float
    ) -> float:
        confidence = trend_consistency

        if abs(trend_strength) > 0.5:
            confidence = min(1.0, confidence + 0.2)

        if volatility > 0.40:
            confidence = min(1.0, confidence + 0.3)

        if abs(trend_strength) < 0.1 and volatility < 0.15:
            confidence *= 0.7

        return max(0.3, min(1.0, confidence))

    def _default_regime(self) -> RegimeInfo:
        return RegimeInfo(
            regime_type="RANGING",
            confidence=0.3,
            volatility=0.15,
            trend_strength=0.0,
            detected_at=datetime.now().isoformat(),
            characteristics={},
        )

    def _create_data_repository(self):
        cfg = self.settings or build_settings()
        try:
            return DataManagerRepository(settings=cfg)
        except TypeError:
            return DataManagerRepository()


def get_market_regime(ticker: str = "SPY") -> str:
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.regime_type


def is_bull_market(ticker: str = "SPY", confidence_threshold: float = 0.6) -> bool:
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.regime_type == "BULL" and regime.confidence >= confidence_threshold


def is_high_volatility(ticker: str = "SPY", threshold: float = 0.25) -> bool:
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(ticker)
    return regime.volatility >= threshold
