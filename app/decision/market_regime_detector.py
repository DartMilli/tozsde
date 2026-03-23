from app.core.decision.market_regime_detector import (  # noqa: F401
    RegimeInfo,
    MarketRegimeDetector as CoreMarketRegimeDetector,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


class MarketRegimeDetector(CoreMarketRegimeDetector):
    def _create_data_repository(self):
        cfg = self.settings
        try:
            return DataManager(settings=cfg)
        except TypeError:
            return DataManager()


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
