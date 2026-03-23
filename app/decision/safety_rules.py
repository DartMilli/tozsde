from app.core.decision.safety_rules import SafetyRuleEngine as CoreSafetyRuleEngine
from app.data_access.data_loader import get_market_volatility_index
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository(settings=None):
    try:
        return DataManager(settings=settings)
    except TypeError:
        return DataManager()


class SafetyRuleEngine(CoreSafetyRuleEngine):
    def _create_data_repository(self):
        return _create_data_repository(settings=self._get_settings())

    def _get_market_volatility_index(self):
        return get_market_volatility_index()
