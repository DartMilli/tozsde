from app.analysis import get_settings
from app.core.analysis.decision_quality_analyzer import (
    DecisionQualityAnalyzer as CoreDecisionQualityAnalyzer,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository():
    try:
        return DataManager(settings=get_settings())
    except Exception:
        return DataManager()


class DecisionQualityAnalyzer(CoreDecisionQualityAnalyzer):
    def __init__(self):
        super().__init__(
            settings=get_settings(),
            data_manager=_create_data_repository(),
        )
