from app.analysis import get_settings
from app.core.analysis.wf_stability_analyzer import (
    WalkForwardStabilityAnalyzer as CoreWalkForwardStabilityAnalyzer,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository():
    try:
        return DataManager(settings=get_settings())
    except TypeError:
        return DataManager()


class WalkForwardStabilityAnalyzer(CoreWalkForwardStabilityAnalyzer):
    def __init__(self):
        super().__init__(
            settings=get_settings(),
            data_manager=_create_data_repository(),
        )
