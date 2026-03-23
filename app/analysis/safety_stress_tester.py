from app.analysis import get_settings
from app.core.analysis.safety_stress_tester import (
    SafetyStressTester as CoreSafetyStressTester,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository():
    try:
        return DataManager(settings=get_settings())
    except Exception:
        return DataManager()


class SafetyStressTester(CoreSafetyStressTester):
    def __init__(self):
        super().__init__(
            settings=get_settings(),
            data_manager=_create_data_repository(),
        )
