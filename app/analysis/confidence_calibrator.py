from app.analysis import get_settings
from app.core.analysis.confidence_calibrator import (  # noqa: F401
    CalibrationResult,
    ConfidenceCalibrator as CoreConfidenceCalibrator,
)
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository():
    try:
        return DataManager(settings=get_settings())
    except TypeError:
        return DataManager()


class ConfidenceCalibrator(CoreConfidenceCalibrator):
    def __init__(self):
        settings = get_settings()
        data_manager = _create_data_repository()
        super().__init__(settings=settings, data_manager=data_manager)
