from app.analysis import get_settings
from app.core.analysis.phase6_validator import Phase6Validator as CorePhase6Validator
from app.infrastructure.repositories import DataManagerRepository

DataManager = DataManagerRepository


def _create_data_repository(settings=None):
    cfg = settings if settings is not None else get_settings()
    try:
        return DataManager(settings=cfg)
    except TypeError:
        return DataManager()


class Phase6Validator(CorePhase6Validator):
    def __init__(self, dm: DataManagerRepository | None = None, settings=None):
        cfg = settings if settings is not None else get_settings()
        super().__init__(
            dm=dm or _create_data_repository(settings=cfg),
            settings=cfg,
        )
