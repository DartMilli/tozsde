from dataclasses import dataclass
from typing import Any

from app.config.build_settings import build_settings
from app.infrastructure.repositories.sqlite_ohlcv_repository import (
    SqliteOhlcvRepository,
)
from app.infrastructure.repositories import DataManagerRepository
from app.application.use_cases import (
    RunDailyPipelineUseCase,
    RunWalkForwardUseCase,
    RunWeeklyReliabilityUseCase,
    RunMonthlyRetrainingUseCase,
    RunPhase5ValidationUseCase,
    RunHistoricalPaperUseCase,
    TrainRLModelUseCase,
    ValidateModelUseCase,
)


@dataclass
class ApplicationContainer:
    settings: Any
    ohlcv_repo: Any
    data_manager: Any
    data_manager_repo: Any
    daily_pipeline: Any
    walk_forward: Any
    weekly_reliability: Any
    monthly_retraining: Any
    phase5_validation: Any
    historical_paper: Any
    train_rl: Any
    validate_model: Any

    # Add service/repository getters as needed for DI root
    def get_supported_tickers(self):
        try:
            if hasattr(self.settings, "TICKERS"):
                val = getattr(self.settings, "TICKERS")
                if isinstance(val, list) and val:
                    return val
        except Exception:
            pass

        try:
            from app.data_access.data_loader import get_supported_ticker_list

            tickers = get_supported_ticker_list()
            excluded = set(getattr(self.settings, "EXCLUDED_TICKERS", []))
            return [t for t in tickers if t not in excluded]
        except Exception:
            return []


def build_application(ensure_dirs: bool = True) -> ApplicationContainer:
    settings = build_settings(ensure_dirs=ensure_dirs)

    def _supported_tickers():
        try:
            if hasattr(settings, "TICKERS"):
                val = getattr(settings, "TICKERS")
                if isinstance(val, list) and val:
                    return val
        except Exception:
            pass

        try:
            from app.data_access.data_loader import get_supported_ticker_list

            tickers = get_supported_ticker_list()
            excluded = set(getattr(settings, "EXCLUDED_TICKERS", []))
            return [t for t in tickers if t not in excluded]
        except Exception:
            return []

    dm = DataManagerRepository(settings=settings)
    ohlcv_repo = SqliteOhlcvRepository(data_manager=dm)
    dm_repo = DataManagerRepository(data_manager=dm, settings=settings)
    daily_pipeline = RunDailyPipelineUseCase(
        settings=settings,
        data_manager=dm,
    )
    walk_forward = RunWalkForwardUseCase(
        settings=settings,
        data_manager=dm,
    )
    weekly_reliability = RunWeeklyReliabilityUseCase(
        settings=settings,
        ticker_provider=_supported_tickers,
    )
    monthly_retraining = RunMonthlyRetrainingUseCase(
        settings=settings,
        ticker_provider=_supported_tickers,
    )
    phase5_validation = RunPhase5ValidationUseCase(data_manager=dm)
    historical_paper = RunHistoricalPaperUseCase()
    train_rl = TrainRLModelUseCase(settings=settings)
    validate_model = ValidateModelUseCase(settings=settings)
    # Add other repositories/services here as needed, all injected with dm/settings
    # Inject settings into modules that support DI-style `set_settings`
    try:
        from app.reporting import plotter as _plotter

        if hasattr(_plotter, "set_settings"):
            _plotter.set_settings(settings)
    except Exception:
        pass
    try:
        from app.notifications import alerter as _alerter

        if hasattr(_alerter, "set_settings"):
            _alerter.set_settings(settings)
    except Exception:
        pass
    try:
        from app.reporting import audit_builder as _audit

        if hasattr(_audit, "set_settings"):
            _audit.set_settings(settings)
    except Exception:
        pass
    container = ApplicationContainer(
        settings=settings,
        ohlcv_repo=ohlcv_repo,
        data_manager=dm,
        data_manager_repo=dm_repo,
        daily_pipeline=daily_pipeline,
        walk_forward=walk_forward,
        weekly_reliability=weekly_reliability,
        monthly_retraining=monthly_retraining,
        phase5_validation=phase5_validation,
        historical_paper=historical_paper,
        train_rl=train_rl,
        validate_model=validate_model,
    )
    return container
