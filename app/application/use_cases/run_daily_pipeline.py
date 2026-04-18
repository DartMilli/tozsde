from app.services.trading_pipeline import TradingPipelineService
from app.backtesting.history_store import HistoryStore
from app.models.model_trainer import TradingEnv
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.services.execution_engines import NoopExecutionEngine, LiveExecutionEngine
from app.services.paper_execution import PaperExecutionEngine
from app.infrastructure.repositories.sqlite_ohlcv_repository import (
    SqliteOhlcvRepository,
)
from app.infrastructure.repositories.sqlite_decision_repository import (
    SqliteDecisionRepository,
)
from app.infrastructure.repositories.sqlite_model_repository import (
    SqliteModelRepository,
)
from app.infrastructure.repositories.sqlite_metrics_repository import (
    SqliteMetricsRepository,
)
from app.infrastructure.repositories import DataManagerRepository
from app.application.use_cases.daily_pipeline_use_case import DailyPipelineUseCase
from app.application.use_cases.result import UseCaseResult, ok


class RunDailyPipelineUseCase:
    def __init__(self, settings, data_manager=None, logger=None):
        self._settings = settings
        self._data_manager = data_manager
        self._logger = logger
        self._daily_pipeline: DailyPipelineUseCase | None = None

    def _build_pipeline(self) -> DailyPipelineUseCase:
        settings = self._settings
        data_manager = self._data_manager
        logger = self._logger

        ohlcv_repo = SqliteOhlcvRepository(data_manager=data_manager)  # noqa: F841
        decision_repo = SqliteDecisionRepository(settings)
        model_repo = SqliteModelRepository(settings)  # noqa: F841
        metrics_repo = SqliteMetricsRepository(settings)  # noqa: F841
        state_repo = DataManagerRepository(settings=settings)

        execution_mode = getattr(settings, "EXECUTION_MODE", "paper")
        if execution_mode == "paper":
            execution_engine = PaperExecutionEngine(
                state_repo, logger, settings=settings
            )
        elif execution_mode == "live":
            broker_adapter = getattr(settings, "BROKER_ADAPTER", "noop")
            if broker_adapter == "alpaca":
                from app.services.execution_engines import AlpacaExecutionEngine

                execution_engine = AlpacaExecutionEngine(logger=logger)
            else:
                execution_engine = LiveExecutionEngine(logger=logger)
        else:
            execution_engine = NoopExecutionEngine(logger)

        pipeline = TradingPipelineService(
            history_store=HistoryStore(history_repo=decision_repo, settings=settings),
            settings=settings,
            logger=logger,
            data_fetcher=MarketDataFetcher(),
            model_runner=ModelEnsembleRunner(
                model_dir=getattr(settings, "MODEL_DIR", None),
                env_class=TradingEnv,
                settings=settings,
            ),
            email_notifier=EmailNotifier(),
            execution_engine=execution_engine,
            state_repo=state_repo,
        )
        return DailyPipelineUseCase(pipeline)

    def run(self, dry_run: bool = False, ticker: str = None) -> UseCaseResult:
        if self._daily_pipeline is None:
            self._daily_pipeline = self._build_pipeline()
        inner = self._daily_pipeline.run(dry_run=dry_run, ticker=ticker)
        if inner.get("status") == "error":
            return inner
        return ok(
            "run_daily_pipeline",
            data=inner.get("data"),
            dry_run=dry_run,
            ticker=ticker,
        )
