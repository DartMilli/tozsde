from app.services.trading_pipeline import TradingPipelineService
from app.backtesting.history_store import HistoryStore
from app.models.model_trainer import TradingEnv
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.services.execution_engines import NoopExecutionEngine
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
        self.settings = settings
        self.ohlcv_repo = SqliteOhlcvRepository(data_manager=data_manager)
        self.decision_repo = SqliteDecisionRepository(settings)
        self.model_repo = SqliteModelRepository(settings)
        self.metrics_repo = SqliteMetricsRepository(settings)
        state_repo = DataManagerRepository(settings=settings)
        if getattr(settings, "EXECUTION_MODE", "paper") == "paper":
            execution_engine = PaperExecutionEngine(
                state_repo, logger, settings=settings
            )
        else:
            execution_engine = NoopExecutionEngine(logger)

        self.pipeline = TradingPipelineService(
            history_store=HistoryStore(
                history_repo=self.decision_repo, settings=settings
            ),
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
        self.daily_pipeline = DailyPipelineUseCase(self.pipeline)

    def run(self, dry_run: bool = False, ticker: str = None) -> UseCaseResult:
        result = self.daily_pipeline.run(dry_run=dry_run, ticker=ticker)
        return ok(
            "run_daily_pipeline",
            data=result,
            dry_run=dry_run,
            ticker=ticker,
        )
