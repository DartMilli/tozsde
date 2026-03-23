from app.backtesting.walk_forward import run_walk_forward
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
from app.application.use_cases.result import UseCaseResult, ok


class RunWalkForwardUseCase:
    def __init__(self, settings, data_manager=None):
        # Prefer injected data_manager; repository adaptors wrap it
        self.settings = settings
        self.ohlcv_repo = SqliteOhlcvRepository(data_manager=data_manager)
        self.decision_repo = SqliteDecisionRepository(settings)
        self.model_repo = SqliteModelRepository(settings)
        self.metrics_repo = SqliteMetricsRepository(settings)

    def run(self, ticker: str = None) -> UseCaseResult:
        if ticker:
            return ok(
                "run_walk_forward",
                data={ticker: run_walk_forward(ticker, metrics_repo=self.metrics_repo)},
                ticker=ticker,
            )

        try:
            from app.data_access.data_loader import get_supported_ticker_list

            tickers = get_supported_ticker_list()
            excluded = set(getattr(self.settings, "EXCLUDED_TICKERS", []))
            tickers = [symbol for symbol in tickers if symbol not in excluded]
        except Exception:
            tickers = []

        results = {}
        for symbol in tickers:
            results[symbol] = run_walk_forward(symbol, metrics_repo=self.metrics_repo)
        return ok(
            "run_walk_forward",
            data=results,
            ticker=ticker,
            total_tickers=len(tickers),
        )
