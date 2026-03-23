from app.validation.validation_runner import ValidationRunner
from app.infrastructure.repositories.sqlite_model_repository import (
    SqliteModelRepository,
)
from app.infrastructure.repositories.sqlite_metrics_repository import (
    SqliteMetricsRepository,
)
from app.application.use_cases.result import UseCaseResult, ok


class ValidateModelUseCase:
    def __init__(self, settings):
        self.settings = settings
        self.model_repo = SqliteModelRepository(settings)
        self.metrics_repo = SqliteMetricsRepository(settings)

    def run(self, mode: str = "quick") -> UseCaseResult:
        runner = ValidationRunner(mode=mode)
        runner.execute()
        return ok(
            "validate_model",
            data=runner.results,
            mode=mode,
        )
