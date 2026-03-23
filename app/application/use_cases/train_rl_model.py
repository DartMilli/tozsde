from app.models.model_trainer import train_rl_agent
from app.infrastructure.repositories.sqlite_model_repository import (
    SqliteModelRepository,
)
from app.application.use_cases.result import UseCaseResult, ok


class TrainRLModelUseCase:
    def __init__(self, settings):
        self.settings = settings
        self.model_repo = SqliteModelRepository(settings)

    def run(self, ticker: str, **kwargs) -> UseCaseResult:
        train_rl_agent(ticker=ticker, **kwargs)
        return ok(
            "train_rl_model",
            data={"completed": True},
            ticker=ticker,
        )
