from app.backtesting.historical_paper_runner import HistoricalPaperRunner
from app.application.use_cases.result import ok, UseCaseResult


class RunHistoricalPaperUseCase:
    def __init__(self, logger=None):
        self.logger = logger

    def run(self, ticker: str, start_date: str, end_date: str) -> UseCaseResult:
        runner = HistoricalPaperRunner(logger=self.logger)
        result = runner.run(ticker=ticker, start_date=start_date, end_date=end_date)
        return ok(
            "run_historical_paper",
            data=result,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
