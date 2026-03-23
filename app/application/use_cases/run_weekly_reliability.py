from datetime import date, timedelta

from app.models.model_reliability import (
    ModelReliabilityAnalyzer,
    save_reliability_scores,
)
from app.application.use_cases.result import ok, UseCaseResult


class RunWeeklyReliabilityUseCase:
    def __init__(
        self,
        settings,
        ticker_provider,
        analyzer_factory=ModelReliabilityAnalyzer,
        save_scores_fn=save_reliability_scores,
    ):
        self.settings = settings
        self._ticker_provider = ticker_provider
        self._analyzer_factory = analyzer_factory
        self._save_scores_fn = save_scores_fn

    def run(self, dry_run: bool = False) -> UseCaseResult:
        processed = 0
        saved = 0
        failed = 0

        analyzer = self._analyzer_factory()
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=self.settings.RELIABILITY_PERIOD_DAYS)

        for ticker_symbol in self._ticker_provider():
            try:
                scores = analyzer.analyze(
                    ticker=ticker_symbol,
                    start=start,
                    end=end,
                )
                processed += 1
                if not dry_run:
                    self._save_scores_fn(ticker_symbol, end.isoformat(), scores)
                    saved += 1
            except Exception:
                failed += 1

        return ok(
            "run_weekly_reliability",
            data={
                "processed": processed,
                "saved": saved,
                "failed": failed,
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
            },
            dry_run=dry_run,
        )
