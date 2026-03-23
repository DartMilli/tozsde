from app.backtesting.walk_forward import run_walk_forward
from app.models.model_trainer import train_rl_agent
from app.optimization.fitness import normalize_wf_score
from app.application.use_cases.result import ok, UseCaseResult


class RunMonthlyRetrainingUseCase:
    def __init__(
        self,
        settings,
        ticker_provider,
        walk_forward_fn=run_walk_forward,
        train_rl_fn=train_rl_agent,
    ):
        self.settings = settings
        self._ticker_provider = ticker_provider
        self._walk_forward_fn = walk_forward_fn
        self._train_rl_fn = train_rl_fn

    def run(self, dry_run: bool = False) -> UseCaseResult:
        processed = 0
        trained = 0
        failed = 0
        skipped = 0

        for ticker_symbol in self._ticker_provider():
            try:
                wf_summary = self._walk_forward_fn(ticker_symbol)
                if not wf_summary:
                    skipped += 1
                    continue

                wf_score = wf_summary.get("normalized_score")
                if wf_score is None:
                    raw_fitness = wf_summary.get("raw_fitness")
                    if raw_fitness is None:
                        raw_fitness = wf_summary.get("wf_fitness", 0.0)
                    wf_score = normalize_wf_score(raw_fitness)

                processed += 1
                if self.settings.ENABLE_RL:
                    self._train_rl_fn(
                        ticker=ticker_symbol,
                        wf_score=wf_score,
                        wf_summary=wf_summary,
                    )
                    trained += 1
            except Exception:
                failed += 1

        return ok(
            "run_monthly_retraining",
            data={
                "processed": processed,
                "trained": trained,
                "failed": failed,
                "skipped": skipped,
                "rl_enabled": bool(self.settings.ENABLE_RL),
            },
            dry_run=dry_run,
        )
