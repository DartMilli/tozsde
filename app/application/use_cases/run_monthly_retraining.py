from app.backtesting.walk_forward import run_walk_forward
from app.models.model_trainer import train_rl_agent
from app.optimization.fitness import normalize_wf_score
from app.application.use_cases.result import ok, UseCaseResult
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class RunMonthlyRetrainingUseCase:
    def __init__(
        self,
        settings,
        ticker_provider,
        walk_forward_fn=run_walk_forward,
        train_rl_fn=train_rl_agent,
        shadow_evaluator=None,
    ):
        self.settings = settings
        self._ticker_provider = ticker_provider
        self._walk_forward_fn = walk_forward_fn
        self._train_rl_fn = train_rl_fn
        self._shadow_evaluator = shadow_evaluator

    def _get_shadow_evaluator(self):
        if self._shadow_evaluator is None:
            from app.models.shadow_evaluator import ShadowEvaluator

            self._shadow_evaluator = ShadowEvaluator(settings=self.settings)
        return self._shadow_evaluator

    def run(self, dry_run: bool = False) -> UseCaseResult:
        processed = 0
        trained = 0
        failed = 0
        skipped = 0
        shadow_registered = 0

        if not self.settings.ENABLE_RL:
            logger.warning(
                "ENABLE_RL=false: RL model retraining is disabled. "
                "Set the ENABLE_RL=true environment variable to enable monthly RL retraining."
            )

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
                if self.settings.ENABLE_RL and not dry_run:
                    train_result = self._train_rl_fn(
                        ticker=ticker_symbol,
                        wf_score=wf_score,
                        wf_summary=wf_summary,
                        auto_promote=not getattr(
                            self.settings,
                            "ENABLE_SHADOW_EVAL",
                            False,
                        ),
                    )
                    trained += 1
                    if (
                        getattr(self.settings, "ENABLE_SHADOW_EVAL", False)
                        and train_result
                    ):
                        metadata = dict(train_result.get("metadata", {}))
                        metadata.update(
                            {
                                "model_id": train_result.get("model_id"),
                                "model_type": train_result.get("model_type"),
                                "wf_score": train_result.get("wf_score"),
                                "wf_summary": wf_summary,
                            }
                        )
                        self._get_shadow_evaluator().register_challenger(
                            ticker=ticker_symbol,
                            model_path=train_result.get("model_path"),
                            meta=metadata,
                        )
                        shadow_registered += 1
            except Exception as exc:
                logger.error(
                    "MonthlyRetraining failed for %s: %s",
                    ticker_symbol,
                    exc,
                    exc_info=True,
                )
                failed += 1

        return ok(
            "run_monthly_retraining",
            data={
                "processed": processed,
                "trained": trained,
                "failed": failed,
                "skipped": skipped,
                "rl_enabled": bool(self.settings.ENABLE_RL),
                "shadow_registered": shadow_registered,
            },
            dry_run=dry_run,
        )
