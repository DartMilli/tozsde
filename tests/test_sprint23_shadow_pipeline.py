from __future__ import annotations

from dataclasses import replace
from datetime import date
from unittest.mock import MagicMock, patch

from app.application.use_cases.daily_pipeline_use_case import DailyPipelineUseCase
from app.application.use_cases.run_monthly_retraining import RunMonthlyRetrainingUseCase
from app.infrastructure.repositories import DataManagerRepository
from app.models.shadow_evaluator import ShadowEvaluator


def _insert_outcome(
    repo: DataManagerRepository, ticker: str, day: str, future_return: float
) -> None:
    with repo.connection() as conn:
        conn.execute(
            """
            INSERT INTO outcomes (
                ticker, decision_timestamp, pnl_pct, success, future_return,
                evaluated_at, exit_reason, horizon_days, outcome_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                f"{day}T00:00:00",
                future_return,
                1 if future_return > 0 else 0,
                future_return,
                f"{day}T12:00:00",
                "test",
                1,
                "{}",
            ),
        )
        conn.commit()


def test_shadow_evaluator_register_challenger_persists_candidate_metadata(
    test_settings,
    test_db,
):
    settings = replace(test_settings, ENABLE_SHADOW_EVAL=True)
    repo = DataManagerRepository(settings=settings)
    evaluator = ShadowEvaluator(settings=settings, data_repository=repo)

    evaluator.register_challenger(
        ticker="VOO",
        model_path="/tmp/challenger.zip",
        meta={
            "model_id": "challenger-1",
            "model_type": "DQN",
            "wf_score": 0.82,
        },
    )

    saved = repo.fetch_model("challenger-1")
    assert saved["status"] == "candidate", f"Expected candidate status, got: {saved}"
    assert saved["metadata"].get(
        "shadow_registered_at"
    ), f"Missing shadow metadata: {saved}"


def test_shadow_evaluator_promotion_passes_when_challenger_sharpe_is_better(
    test_settings,
    test_db,
):
    settings = replace(
        test_settings,
        ENABLE_SHADOW_EVAL=True,
        SHADOW_EVAL_DAYS=3,
        SHADOW_PROMOTION_THRESHOLD=1.1,
    )
    repo = DataManagerRepository(settings=settings)
    evaluator = ShadowEvaluator(settings=settings, data_repository=repo)

    repo.register_model(
        "champion-1", "VOO", "DQN", 0.60, "/champion.zip", status="active"
    )
    repo.register_model(
        "challenger-1", "VOO", "DQN", 0.80, "/challenger.zip", status="candidate"
    )

    for offset, future_return in enumerate([0.01, 0.015, 0.02], start=1):
        day = f"2026-04-0{offset}"
        repo.save_shadow_evaluation(
            ticker="VOO",
            date=day,
            champion_model="champion-1",
            challenger_model="challenger-1",
            champion_action=0,
            challenger_action=1,
        )
        _insert_outcome(repo, "VOO", day, future_return)

    promotion = evaluator.evaluate_promotion("VOO", "challenger-1")

    assert promotion["promote"] is True, f"Expected promotion, got: {promotion}"
    assert (
        repo.fetch_model("challenger-1")["status"] == "active"
    ), "Challenger should be active after promotion"
    assert (
        repo.fetch_model("champion-1")["status"] == "archived"
    ), "Champion should be archived after promotion"


def test_shadow_evaluator_promotion_requires_enough_days(test_settings, test_db):
    settings = replace(
        test_settings,
        ENABLE_SHADOW_EVAL=True,
        SHADOW_EVAL_DAYS=3,
        SHADOW_PROMOTION_THRESHOLD=1.1,
    )
    repo = DataManagerRepository(settings=settings)
    evaluator = ShadowEvaluator(settings=settings, data_repository=repo)

    repo.register_model(
        "champion-1", "VOO", "DQN", 0.60, "/champion.zip", status="active"
    )
    repo.register_model(
        "challenger-1", "VOO", "DQN", 0.80, "/challenger.zip", status="candidate"
    )

    for offset, future_return in enumerate([0.01, 0.015], start=1):
        day = f"2026-04-0{offset}"
        repo.save_shadow_evaluation(
            ticker="VOO",
            date=day,
            champion_model="champion-1",
            challenger_model="challenger-1",
            champion_action=0,
            challenger_action=1,
        )
        _insert_outcome(repo, "VOO", day, future_return)

    promotion = evaluator.evaluate_promotion("VOO", "challenger-1")

    assert (
        promotion["promote"] is False
    ), f"Promotion should wait for enough days, got: {promotion}"
    assert (
        promotion["days_evaluated"] == 2
    ), f"Expected 2 evaluated days, got: {promotion}"


def test_run_monthly_retraining_shadow_mode_registers_challenger(test_settings):
    settings = replace(test_settings, ENABLE_RL=True, ENABLE_SHADOW_EVAL=True)
    train_fn = MagicMock(
        return_value={
            "model_id": "challenger-1",
            "model_type": "DQN",
            "wf_score": 0.74,
            "model_path": "/challenger.zip",
            "metadata": {"reward_strategy": "price_diff"},
        }
    )
    shadow_evaluator = MagicMock()

    use_case = RunMonthlyRetrainingUseCase(
        settings=settings,
        ticker_provider=lambda: ["VOO"],
        walk_forward_fn=lambda ticker: {"normalized_score": 0.74},
        train_rl_fn=train_fn,
        shadow_evaluator=shadow_evaluator,
    )

    result = use_case.run(dry_run=False)

    assert (
        result["data"]["shadow_registered"] == 1
    ), f"Expected shadow registration, got: {result}"
    train_fn.assert_called_once()
    shadow_evaluator.register_challenger.assert_called_once()


def test_daily_pipeline_shadow_evaluation_calls_evaluator(test_settings):
    settings = replace(test_settings, ENABLE_SHADOW_EVAL=True)
    pipeline = MagicMock()
    pipeline._get_settings.return_value = settings
    pipeline.history_store = MagicMock()
    pipeline.data_fetcher = MagicMock()
    pipeline.email_notifier = MagicMock()
    pipeline.logger = MagicMock()

    use_case = DailyPipelineUseCase(pipeline)
    candidate = {
        "ticker": "VOO",
        "payload": {"model_version": "champion-1"},
        "decision": {"action_code": 1, "confidence": 0.8},
    }

    with patch("app.models.shadow_evaluator.ShadowEvaluator") as shadow_cls:
        shadow_cls.return_value.evaluate_daily_shadows.return_value = [
            {"promote": False}
        ]
        use_case._run_shadow_evaluation([candidate], date(2026, 4, 14), dry_run=False)

    shadow_cls.assert_called_once()
    shadow_cls.return_value.evaluate_daily_shadows.assert_called_once_with(
        champion_candidate=candidate,
        as_of_date=date(2026, 4, 14),
    )
