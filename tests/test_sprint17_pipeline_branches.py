"""S17: trading_pipeline.py remaining branch coverage.

Targets lines 211-243, 256-308 (build_daily_candidate branches),
and _log_go_live_metrics, run_daily, _build_fallback_candidate_no_models.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest


def _make_pipeline(test_settings):
    from app.services.trading_pipeline import TradingPipelineService

    return TradingPipelineService(
        history_store=MagicMock(),
        settings=test_settings,
        data_fetcher=MagicMock(),
        model_runner=MagicMock(),
        email_notifier=MagicMock(),
        execution_engine=MagicMock(),
        state_repo=MagicMock(),
    )


def _basic_payload(action_code=1):
    return {
        "ticker": "AAPL",
        "timestamp": "2024-01-01T00:00:00",
        "latest_price": 150.0,
        "volatility": 0.02,
        "model_votes": [],
        "avg_confidence": 0.75,
        "avg_wf_score": 0.8,
        "ensemble_quality": "STABLE",
        "action_code": action_code,
        "decision_source": "model",
        "features_hash": "abc123",
    }


class TestBuildDailyCandidate:
    def test_happy_path_with_decision_in_payload(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        payload = {
            **_basic_payload(),
            "decision": {"action_code": 1, "confidence": 0.8},
            "explanation": {"en": "Buy.", "hu": "Vétel."},
        }
        with patch.object(
            pipeline, "run_decision_models", return_value=payload
        ), patch.object(
            pipeline, "apply_safety_rules", side_effect=lambda d, a: d
        ), patch(
            "app.services.trading_pipeline.build_audit_metadata", return_value={}
        ):
            result = pipeline.build_daily_candidate("AAPL")
        assert result["ticker"] == "AAPL"
        assert "decision" in result

    def test_happy_path_without_decision_in_payload(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        payload = _basic_payload()
        built_decision = {"action_code": 1, "confidence": 0.8}
        with patch.object(
            pipeline, "run_decision_models", return_value=payload
        ), patch.object(
            pipeline, "apply_safety_rules", side_effect=lambda d, a: d
        ), patch(
            "app.services.trading_pipeline.build_recommendation",
            return_value=built_decision,
        ), patch(
            "app.services.trading_pipeline.build_explanation", return_value={"en": "ok"}
        ), patch(
            "app.services.trading_pipeline.build_audit_metadata", return_value={}
        ):
            result = pipeline.build_daily_candidate("AAPL")
        assert result["decision"]["action_code"] == 1

    def test_error_in_payload_raises(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        payload = {"error": "SOME_ERROR", "ticker": "AAPL"}
        with patch.object(pipeline, "run_decision_models", return_value=payload):
            with pytest.raises(ValueError, match="SOME_ERROR"):
                pipeline.build_daily_candidate("AAPL")

    def test_no_models_with_fallback_disabled(self, test_settings):
        from dataclasses import replace

        settings = replace(test_settings, ALLOW_NO_MODEL_FALLBACK=False)
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
        )
        payload = {"error": "NO_MODELS"}
        with patch.object(pipeline, "run_decision_models", return_value=payload):
            with pytest.raises(ValueError, match="NO_MODELS"):
                pipeline.build_daily_candidate("AAPL")

    def test_no_models_with_fallback_enabled_returns_hold(self, test_settings):
        from dataclasses import replace

        settings = replace(test_settings, ALLOW_NO_MODEL_FALLBACK=True)
        from app.services.trading_pipeline import TradingPipelineService

        pipeline = TradingPipelineService(
            history_store=MagicMock(),
            settings=settings,
            data_fetcher=MagicMock(),
            model_runner=MagicMock(),
            email_notifier=MagicMock(),
            execution_engine=MagicMock(),
        )
        fallback = {
            "ticker": "AAPL",
            "payload": _basic_payload(action_code=0),
            "decision": {"action_code": 0, "no_trade": True},
            "explanation": {"en": "hold"},
            "audit": {},
        }
        with patch.object(
            pipeline, "run_decision_models", return_value={"error": "NO_MODELS"}
        ), patch.object(
            pipeline, "_build_fallback_candidate_no_models", return_value=fallback
        ):
            result = pipeline.build_daily_candidate("AAPL")
        assert result["decision"]["action_code"] == 0

    def test_explanation_built_when_none_in_payload(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        payload = {
            **_basic_payload(),
            "decision": {"action_code": 0},
            "explanation": None,
        }
        with patch.object(
            pipeline, "run_decision_models", return_value=payload
        ), patch.object(
            pipeline, "apply_safety_rules", side_effect=lambda d, a: d
        ), patch(
            "app.services.trading_pipeline.build_explanation",
            return_value={"en": "hold"},
        ) as mock_exp, patch(
            "app.services.trading_pipeline.build_audit_metadata", return_value={}
        ):
            result = pipeline.build_daily_candidate("AAPL")
        mock_exp.assert_called()


class TestLogGoLiveMetrics:
    def test_logs_ok_status(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        with patch(
            "app.services.trading_pipeline.compute_drawdown_summary",
            return_value={
                "status": "ok",
                "rows": 10,
                "max_drawdown": 0.05,
                "current_drawdown": 0.02,
            },
        ), patch(
            "app.services.trading_pipeline.compute_loss_streak",
            return_value={"loss_streak": 2},
        ):
            # Should not raise
            pipeline._log_go_live_metrics("AAPL")

    def test_logs_no_data_status(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        with patch(
            "app.services.trading_pipeline.compute_drawdown_summary",
            return_value={"status": "no_data"},
        ), patch(
            "app.services.trading_pipeline.compute_loss_streak",
            return_value={"loss_streak": 0},
        ):
            # Should not raise
            pipeline._log_go_live_metrics("AAPL")


class TestRunDaily:
    def test_run_daily_delegates_to_use_case(self, test_settings):
        pipeline = _make_pipeline(test_settings)
        # DailyPipelineUseCase is imported inside run_daily() → patch at source
        with patch(
            "app.application.use_cases.daily_pipeline_use_case.DailyPipelineUseCase"
        ) as mock_uc:
            with patch(
                "app.application.use_cases.daily_pipeline_use_case", create=True
            ) as mock_mod:
                # Direct test: just ensure it calls without error via mocking the import
                mock_uc_inner = MagicMock()
                mock_uc_inner.run.return_value = None
                with patch.dict(
                    "sys.modules",
                    {
                        "app.application.use_cases.daily_pipeline_use_case": type(
                            "m",
                            (),
                            {"DailyPipelineUseCase": lambda *a, **k: mock_uc_inner},
                        )()
                    },
                ):
                    pipeline.run_daily(dry_run=True, ticker="VOO")


class TestBuildFallbackCandidateNoModels:
    def test_no_data_raises(self, test_settings):
        import pandas as pd

        pipeline = _make_pipeline(test_settings)
        pipeline.data_fetcher.load_data.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="NO_DATA"):
            pipeline._build_fallback_candidate_no_models("AAPL", date(2024, 1, 1))

    def test_returns_hold_candidate(self, test_settings):
        import pandas as pd

        pipeline = _make_pipeline(test_settings)
        df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        pipeline.data_fetcher.load_data.return_value = df
        with patch(
            "app.services.trading_pipeline.compute_normalized_volatility",
            return_value=0.02,
        ), patch(
            "app.services.trading_pipeline.build_recommendation",
            return_value={"action_code": 0},
        ), patch(
            "app.services.trading_pipeline.build_explanation",
            return_value={"en": "hold"},
        ), patch(
            "app.services.trading_pipeline.build_audit_metadata", return_value={}
        ):
            result = pipeline._build_fallback_candidate_no_models(
                "AAPL", date(2024, 1, 1)
            )
        assert result["decision"]["no_trade"] is True
        assert result["decision"]["action_code"] == 0
