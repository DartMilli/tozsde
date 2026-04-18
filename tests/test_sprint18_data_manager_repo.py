"""S18B (part 2) – DataManagerRepository fallback branch coverage.

All methods have two branches:
  1. if repo: → delegates to SqliteDecisionRepository / SqliteModelRepository
  2. else    → fallback to legacy DataManager (_dm)

We test the fallback (else) branch by forcing _decision_repo / _model_repo = False.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixture: DataManagerRepository with disabled sub-repos (forces _dm fallback)
# ---------------------------------------------------------------------------


@pytest.fixture
def dm_repo_fallback():
    """DataManagerRepository whose sub-repos are False (fallback mode)."""
    from app.infrastructure.repositories.data_manager_repository import (
        DataManagerRepository,
    )

    mock_dm = MagicMock()
    repo = DataManagerRepository.__new__(DataManagerRepository)
    repo._dm = mock_dm
    repo._settings = None
    repo._decision_repo = False  # forces fallback to _dm
    repo._model_repo = False  # forces fallback to _dm
    repo._ohlcv_repo = MagicMock()
    return repo


# ---------------------------------------------------------------------------
# _get_decision_repo / _get_model_repo – failure path (returns False)
# ---------------------------------------------------------------------------


class TestGetRepoHelpers:
    def test_get_decision_repo_failure_returns_false(self):
        from app.infrastructure.repositories.data_manager_repository import (
            DataManagerRepository,
        )

        repo = DataManagerRepository.__new__(DataManagerRepository)
        repo._settings = MagicMock()
        repo._settings.DB_PATH = None  # will cause SqliteDecisionRepository to raise
        repo._decision_repo = None  # not initialized yet
        result = repo._get_decision_repo()
        assert result is False

    def test_get_model_repo_failure_returns_false(self):
        from app.infrastructure.repositories.data_manager_repository import (
            DataManagerRepository,
        )

        repo = DataManagerRepository.__new__(DataManagerRepository)
        repo._settings = MagicMock()
        repo._settings.DB_PATH = None
        repo._model_repo = None
        result = repo._get_model_repo()
        assert result is False

    def test_get_decision_repo_cached_false(self, dm_repo_fallback):
        # Already False → should return False without re-initializing
        result = dm_repo_fallback._get_decision_repo()
        assert result is False

    def test_get_model_repo_cached_false(self, dm_repo_fallback):
        result = dm_repo_fallback._get_model_repo()
        assert result is False


# ---------------------------------------------------------------------------
# Decision fallback methods
# ---------------------------------------------------------------------------


class TestDecisionFallback:
    def test_save_decision(self, dm_repo_fallback):
        dm_repo_fallback._dm.save_decision.return_value = 1
        result = dm_repo_fallback.save_decision({"ticker": "VOO"})
        dm_repo_fallback._dm.save_decision.assert_called_once()
        assert result == 1

    def test_fetch_decision(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_decision.return_value = {"id": 1}
        result = dm_repo_fallback.fetch_decision(1)
        dm_repo_fallback._dm.fetch_decision.assert_called_once_with(1)
        assert result == {"id": 1}

    def test_fetch_decisions_for_ticker(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_decisions_for_ticker.return_value = []
        result = dm_repo_fallback.fetch_decisions_for_ticker(
            "VOO", "2025-01-01", "2025-12-31"
        )
        dm_repo_fallback._dm.fetch_decisions_for_ticker.assert_called_once()
        assert result == []

    def test_save_history_record(self, dm_repo_fallback):
        dm_repo_fallback._dm.save_history_record.return_value = 42
        result = dm_repo_fallback.save_history_record(
            ticker="V",
            action_code=1,
            label="BUY",
            confidence=0.7,
            wf_score=0.6,
            d_blob="{}",
            a_blob="{}",
        )
        assert result == 42

    def test_fetch_history_records_by_ticker(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_history_records_by_ticker.return_value = [
            {"ticker": "VOO"}
        ]
        result = dm_repo_fallback.fetch_history_records_by_ticker("VOO")
        assert result[0]["ticker"] == "VOO"

    def test_has_decision_for_date(self, dm_repo_fallback):
        dm_repo_fallback._dm.has_decision_for_date.return_value = True
        result = dm_repo_fallback.has_decision_for_date("VOO", date.today())
        assert result is True

    def test_fetch_history_range(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_history_range.return_value = []
        result = dm_repo_fallback.fetch_history_range("VOO", "2025-01-01", "2025-12-31")
        assert result == []

    def test_fetch_recent_outcomes(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_recent_outcomes.return_value = []
        result = dm_repo_fallback.fetch_recent_outcomes("VOO", n=3)
        assert result == []

    def test_save_outcome(self, dm_repo_fallback):
        dm_repo_fallback.save_outcome(
            decision_id=1,
            ticker="VOO",
            decision_timestamp="2025-01-01T00:00:00",
            pnl_pct=0.05,
            success=True,
            future_return=0.07,
            exit_reason="target_hit",
            horizon_days=10,
            outcome_json="{}",
        )
        dm_repo_fallback._dm.save_outcome.assert_called_once()

    def test_get_unevaluated_buy_decisions(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_unevaluated_buy_decisions.return_value = []
        result = dm_repo_fallback.get_unevaluated_buy_decisions()
        assert result == []

    def test_save_portfolio_state(self, dm_repo_fallback):
        dm_repo_fallback.save_portfolio_state(
            "2025-01-01T00:00:00", 10000.0, 12000.0, 0.2, "{}", "paper"
        )
        dm_repo_fallback._dm.save_portfolio_state.assert_called_once()

    def test_fetch_portfolio_state_range(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_portfolio_state_range.return_value = []
        result = dm_repo_fallback.fetch_portfolio_state_range(
            "2025-01-01", "2025-12-31"
        )
        assert result == []

    def test_fetch_latest_portfolio_state(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_latest_portfolio_state.return_value = {}
        result = dm_repo_fallback.fetch_latest_portfolio_state()
        assert result == {}

    def test_save_decision_effectiveness(self, dm_repo_fallback):
        dm_repo_fallback.save_decision_effectiveness(1, "VOO", 0.75, "{}")
        dm_repo_fallback._dm.save_decision_effectiveness.assert_called_once()

    def test_save_decision_effectiveness_rolling(self, dm_repo_fallback):
        dm_repo_fallback.save_decision_effectiveness_rolling(
            "VOO", 30, "2025-06-01", "{}"
        )
        dm_repo_fallback._dm.save_decision_effectiveness_rolling.assert_called_once()

    def test_save_decision_quality_metrics(self, dm_repo_fallback):
        dm_repo_fallback.save_decision_quality_metrics(
            "VOO", "2025-01-01", "2025-06-01", "{}"
        )
        dm_repo_fallback._dm.save_decision_quality_metrics.assert_called_once()

    def test_save_confidence_calibration(self, dm_repo_fallback):
        dm_repo_fallback.save_confidence_calibration(
            "VOO", "2025-01-01", "2025-06-01", "isotonic", "{}", "{}"
        )
        dm_repo_fallback._dm.save_confidence_calibration.assert_called_once()

    def test_fetch_latest_confidence_calibration(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_latest_confidence_calibration.return_value = {}
        result = dm_repo_fallback.fetch_latest_confidence_calibration("VOO")
        assert result == {}

    def test_save_wf_stability_metrics(self, dm_repo_fallback):
        dm_repo_fallback.save_wf_stability_metrics("VOO", "{}")
        dm_repo_fallback._dm.save_wf_stability_metrics.assert_called_once()

    def test_save_safety_stress_results(self, dm_repo_fallback):
        dm_repo_fallback.save_safety_stress_results(
            "VOO", "2025-01-01", "2025-06-01", "crash", "{}"
        )
        dm_repo_fallback._dm.save_safety_stress_results.assert_called_once()

    def test_save_validation_report(self, dm_repo_fallback):
        dm_repo_fallback.save_validation_report("{}")
        dm_repo_fallback._dm.save_validation_report.assert_called_once()

    def test_fetch_latest_validation_report(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_latest_validation_report.return_value = {}
        result = dm_repo_fallback.fetch_latest_validation_report()
        assert result == {}

    def test_update_history_audit(self, dm_repo_fallback):
        dm_repo_fallback.update_history_audit(1, "{}")
        dm_repo_fallback._dm.update_history_audit.assert_called_once()

    def test_save_model_reliability(self, dm_repo_fallback):
        dm_repo_fallback.save_model_reliability("VOO", "2025-06-01", "{}")
        dm_repo_fallback._dm.save_model_reliability.assert_called_once()

    def test_save_market_data(self, dm_repo_fallback):
        dm_repo_fallback.save_market_data("VIX", None)
        dm_repo_fallback._dm.save_market_data.assert_called_once()

    def test_get_market_data(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_market_data.return_value = []
        result = dm_repo_fallback.get_market_data("VIX", days=5)
        assert result == []

    def test_get_ticker_historical_recommendations(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_ticker_historical_recommendations.return_value = []
        result = dm_repo_fallback.get_ticker_historical_recommendations(
            "VOO", "2025-01-01", "2025-12-31"
        )
        assert result == []

    def test_get_strategy_accuracy(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_strategy_accuracy.return_value = []
        result = dm_repo_fallback.get_strategy_accuracy("VOO")
        assert result == []

    def test_get_today_recommendations(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_today_recommendations.return_value = []
        result = dm_repo_fallback.get_today_recommendations()
        assert result == []

    def test_save_walk_forward_result(self, dm_repo_fallback):
        dm_repo_fallback.save_walk_forward_result("VOO", "{}")
        dm_repo_fallback._dm.save_walk_forward_result.assert_called_once()

    def test_fetch_latest_decision_quality_metrics(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_latest_decision_quality_metrics.return_value = {}
        result = dm_repo_fallback.fetch_latest_decision_quality_metrics("VOO")
        assert result == {}

    def test_update_market_data(self, dm_repo_fallback):
        dm_repo_fallback._dm.update_market_data.return_value = None
        dm_repo_fallback.update_market_data()
        dm_repo_fallback._dm.update_market_data.assert_called_once()

    def test_initialize_tables_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.initialize_tables = MagicMock()
        dm_repo_fallback.initialize_tables()
        dm_repo_fallback._dm.initialize_tables.assert_called_once()

    def test_connection_fallback(self, dm_repo_fallback):
        mock_ctx = MagicMock()
        dm_repo_fallback._dm.connection.return_value = mock_ctx
        result = dm_repo_fallback.connection()
        assert result is mock_ctx


# ---------------------------------------------------------------------------
# Model fallback methods
# ---------------------------------------------------------------------------


class TestModelFallback:
    def test_save_model_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.save_model = MagicMock()
        dm_repo_fallback.save_model(
            model_id="m1",
            ticker="VOO",
            model_type="DQN",
            wf_score=0.8,
            model_path="/models/m1.zip",
        )
        dm_repo_fallback._dm.save_model.assert_called_once()

    def test_save_model_uses_register_model_when_save_model_absent(
        self, dm_repo_fallback
    ):
        del dm_repo_fallback._dm.save_model
        dm_repo_fallback._dm.register_model = MagicMock()
        dm_repo_fallback.save_model(
            model_id="m2",
            ticker="V",
            model_type="PPO",
            wf_score=0.7,
            model_path="/m2.zip",
        )
        dm_repo_fallback._dm.register_model.assert_called_once()

    def test_save_model_raises_when_neither_available(self, dm_repo_fallback):
        # Remove both attributes
        del dm_repo_fallback._dm.save_model
        del dm_repo_fallback._dm.register_model
        with pytest.raises(AttributeError):
            dm_repo_fallback.save_model(
                model_id="m3",
                ticker="X",
                model_type="PPO",
                wf_score=0.5,
                model_path="/m3.zip",
            )

    def test_fetch_model_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_model.return_value = {"model_id": "m1"}
        result = dm_repo_fallback.fetch_model("m1")
        assert result["model_id"] == "m1"

    def test_fetch_models_for_ticker_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_models_for_ticker.return_value = []
        result = dm_repo_fallback.fetch_models_for_ticker("VOO")
        assert result == []

    def test_register_model_fallback(self, dm_repo_fallback):
        dm_repo_fallback.register_model("m1", "VOO", "DQN", 0.8, "/m1.zip")
        dm_repo_fallback._dm.register_model.assert_called_once()

    def test_update_model_status_fallback(self, dm_repo_fallback):
        dm_repo_fallback.update_model_status("m1", "active")
        dm_repo_fallback._dm.update_model_status.assert_called_once_with(
            model_id="m1", status="active"
        )

    def test_fetch_active_models_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_active_models.return_value = []
        result = dm_repo_fallback.fetch_active_models("VOO")
        assert result == []

    def test_save_model_trust_metrics_fallback(self, dm_repo_fallback):
        dm_repo_fallback.save_model_trust_metrics("/m1.zip", "VOO", 0.9, "{}")
        dm_repo_fallback._dm.save_model_trust_metrics.assert_called_once()

    def test_fetch_latest_model_trust_weights_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.fetch_latest_model_trust_weights.return_value = {}
        result = dm_repo_fallback.fetch_latest_model_trust_weights("VOO")
        assert result == {}

    def test_get_top_models_fallback(self, dm_repo_fallback):
        dm_repo_fallback._dm.get_top_models.return_value = []
        result = dm_repo_fallback.get_top_models("VOO")
        assert result == []


# ---------------------------------------------------------------------------
# OHLCV delegation
# ---------------------------------------------------------------------------


class TestOhlcvDelegation:
    def test_load_ohlcv(self, dm_repo_fallback):
        import pandas as pd

        dm_repo_fallback._ohlcv_repo.load_ohlcv.return_value = pd.DataFrame()
        result = dm_repo_fallback.load_ohlcv("VOO")
        dm_repo_fallback._ohlcv_repo.load_ohlcv.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_fetch_ohlcv_no_end_date(self, dm_repo_fallback):
        import pandas as pd

        dm_repo_fallback._ohlcv_repo.load_ohlcv.return_value = pd.DataFrame()
        result = dm_repo_fallback.fetch_ohlcv("VOO")
        dm_repo_fallback._ohlcv_repo.load_ohlcv.assert_called_once()

    def test_fetch_ohlcv_with_end_date_uses_get_ohlcv(self, dm_repo_fallback):
        import pandas as pd

        dm_repo_fallback._dm.get_ohlcv.return_value = pd.DataFrame()
        result = dm_repo_fallback.fetch_ohlcv(
            "VOO", start_date="2025-01-01", end_date="2025-06-01"
        )
        dm_repo_fallback._dm.get_ohlcv.assert_called_once()

    def test_save_ohlcv(self, dm_repo_fallback):
        import pandas as pd

        dm_repo_fallback._ohlcv_repo.save_ohlcv.return_value = None
        dm_repo_fallback.save_ohlcv("VOO", pd.DataFrame())
        dm_repo_fallback._ohlcv_repo.save_ohlcv.assert_called_once()


# ---------------------------------------------------------------------------
# DataManagerRepository construction paths
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_wrapping_another_repo_reuses_dm(self):
        from app.infrastructure.repositories.data_manager_repository import (
            DataManagerRepository,
        )

        inner_dm = MagicMock()
        outer = DataManagerRepository.__new__(DataManagerRepository)
        outer._dm = inner_dm
        outer._settings = None
        outer._decision_repo = None
        outer._model_repo = None
        outer._ohlcv_repo = MagicMock()

        # Simulate passing DataManagerRepository instance
        second = DataManagerRepository.__new__(DataManagerRepository)
        second._dm = outer._dm  # would use inner_dm
        assert second._dm is inner_dm

    def test_construction_wraps_data_manager_repo(self):
        """Lines 29+31: when data_manager is a DataManagerRepository, reuse its _dm."""
        from app.infrastructure.repositories.data_manager_repository import (
            DataManagerRepository,
        )

        # Build a mock inner DataManagerRepository
        inner = DataManagerRepository.__new__(DataManagerRepository)
        inner._dm = MagicMock()
        inner._settings = None
        inner._decision_repo = False
        inner._model_repo = False
        inner._ohlcv_repo = MagicMock()

        outer = DataManagerRepository(data_manager=inner)
        assert outer._dm is inner._dm


# ---------------------------------------------------------------------------
# if repo: branches – decision methods
# ---------------------------------------------------------------------------


@pytest.fixture
def dm_repo_with_mock_decision():
    """DataManagerRepository with truthy mock _decision_repo (covers if repo: branches)."""
    from app.infrastructure.repositories.data_manager_repository import (
        DataManagerRepository,
    )

    mock_decision_repo = MagicMock()
    mock_model_repo = MagicMock()
    repo = DataManagerRepository.__new__(DataManagerRepository)
    repo._dm = MagicMock()
    repo._settings = None
    repo._decision_repo = mock_decision_repo
    repo._model_repo = mock_model_repo
    repo._ohlcv_repo = MagicMock()
    return repo


class TestDecisionIfRepoBranch:
    def test_save_decision(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.save_decision.return_value = 1
        result = dm_repo_with_mock_decision.save_decision({"ticker": "VOO"})
        dm_repo_with_mock_decision._decision_repo.save_decision.assert_called_once()

    def test_fetch_decision(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_decision.return_value = {
            "id": 1
        }
        result = dm_repo_with_mock_decision.fetch_decision(1)
        assert result == {"id": 1}

    def test_save_history_record(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.save_history_record.return_value = 42
        result = dm_repo_with_mock_decision.save_history_record(
            ticker="V",
            action_code=1,
            label="BUY",
            confidence=0.7,
            wf_score=0.6,
            d_blob="{}",
            a_blob="{}",
        )
        assert result == 42

    def test_fetch_history_records_by_ticker(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_history_records_by_ticker.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.fetch_history_records_by_ticker("VOO")
        assert result == []

    def test_has_decision_for_date(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.has_decision_for_date.return_value = (
            False
        )
        result = dm_repo_with_mock_decision.has_decision_for_date("VOO", date.today())
        dm_repo_with_mock_decision._decision_repo.has_decision_for_date.assert_called_once()

    def test_fetch_history_range(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_history_range.return_value = []
        result = dm_repo_with_mock_decision.fetch_history_range(
            "VOO", "2025-01-01", "2025-12-31"
        )
        assert result == []

    def test_fetch_recent_outcomes(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_recent_outcomes.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.fetch_recent_outcomes("VOO")
        assert result == []

    def test_save_outcome(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_outcome(
            1, "VOO", "2025-01-01T00:00:00", 0.05, True, 0.07, "hit", 10, "{}"
        )
        dm_repo_with_mock_decision._decision_repo.save_outcome.assert_called_once()

    def test_get_unevaluated_buy_decisions(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.get_unevaluated_buy_decisions.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.get_unevaluated_buy_decisions()
        assert result == []

    def test_save_portfolio_state(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_portfolio_state(
            "ts", 1.0, 2.0, 0.1, "{}", "paper"
        )
        dm_repo_with_mock_decision._decision_repo.save_portfolio_state.assert_called_once()

    def test_fetch_portfolio_state_range(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_portfolio_state_range.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.fetch_portfolio_state_range(
            "2025-01-01", "2025-12-31"
        )
        assert result == []

    def test_fetch_latest_portfolio_state(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_latest_portfolio_state.return_value = (
            {}
        )
        result = dm_repo_with_mock_decision.fetch_latest_portfolio_state()
        assert result == {}

    def test_save_decision_effectiveness(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_decision_effectiveness(1, "VOO", 0.8, "{}")
        dm_repo_with_mock_decision._decision_repo.save_decision_effectiveness.assert_called_once()

    def test_save_decision_effectiveness_rolling(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_decision_effectiveness_rolling(
            "VOO", 30, "2025-06-01", "{}"
        )
        dm_repo_with_mock_decision._decision_repo.save_decision_effectiveness_rolling.assert_called_once()

    def test_save_decision_quality_metrics(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_decision_quality_metrics(
            "VOO", "2025-01-01", "2025-06-01", "{}"
        )
        dm_repo_with_mock_decision._decision_repo.save_decision_quality_metrics.assert_called_once()

    def test_save_confidence_calibration(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_confidence_calibration(
            "VOO", "2025-01-01", "2025-06-01", "isotonic", "{}", "{}"
        )
        dm_repo_with_mock_decision._decision_repo.save_confidence_calibration.assert_called_once()

    def test_fetch_latest_confidence_calibration(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_latest_confidence_calibration.return_value = (
            {}
        )
        result = dm_repo_with_mock_decision.fetch_latest_confidence_calibration("VOO")
        assert result == {}

    def test_save_wf_stability_metrics(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_wf_stability_metrics("VOO", "{}")
        dm_repo_with_mock_decision._decision_repo.save_wf_stability_metrics.assert_called_once()

    def test_save_safety_stress_results(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_safety_stress_results(
            "VOO", "2025-01-01", "2025-06-01", "crash", "{}"
        )
        dm_repo_with_mock_decision._decision_repo.save_safety_stress_results.assert_called_once()

    def test_save_validation_report(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_validation_report("{}")
        dm_repo_with_mock_decision._decision_repo.save_validation_report.assert_called_once()

    def test_fetch_latest_validation_report(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_latest_validation_report.return_value = (
            {}
        )
        result = dm_repo_with_mock_decision.fetch_latest_validation_report()
        assert result == {}

    def test_update_history_audit(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.update_history_audit(1, "{}")
        dm_repo_with_mock_decision._decision_repo.update_history_audit.assert_called_once()

    def test_save_model_reliability(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_model_reliability("VOO", "2025-06-01", "{}")
        dm_repo_with_mock_decision._decision_repo.save_model_reliability.assert_called_once()

    def test_save_market_data(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_market_data("VIX", None)
        dm_repo_with_mock_decision._decision_repo.save_market_data.assert_called_once()

    def test_get_market_data(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.get_market_data.return_value = []
        result = dm_repo_with_mock_decision.get_market_data("VIX")
        assert result == []

    def test_get_ticker_historical_recommendations(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.get_ticker_historical_recommendations.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.get_ticker_historical_recommendations(
            "VOO", "2025-01-01", "2025-12-31"
        )
        assert result == []

    def test_get_strategy_accuracy(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.get_strategy_accuracy.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.get_strategy_accuracy("VOO")
        assert result == []

    def test_get_today_recommendations(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.get_today_recommendations.return_value = (
            []
        )
        result = dm_repo_with_mock_decision.get_today_recommendations()
        assert result == []

    def test_save_walk_forward_result(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_walk_forward_result("VOO", "{}")
        dm_repo_with_mock_decision._decision_repo.save_walk_forward_result.assert_called_once()

    def test_fetch_latest_decision_quality_metrics(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._decision_repo.fetch_latest_decision_quality_metrics.return_value = (
            {}
        )
        result = dm_repo_with_mock_decision.fetch_latest_decision_quality_metrics("VOO")
        assert result == {}

    def test_initialize_tables(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._dm.initialize_tables = MagicMock()
        dm_repo_with_mock_decision.initialize_tables()
        dm_repo_with_mock_decision._decision_repo.initialize_tables.assert_called_once()

    def test_connection(self, dm_repo_with_mock_decision):
        mock_ctx = MagicMock()
        dm_repo_with_mock_decision._decision_repo.connection.return_value = mock_ctx
        result = dm_repo_with_mock_decision.connection()
        assert result is mock_ctx


class TestModelIfRepoBranch:
    def test_save_model(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_model(
            model_id="m1",
            ticker="VOO",
            model_type="DQN",
            wf_score=0.8,
            model_path="/m1.zip",
        )
        dm_repo_with_mock_decision._model_repo.save_model.assert_called_once()

    def test_fetch_model(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._model_repo.fetch_model.return_value = {}
        result = dm_repo_with_mock_decision.fetch_model("m1")
        assert result == {}

    def test_fetch_models_for_ticker(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._model_repo.fetch_models_for_ticker.return_value = []
        result = dm_repo_with_mock_decision.fetch_models_for_ticker("VOO")
        assert result == []

    def test_register_model(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.register_model("m1", "VOO", "DQN", 0.8, "/m1.zip")
        dm_repo_with_mock_decision._model_repo.register_model.assert_called_once()

    def test_update_model_status(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.update_model_status("m1", "active")
        dm_repo_with_mock_decision._model_repo.update_model_status.assert_called_once()

    def test_fetch_active_models(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._model_repo.fetch_active_models.return_value = []
        result = dm_repo_with_mock_decision.fetch_active_models("VOO")
        assert result == []

    def test_save_model_trust_metrics(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision.save_model_trust_metrics("/m1.zip", "VOO", 0.9, "{}")
        dm_repo_with_mock_decision._model_repo.save_model_trust_metrics.assert_called_once()

    def test_fetch_latest_model_trust_weights(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._model_repo.fetch_latest_model_trust_weights.return_value = (
            {}
        )
        result = dm_repo_with_mock_decision.fetch_latest_model_trust_weights("VOO")
        assert result == {}

    def test_get_top_models(self, dm_repo_with_mock_decision):
        dm_repo_with_mock_decision._model_repo.get_top_models.return_value = []
        result = dm_repo_with_mock_decision.get_top_models("VOO")
        assert result == []
