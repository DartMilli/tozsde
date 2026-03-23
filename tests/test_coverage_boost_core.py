import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.analysis.analyzer import get_params, compute_signals
from app.decision.ensemble_quality import bucket_ensemble_quality, EnsembleQualityBucket
from app.decision.decision_reliability import assess_decision_reliability
from app.decision.drift_detector import (
    PerformanceDriftDetector,
    batch_check_drift,
    get_drifting_tickers,
)
from app.decision.safety_rules import SafetyRuleEngine
from app.backtesting.history_store import HistoryStore
from app.models.model_reliability import ModelReliabilityAnalyzer
from app.decision.portfolio_correlation_manager import PortfolioCorrelationManager
from app.decision.etf_allocator import (
    ETFAllocator,
    get_low_cost_etf,
    estimate_portfolio_cost,
    classify_asset_type,
    AssetType,
)
from app.data_access.data_manager import DataManager
from app.reporting.audit_builder import build_audit_metadata
import main


def test_analyzer_get_params_with_file(tmp_path, monkeypatch, test_settings):
    from app.analysis import set_settings as set_analysis_settings

    params = {"TEST": {"bbands_stddev": "2.5", "sma_period": 10}}
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params))

    set_analysis_settings(replace(test_settings, PARAMS_FILE_PATH=params_path))

    loaded = get_params("TEST")
    assert loaded["bbands_stddev"] == 2.5
    assert loaded["sma_period"] == 10


def test_analyzer_get_params_missing_file(tmp_path, monkeypatch, test_settings):
    from app.analysis import set_settings as set_analysis_settings

    params_path = tmp_path / "missing.json"
    set_analysis_settings(replace(test_settings, PARAMS_FILE_PATH=params_path))
    loaded = get_params("MISSING")
    assert "sma_period" in loaded


def test_compute_signals_empty_df_returns_empty():
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    signals, indicators = compute_signals(df, "TEST", None)
    assert signals == []
    assert indicators["SMA"] is None


def test_ensemble_quality_buckets():
    assert bucket_ensemble_quality(1.0) in {
        EnsembleQualityBucket.STRONG,
        EnsembleQualityBucket.NORMAL,
        EnsembleQualityBucket.WEAK,
        EnsembleQualityBucket.CHAOTIC,
    }


def test_decision_reliability_levels(test_settings):
    low = assess_decision_reliability(0.1, 0.2, settings=test_settings)
    assert low.trade_allowed is False

    strong = assess_decision_reliability(
        test_settings.STRONG_CONFIDENCE_THRESHOLD + 0.05,
        test_settings.STRONG_WF_THRESHOLD + 0.05,
        settings=test_settings,
    )
    assert strong.trade_allowed is True


def test_drift_detector_alerts(monkeypatch):
    detector = PerformanceDriftDetector(
        lookback_days=3, drift_threshold=0.1, critical_threshold=0.2
    )

    monkeypatch.setattr(
        detector, "_load_historical_scores", lambda ticker: [0.8, 0.75, 0.7]
    )
    info = detector.check_drift("TEST", current_score=0.5)
    assert info["alert_level"] in {"WARNING", "CRITICAL"}

    alert = detector.generate_alert("TEST", info)
    assert alert is not None


def test_batch_check_drift_handles_error(monkeypatch):
    detector = PerformanceDriftDetector()
    monkeypatch.setattr(
        detector,
        "check_drift",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        "app.decision.drift_detector.PerformanceDriftDetector", lambda: detector
    )

    results = batch_check_drift({"A": 0.5})
    assert results["A"]["alert_level"] == "ERROR"


def test_get_drifting_tickers(monkeypatch):
    def fake_check(scores_dict):
        return {
            "A": {"drifting": True, "alert_level": "WARNING"},
            "B": {"drifting": False, "alert_level": "NONE"},
        }

    monkeypatch.setattr("app.decision.drift_detector.batch_check_drift", fake_check)
    drifting = get_drifting_tickers({"A": 0.5, "B": 0.6}, alert_level="WARNING")
    assert drifting == ["A"]


def test_safety_rules_overrides(monkeypatch):
    class DummyHistory:
        def load_range(self, ticker, start, end):
            return [{"action_code": 1}, {"action_code": 1}]

        def load_recent_outcomes(self, ticker, n=3):
            return [{"success": False}, {"success": False}, {"success": False}]

    monkeypatch.setattr(
        "app.decision.safety_rules.get_market_volatility_index", lambda: 35.0
    )

    class DummyDM:
        def get_market_regime_is_bear(self, ref_date=None):
            return True

    monkeypatch.setattr("app.decision.safety_rules.DataManager", lambda: DummyDM())

    engine = SafetyRuleEngine(DummyHistory())
    monkeypatch.setattr(engine, "_is_bear_market", lambda *args, **kwargs: True)
    decision = {
        "action_code": 1,
        "action": "BUY",
        "ensemble_quality": "CHAOTIC",
        "strength": "NORMAL",
        "no_trade": False,
        "reasons": [],
    }

    result = engine.apply("TEST", decision, date.today())
    assert result["no_trade"] is True
    assert result.get("safety_override") is True


def test_history_store_load_range(monkeypatch):
    hs = HistoryStore()

    def fake_fetch_history_range(ticker, start_iso, end_iso):
        return [("2026-02-01T00:00:00", "BUY", json.dumps({"action_code": 1}), "{}")]

    monkeypatch.setattr(hs.dm, "fetch_history_range", fake_fetch_history_range)
    out = hs.load_range("TEST", date.today() - timedelta(days=1), date.today())
    assert out[0]["action_code"] == 1


def test_model_reliability_scores(tmp_path, monkeypatch, test_settings):
    from app.models import set_settings as set_model_settings

    decision_dir = tmp_path / "decision"
    outcome_dir = tmp_path / "outcome"
    decision_dir.mkdir()
    outcome_dir.mkdir()

    model_settings = replace(
        test_settings,
        DECISION_LOG_DIR=decision_dir,
        DECISION_OUTCOME_DIR=outcome_dir,
        RELIABILITY_SOURCE="file",
    )
    set_model_settings(model_settings)

    day = "2026-02-01"
    (decision_dir / day).mkdir()
    (outcome_dir / day).mkdir()

    (decision_dir / day / "TEST.json").write_text(
        json.dumps({"model_votes": [{"model_path": "m", "confidence": 0.8}]})
    )
    (outcome_dir / day / "TEST.json").write_text(
        json.dumps({"success": True, "future_return": 0.01})
    )

    analyzer = ModelReliabilityAnalyzer(settings=model_settings)
    scores = analyzer.analyze(
        "TEST", date.fromisoformat(day), date.fromisoformat(day), min_samples=1
    )
    assert scores["m"]["reliability_score"] >= 0


def test_portfolio_correlation_manager(monkeypatch, sample_df):
    manager = PortfolioCorrelationManager(lookback_days=10)

    class DummyDM:
        def load_ohlcv(self, ticker):
            return sample_df

    monkeypatch.setattr(manager, "dm", DummyDM())
    corr = manager.compute_correlation_matrix(["A", "B"], use_cache=False)
    assert not corr.empty

    score = manager.calculate_diversification_score({"A": 0.5, "B": 0.5})
    assert 0.0 <= score <= 1.0

    selected = manager.optimize_for_low_correlation(["A", "B", "C"], target_size=2)
    assert len(selected) >= 1

    high_corr = manager.get_highly_correlated_pairs(["A", "B"], threshold=0.1)
    assert isinstance(high_corr, list)

    manager.clear_cache()


def test_data_manager_recent_metrics(test_db):
    dm = test_db
    dm.log_pipeline_execution("A", "success", 1.0)
    dm.log_pipeline_execution("B", "error", 0.5, error_message="fail")
    metrics = dm.get_recent_metrics(hours=24)
    assert metrics["total_executions"] >= 2


def test_data_manager_market_data_roundtrip(test_db, sample_df):
    dm = test_db
    df = sample_df.copy()
    df.index.name = "Date"
    df = df.rename(columns={"Close": "Close"})
    dm.save_market_data("VIX", df)
    rows = dm.get_market_data("VIX", days=1)
    assert len(rows) >= 1


def test_audit_builder_metadata(monkeypatch, test_settings):
    from app.reporting import audit_builder

    audit_builder.set_settings(replace(test_settings, ENABLE_DRIFT_DETECTION=False))
    payload = {"ticker": "TEST", "model_votes": [], "volatility": 0.01}
    decision = {
        "action_code": 1,
        "action": "BUY",
        "confidence": 0.6,
        "wf_score": 0.7,
        "ensemble_quality": 0.5,
        "quality_score": 0.7,
        "no_trade": False,
    }

    audit = build_audit_metadata(payload, decision)
    assert "confidence_bucket" in audit
    assert "ensemble_quality" in audit


def test_etf_allocator_mix_and_costs():
    allocator = ETFAllocator()
    comp = allocator.compare_etf_costs("VOO", "SPY")
    assert comp.cheaper_option in {"VOO", "SPY"}

    mix = allocator.calculate_optimal_mix(
        target_sectors={"Technology": 0.5},
        available_etfs=["VOO", "SPY"],
        available_stocks=["AAPL", "MSFT"],
    )
    assert mix.etf_weights or mix.stock_weights

    assert get_low_cost_etf(None) == "VOO"
    cost = estimate_portfolio_cost({"VOO": 1.0})
    assert cost > 0
    assert classify_asset_type("VOO") == AssetType.ETF


def test_main_weekly_and_monthly(monkeypatch, test_settings):
    class DummyWeeklyUseCase:
        def run(self, dry_run=False):
            return {"status": "ok"}

    class DummyMonthlyUseCase:
        def run(self, dry_run=False):
            return {"status": "ok"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type(
            "C",
            (),
            {
                "weekly_reliability": DummyWeeklyUseCase(),
                "monthly_retraining": DummyMonthlyUseCase(),
            },
        )(),
    )

    main.run_weekly(dry_run=False)
    main.run_monthly(dry_run=False)


def test_main_manual_commands(monkeypatch):
    class DummyWalkUseCase:
        def run(self, ticker=None):
            return {"status": "ok", "ticker": ticker}

    class DummyTrainUseCase:
        def run(self, ticker=None, **kwargs):
            return {"status": "ok", "ticker": ticker}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type(
            "C",
            (),
            {"walk_forward": DummyWalkUseCase(), "train_rl": DummyTrainUseCase()},
        )(),
    )

    main.run_walk_forward_manual("TEST")
    main.run_train_rl_manual("TEST")


def test_main_parse_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "daily", "--dry-run"])
    args = main.parse_arguments()
    assert args.command == "daily"
    assert args.dry_run is True
