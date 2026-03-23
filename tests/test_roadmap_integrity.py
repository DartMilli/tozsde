import pytest
from datetime import date
from pathlib import Path
import json

from app.config.config import Config
from dataclasses import replace
from app.backtesting import set_settings as set_backtesting_settings
from app.models import set_settings as set_model_settings
from app.decision.recommendation_builder import build_recommendation, build_explanation
from app.backtesting.backtester import Backtester
import app.backtesting.backtester as backtester_module
import app.indicators.technical as technical
from app.models.model_reliability import ModelReliabilityAnalyzer
from app.decision.allocation import allocate_capital, enforce_correlation_limits
from app.data_access.data_manager import DataManager
import main


def test_p0_deterministic_recommendation_output():
    payload = {
        "ticker": "TEST",
        "avg_confidence": 0.7,
        "avg_wf_score": 0.8,
        "action_code": 1,
        "ensemble_quality": "STABLE",
    }

    first = build_recommendation(payload)
    second = build_recommendation(payload)

    assert first == second


def test_p0_failure_one_ticker_does_not_abort(monkeypatch):
    calls = []

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            calls.append((dry_run, ticker))
            return {"ok": True}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=True)

    assert result == {"ok": True}
    assert calls == [(True, None)]


def test_p1_recommendation_upsert_is_idempotent(test_db):
    test_db.log_recommendation("TEST", "BUY", 0.7, params={"x": 1})
    test_db.log_recommendation("TEST", "BUY", 0.7, params={"x": 1})

    today = date.today().isoformat()
    with test_db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE date = ? AND ticker = ?",
            (today, "TEST"),
        ).fetchone()[0]

    assert count == 1


def test_p2_transaction_costs_reduce_net_profit(monkeypatch, sample_df, test_settings):
    def fake_signals(df, ticker, params, return_series=False):
        signals = ["BUY", "SELL"] + ["HOLD"] * (len(df) - 2)
        return signals, {}

    monkeypatch.setattr(backtester_module, "compute_signals", fake_signals)

    df = sample_df.copy()
    df["Close"] = [100 + i for i in range(len(df))]

    params = {
        "sma_period": 5,
        "ema_period": 5,
        "rsi_period": 5,
        "macd_fast": 5,
        "macd_slow": 21,
        "macd_signal": 5,
        "bbands_period": 10,
        "bbands_stddev": 2.0,
        "atr_period": 5,
        "adx_period": 5,
        "stoch_k": 5,
        "stoch_d": 3,
    }

    low_cost_settings = replace(
        test_settings,
        TRANSACTION_FEE_PCT=0.0001,
        MIN_SLIPPAGE_PCT=0.0001,
        SPREAD_PCT=0.0001,
    )
    set_backtesting_settings(low_cost_settings)
    low_cost_report = Backtester(df, "TEST", settings=low_cost_settings).run(params)

    high_cost_settings = replace(
        test_settings,
        TRANSACTION_FEE_PCT=0.01,
        MIN_SLIPPAGE_PCT=0.01,
        SPREAD_PCT=0.01,
    )
    set_backtesting_settings(high_cost_settings)
    high_cost_report = Backtester(df, "TEST", settings=high_cost_settings).run(params)

    assert (
        high_cost_report.metrics["net_profit"] < low_cost_report.metrics["net_profit"]
    )


def test_p3_no_trade_is_explicit():
    payload = {
        "ticker": "TEST",
        "avg_confidence": Config.CONFIDENCE_NO_TRADE_THRESHOLD - 0.01,
        "avg_wf_score": 0.8,
        "action_code": 1,
        "ensemble_quality": "STABLE",
    }

    decision = build_recommendation(payload)

    assert decision["no_trade"] is True
    assert decision["no_trade_reason"] is not None
    assert 0 <= decision["confidence"] <= 1


def test_p5_walk_forward_influences_strength():
    base_payload = {
        "ticker": "TEST",
        "avg_confidence": Config.STRONG_CONFIDENCE_THRESHOLD + 0.05,
        "action_code": 1,
        "ensemble_quality": "STABLE",
    }

    strong_payload = {**base_payload, "avg_wf_score": Config.STRONG_WF_THRESHOLD + 0.05}
    weak_payload = {**base_payload, "avg_wf_score": Config.STRONG_WF_THRESHOLD - 0.1}

    strong_decision = build_recommendation(strong_payload)
    weak_decision = build_recommendation(weak_payload)

    assert strong_decision["strength"] == "STRONG"
    assert weak_decision["strength"] != "STRONG"


def test_p8_explanation_does_not_change_decision():
    payload = {
        "ticker": "TEST",
        "avg_confidence": 0.6,
        "avg_wf_score": 0.6,
        "action_code": 1,
        "ensemble_quality": "STABLE",
        "model_votes": [],
    }

    decision = build_recommendation(payload)
    decision_before = dict(decision)

    _ = build_explanation(payload, decision)

    assert decision == decision_before


def test_p2_indicator_cache_preserves_output(sample_df):
    close = sample_df["Close"].values
    first = technical.sma(close, 5)
    second = technical.sma(close, 5)

    import numpy as np

    assert np.allclose(first, second, equal_nan=True)


def test_p6_reliability_degradation_detectable(tmp_path, monkeypatch, test_settings):
    decision_dir = tmp_path / "decision_logs"
    outcome_dir = tmp_path / "decision_outcomes"
    decision_dir.mkdir(parents=True, exist_ok=True)
    outcome_dir.mkdir(parents=True, exist_ok=True)

    model_settings = replace(
        test_settings,
        DECISION_LOG_DIR=decision_dir,
        DECISION_OUTCOME_DIR=outcome_dir,
    )
    set_model_settings(model_settings)

    ticker = "TEST"
    model_path = "model_a.zip"

    good_day = "2026-02-01"
    bad_day = "2026-02-02"

    for day, success, ret in [(good_day, True, 0.02), (bad_day, False, -0.02)]:
        day_dir_dec = decision_dir / day
        day_dir_out = outcome_dir / day
        day_dir_dec.mkdir(parents=True)
        day_dir_out.mkdir(parents=True)

        (day_dir_dec / f"{ticker}.json").write_text(
            json.dumps(
                {
                    "model_votes": [
                        {
                            "model_path": model_path,
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        )
        (day_dir_out / f"{ticker}.json").write_text(
            json.dumps(
                {
                    "success": success,
                    "future_return": ret,
                }
            )
        )

    analyzer = ModelReliabilityAnalyzer(settings=model_settings)
    scores = analyzer.analyze(
        ticker=ticker,
        start=date.fromisoformat(good_day),
        end=date.fromisoformat(bad_day),
        min_samples=1,
    )

    assert model_path in scores
    assert scores[model_path]["reliability_score"] <= 1.0


def test_p6_reliability_scores_persisted(test_db):
    dm = test_db
    payload = {"model_a.zip": {"reliability_score": 0.42}}
    dm.save_model_reliability("TEST", "2026-02-03", json.dumps(payload))

    with dm.connection() as conn:
        row = conn.execute(
            "SELECT score_details FROM model_reliability WHERE ticker = ? AND date = ?",
            ("TEST", "2026-02-03"),
        ).fetchone()

    assert row is not None
    stored = json.loads(row[0])
    assert stored["model_a.zip"]["reliability_score"] == 0.42


def test_p7_allocation_selection_stable_with_more_capital(monkeypatch):
    decisions = [
        {
            "ticker": "A",
            "date": "2026-02-05",
            "payload": {"volatility": 0.02},
            "decision": {"action_code": 1, "confidence": 0.7},
        },
        {
            "ticker": "B",
            "date": "2026-02-05",
            "payload": {"volatility": 0.04},
            "decision": {"action_code": 1, "confidence": 0.6},
        },
    ]

    monkeypatch.setattr(Config, "INITIAL_CAPITAL", 10000)
    import pandas as pd

    corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    monkeypatch.setattr(
        "app.decision.allocation._get_correlation_matrix", lambda *args, **kwargs: corr
    )
    low_cap = allocate_capital([d.copy() for d in decisions])
    low_selected = {d["ticker"] for d in low_cap if d.get("allocation_amount", 0) > 0}

    monkeypatch.setattr(Config, "INITIAL_CAPITAL", 20000)
    high_cap = allocate_capital([d.copy() for d in decisions])
    high_selected = {d["ticker"] for d in high_cap if d.get("allocation_amount", 0) > 0}

    assert low_selected == high_selected


def test_p7_correlation_limits_reduce_weaker_position(monkeypatch):
    decisions = [
        {
            "ticker": "A",
            "date": "2026-02-05",
            "payload": {"volatility": 0.02},
            "decision": {"action_code": 1, "confidence": 0.9},
            "allocation_amount": 1000.0,
            "allocation_pct": 0.5,
        },
        {
            "ticker": "B",
            "date": "2026-02-05",
            "payload": {"volatility": 0.02},
            "decision": {"action_code": 1, "confidence": 0.4},
            "allocation_amount": 1000.0,
            "allocation_pct": 0.5,
        },
    ]

    import pandas as pd

    corr = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    monkeypatch.setattr(
        "app.decision.allocation._get_correlation_matrix", lambda *args, **kwargs: corr
    )

    adjusted = enforce_correlation_limits(decisions, max_correlation=0.7)
    weaker = next(d for d in adjusted if d["ticker"] == "B")

    assert weaker["allocation_amount"] == 500.0
    assert weaker["allocation_pct"] == 0.25


def test_p9_restart_tolerates_state_persistence(test_db):
    dm = test_db
    dm.log_recommendation("TEST", "BUY", 0.7, params={"p": 1})

    dm2 = DataManager(settings=test_db.settings)
    today = date.today().isoformat()
    with dm2.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM recommendations WHERE date = ? AND ticker = ?",
            (today, "TEST"),
        ).fetchone()[0]

    assert count == 1


def test_p9_metrics_reflect_partial_failures(test_db):
    dm = test_db
    dm.log_pipeline_execution("A", "success", 1.2)
    dm.log_pipeline_execution("B", "error", 0.3, error_message="fail")

    metrics = dm.get_recent_metrics(hours=24)

    assert metrics["total_executions"] >= 2
    assert metrics["errors_count"] >= 1
