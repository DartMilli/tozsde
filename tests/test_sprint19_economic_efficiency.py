import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from app.backtesting.history_store import HistoryStore
from app.config.build_settings import build_settings
from app.core.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.core.decision.expectancy_gate import ExpectancyGate
from app.core.decision.recommender import generate_daily_recommendation_payload
from app.models.model_reliability import load_latest_reliability_scores


def _insert_completed_trade(
    dm,
    ticker: str,
    timestamp: str,
    action_code: int,
    confidence: float,
    regime: str,
    confidence_bucket: str,
    pnl_pct: float,
) -> None:
    decision_id = dm.save_history_record(
        ticker=ticker,
        model_id=None,
        model_version="test-model",
        action_code=action_code,
        label={0: "HOLD", 1: "BUY", 2: "SELL"}[action_code],
        confidence=confidence,
        wf_score=0.7,
        reliability_score=0.8,
        execution_price=100.0,
        features_hash="hash",
        d_blob=json.dumps(
            {
                "action_code": 0,
                "original_action": action_code,
                "confidence": confidence,
            }
        ),
        a_blob=json.dumps(
            {
                "regime": regime,
                "confidence_bucket": confidence_bucket,
            }
        ),
        explanation_json="{}",
        model_votes_json="[]",
        safety_overrides_json="{}",
        position_sizing_json="{}",
        decision_source="test",
        as_of_date=timestamp,
        timestamp=timestamp,
    )
    dm.save_outcome(
        decision_id=decision_id,
        ticker=ticker,
        decision_timestamp=timestamp,
        pnl_pct=pnl_pct,
        success=pnl_pct > 0,
        future_return=pnl_pct,
        exit_reason="test",
        horizon_days=5,
        outcome_json=json.dumps({"pnl_pct": pnl_pct}),
    )


def test_build_settings_enable_reliability_defaults_true(monkeypatch):
    monkeypatch.delenv("ENABLE_RELIABILITY", raising=False)
    env_file = Path(__file__).parent / ".tmp_default_settings.env"
    env_file.write_text("LOGGING_LEVEL=INFO\n")
    try:
        settings = build_settings(env_file=env_file, ensure_dirs=False)
    finally:
        env_file.unlink(missing_ok=True)
    assert settings.ENABLE_RELIABILITY is True


def test_load_latest_reliability_scores_respects_injected_settings_and_as_of_date(
    test_settings,
):
    older = Path(test_settings.MODEL_RELIABILITY_DIR) / "AAA_2026-04-01.json"
    newer = Path(test_settings.MODEL_RELIABILITY_DIR) / "AAA_2026-04-20.json"
    older.write_text(json.dumps({"m1": {"reliability_score": 0.4}}))
    newer.write_text(json.dumps({"m2": {"reliability_score": 0.9}}))

    scores = load_latest_reliability_scores(
        "AAA",
        as_of_date="2026-04-14",
        settings=test_settings,
    )

    assert scores == {"m1": 0.4}


def test_aggregate_weighted_ensemble_prefers_more_reliable_model():
    settings = MagicMock()
    settings.MAX_MODEL_AGE_DAYS = 365
    settings.MIN_WF_WEIGHT = 0.0
    settings.RANK_ALPHA = 1.0
    settings.RECENCY_HALF_LIFE_DAYS = 90
    settings.MODEL_DEMOTION_THRESHOLD = 0.0

    today = date.today()
    action, confidence, _ = aggregate_weighted_ensemble(
        votes=[1, 2],
        confidences=[0.55, 0.70],
        wf_scores=[1.0, 1.0],
        model_votes=[
            {"rank": 1, "trained_at": today, "model_path": "m1"},
            {"rank": 1, "trained_at": today, "model_path": "m2"},
        ],
        reliability_scores={"m1": 1.0, "m2": 0.1},
        settings=settings,
    )

    assert action == 1
    assert confidence > 0.85


def test_aggregate_weighted_ensemble_demotes_low_reliability_model():
    settings = MagicMock()
    settings.MAX_MODEL_AGE_DAYS = 365
    settings.MIN_WF_WEIGHT = 0.0
    settings.RANK_ALPHA = 1.0
    settings.RECENCY_HALF_LIFE_DAYS = 90
    settings.MODEL_DEMOTION_THRESHOLD = 0.3

    today = date.today()
    action, confidence, _ = aggregate_weighted_ensemble(
        votes=[1, 2],
        confidences=[0.45, 0.90],
        wf_scores=[1.0, 1.0],
        model_votes=[
            {"rank": 1, "trained_at": today, "model_path": "m1"},
            {"rank": 1, "trained_at": today, "model_path": "m2"},
        ],
        reliability_scores={"m1": 1.0, "m2": 0.2},
        settings=settings,
    )

    assert action == 1
    assert confidence > 0.95


def test_expectancy_gate_positive_edge_passes(test_db, test_settings):
    base_date = datetime(2026, 4, 14)
    for offset in range(10):
        _insert_completed_trade(
            dm=test_db,
            ticker="VOO",
            timestamp=(base_date - timedelta(days=offset + 1)).date().isoformat(),
            action_code=1,
            confidence=0.85,
            regime="TREND",
            confidence_bucket="HIGH",
            pnl_pct=0.01,
        )

    gate = ExpectancyGate(settings=test_settings, data_manager=test_db)
    result = gate.evaluate(
        ticker="VOO",
        action_code=1,
        confidence_bucket="HIGH",
        regime="TREND",
        as_of_date=base_date.date(),
    )

    assert result.gate_pass is True
    assert result.sample_count == 10
    assert result.expected_net_pnl > 0


def test_expectancy_gate_negative_edge_blocks(test_db, test_settings):
    base_date = datetime(2026, 4, 14)
    for offset in range(12):
        _insert_completed_trade(
            dm=test_db,
            ticker="VOO",
            timestamp=(base_date - timedelta(days=offset + 1)).date().isoformat(),
            action_code=1,
            confidence=0.85,
            regime="TREND",
            confidence_bucket="HIGH",
            pnl_pct=0.001,
        )

    gate = ExpectancyGate(settings=test_settings, data_manager=test_db)
    result = gate.evaluate(
        ticker="VOO",
        action_code=1,
        confidence_bucket="HIGH",
        regime="TREND",
        as_of_date=base_date.date(),
    )

    assert result.gate_pass is False
    assert result.sample_count == 12
    assert result.expected_net_pnl < 0


def test_expectancy_gate_insufficient_samples_passes(test_db, test_settings):
    base_date = datetime(2026, 4, 14)
    for offset in range(3):
        _insert_completed_trade(
            dm=test_db,
            ticker="VOO",
            timestamp=(base_date - timedelta(days=offset + 1)).date().isoformat(),
            action_code=1,
            confidence=0.50,
            regime="TREND",
            confidence_bucket="MEDIUM",
            pnl_pct=-0.01,
        )

    gate = ExpectancyGate(settings=test_settings, data_manager=test_db)
    result = gate.evaluate(
        ticker="VOO",
        action_code=1,
        confidence_bucket="MEDIUM",
        regime="TREND",
        as_of_date=base_date.date(),
    )

    assert result.gate_pass is True
    assert result.reason == "INSUFFICIENT_SAMPLES"


def test_recommender_applies_reliability_and_expectancy_gate(test_db, test_settings):
    base_date = date(2026, 4, 14)
    for offset in range(10):
        _insert_completed_trade(
            dm=test_db,
            ticker="TEST",
            timestamp=(base_date - timedelta(days=offset + 1)).isoformat(),
            action_code=1,
            confidence=0.9,
            regime="TREND",
            confidence_bucket="HIGH",
            pnl_pct=0.001,
        )

    settings = replace(test_settings, ENABLE_EXPECTANCY_GATE=True)
    (Path(settings.MODEL_RELIABILITY_DIR) / "TEST_2026-04-14.json").write_text(
        json.dumps(
            {
                "model_a": {"reliability_score": 0.95},
                "model_b": {"reliability_score": 0.1},
            }
        )
    )

    class DummyRunner:
        def run_ensemble(self, **kwargs):
            return (
                [1, 2],
                [0.60, 0.80],
                [0.80, 0.80],
                [
                    {
                        "model_path": "model_a",
                        "model_name": "model_a",
                        "rank": 1,
                        "trained_at": base_date,
                        "action_label": "BUY",
                        "confidence": 0.60,
                        "wf_score": 0.80,
                    },
                    {
                        "model_path": "model_b",
                        "model_name": "model_b",
                        "rank": 1,
                        "trained_at": base_date,
                        "action_label": "SELL",
                        "confidence": 0.80,
                        "wf_score": 0.80,
                    },
                ],
                [],
            )

    class DummySafetyRuleEngine:
        def __init__(self, *args, **kwargs):
            pass

    class DummyDecisionEngine:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ticker: str, decision: dict) -> dict:
            return decision

    df = pd.DataFrame(
        {
            "Open": [99.5, 100.5, 101.5, 102.5],
            "High": [100.5, 101.5, 102.5, 103.5],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0, 103.0],
            "Volume": [1000000, 1000000, 1000000, 1000000],
            "ADX": [30.0, 31.0, 32.0, 33.0],
        },
        index=pd.date_range("2026-04-10", periods=4, freq="D"),
    )

    payload = generate_daily_recommendation_payload(
        "TEST",
        history_store=HistoryStore(history_repo=test_db),
        model_runner=DummyRunner(),
        as_of_date=base_date,
        settings=settings,
        load_data_fn=lambda ticker, start, end: df.copy(),
        prepare_df_fn=lambda frame, ticker: frame,
        safety_rule_engine_cls=DummySafetyRuleEngine,
        decision_engine_cls=DummyDecisionEngine,
    )

    assert payload["decision"]["original_action"] == 1
    assert payload["decision"]["action_code"] == 0
    assert payload["decision"]["no_trade"] is True
    assert payload["decision"]["no_trade_reason"].startswith("EXPECTANCY_NEGATIVE")
    assert payload["reliability_scores"]["model_a"] == 0.95
    assert payload["expectancy"]["gate_pass"] is False
