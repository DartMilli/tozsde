from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from app.core.decision.market_regime_detector import RegimeInfo
from app.core.decision.position_sizer import apply_position_sizing
from app.core.decision.regime_policy import get_regime_policy
from app.core.decision.recommender import generate_daily_recommendation_payload
from app.core.decision.safety_rules import SafetyRuleEngine


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
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


class DummyRunner:
    def __init__(self, confidence: float = 0.6):
        self.confidence = confidence

    def run_ensemble(self, **kwargs):
        return (
            [1],
            [self.confidence],
            [0.8],
            [
                {
                    "model_path": "model_a",
                    "model_name": "model_a",
                    "rank": 1,
                    "trained_at": date(2026, 4, 14),
                    "action_label": "BUY",
                    "confidence": self.confidence,
                    "wf_score": 0.8,
                }
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


def test_get_regime_policy_applies_overrides(test_settings):
    settings = replace(
        test_settings,
        REGIME_POLICY_OVERRIDES={"BULL": {"confidence_floor": 0.18}},
    )

    policy = get_regime_policy("BULL", settings=settings)

    assert policy.confidence_floor == 0.18
    assert policy.allow_new_buys is True


def test_apply_position_sizing_uses_regime_max_position_pct(test_settings):
    item = {
        "decision": {
            "action_code": 1,
            "confidence": 0.9,
            "wf_score": 0.9,
            "regime_policy": {"max_position_pct": 0.08},
        },
        "allocation_amount": 5000.0,
        "payload": {},
    }

    sized = apply_position_sizing(item, equity=10000.0, settings=test_settings)

    assert sized["allocation_amount"] == 800.0
    assert sized["position_sizing"]["regime_max_position_pct"] == 0.08


def test_safety_rule_engine_strict_cooldown_extends_lookback(test_settings):
    trade_date = (date(2026, 4, 14) - timedelta(days=6)).isoformat()

    class FakeHistory:
        def load_range(self, ticker, start, end):
            if start <= date.fromisoformat(trade_date) <= end:
                return [{"action_code": 1}]
            return []

        def load_recent_outcomes(self, ticker, n=3):
            return []

    engine = SafetyRuleEngine(
        FakeHistory(), settings=replace(test_settings, COOLDOWN_MAX_TRADES=1)
    )

    assert engine._in_cooldown("TEST", date(2026, 4, 14), strictness="NORMAL") is False
    assert engine._in_cooldown("TEST", date(2026, 4, 14), strictness="STRICT") is True


def test_recommender_regime_policy_blocks_new_buys_in_bear(monkeypatch, test_settings):
    settings = replace(
        test_settings, ENABLE_REGIME_POLICY=True, ENABLE_EXPECTANCY_GATE=False
    )

    class DummyDetector:
        def __init__(self, settings=None):
            pass

        def detect_regime(self, ticker="SPY"):
            return RegimeInfo(
                regime_type="BEAR",
                confidence=0.9,
                volatility=0.2,
                trend_strength=-0.8,
                detected_at="2026-04-14T00:00:00",
                characteristics={},
            )

    monkeypatch.setattr(
        "app.core.decision.market_regime_detector.MarketRegimeDetector",
        DummyDetector,
    )

    history_store = type(
        "H",
        (),
        {
            "save_decision": lambda *a, **k: None,
            "load_range": lambda *a, **k: [],
            "load_recent_outcomes": lambda *a, **k: [],
        },
    )()

    payload = generate_daily_recommendation_payload(
        "TEST",
        history_store=history_store,
        model_runner=DummyRunner(confidence=0.9),
        as_of_date=date(2026, 4, 14),
        settings=settings,
        load_data_fn=lambda ticker, start, end: _sample_df().copy(),
        prepare_df_fn=lambda frame, ticker: frame,
        safety_rule_engine_cls=DummySafetyRuleEngine,
        decision_engine_cls=DummyDecisionEngine,
    )

    assert payload["regime"] == "BEAR"
    assert payload["decision"]["no_trade"] is True
    assert payload["decision"]["no_trade_reason"] == "REGIME_BLOCK_NEW_BUYS"


def test_recommender_regime_policy_lowers_confidence_floor_in_bull(
    monkeypatch, test_settings
):
    settings = replace(
        test_settings, ENABLE_REGIME_POLICY=True, ENABLE_EXPECTANCY_GATE=False
    )
    monkeypatch.setattr(
        "app.core.decision.recommender.scale_confidence_by_volatility",
        lambda confidence, volatility: confidence,
    )
    monkeypatch.setattr(
        "app.core.decision.recommender._quality_label",
        lambda eq_float, settings: "WEAK",
    )

    class DummyDetector:
        def __init__(self, settings=None):
            pass

        def detect_regime(self, ticker="SPY"):
            return RegimeInfo(
                regime_type="BULL",
                confidence=0.9,
                volatility=0.15,
                trend_strength=0.8,
                detected_at="2026-04-14T00:00:00",
                characteristics={},
            )

    monkeypatch.setattr(
        "app.core.decision.market_regime_detector.MarketRegimeDetector",
        DummyDetector,
    )

    payload = generate_daily_recommendation_payload(
        "TEST",
        history_store=type(
            "H",
            (),
            {
                "save_decision": lambda *a, **k: None,
                "load_range": lambda *a, **k: [],
                "load_recent_outcomes": lambda *a, **k: [],
            },
        )(),
        model_runner=DummyRunner(confidence=0.22),
        as_of_date=date(2026, 4, 14),
        settings=settings,
        load_data_fn=lambda ticker, start, end: _sample_df().copy(),
        prepare_df_fn=lambda frame, ticker: frame,
        safety_rule_engine_cls=DummySafetyRuleEngine,
        decision_engine_cls=DummyDecisionEngine,
    )

    assert payload["regime"] == "BULL"
    assert payload["decision"]["no_trade"] is False
    assert payload["decision"]["action_code"] == 1
