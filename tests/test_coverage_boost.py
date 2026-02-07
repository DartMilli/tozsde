import json
from datetime import date
import numpy as np
import pandas as pd
import pytest

from app.config.config import Config
from app.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.decision.ensemble_quality import bucket_ensemble_quality
from app.decision.decision_reliability import assess_decision_reliability
from app.decision.decision_event import build_decision_event
from app.decision.volatility import (
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)
from app.decision.safety_rules import SafetyRuleEngine
from app.notifications.mailer import send_email
from app.notifications.alerter import ErrorAlerter
from app.models.model_trainer import TradingEnv, train_rl_agent, backtest_rl_model
from app.models.rl_inference import RLModelEnsembleRunner
from app.decision.recommender import generate_daily_recommendation_payload
from app.data_access.data_manager import DataManager


def _sample_env_df(rows=20):
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    base = np.linspace(100, 110, rows)
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": base,
            "Volume": 1000,
            "SMA": base,
            "BB_upper": base + 2,
            "BB_lower": base - 2,
            "RSI": 50,
            "ADX": 25,
            "MACD": 0.1,
            "MACD_SIGNAL": 0.05,
            "ATR": 1.0,
        },
        index=dates,
    )
    return df


def test_trading_env_reward_strategies():
    df = _sample_env_df()
    strategies = TradingEnv.get_reward_strategies()

    for strategy in strategies:
        env = TradingEnv(df, reward_strategy=strategy)
        obs, info = env.reset()
        assert len(obs) == env.observation_space.shape[0]
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, (int, float))


def test_train_rl_agent_minimal(monkeypatch, tmp_path):
    df = _sample_env_df()

    class DummyModel:
        def __init__(self, **kwargs):
            pass

        def learn(self, total_timesteps, reset_num_timesteps=False):
            return None

        def save(self, path):
            Path = __import__("pathlib").Path
            Path(path).write_text("ok")

    class DummyDM:
        def register_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr("app.models.model_trainer.load_data", lambda *a, **k: df)
    monkeypatch.setattr("app.models.model_trainer.prepare_df", lambda *a, **k: df)
    monkeypatch.setattr("app.models.model_trainer.DQN", DummyModel)
    monkeypatch.setattr("app.models.model_trainer.PPO", DummyModel)
    monkeypatch.setattr("app.models.model_trainer.DataManager", lambda: DummyDM())
    monkeypatch.setattr("app.models.model_trainer.DummyVecEnv", lambda fns: fns[0]())

    model_path = tmp_path / "model.zip"
    train_rl_agent(
        "TEST",
        model_type="DQN",
        timesteps=0,
        model_path=str(model_path),
        reward_strategy="portfolio_value",
    )

    assert model_path.exists()


def test_backtest_rl_model_minimal(monkeypatch):
    df = _sample_env_df()

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return 0, None

    class DummyVec:
        def __init__(self, fns):
            self.env = fns[0]()

        def reset(self):
            obs, _ = self.env.reset()
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            return obs, [reward], [terminated], [info]

    monkeypatch.setattr("app.models.model_trainer.load_data", lambda *a, **k: df)
    monkeypatch.setattr("app.models.model_trainer.prepare_df", lambda *a, **k: df)
    monkeypatch.setattr(
        "app.models.model_trainer.PPO", type("P", (), {"load": lambda p: DummyModel()})
    )
    monkeypatch.setattr(
        "app.models.model_trainer.DQN", type("D", (), {"load": lambda p: DummyModel()})
    )
    monkeypatch.setattr("app.models.model_trainer.DummyVecEnv", DummyVec)

    result = backtest_rl_model(
        "TEST", "ppo_model.zip", reward_strategy="portfolio_value"
    )
    assert "composite_score" in result


def test_ensemble_aggregation_and_quality():
    votes = [1, 1, 2]
    confidences = [0.8, 0.6, 0.4]
    wf_scores = [0.9, 0.8, 0.7]
    model_votes = [
        {"rank": 1, "trained_at": date.today()},
        {"rank": 2, "trained_at": date.today()},
        {"rank": 3, "trained_at": date.today()},
    ]

    action, conf, quality = aggregate_weighted_ensemble(
        votes, confidences, wf_scores, model_votes
    )
    assert action in {0, 1, 2}
    bucket = bucket_ensemble_quality(quality)
    assert bucket.value in {"STRONG", "NORMAL", "WEAK", "CHAOTIC"}


def test_decision_reliability_and_event():
    result = assess_decision_reliability(0.8, 0.8)
    assert result.trade_allowed is True

    payload = {"ticker": "TEST", "timestamp": "2026-02-05", "model_votes": [1, 2]}
    decision = {"action": "BUY", "confidence": 0.8, "no_trade": False}
    audit = {
        "decision_level": "STRONG",
        "trade_allowed": True,
        "confidence_bucket": "HIGH",
        "quality_score": 0.9,
        "ensemble_quality": "STRONG",
    }

    event = build_decision_event(payload, decision, audit)
    assert event["model_count"] == 2


def test_volatility_scaling():
    df = _sample_env_df()
    vol = compute_normalized_volatility(df)
    scaled = scale_confidence_by_volatility(0.8, vol, soft_cap=0.0001)
    assert 0.0 <= scaled <= 0.8


def test_safety_rules_overrides(monkeypatch):
    class DummyHistory:
        def load_range(self, *args, **kwargs):
            return [{"action_code": 1}, {"action_code": 1}]

        def load_recent_outcomes(self, *args, **kwargs):
            return [{"success": False}]

    engine = SafetyRuleEngine(DummyHistory())
    monkeypatch.setattr(
        "app.decision.safety_rules.get_market_volatility_index", lambda: 50
    )
    monkeypatch.setattr(SafetyRuleEngine, "_is_bear_market", lambda *a, **k: False)

    decision = {
        "action_code": 1,
        "action": "BUY",
        "strength": "NORMAL",
        "ensemble_quality": "CHAOTIC",
    }

    result = engine.apply("TEST", decision, date.today())
    assert result["no_trade"] is True


def test_mailer_and_alerter(monkeypatch):
    class DummySMTP:
        def __init__(self, host, port):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            return None

        def login(self, user, password):
            return None

        def send_message(self, msg):
            return None

    monkeypatch.setattr("smtplib.SMTP", DummySMTP)
    monkeypatch.setenv("EMAIL_HOST", "localhost")
    monkeypatch.setenv("EMAIL_USER", "user")
    monkeypatch.setenv("EMAIL_PASSWORD", "pass")

    assert send_email("sub", "body", "to@test") is True

    monkeypatch.setattr("app.notifications.alerter.send_email", lambda *a, **k: True)
    assert ErrorAlerter.alert("DB_CONNECTION_FAILED", "boom") is True


def test_rl_inference_run_ensemble(monkeypatch):
    runner = RLModelEnsembleRunner(model_dir=".", env_class=lambda df: df)

    class DummyVec:
        def __init__(self, fns):
            self.env = fns[0]()

    monkeypatch.setattr("app.models.rl_inference.DummyVecEnv", DummyVec)

    monkeypatch.setattr(
        RLModelEnsembleRunner,
        "_find_latest_model",
        lambda *args, **kwargs: "model.zip",
    )
    monkeypatch.setattr(
        RLModelEnsembleRunner,
        "_detect_model_type",
        lambda *args, **kwargs: (object, "PPO"),
    )
    monkeypatch.setattr(
        RLModelEnsembleRunner,
        "_run_single_model",
        lambda *args, **kwargs: (1, 0.8, 0.7, {"rank": 1}),
    )

    votes, confidences, wf_scores, model_votes = runner.run_ensemble(
        df=pd.DataFrame({"Close": [1, 2, 3]}),
        ticker="TEST",
        top_n=1,
        debug=False,
    )

    assert votes == [1]
    assert model_votes[0]["action"] == 1


def test_recommender_payload(monkeypatch):
    df = _sample_env_df()

    monkeypatch.setattr("app.decision.recommender.load_data", lambda *a, **k: df)
    monkeypatch.setattr("app.decision.recommender.prepare_df", lambda *a, **k: df)
    monkeypatch.setattr(
        "app.decision.recommender.RLModelEnsembleRunner.run_ensemble",
        lambda *a, **k: (
            [1],
            [0.8],
            [0.7],
            [{"action": 1, "action_label": "BUY", "confidence": 0.8, "wf_score": 0.7}],
            [],
        ),
    )

    monkeypatch.setattr(
        "app.decision.safety_rules.get_market_volatility_index", lambda: None
    )
    monkeypatch.setattr(
        "app.decision.safety_rules.SafetyRuleEngine._is_bear_market",
        lambda *a, **k: False,
    )

    history_stub = type(
        "H",
        (),
        {
            "save_decision": lambda *a, **k: None,
            "load_range": lambda *a, **k: [],
            "load_recent_outcomes": lambda *a, **k: [],
        },
    )()

    payload = generate_daily_recommendation_payload("TEST", history_store=history_stub)
    assert "decision" in payload


def test_reliability_saved_in_db(test_db):
    dm = test_db
    dm.save_model_reliability(
        "TEST", "2026-02-05", json.dumps({"m": {"reliability_score": 0.5}})
    )
    with dm.connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM model_reliability WHERE ticker = ?",
            ("TEST",),
        ).fetchone()[0]
    assert row == 1
