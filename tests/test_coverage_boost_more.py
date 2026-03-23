import json
from types import SimpleNamespace
from dataclasses import replace
from datetime import date, datetime, timedelta

import pandas as pd
import pytest


def test_run_daily_dry_run_success(monkeypatch, test_settings):
    import main

    calls = []

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            calls.append((dry_run, ticker))
            return {"status": "ok"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=True, ticker="AAA")
    assert result == {"status": "ok"}
    assert calls == [(True, "AAA")]


def test_run_daily_error_alert(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    with pytest.raises(RuntimeError):
        main.run_daily(dry_run=False, ticker="AAA")


def test_run_daily_no_candidates(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            return {"status": "no_candidates"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=True, ticker="AAA")
    assert result == {"status": "no_candidates"}


def test_run_daily_email_failure(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            return {"status": "email_failed"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=False, ticker="AAA")
    assert result == {"status": "email_failed"}


def test_run_daily_build_recommendation(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            return {"status": "built"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=True)
    assert result == {"status": "built"}


def test_run_daily_email_success(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            return {"status": "email_sent"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=False, ticker="AAA")
    assert result == {"status": "email_sent"}


def test_run_daily_no_email_lines(monkeypatch):
    import main

    class FakeDailyPipeline:
        def run(self, dry_run=False, ticker=None):
            return {"status": "no_email_lines"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"daily_pipeline": FakeDailyPipeline()})(),
    )

    result = main.run_daily(dry_run=False, ticker="AAA")
    assert result == {"status": "no_email_lines"}


def test_run_weekly_saves_scores(monkeypatch, test_settings):
    import main

    saved = []

    class FakeWeeklyUseCase:
        def run(self, dry_run=False):
            saved.append(dry_run)
            return {"status": "ok"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"weekly_reliability": FakeWeeklyUseCase()})(),
    )

    main.run_weekly(dry_run=False)

    assert saved


def test_run_weekly_errors(monkeypatch, test_settings):
    import main

    class FakeWeeklyUseCase:
        def run(self, dry_run=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"weekly_reliability": FakeWeeklyUseCase()})(),
    )

    with pytest.raises(RuntimeError):
        main.run_weekly(dry_run=False)


def test_run_weekly_outer_error(monkeypatch):
    import main

    class FakeWeeklyUseCase:
        def run(self, dry_run=False):
            raise RuntimeError("init fail")

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"weekly_reliability": FakeWeeklyUseCase()})(),
    )
    with pytest.raises(RuntimeError):
        main.run_weekly(dry_run=False)


def test_run_monthly_with_rl(monkeypatch, test_settings):
    import main

    calls = []

    class FakeMonthlyUseCase:
        def run(self, dry_run=False):
            calls.append(dry_run)
            return {"status": "ok"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"monthly_retraining": FakeMonthlyUseCase()})(),
    )

    main.run_monthly(dry_run=False)

    assert calls


def test_run_monthly_without_rl(monkeypatch, test_settings):
    import main

    class FakeMonthlyUseCase:
        def run(self, dry_run=False):
            return {"status": "ok"}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"monthly_retraining": FakeMonthlyUseCase()})(),
    )

    main.run_monthly(dry_run=False)


def test_run_monthly_error(monkeypatch, test_settings):
    import main

    class FakeMonthlyUseCase:
        def run(self, dry_run=False):
            raise RuntimeError("fail")

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type("C", (), {"monthly_retraining": FakeMonthlyUseCase()})(),
    )

    with pytest.raises(RuntimeError):
        main.run_monthly(dry_run=False)


def test_manual_walk_forward_and_rl(monkeypatch):
    import main

    class FakeWalkUseCase:
        def run(self, ticker=None):
            return {"status": "ok", "ticker": ticker}

    class FakeTrainUseCase:
        def run(self, ticker=None, **kwargs):
            return {"status": "ok", "ticker": ticker}

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type(
            "C", (), {"walk_forward": FakeWalkUseCase(), "train_rl": FakeTrainUseCase()}
        )(),
    )

    main.run_walk_forward_manual("AAA", dry_run=False)
    main.run_train_rl_manual("AAA", dry_run=False)


def test_manual_failures(monkeypatch):
    import main

    class FakeWalkUseCase:
        def run(self, ticker=None):
            raise RuntimeError("fail")

    class FakeTrainUseCase:
        def run(self, ticker=None, **kwargs):
            raise RuntimeError("fail")

    monkeypatch.setattr(
        main,
        "_APP_CONTAINER",
        type(
            "C", (), {"walk_forward": FakeWalkUseCase(), "train_rl": FakeTrainUseCase()}
        )(),
    )

    with pytest.raises(RuntimeError):
        main.run_walk_forward_manual("AAA", dry_run=False)
    with pytest.raises(RuntimeError):
        main.run_train_rl_manual("AAA", dry_run=False)


def test_main_dispatch(monkeypatch):
    import main

    calls = []

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="daily", dry_run=True, ticker="AAA", loglevel="INFO"
        ),
    )
    monkeypatch.setattr(main, "run_daily", lambda **kwargs: calls.append(kwargs))

    main.main()

    assert calls


def test_main_keyboard_interrupt(monkeypatch):
    import main

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="daily", dry_run=True, ticker="AAA", loglevel="INFO"
        ),
    )
    monkeypatch.setattr(
        main, "run_daily", lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    )

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0


def test_parse_arguments(monkeypatch):
    import main
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--loglevel", "DEBUG", "daily", "--dry-run", "--ticker", "AAA"],
    )

    args = main.parse_arguments()
    assert args.command == "daily"
    assert args.dry_run is True


def test_main_dispatch_weekly_monthly(monkeypatch):
    import main

    calls = []

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="weekly", dry_run=True, loglevel="INFO", ticker=None
        ),
    )
    monkeypatch.setattr(main, "run_weekly", lambda **kwargs: calls.append("weekly"))

    main.main()

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="monthly", dry_run=True, loglevel="INFO", ticker=None
        ),
    )
    monkeypatch.setattr(main, "run_monthly", lambda **kwargs: calls.append("monthly"))

    main.main()

    assert "weekly" in calls and "monthly" in calls


def test_main_dispatch_manual(monkeypatch):
    import main

    calls = []

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="walk-forward", dry_run=True, loglevel="INFO", ticker="AAA"
        ),
    )
    monkeypatch.setattr(
        main, "run_walk_forward_manual", lambda **kwargs: calls.append("wf")
    )

    main.main()

    monkeypatch.setattr(
        main,
        "parse_arguments",
        lambda: SimpleNamespace(
            command="train-rl", dry_run=True, loglevel="INFO", ticker="AAA"
        ),
    )
    monkeypatch.setattr(
        main, "run_train_rl_manual", lambda **kwargs: calls.append("rl")
    )

    main.main()

    assert "wf" in calls and "rl" in calls


def test_alerter_paths(monkeypatch):
    from app.notifications import alerter

    sent = []

    monkeypatch.setattr(
        alerter, "send_email", lambda *args, **kwargs: sent.append(args)
    )

    assert alerter.ErrorAlerter.alert(
        "DB_CONNECTION_FAILED",
        "db down",
        details={"x": 1},
        severity="auto",
    )
    assert alerter.ErrorAlerter.alert(
        "HIGH_DRIFT",
        "drift",
        details={"x": 2},
        severity="warning",
    )
    assert not alerter.ErrorAlerter.alert(
        "LOW_CONFIDENCE",
        "low",
        details={"x": 3},
        severity="auto",
    )

    assert sent


def test_alerter_decorator(monkeypatch):
    from app.notifications.alerter import catch_and_alert, ErrorAlerter

    seen = []

    monkeypatch.setattr(ErrorAlerter, "alert", lambda **kwargs: seen.append(kwargs))

    @catch_and_alert
    def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        boom()

    assert seen


def test_alerter_remediation_steps():
    from app.notifications.alerter import ErrorAlerter

    assert (
        "database"
        in ErrorAlerter._get_remediation_steps("DB_CONNECTION_FAILED").lower()
    )
    assert "review" in ErrorAlerter._get_remediation_steps("UNKNOWN").lower()


def test_safety_rules_apply(monkeypatch):
    from app.decision.safety_rules import SafetyRuleEngine

    class FakeHistory:
        def load_range(self, *args, **kwargs):
            return [{"action_code": 1}]

        def load_recent_outcomes(self, *args, **kwargs):
            return [{"success": False}]

    monkeypatch.setattr(
        "app.decision.safety_rules.get_market_volatility_index", lambda: 50.0
    )

    class FakeDM:
        def get_market_regime_is_bear(self, **kwargs):
            return True

    monkeypatch.setattr("app.decision.safety_rules.DataManager", FakeDM)

    engine = SafetyRuleEngine(FakeHistory())

    decision = {
        "action_code": 1,
        "action": "BUY",
        "strength": "WEAK",
        "ensemble_quality": "CHAOTIC",
    }

    out = engine.apply("AAA", decision, date.today())

    assert out["no_trade"]
    assert out["safety_override"]


def test_safety_rules_bear_warning(monkeypatch):
    from app.decision.safety_rules import SafetyRuleEngine

    class FakeHistory:
        def load_range(self, *args, **kwargs):
            return []

        def load_recent_outcomes(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        "app.decision.safety_rules.get_market_volatility_index", lambda: None
    )

    class FakeDM:
        def get_market_regime_is_bear(self, **kwargs):
            return True

    monkeypatch.setattr("app.decision.safety_rules.DataManager", FakeDM)

    engine = SafetyRuleEngine(FakeHistory())

    decision = {
        "action_code": 1,
        "action": "BUY",
        "strength": "STRONG",
        "ensemble_quality": "NORMAL",
    }

    out = engine.apply("AAA", decision, date.today())

    assert "warnings" in out


def test_decision_reliability_and_ensemble_quality(test_settings):
    from app.decision.decision_reliability import assess_decision_reliability
    from app.decision.ensemble_quality import (
        bucket_ensemble_quality,
        EnsembleQualityBucket,
    )

    low = assess_decision_reliability(
        test_settings.CONFIDENCE_NO_TRADE_THRESHOLD - 0.01,
        0.1,
        settings=test_settings,
    )
    assert not low.trade_allowed

    strong = assess_decision_reliability(
        test_settings.STRONG_CONFIDENCE_THRESHOLD + 0.01,
        test_settings.STRONG_WF_THRESHOLD + 0.01,
        settings=test_settings,
    )
    assert strong.confidence_level == "STRONG"

    weak = assess_decision_reliability(
        test_settings.WEAK_CONFIDENCE_THRESHOLD - 0.01,
        0.1,
        settings=test_settings,
    )
    assert weak.confidence_level == "WEAK"

    normal = assess_decision_reliability(
        test_settings.WEAK_CONFIDENCE_THRESHOLD + 0.01,
        0.1,
        settings=test_settings,
    )
    assert normal.confidence_level == "NORMAL"

    assert (
        bucket_ensemble_quality(
            test_settings.ENSEMBLE_QUALITY_THRESHOLDS["STRONG"],
            settings=test_settings,
        )
        == EnsembleQualityBucket.STRONG
    )
    assert (
        bucket_ensemble_quality(0.0, settings=test_settings)
        == EnsembleQualityBucket.CHAOTIC
    )


def test_audit_builder_paths(monkeypatch, test_settings):
    from app.reporting.audit_builder import (
        decision_reliability_level,
        compute_consistency_flags,
        confidence_bucket,
        build_audit_metadata,
    )
    from dataclasses import replace
    from app.reporting import audit_builder

    assert decision_reliability_level(None, None) == ("UNKNOWN", False)
    assert confidence_bucket(0.2) == "VERY_LOW"

    payload = {"model_votes": [{"action": 1}, {"action": 1}, {"action": 0}]}
    decision = {
        "action_code": 1,
        "confidence": 0.8,
        "wf_score": 0.7,
        "ensemble_quality": 0.8,
        "quality_score": 0.9,
    }

    flags = compute_consistency_flags(payload, decision)
    assert flags["majority_action"] == 1

    audit_builder.set_settings(replace(test_settings, ENABLE_DRIFT_DETECTION=False))
    audit = build_audit_metadata(payload, decision)
    assert audit["decision_level"] in {"STRONG", "NORMAL", "WEAK", "NO_TRADE"}


def test_model_reliability_scores(tmp_path, monkeypatch, test_settings):
    from app.models import model_reliability
    from app.models import set_settings as set_model_settings

    model_settings = replace(test_settings, MODEL_RELIABILITY_DIR=tmp_path)
    set_model_settings(model_settings)
    analyzer = model_reliability.ModelReliabilityAnalyzer(settings=model_settings)

    rows = [
        {"success": 1, "future_return": 0.01, "confidence": 0.8},
        {"success": 0, "future_return": -0.01, "confidence": 0.2},
        {"success": 1, "future_return": 0.02, "confidence": 0.9},
        {"success": 1, "future_return": 0.0, "confidence": 0.7},
        {"success": 1, "future_return": 0.03, "confidence": 0.6},
    ]

    scores = analyzer._compute_scores({"model": rows}, min_samples=5)

    assert "model" in scores
    assert 0 <= scores["model"]["reliability_score"] <= 1

    set_model_settings(model_settings)

    (tmp_path / "AAA_2024-01-01.json").write_text(
        json.dumps({"m1": {"reliability_score": 0.4}})
    )
    (tmp_path / "AAA_2024-02-01.json").write_text(
        json.dumps({"m2": {"reliability_score": 0.8}})
    )

    latest = model_reliability.load_latest_reliability_scores("AAA")
    assert latest["m2"] == 0.8

    model_reliability.save_reliability_scores(
        "AAA",
        "2024-02-01",
        {"x": {}},
        settings=model_settings,
    )
    out_file = tmp_path / "AAA_2024-02-01.json"
    assert out_file.exists()

    set_model_settings(replace(test_settings, MODEL_RELIABILITY_DIR=tmp_path / "empty"))
    assert model_reliability.load_latest_reliability_scores("ZZZ") == {}


def test_analyzer_compute_signals(tmp_path, monkeypatch, test_settings):
    from app.analysis import analyzer, set_settings as set_analysis_settings

    set_analysis_settings(
        replace(test_settings, PARAMS_FILE_PATH=tmp_path / "missing.json")
    )

    params = analyzer.get_params("AAA")
    assert params

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(30),
            "High": range(1, 31),
            "Low": range(0, 30),
            "Close": range(1, 31),
        },
        index=dates,
    )

    signals, indicators = analyzer.compute_signals(df, "AAA", params)
    assert isinstance(signals, list)
    assert indicators

    series, _ = analyzer.compute_signals(df, "AAA", params, return_series=True)
    assert len(series) == len(df)

    empty_signals, empty_indicators = analyzer.compute_signals(
        df.iloc[0:0], "AAA", params
    )
    assert empty_signals == []
    assert empty_indicators["SMA"] is None


def test_analyzer_get_params_file(tmp_path, monkeypatch, test_settings):
    from app.analysis import analyzer, set_settings as set_analysis_settings

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"AAA": {"bbands_stddev": 2}}))

    set_analysis_settings(replace(test_settings, PARAMS_FILE_PATH=params_path))
    params = analyzer.get_params("AAA")
    assert isinstance(params["bbands_stddev"], float)


def test_analyzer_get_params_invalid_json(tmp_path, monkeypatch, test_settings):
    from app.analysis import analyzer, set_settings as set_analysis_settings

    params_path = tmp_path / "params.json"
    params_path.write_text("not-json")
    set_analysis_settings(replace(test_settings, PARAMS_FILE_PATH=params_path))

    params = analyzer.get_params("AAA")
    assert params


def test_analyzer_get_params_missing_ticker(tmp_path, monkeypatch, test_settings):
    from app.analysis import analyzer, set_settings as set_analysis_settings

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"BBB": {"bbands_stddev": 2}}))
    set_analysis_settings(replace(test_settings, PARAMS_FILE_PATH=params_path))

    params = analyzer.get_params("AAA")
    assert params


def test_analyzer_all_signals(monkeypatch):
    from app.analysis import analyzer
    import numpy as np

    monkeypatch.setattr(
        analyzer.ta, "sma", lambda *args, **kwargs: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "ema", lambda *args, **kwargs: np.array([0.5, 1.5])
    )
    monkeypatch.setattr(
        analyzer.ta, "rsi", lambda *args, **kwargs: np.array([20.0, 40.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "macd",
        lambda *args, **kwargs: (np.array([0.0, 1.0]), np.array([1.0, 0.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "bbands",
        lambda *args, **kwargs: (np.array([10.0]), np.array([5.0]), np.array([2.0])),
    )
    monkeypatch.setattr(
        analyzer.ta, "atr", lambda *args, **kwargs: np.array([1.0, 2.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "adx",
        lambda *args, **kwargs: (np.array([30.0]), np.array([1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "stoch",
        lambda *args, **kwargs: (np.array([0.0, 2.0]), np.array([1.0, 1.0])),
    )

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [1.0, 1.0],
        },
        index=dates,
    )

    params = analyzer.get_default_params()
    signals, _ = analyzer.compute_signals(df, "AAA", params)

    assert signals


def test_analyzer_sell_signals(monkeypatch):
    from app.analysis import analyzer
    import numpy as np

    monkeypatch.setattr(
        analyzer.ta, "sma", lambda *args, **kwargs: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "ema", lambda *args, **kwargs: np.array([1.5, 0.5])
    )
    monkeypatch.setattr(
        analyzer.ta, "rsi", lambda *args, **kwargs: np.array([80.0, 60.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "macd",
        lambda *args, **kwargs: (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "bbands",
        lambda *args, **kwargs: (np.array([1.0]), np.array([0.5]), np.array([0.2])),
    )
    monkeypatch.setattr(
        analyzer.ta, "atr", lambda *args, **kwargs: np.array([1.0, 2.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "adx",
        lambda *args, **kwargs: (np.array([30.0]), np.array([1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "stoch",
        lambda *args, **kwargs: (np.array([2.0, 0.0]), np.array([1.0, 1.0])),
    )

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [2.0, 2.0],
        },
        index=dates,
    )

    params = analyzer.get_default_params()
    signals, _ = analyzer.compute_signals(df, "AAA", params)

    assert signals


def test_analyzer_return_series_signals(monkeypatch):
    from app.analysis import analyzer
    import numpy as np

    monkeypatch.setattr(analyzer, "get_params", lambda t: analyzer.get_default_params())
    monkeypatch.setattr(
        analyzer.ta, "sma", lambda *args, **kwargs: np.array([2.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "ema", lambda *args, **kwargs: np.array([1.0, 2.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "rsi", lambda *args, **kwargs: np.array([50.0, 55.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "macd",
        lambda *args, **kwargs: (np.array([0.0, 0.1]), np.array([0.0, 0.05])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "bbands",
        lambda *args, **kwargs: (np.array([2.0]), np.array([1.0]), np.array([0.5])),
    )
    monkeypatch.setattr(
        analyzer.ta, "atr", lambda *args, **kwargs: np.array([1.0, 1.1])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "adx",
        lambda *args, **kwargs: (np.array([20.0]), np.array([1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "stoch",
        lambda *args, **kwargs: (np.array([1.0, 2.0]), np.array([1.0, 1.5])),
    )

    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0, 1.0],
            "High": [2.0, 2.0, 2.0],
            "Low": [0.5, 0.5, 0.5],
            "Close": [1.0, 1.0, 1.0],
        },
        index=dates,
    )

    series, _ = analyzer.compute_signals(df, "AAA", None, return_series=True)
    assert "BUY" in series


def test_analyzer_no_indicators(monkeypatch):
    from app.analysis import analyzer
    import numpy as np

    monkeypatch.setattr(
        analyzer.ta, "sma", lambda *args, **kwargs: np.array([np.nan, np.nan])
    )
    monkeypatch.setattr(
        analyzer.ta, "ema", lambda *args, **kwargs: np.array([np.nan, np.nan])
    )
    monkeypatch.setattr(
        analyzer.ta, "rsi", lambda *args, **kwargs: np.array([np.nan, np.nan])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "macd",
        lambda *args, **kwargs: (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        ),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "bbands",
        lambda *args, **kwargs: (
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
        ),
    )
    monkeypatch.setattr(
        analyzer.ta, "atr", lambda *args, **kwargs: np.array([np.nan, np.nan])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "adx",
        lambda *args, **kwargs: (
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
        ),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "stoch",
        lambda *args, **kwargs: (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        ),
    )

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [1.0, 1.0],
        },
        index=dates,
    )

    signals, _ = analyzer.compute_signals(df, "AAA", analyzer.get_default_params())
    assert signals == []


def test_analyzer_params_none(monkeypatch):
    from app.analysis import analyzer
    import numpy as np

    monkeypatch.setattr(analyzer, "get_params", lambda t: analyzer.get_default_params())
    monkeypatch.setattr(
        analyzer.ta, "sma", lambda *args, **kwargs: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "ema", lambda *args, **kwargs: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta, "rsi", lambda *args, **kwargs: np.array([50.0, 50.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "macd",
        lambda *args, **kwargs: (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "bbands",
        lambda *args, **kwargs: (np.array([1.0]), np.array([1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta, "atr", lambda *args, **kwargs: np.array([1.0, 1.0])
    )
    monkeypatch.setattr(
        analyzer.ta,
        "adx",
        lambda *args, **kwargs: (np.array([10.0]), np.array([1.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        analyzer.ta,
        "stoch",
        lambda *args, **kwargs: (np.array([1.0, 1.0]), np.array([1.0, 1.0])),
    )

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Open": [1.0, 1.0],
            "High": [2.0, 2.0],
            "Low": [0.5, 0.5],
            "Close": [1.0, 1.0],
        },
        index=dates,
    )

    signals, _ = analyzer.compute_signals(df, "AAA", None)
    assert isinstance(signals, list)


def test_analyzer_main_block(monkeypatch, tmp_path):
    import runpy
    import types
    import sys

    dummy_loader = types.ModuleType("app.data_access.data_loader")

    def fake_load_data(*args, **kwargs):
        return pd.DataFrame()

    def fake_get_supported_ticker_list():
        return ["AAA"]

    dummy_loader.load_data = fake_load_data
    dummy_loader.get_supported_ticker_list = fake_get_supported_ticker_list

    monkeypatch.setitem(sys.modules, "app.data_access.data_loader", dummy_loader)
    sys.modules.pop("app.analysis.analyzer", None)

    runpy.run_module("app.analysis.analyzer", run_name="__main__")


def test_data_loader_load_data(monkeypatch):
    from app.data_access import data_loader

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(5),
            "High": range(1, 6),
            "Low": range(0, 5),
            "Close": range(1, 6),
            "Volume": range(10, 15),
        },
        index=dates,
    )

    class FakeDM:
        def __init__(self):
            self.calls = 0

        def load_ohlcv(self, ticker, start_date=None):
            self.calls += 1
            return df if self.calls > 1 else pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)

    called = []
    monkeypatch.setattr(
        data_loader,
        "download_and_save_data",
        lambda *args, **kwargs: called.append(1) or True,
    )

    out = data_loader.load_data("AAA", start="2024-01-01", end="2024-01-10")

    assert not out.empty
    assert called


def test_data_loader_load_data_no_download(monkeypatch):
    from app.data_access import data_loader

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(5),
            "High": range(1, 6),
            "Low": range(0, 5),
            "Close": range(1, 6),
            "Volume": range(10, 15),
        },
        index=dates,
    )

    class FakeDM:
        def load_ohlcv(self, ticker, start_date=None):
            return df

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)

    monkeypatch.setattr(
        data_loader,
        "download_and_save_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(Exception("should not")),
    )

    out = data_loader.load_data("AAA", start="2024-01-01", end="2024-01-03")

    assert not out.empty


def test_data_loader_download_and_vix(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def __init__(self):
            self.saved = []

        def save_ohlcv(self, ticker, df):
            self.saved.append(ticker)

        def get_market_data(self, symbol, days=1):
            return [(datetime.now().strftime("%Y-%m-%d"), 12.3)]

        def save_market_data(self, symbol, df):
            self.saved.append(symbol)

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return pd.DataFrame()

        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="5d"):
                return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert (
        data_loader.download_and_save_data("AAA", date.today(), date.today()) is False
    )
    assert data_loader.get_market_volatility_index() == 12.3


def test_data_loader_download_success(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def __init__(self):
            self.saved = []

        def save_ohlcv(self, ticker, df):
            self.saved.append(ticker)

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            dates = pd.date_range("2024-01-01", periods=2, freq="D")
            return pd.DataFrame(
                {
                    "Open": [1, 2],
                    "High": [1, 2],
                    "Low": [1, 2],
                    "Close": [1, 2],
                    "Volume": [1, 2],
                },
                index=dates,
            )

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.download_and_save_data("AAA", date.today(), date.today()) is True


def test_data_loader_vix_fallback(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def __init__(self):
            self.saved = []

        def get_market_data(self, symbol, days=1):
            return []

        def save_market_data(self, symbol, df):
            self.saved.append(symbol)

    class FakeYF:
        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="5d"):
                return pd.DataFrame({"Close": [10.0, 11.0]})

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.get_market_volatility_index() == 11.0


def test_data_loader_run_full_download(tmp_path, monkeypatch, test_settings):
    from app.data_access import data_loader
    from app import data_access

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    apple_df = pd.DataFrame({"Close": [1.0, 2.0]}, index=dates)

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return apple_df

    monkeypatch.setattr(data_loader, "yf", FakeYF)

    def fake_load_data(ticker, start, end):
        return apple_df.iloc[:1]

    called = []
    monkeypatch.setattr(data_loader, "load_data", fake_load_data)
    monkeypatch.setattr(
        data_loader,
        "download_and_save_data",
        lambda *args, **kwargs: called.append(1) or False,
    )

    failed_path = tmp_path / "failed.json"
    data_access.set_settings(replace(test_settings, FAILED_DAYS_FILE_PATH=failed_path))

    data_loader.run_full_download(
        tickers=["AAA"], start_date="2024-01-01", end_date="2024-01-03"
    )

    assert failed_path.exists()
    assert called


def test_data_loader_vix_no_data(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def get_market_data(self, symbol, days=1):
            return []

        def save_market_data(self, symbol, df):
            return None

    class FakeYF:
        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="5d"):
                return pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.get_market_volatility_index() is None


def test_data_loader_supported_tickers():
    from app.data_access.data_loader import (
        get_supported_tickers,
        get_supported_ticker_list,
    )

    tickers = get_supported_tickers()
    assert "VOO" in tickers
    assert "VOO" in list(get_supported_ticker_list())


def test_data_loader_default_dates(monkeypatch):
    from app.data_access import data_loader

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(5),
            "High": range(1, 6),
            "Low": range(0, 5),
            "Close": range(1, 6),
            "Volume": range(10, 15),
        },
        index=dates,
    )

    class FakeDM:
        def __init__(self):
            self.calls = 0

        def load_ohlcv(self, ticker, start_date=None):
            self.calls += 1
            return df if self.calls > 1 else pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(
        data_loader, "download_and_save_data", lambda *args, **kwargs: True
    )

    out = data_loader.load_data("AAA")
    assert not out.empty


def test_data_loader_run_full_download_success(tmp_path, monkeypatch, test_settings):
    from app.data_access import data_loader
    from app import data_access

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    apple_df = pd.DataFrame({"Close": [1.0, 2.0]}, index=dates)

    class FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return apple_df

    monkeypatch.setattr(data_loader, "yf", FakeYF)
    monkeypatch.setattr(data_loader, "load_data", lambda *args, **kwargs: apple_df)
    monkeypatch.setattr(
        data_loader, "download_and_save_data", lambda *args, **kwargs: True
    )

    failed_path = tmp_path / "failed2.json"
    data_access.set_settings(replace(test_settings, FAILED_DAYS_FILE_PATH=failed_path))

    data_loader.run_full_download(
        tickers=["AAA"], start_date="2024-01-01", end_date="2024-01-03"
    )
    assert failed_path.exists()


def test_data_loader_vix_old_db(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def __init__(self):
            self.saved = []

        def get_market_data(self, symbol, days=1):
            return [("2000-01-01", 10.0)]

        def save_market_data(self, symbol, df):
            self.saved.append(symbol)

    class FakeYF:
        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="5d"):
                return pd.DataFrame({"Close": [10.0, 12.0]})

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.get_market_volatility_index() == 12.0


def test_data_loader_vix_error_fallback(monkeypatch):
    from app.data_access import data_loader

    class FakeDM:
        def __init__(self):
            self.calls = 0

        def get_market_data(self, symbol, days=1):
            self.calls += 1
            if days == 1:
                return []
            return [("2000-01-01", 9.0)]

        def save_market_data(self, symbol, df):
            return None

    class FakeYF:
        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="5d"):
                raise RuntimeError("fail")

    monkeypatch.setattr(data_loader, "DataManager", FakeDM)
    monkeypatch.setattr(data_loader, "yf", FakeYF)

    assert data_loader.get_market_volatility_index() == 9.0


def test_data_manager_metrics(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "test.db"
    dm = DataManager(settings=replace(test_settings, DB_PATH=db_path))
    dm.initialize_tables()

    dates = pd.date_range("2024-01-01", periods=210, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(210),
            "High": range(1, 211),
            "Low": range(0, 210),
            "Close": list(range(210, 0, -1)),
            "Volume": range(100, 310),
        },
        index=dates,
    )
    df.index.name = "Date"

    dm.save_ohlcv("SPY", df)
    loaded = dm.load_ohlcv("SPY")
    assert not loaded.empty

    assert bool(dm.get_market_regime_is_bear(ref_date=dates[-1].date())) is True

    dm.save_market_data("^VIX", pd.DataFrame({"Close": [10.0, 11.0]}, index=dates[:2]))
    assert dm.get_market_data("^VIX", days=1)

    dm.log_pipeline_execution("AAA", "success", 1.2)
    dm.log_backtest_execution("AAA", 0.5, 2, 1.1)

    metrics = dm.get_recent_metrics(hours=24)
    assert metrics["total_executions"] >= 1

    summary = dm.get_daily_summary(datetime.now().strftime("%Y-%m-%d"))
    assert "executions" in summary


def test_data_manager_history_and_correlation(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "test2.db"
    dm = DataManager(settings=replace(test_settings, DB_PATH=db_path))
    dm.initialize_tables()

    today = datetime.now().strftime("%Y-%m-%d")
    dm.log_recommendation("AAA", "BUY", 0.9, params={"wf_score": 0.7})
    assert dm.get_today_recommendations()
    assert dm.get_ticker_historical_recommendations("AAA", today, today)

    with dm.connection() as conn:
        conn.execute(
            """
            INSERT INTO decision_history (timestamp, ticker, action_code, action_label, confidence, wf_score, decision_blob, audit_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                "AAA",
                1,
                "BUY",
                0.9,
                0.8,
                json.dumps({"action_code": 1}),
                json.dumps({"note": "x"}),
            ),
        )
        conn.commit()

    assert dm.get_strategy_accuracy("AAA")
    assert dm.get_unevaluated_buy_decisions()

    dm.update_history_audit(1, json.dumps({"outcome": "ok"}))
    assert dm.fetch_history_range("AAA", "2000-01-01", "2100-01-01")

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(10),
            "High": range(1, 11),
            "Low": range(0, 10),
            "Close": range(1, 11),
            "Volume": range(10, 20),
        },
        index=dates,
    )
    df.index.name = "Date"

    dm.save_ohlcv("AAA", df)
    dm.save_ohlcv("BBB", df)

    corr = dm.get_correlation_matrix(["AAA", "BBB"], lookback_days=5)
    assert not corr.empty


def test_data_manager_empty_and_records(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "test3.db"
    dm = DataManager(settings=replace(test_settings, DB_PATH=db_path))
    dm.initialize_tables()

    assert dm.get_market_regime_is_bear() is False
    assert dm.get_correlation_matrix(["AAA", "BBB"]).empty

    dm.save_history_record(
        ticker="AAA",
        action_code=1,
        label="BUY",
        confidence=0.9,
        wf_score=0.8,
        d_blob=json.dumps({"action_code": 1}),
        a_blob=json.dumps({"note": "x"}),
    )
    records = dm.fetch_history_records_by_ticker("AAA")
    assert records and records[0]["ticker"] == "AAA"

    assert dm.fetch_recent_outcomes("AAA") == []


def test_data_manager_error_paths(tmp_path, monkeypatch, test_settings):
    from app.data_access.data_manager import DataManager

    db_path = tmp_path / "test4.db"
    dm = DataManager(settings=replace(test_settings, DB_PATH=db_path))
    dm.initialize_tables()

    monkeypatch.setattr(
        dm, "_get_conn", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )

    assert dm.log_pipeline_execution("AAA", "success", 1.0) is False
    assert dm.log_backtest_execution("AAA", 0.1, 1, 1.0) is False
    assert dm.get_recent_metrics(hours=1)["total_executions"] == 0
    assert dm.get_daily_summary("2024-01-01")["executions"] == 0


def test_history_store_iteration(monkeypatch):
    from app.backtesting.history_store import HistoryStore

    class FakeDM:
        def fetch_history_records_by_ticker(self, ticker):
            return [
                {
                    "timestamp": "2024-01-01",
                    "ticker": ticker,
                    "decision": {},
                    "audit": {},
                }
            ]

        def fetch_history_range(self, ticker, start_iso, end_iso):
            return [
                ("2024-01-01T00:00:00", "BUY", json.dumps({"action_code": 1}), "{}")
            ]

        def fetch_recent_outcomes(self, ticker, n=3):
            return [{"success": True}]

    hs = HistoryStore()
    hs.dm = FakeDM()

    assert list(hs.iter_records("AAA"))
    assert hs.load_range("AAA", date(2024, 1, 1), date(2024, 1, 2))
    assert hs.load_recent_outcomes("AAA")


def test_drift_detector_paths(monkeypatch):
    from app.decision.drift_detector import (
        PerformanceDriftDetector,
        batch_check_drift,
        get_drifting_tickers,
    )

    detector = PerformanceDriftDetector(
        lookback_days=3, drift_threshold=0.1, critical_threshold=0.2
    )

    monkeypatch.setattr(detector, "_load_historical_scores", lambda t: [])
    no_drift = detector.check_drift("AAA", 0.5)
    assert no_drift["alert_level"] == "NONE"

    monkeypatch.setattr(detector, "_load_historical_scores", lambda t: [0.8, 0.8, 0.8])
    drift = detector.check_drift("AAA", 0.4)
    assert drift["drifting"]
    assert detector.generate_alert("AAA", drift)

    monkeypatch.setattr(
        "app.decision.drift_detector.PerformanceDriftDetector.check_drift",
        lambda self, t, c: {"drifting": True, "alert_level": "WARNING"},
    )
    assert batch_check_drift({"AAA": 0.5})["AAA"]["alert_level"] == "WARNING"
    assert "AAA" in get_drifting_tickers({"AAA": 0.5}, alert_level="WARNING")


def test_drift_detector_metrics(monkeypatch):
    from app.decision.drift_detector import (
        PerformanceDriftDetector,
        get_drifting_tickers,
    )

    detector = PerformanceDriftDetector()
    metrics = detector._compute_drift_metrics([1.0, 1.1, 1.2, 1.3], 1.0)
    assert metrics["trend"] in {"improving", "degrading", "stable"}

    no_alert = detector.generate_alert("AAA", {"drifting": False})
    assert no_alert is None

    assert get_drifting_tickers({"AAA": 0.5}, alert_level="UNKNOWN") in ([], ["AAA"])


def test_drift_detector_load_history(tmp_path, monkeypatch, test_settings):
    from app.decision.drift_detector import PerformanceDriftDetector
    import sqlite3

    db_path = tmp_path / "drift.db"
    settings = replace(test_settings, DB_PATH=db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE decision_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action_code INTEGER,
            action_label TEXT,
            confidence REAL,
            wf_score REAL,
            decision_blob TEXT,
            audit_blob TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO decision_history (timestamp, ticker, wf_score) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), "AAA", 0.7),
    )
    conn.commit()
    conn.close()

    detector = PerformanceDriftDetector(lookback_days=5, settings=settings)
    scores = detector._load_historical_scores("AAA")
    assert scores


def test_drift_detector_batch_error(monkeypatch):
    from app.decision.drift_detector import batch_check_drift

    monkeypatch.setattr(
        "app.decision.drift_detector.PerformanceDriftDetector.check_drift",
        lambda self, t, c: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    result = batch_check_drift({"AAA": 0.5})
    assert result["AAA"]["alert_level"] == "ERROR"


def test_mailer_send_email(monkeypatch):
    from app.notifications.mailer import send_email
    import smtplib

    class FakeSMTP:
        def __init__(self, host, port):
            self.host = host
            self.port = port

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

    monkeypatch.setenv("EMAIL_HOST", "smtp.test")
    monkeypatch.setenv("EMAIL_PORT", "587")
    monkeypatch.setenv("EMAIL_USER", "u")
    monkeypatch.setenv("EMAIL_PASSWORD", "p")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    assert send_email("s", "b", "t") is True


def test_mailer_send_email_failure(monkeypatch):
    from app.notifications.mailer import send_email
    import smtplib

    class FakeSMTP:
        def __init__(self, host, port):
            raise RuntimeError("fail")

    monkeypatch.setenv("EMAIL_HOST", "smtp.test")
    monkeypatch.setenv("EMAIL_PORT", "587")
    monkeypatch.setenv("EMAIL_USER", "u")
    monkeypatch.setenv("EMAIL_PASSWORD", "p")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    assert send_email("s", "b", "t") is False


def test_logger_setup(monkeypatch, tmp_path, test_settings):
    from app.infrastructure.logger import setup_logger, is_logger_debug

    settings = replace(test_settings, LOG_DIR=tmp_path, LOGGING_LEVEL="DEBUG")
    logger = setup_logger("test_logger_setup", settings=settings)
    assert is_logger_debug(logger)


def test_audit_builder_drift_and_chaotic(monkeypatch, test_settings):
    from app.reporting import audit_builder

    audit_builder.set_settings(replace(test_settings, ENABLE_DRIFT_DETECTION=True))

    class FakeDetector:
        def check_drift(self, ticker, current_score):
            return "WARNING"

    monkeypatch.setattr(
        "app.decision.drift_detector.PerformanceDriftDetector", FakeDetector
    )

    payload = {"ticker": "AAA", "model_votes": []}
    decision = {
        "action_code": 1,
        "confidence": 0.9,
        "wf_score": 0.8,
        "ensemble_quality": 0.0,
        "quality_score": 0.2,
    }

    audit = audit_builder.build_audit_metadata(payload, decision)
    assert audit["decision_level"] == "NO_TRADE"


def test_audit_builder_no_votes():
    from app.reporting import audit_builder

    flags = audit_builder.compute_consistency_flags(
        {"model_votes": []}, {"action_code": 0}
    )
    assert flags["majority_action"] is None
    assert flags["vote_divergence"] is None


def test_audit_builder_summary():
    from app.reporting import audit_builder

    audit = {
        "quality_score": 0.5,
        "ensemble_quality": "NORMAL",
        "confidence_bucket": "MEDIUM",
        "volatility_bucket": "LOW",
        "decision_level": "NORMAL",
        "trade_allowed": True,
    }
    payload = {"model_votes": [1, 2, 3]}
    decision = {"confidence": 0.6, "wf_score": None}

    summary = audit_builder.build_audit_summary(audit, payload, decision)
    assert summary["model_count"] == 3


def test_audit_builder_reliability_levels():
    from app.reporting.audit_builder import decision_reliability_level

    assert decision_reliability_level(0.6, None)[0] == "NORMAL"
    assert decision_reliability_level(0.4, None)[0] == "WEAK"


def test_config_and_pi_config(monkeypatch, tmp_path, test_settings):
    from app.config import pi_config
    from app.config.config import Config, _enforce_secret_key_policy
    from dataclasses import replace

    assert pi_config._parse_bool("true") is True
    assert pi_config._parse_bool("0") is False

    env = {"PI_MODE": "true", "PI_BASE_DIR": str(tmp_path)}

    settings = replace(test_settings, PI_MODE=False)
    result = pi_config.apply_pi_config(settings=settings, env=env)
    assert result.PI_MODE is True

    monkeypatch.setenv("ENV", "production")
    monkeypatch.setattr(Config, "SECRET_KEY", "safe_key")
    _enforce_secret_key_policy()


def test_pi_config_detection(monkeypatch, tmp_path):
    from app.config import pi_config

    class FakePlatform:
        @staticmethod
        def system():
            return "Linux"

        @staticmethod
        def machine():
            return "armv7l"

    assert pi_config.detect_pi_mode(env={}, platform_module=FakePlatform) is True

    model_path = tmp_path / "model"
    model_path.write_text("Raspberry Pi 4")

    class FakePlatform2:
        @staticmethod
        def system():
            return "Windows"

        @staticmethod
        def machine():
            return "x86"

    assert (
        pi_config.detect_pi_mode(
            env={}, platform_module=FakePlatform2, model_path=model_path
        )
        is True
    )


def test_pi_config_env_override():
    from app.config import pi_config

    assert pi_config._parse_bool(None) is None
    assert pi_config._parse_bool("maybe") is None
    assert pi_config.detect_pi_mode(env={"PI_MODE": "true"}) is True
    assert pi_config.detect_pi_mode(env={"PI_MODE": "false"}) is False


def test_pi_config_no_apply(monkeypatch, test_settings):
    from app.config import pi_config
    from dataclasses import replace

    settings = replace(test_settings, PI_MODE=False)
    result = pi_config.apply_pi_config(settings=settings, env={"PI_MODE": "false"})
    assert result is settings


def test_pi_config_error_platform(tmp_path):
    from app.config import pi_config

    class FakePlatform:
        @staticmethod
        def system():
            raise RuntimeError("fail")

        @staticmethod
        def machine():
            raise RuntimeError("fail")

    assert pi_config.detect_pi_mode(env={}, platform_module=FakePlatform) is False


def test_pi_config_apply_dirs(tmp_path, test_settings):
    from app.config import pi_config
    from dataclasses import replace

    env = {"PI_MODE": "true", "PI_BASE_DIR": str(tmp_path)}
    settings = replace(test_settings, PI_MODE=False)

    result = pi_config.apply_pi_config(settings=settings, env=env)
    assert result.PI_MODE is True


def test_pi_config_paths(tmp_path):
    from app.config import pi_config

    paths = pi_config._build_pi_paths(tmp_path)
    assert "DATA_DIR" in paths and "LOG_DIR" in paths


def test_pi_config_model_path_non_pi(tmp_path):
    from app.config import pi_config

    model_path = tmp_path / "model"
    model_path.write_text("not a pi")

    class FakePlatform:
        @staticmethod
        def system():
            return "Windows"

        @staticmethod
        def machine():
            return "x86"

    assert (
        pi_config.detect_pi_mode(
            env={}, platform_module=FakePlatform, model_path=model_path
        )
        is False
    )


def test_etf_allocator_core():
    from app.decision.etf_allocator import (
        ETFAllocator,
        AssetType,
        get_low_cost_etf,
        estimate_portfolio_cost,
        classify_asset_type,
    )

    allocator = ETFAllocator()

    assert allocator.classify_asset_type("VOO") == AssetType.ETF
    assert allocator.classify_asset_type("AAPL") == AssetType.STOCK
    assert allocator.classify_asset_type("UNKNOWN") == AssetType.STOCK

    exposure = allocator.analyze_sector_exposure(
        {"VOO": 0.5, "QQQ": 0.2, "XLK": 0.1, "AAPL": 0.2}
    )
    assert exposure

    comp = allocator.compare_etf_costs("VOO", "SPY")
    assert comp.cheaper_option in {"VOO", "SPY"}

    mix = allocator.calculate_optimal_mix(
        target_sectors={"Technology": 0.5},
        available_etfs=["VOO"],
        available_stocks=["AAPL"],
    )
    assert mix.etf_weights
    assert mix.optimization_notes

    assert allocator._get_sector_from_etf("XLK") == "Technology"
    assert allocator._calculate_portfolio_cost({"VOO": 1.0}, {}) > 0
    score = allocator._calculate_diversification_score({"VOO": 0.5}, {"AAPL": 0.5})
    assert 0.0 <= score <= 1.0

    assert get_low_cost_etf("Technology") == "XLK"
    assert get_low_cost_etf() == "VOO"
    assert estimate_portfolio_cost({"VOO": 1.0}) > 0
    assert classify_asset_type("VOO") == AssetType.ETF


def test_portfolio_correlation_manager(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Close": range(1, 41)}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)

    corr = manager.compute_correlation_matrix(["AAA", "BBB"], use_cache=False)
    assert not corr.empty

    corr_cached = manager.compute_correlation_matrix(["AAA", "BBB"], use_cache=True)
    assert not corr_cached.empty

    score = manager.calculate_diversification_score({"AAA": 0.5, "BBB": 0.5})
    assert 0.0 <= score <= 1.0

    risk = manager.decompose_portfolio_risk({"AAA": 0.5, "BBB": 0.5})
    assert risk.total_risk >= 0

    selected = manager.optimize_for_low_correlation(
        ["AAA", "BBB", "CCC"], target_size=2, max_correlation=1.0
    )
    assert len(selected) == 2

    pairs = manager.get_highly_correlated_pairs(["AAA", "BBB"], threshold=0.0)
    assert pairs

    manager.clear_cache()
    assert manager._correlation_cache == {}

    assert pcm.check_portfolio_diversification({"AAA": 1.0}) == 0.0
    assert pcm.find_uncorrelated_assets("AAA", ["BBB"], max_correlation=1.0) == ["BBB"]


def test_portfolio_correlation_empty(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame()

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    assert manager.compute_correlation_matrix(["AAA"]).empty
    assert manager.calculate_diversification_score({"AAA": 1.0}) == 0.0
    assert manager.optimize_for_low_correlation([], target_size=2) == []


def test_portfolio_correlation_constraints(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Close": range(1, 41)}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)

    corr = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: corr
    )

    selected = manager.optimize_for_low_correlation(
        ["AAA", "BBB", "CCC"], target_size=2, max_correlation=0.1
    )
    assert len(selected) == 1


def test_portfolio_correlation_cache_and_defaults(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Close": range(1, 41)}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)
    manager._correlation_cache["AAA_BBB"] = pd.DataFrame()
    manager._cache_timestamp = datetime.now() - timedelta(hours=25)

    assert manager._is_cache_valid() is False

    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: pd.DataFrame()
    )
    assert manager.calculate_diversification_score({"AAA": 0.5, "BBB": 0.5}) == 0.5


def test_portfolio_correlation_cache_hit(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            raise RuntimeError("should not be called")

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    cached = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    manager._correlation_cache["AAA"] = cached
    manager._cache_timestamp = datetime.now()

    out = manager.compute_correlation_matrix(["AAA"], use_cache=True)
    assert out.equals(cached)


def test_portfolio_correlation_empty_matrix(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame()

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    selected = manager.optimize_for_low_correlation(
        ["AAA", "BBB", "CCC"], target_size=2
    )
    assert selected == ["AAA", "BBB"]


def test_portfolio_correlation_pairs_and_uncorrelated(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame()

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    assert manager.get_highly_correlated_pairs(["AAA", "BBB"]) == []
    assert pcm.find_uncorrelated_assets("AAA", ["BBB"]) == []


def test_portfolio_correlation_cache_invalid(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame({"Close": range(40)})

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    manager._cache_timestamp = None
    assert manager._is_cache_valid() is False


def test_portfolio_correlation_small_candidates(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame({"Close": range(40)})

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager()
    assert manager.optimize_for_low_correlation(["AAA"], target_size=3) == ["AAA"]


def test_portfolio_correlation_partial_data(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates_short = pd.date_range("2024-01-01", periods=10, freq="D")
    dates_long = pd.date_range("2024-01-01", periods=40, freq="D")

    df_short = pd.DataFrame({"Close": range(10)}, index=dates_short)
    df_long = pd.DataFrame({"Close": range(40)}, index=dates_long)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df_short if ticker == "AAA" else df_long

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)
    corr = manager.compute_correlation_matrix(["AAA", "BBB"], use_cache=False)
    assert not corr.empty


def test_portfolio_correlation_keyerror(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    class FakeDM:
        def load_ohlcv(self, ticker):
            return pd.DataFrame({"Close": range(40)})

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)
    corr = pd.DataFrame([[1.0]], index=["AAA"], columns=["AAA"])
    monkeypatch.setattr(
        manager, "compute_correlation_matrix", lambda *args, **kwargs: corr
    )

    selected = manager.optimize_for_low_correlation(["AAA", "BBB"], target_size=2)
    assert selected


def test_portfolio_diversification_zero_variance(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Close": [1.0] * 40}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)
    monkeypatch.setattr(
        manager,
        "compute_correlation_matrix",
        lambda *args, **kwargs: pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
        ),
    )

    score = manager.calculate_diversification_score({"AAA": 0.5, "BBB": 0.5})
    assert score == 0.0


def test_portfolio_correlation_cache_with_invalid_entry(monkeypatch):
    from app.decision import portfolio_correlation_manager as pcm

    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Close": range(1, 41)}, index=dates)

    class FakeDM:
        def load_ohlcv(self, ticker):
            return df

    monkeypatch.setattr(pcm, "DataManager", lambda: FakeDM())

    manager = pcm.PortfolioCorrelationManager(lookback_days=10)
    manager._correlation_cache["AAA_BBB"] = pd.DataFrame(
        [[1.0]], index=["AAA"], columns=["AAA"]
    )
    manager._cache_timestamp = datetime.now() - timedelta(hours=25)

    corr = manager.compute_correlation_matrix(["AAA", "BBB"], use_cache=True)
    assert not corr.empty


def test_config_ensure_dirs(monkeypatch, tmp_path):
    from app.config.config import Config

    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(Config, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(Config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(Config, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setattr(Config, "CHART_DIR", tmp_path / "charts")
    monkeypatch.setattr(Config, "TENSORBOARD_DIR", tmp_path / "tensorboard")
    monkeypatch.setattr(Config, "MODEL_RELIABILITY_DIR", tmp_path / "reliability")
    monkeypatch.setattr(Config, "DECISION_LOG_DIR", tmp_path / "decision_logs")
    monkeypatch.setattr(Config, "DECISION_OUTCOME_DIR", tmp_path / "decision_outcomes")

    monkeypatch.setattr(Config, "ENABLE_FLASK", True)
    monkeypatch.setattr(Config, "ENABLE_RL", True)
    monkeypatch.setattr(Config, "ENABLE_RELIABILITY", True)

    Config.ensure_dirs()

    assert Config.DATA_DIR.exists()
    assert Config.LOG_DIR.exists()
