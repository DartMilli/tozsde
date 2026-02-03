"""Edge case tests for analyzer module."""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from app.analysis import analyzer


def _make_df():
    dates = pd.date_range(start="2025-01-01", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [105, 106],
            "Low": [99, 100],
            "Close": [102, 103],
            "Volume": [1000, 1100],
        },
        index=dates,
    )


def test_get_params_returns_default_on_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer.Config, "PARAMS_FILE_PATH", str(tmp_path / "missing.json"))
    params = analyzer.get_params("SPY")
    assert params == analyzer.get_default_params()


def test_get_params_converts_stddev_to_float(monkeypatch, tmp_path):
    params_file = tmp_path / "params.json"
    data = {"SPY": {"bbands_stddev": 2, "sma_period": 10}}
    params_file.write_text(json.dumps(data))

    monkeypatch.setattr(analyzer.Config, "PARAMS_FILE_PATH", str(params_file))

    params = analyzer.get_params("SPY")
    assert isinstance(params["bbands_stddev"], float)
    assert params["bbands_stddev"] == 2.0


def test_compute_signals_empty_df():
    df = pd.DataFrame()
    signals, indicators = analyzer.compute_signals(df, "SPY", params={})
    assert signals == []
    assert all(v is None for v in indicators.values())


def test_compute_signals_generates_multiple_signals(monkeypatch):
    df = _make_df()

    monkeypatch.setattr(analyzer.ta, "sma", lambda *a, **k: np.array([1.0, 1.0]))
    monkeypatch.setattr(analyzer.ta, "ema", lambda *a, **k: np.array([0.5, 1.5]))
    monkeypatch.setattr(analyzer.ta, "rsi", lambda *a, **k: np.array([29.0, 31.0]))

    def _macd(*args, **kwargs):
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    monkeypatch.setattr(analyzer.ta, "macd", _macd)

    def _bbands(*args, **kwargs):
        return np.array([2.0]), np.array([1.0]), np.array([0.0])

    monkeypatch.setattr(analyzer.ta, "bbands", _bbands)
    monkeypatch.setattr(analyzer.ta, "atr", lambda *a, **k: np.array([1.0, 2.0]))

    def _adx(*args, **kwargs):
        return np.array([30.0]), np.array([20.0]), np.array([10.0])

    monkeypatch.setattr(analyzer.ta, "adx", _adx)

    def _stoch(*args, **kwargs):
        return np.array([0.0, 2.0]), np.array([1.0, 1.0])

    monkeypatch.setattr(analyzer.ta, "stoch", _stoch)

    params = analyzer.get_default_params()
    signals, indicators = analyzer.compute_signals(df, "SPY", params=params)

    assert any("BUY: EMA crossed above SMA" in s for s in signals)
    assert any("BUY: RSI broke above 30" in s for s in signals)
    assert any("BUY: MACD crossover" in s for s in signals)
    assert any("SELL: Price above Bollinger Upper Band" in s for s in signals)
    assert any("ALERT: ATR volatility spike" in s for s in signals)
    assert any("INFO: Strong trend detected" in s for s in signals)
    assert any("BUY: Stochastic crossover" in s for s in signals)

    assert indicators["SMA"] is not None
    assert indicators["EMA"] is not None


def test_compute_signals_return_series(monkeypatch):
    df = _make_df()

    # Create one BUY crossover on second bar
    monkeypatch.setattr(analyzer.ta, "sma", lambda *a, **k: np.array([1.0, 1.0]))
    monkeypatch.setattr(analyzer.ta, "ema", lambda *a, **k: np.array([0.5, 1.5]))

    # Minimal required mocks for unused indicators
    monkeypatch.setattr(analyzer.ta, "rsi", lambda *a, **k: np.array([50.0, 50.0]))
    monkeypatch.setattr(analyzer.ta, "macd", lambda *a, **k: (np.array([0.0, 0.0]), np.array([0.0, 0.0])))
    monkeypatch.setattr(analyzer.ta, "bbands", lambda *a, **k: (np.array([2.0]), np.array([1.0]), np.array([0.0])))
    monkeypatch.setattr(analyzer.ta, "atr", lambda *a, **k: np.array([1.0, 1.0]))
    monkeypatch.setattr(analyzer.ta, "adx", lambda *a, **k: (np.array([10.0]), np.array([0.0]), np.array([0.0])))
    monkeypatch.setattr(analyzer.ta, "stoch", lambda *a, **k: (np.array([0.0, 0.0]), np.array([0.0, 0.0])))

    signals_per_bar, indicators = analyzer.compute_signals(
        df, "SPY", params=analyzer.get_default_params(), return_series=True
    )

    assert len(signals_per_bar) == len(df)
    assert signals_per_bar[-1] in ("BUY", "SELL", "HOLD")
    assert indicators["SMA"] is not None
