"""Edge case tests for data_loader (Sprint 10 Week 3)."""

import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

from app.data_access import data_loader, get_settings, set_settings as set_data_settings
from dataclasses import replace


def _make_df(start_date, days=5):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    prices = np.linspace(100, 105, len(dates))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1,
            "Low": prices - 1,
            "Close": prices,
            "Volume": np.full(len(dates), 1000000.0),
        },
        index=dates,
    )
    return df


def test_load_data_uses_db_when_fresh(monkeypatch):
    """Should not download when DB is recent enough."""
    end = datetime(2025, 1, 10).date()
    start = datetime(2024, 12, 1).date()
    warmup_start = start - timedelta(days=get_settings().WARMUP_DAYS)

    df_db = _make_df(end - timedelta(days=4), days=5)

    class DummyDM:
        def load_ohlcv(self, ticker, start_date=None):
            return df_db

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())
    download_called = {"count": 0}

    def _fake_download(*args, **kwargs):
        download_called["count"] += 1
        return True

    monkeypatch.setattr(data_loader, "download_and_save_data", _fake_download)

    result = data_loader.load_data("SPY", start=start, end=end)

    assert not result.empty
    assert download_called["count"] == 0


def test_load_data_downloads_when_missing(monkeypatch):
    """Should download when DB is missing data."""
    end = datetime(2025, 1, 10).date()
    start = datetime(2024, 12, 1).date()

    class DummyDM:
        def __init__(self):
            self.calls = 0

        def load_ohlcv(self, ticker, start_date=None):
            self.calls += 1
            if self.calls == 1:
                return pd.DataFrame()
            return _make_df(start - timedelta(days=get_settings().WARMUP_DAYS), days=10)

    dummy = DummyDM()
    monkeypatch.setattr(data_loader, "DataManager", lambda: dummy)

    download_called = {"count": 0}

    def _fake_download(*args, **kwargs):
        download_called["count"] += 1
        return True

    monkeypatch.setattr(data_loader, "download_and_save_data", _fake_download)

    result = data_loader.load_data("VOO", start=start, end=end)

    assert not result.empty
    assert download_called["count"] == 1


def test_load_data_accepts_string_dates(monkeypatch):
    """Should accept YYYY-MM-DD strings for start/end."""

    class DummyDM:
        def load_ohlcv(self, ticker, start_date=None):
            return _make_df(datetime(2024, 1, 1), days=5)

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())
    monkeypatch.setattr(data_loader, "download_and_save_data", lambda *a, **k: True)

    result = data_loader.load_data("QQQ", start="2024-01-01", end="2024-02-01")
    assert not result.empty


def test_download_and_save_data_handles_empty(monkeypatch):
    """Should return False on empty download."""

    class DummyDM:
        def save_ohlcv(self, ticker, df):
            raise AssertionError("save_ohlcv should not be called on empty df")

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())

    def _fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader.yf, "download", _fake_download)

    assert (
        data_loader.download_and_save_data(
            "SPY", datetime(2025, 1, 1), datetime(2025, 1, 2)
        )
        is False
    )


def test_download_and_save_data_success(monkeypatch):
    """Should save data and return True on successful download."""
    saved = {"called": False, "ticker": None}

    class DummyDM:
        def save_ohlcv(self, ticker, df):
            saved["called"] = True
            saved["ticker"] = ticker

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())

    df = _make_df(datetime(2025, 1, 1), days=3)

    def _fake_download(*args, **kwargs):
        return df

    monkeypatch.setattr(data_loader.yf, "download", _fake_download)

    assert (
        data_loader.download_and_save_data(
            "SPY", datetime(2025, 1, 1), datetime(2025, 1, 4)
        )
        is True
    )
    assert saved["called"] is True
    assert saved["ticker"] == "SPY"


def test_get_market_volatility_index_from_db(monkeypatch):
    """Should return VIX from DB when available for today."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    class DummyDM:
        def get_market_data(self, symbol, days=1):
            return [(today_str, 12.34)]

        def save_market_data(self, symbol, df):
            raise AssertionError("save_market_data should not be called")

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())

    vix = data_loader.get_market_volatility_index()
    assert vix == 12.34


def test_get_market_volatility_index_download_fallback(monkeypatch):
    """Should download VIX when not in DB, then fallback to DB if download fails."""

    class DummyDM:
        def get_market_data(self, symbol, days=1):
            return []

        def save_market_data(self, symbol, df):
            self.saved = True

        def get_market_data_fallback(self, symbol, days=10):
            return [("2025-01-01", 15.0)]

    dummy = DummyDM()

    def _get_market_data(symbol, days=1):
        return [] if days == 1 else [("2025-01-01", 15.0)]

    dummy.get_market_data = _get_market_data

    class DummyTicker:
        def history(self, period="5d"):
            return pd.DataFrame()

    monkeypatch.setattr(data_loader, "DataManager", lambda: dummy)
    monkeypatch.setattr(data_loader.yf, "Ticker", lambda symbol: DummyTicker())

    vix = data_loader.get_market_volatility_index()
    assert vix == 15.0


def test_get_market_volatility_index_download_success(monkeypatch):
    """Should download VIX and save when DB is missing today."""
    saved = {"called": False}

    class DummyDM:
        def get_market_data(self, symbol, days=1):
            return []

        def save_market_data(self, symbol, df):
            saved["called"] = True

    class DummyTicker:
        def history(self, period="5d"):
            dates = pd.date_range(start="2025-01-01", periods=3, freq="D")
            return pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=dates)

    monkeypatch.setattr(data_loader, "DataManager", lambda: DummyDM())
    monkeypatch.setattr(data_loader.yf, "Ticker", lambda symbol: DummyTicker())

    vix = data_loader.get_market_volatility_index()
    assert vix == 12.0
    assert saved["called"] is True


def test_get_supported_tickers_and_list():
    """Should expose supported tickers and keys."""
    tickers = data_loader.get_supported_tickers()
    keys = list(data_loader.get_supported_ticker_list())

    assert "VOO" in tickers
    assert "VOO" in keys


def test_run_full_download_downloads_missing_days(monkeypatch, test_settings, tmp_path):
    """Should call download_and_save_data for missing trading days."""
    dates = pd.date_range(start="2025-01-01", periods=3, freq="D")
    apple_df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

    def _fake_yf_download(*args, **kwargs):
        return apple_df

    monkeypatch.setattr(data_loader.yf, "download", _fake_yf_download)

    def _fake_load_data(ticker, start, end, **kwargs):
        # Return only the first day to force missing days
        return apple_df.iloc[:1]

    monkeypatch.setattr(data_loader, "load_data", _fake_load_data)

    calls = {"count": 0}

    def _fake_download_and_save(*args, **kwargs):
        calls["count"] += 1
        return True

    monkeypatch.setattr(data_loader, "download_and_save_data", _fake_download_and_save)

    settings = replace(
        test_settings,
        TICKERS=["AAA", "BBB"],
        START_DATE="2025-01-01",
        END_DATE="2025-01-03",
        FAILED_DAYS_FILE_PATH=tmp_path / "fails.json",
    )
    set_data_settings(settings)

    # Avoid writing to real file
    monkeypatch.setattr(data_loader.json, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        data_loader, "open", lambda *args, **kwargs: io.StringIO(), raising=False
    )

    data_loader.run_full_download(settings=settings)

    # 2 tickers * 2 missing days = 4 calls
    assert calls["count"] == 4
