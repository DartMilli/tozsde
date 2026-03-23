"""
Pytest configuration and shared fixtures.

Fixtures:
- test_db: Temporary SQLite database for testing
- sample_ohlcv: Sample OHLCV data row
- sample_df: Sample DataFrame with OHLCV data
- mock_config: Configuration override for tests
"""

import pytest
import tempfile
import os
from pathlib import Path
from dataclasses import replace
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import after adding project to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.build_settings import build_settings
from app.data_access.data_manager import DataManager


def _apply_settings(settings):
    try:
        from app import analysis

        analysis.set_settings(settings)
    except Exception:
        pass
    try:
        from app import backtesting

        backtesting.set_settings(settings)
    except Exception:
        pass
    try:
        from app import data_access

        data_access.set_settings(settings)
    except Exception:
        pass
    try:
        from app import decision

        decision.set_settings(settings)
    except Exception:
        pass
    try:
        from app import governance

        governance.set_settings(settings)
    except Exception:
        pass
    try:
        from app import models

        models.set_settings(settings)
    except Exception:
        pass
    try:
        from app import infrastructure

        infrastructure.set_settings(settings)
    except Exception:
        pass
    try:
        from app import ui

        ui.set_settings(settings)
    except Exception:
        pass
    try:
        from app import validation

        validation.set_settings(settings)
    except Exception:
        pass
    try:
        from app.reporting import audit_builder

        audit_builder.set_settings(settings)
    except Exception:
        pass
    try:
        from app.reporting import plotter

        plotter.set_settings(settings)
    except Exception:
        pass
    try:
        from app.notifications import alerter

        alerter.set_settings(settings)
    except Exception:
        pass
    try:
        from app.ui import app as ui_app

        ui_app.settings = settings
        ui_app.app.secret_key = settings.SECRET_KEY
    except Exception:
        pass


def _build_test_settings(tmpdir: Path):
    base = build_settings(ensure_dirs=False)
    data_dir = tmpdir / "data"
    log_dir = tmpdir / "logs"
    reports_dir = tmpdir / "reports"
    chart_dir = tmpdir / "charts"
    diagnostics_dir = tmpdir / "diagnostics"
    history_dir = tmpdir / "decision_history"
    decision_log_dir = tmpdir / "decision_logs"
    decision_outcome_dir = tmpdir / "decision_outcomes"
    model_dir = tmpdir / "models"
    tensorboard_dir = tmpdir / "tensorboard"
    model_reliability_dir = log_dir / "model_reliability"
    data_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    chart_dir.mkdir(exist_ok=True)
    diagnostics_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)
    decision_log_dir.mkdir(exist_ok=True)
    decision_outcome_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    model_reliability_dir.mkdir(exist_ok=True)

    return replace(
        base,
        DATA_DIR=data_dir,
        LOG_DIR=log_dir,
        REPORTS_DIR=reports_dir,
        CHART_DIR=chart_dir,
        DIAGNOSTICS_DIR=diagnostics_dir,
        HISTORY_DIR=history_dir,
        DECISION_LOG_DIR=decision_log_dir,
        DECISION_OUTCOME_DIR=decision_outcome_dir,
        MODEL_DIR=model_dir,
        TENSORBOARD_DIR=tensorboard_dir,
        MODEL_RELIABILITY_DIR=model_reliability_dir,
        DB_PATH=data_dir / "test.db",
        PARAMS_FILE_PATH=data_dir / "optimized_params.json",
        FAILED_DAYS_FILE_PATH=data_dir / "failed_to_download.json",
        MODEL_TEST_RESULT_FILE_PATH=data_dir / "model_test_result.json",
    )


@pytest.fixture(scope="function")
def test_settings(tmp_path):
    settings = _build_test_settings(tmp_path)
    _apply_settings(settings)
    return settings


@pytest.fixture(scope="function")
def test_db(test_settings):
    """Create temporary test database that persists across session."""
    dm = DataManager(settings=test_settings)
    dm.initialize_tables()
    return dm


@pytest.fixture(scope="function")
def populated_ohlcv(test_db):
    """Populate test DB with OHLCV data for correlation-based tests."""
    rng = np.random.RandomState(123)
    dates = pd.date_range(start="2025-01-01", periods=180, freq="D")

    base_returns = rng.normal(0.0005, 0.01, len(dates))

    tickers = ["AAPL", "MSFT", "JNJ", "XOM", "SPY", "VOO", "QQQ"]

    for i, ticker in enumerate(tickers):
        noise = rng.normal(0, 0.005 + i * 0.0002, len(dates))
        returns = base_returns * (0.85 - i * 0.05) + noise
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "Open": prices * (1 + rng.normal(0, 0.002, len(dates))),
                "High": prices * (1 + np.abs(rng.normal(0, 0.003, len(dates)))),
                "Low": prices * (1 - np.abs(rng.normal(0, 0.003, len(dates)))),
                "Close": prices,
                "Volume": rng.uniform(900000, 1100000, len(dates)),
            }
        )
        df.set_index(dates, inplace=True)

        test_db.save_ohlcv(ticker, df)

    return tickers


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data row for testing."""
    return {
        "ticker": "TEST",
        "date": "2025-01-20",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 1000000.0,
    }


@pytest.fixture
def sample_df():
    """Sample DataFrame with 30 days of OHLCV data."""
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")

    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 30)  # 0.1% mean return, 2% std
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "date": dates,
            "Open": prices * (1 + np.random.uniform(-0.01, 0.01, 30)),
            "High": prices * (1 + np.abs(np.random.uniform(0, 0.02, 30))),
            "Low": prices * (1 - np.abs(np.random.uniform(0, 0.02, 30))),
            "Close": prices,
            "Volume": np.random.uniform(900000, 1100000, 30),
        }
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.set_index(dates, inplace=True)  # Set date_range as DatetimeIndex
    return df


@pytest.fixture
def sample_signals():
    """Sample signal series: 0=HOLD, 1=BUY, 2=SELL."""
    return pd.Series([0, 0, 1, 0, 0, 0, 2, 0, 1, 0] * 3)


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    return {
        "INITIAL_CAPITAL": 10000,
        "TRANSACTION_FEE_PCT": 0.001,
        "MIN_SLIPPAGE_PCT": 0.0005,
        "SPREAD_PCT": 0.0005,
        "TRAIN_WINDOW_MONTHS": 24,
        "TEST_WINDOW_MONTHS": 6,
        "WINDOW_STEP_MONTHS": 3,
    }
