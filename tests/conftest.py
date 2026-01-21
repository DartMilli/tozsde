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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import after adding project to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.config import Config
from app.data_access.data_manager import DataManager


@pytest.fixture(scope="session")
def test_db():
    """Create temporary test database that persists across session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Store original paths
        original_data_dir = Config.DATA_DIR
        original_db_path = Config.DB_PATH
        original_log_dir = Config.LOG_DIR

        # Override with temp paths
        Config.DATA_DIR = Path(tmpdir) / "data"
        Config.DATA_DIR.mkdir(exist_ok=True)
        Config.DB_PATH = Config.DATA_DIR / "test.db"
        Config.LOG_DIR = Path(tmpdir) / "logs"
        Config.LOG_DIR.mkdir(exist_ok=True)

        # Initialize database
        dm = DataManager()
        dm.initialize_tables()

        yield dm

        # Restore original paths
        Config.DATA_DIR = original_data_dir
        Config.DB_PATH = original_db_path
        Config.LOG_DIR = original_log_dir


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
