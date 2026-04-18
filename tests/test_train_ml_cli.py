"""
Integration tests for train-ml CLI command and run_train_ml().
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import main as cli_main
from main import run_train_ml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with lowercase OHLCV columns and a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.abs(rng.uniform(0, 0.01, n))),
            "Low": prices * (1 - np.abs(rng.uniform(0, 0.01, n))),
            "Close": prices,
            "Volume": rng.uniform(1e6, 5e6, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Unit-level: run_train_ml()
# ---------------------------------------------------------------------------


class TestRunTrainMl:
    def _patch_loaders(self, df: pd.DataFrame):
        """Return a context manager that mocks load_data and prepare_df."""
        return mock.patch.multiple(
            "main",
            **{},  # run_train_ml imports lazily inside the function
        )

    @pytest.fixture()
    def tmp_model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    def _run(self, ticker: str, tmp_dir: Path, df: pd.DataFrame, **kwargs):
        """Call run_train_ml with loaders mocked to return df."""
        lc_df = df.copy()
        lc_df.columns = [c.lower() for c in lc_df.columns]

        with mock.patch(
            "app.data_access.data_loader.load_data", return_value=df
        ), mock.patch(
            "app.data_access.data_cleaner.prepare_df", return_value=lc_df
        ), mock.patch(
            "app.config.build_settings.build_settings"
        ) as mock_cfg:
            cfg_obj = mock.MagicMock()
            cfg_obj.MODEL_DIR = str(tmp_dir)
            mock_cfg.return_value = cfg_obj
            return run_train_ml(ticker=ticker, dry_run=True, **kwargs)

    def test_run_train_ml_rf_returns_ok(self, tmp_model_dir):
        df = _make_ohlcv(200)
        result = self._run("VOO", tmp_model_dir, df, model_type="rf")
        assert result["ok"] is True, f"Expected ok, got: {result}"
        assert result["ticker"] == "VOO"
        assert result["model_type"] == "rf"
        assert result["samples"] > 0
        assert "train_score" in result
        assert "val_score" in result

    def test_run_train_ml_gb_returns_ok(self, tmp_model_dir):
        df = _make_ohlcv(200)
        result = self._run("VOO", tmp_model_dir, df, model_type="gb")
        assert result["ok"] is True, f"Expected ok, got: {result}"
        assert result["model_type"] == "gb"

    def test_run_train_ml_dry_run_no_files(self, tmp_model_dir):
        df = _make_ohlcv(200)
        result = self._run("VOO", tmp_model_dir, df)
        assert result["dry_run"] is True
        pkl_files = (
            list((tmp_model_dir / "ml_predictor").glob("*.pkl"))
            if (tmp_model_dir / "ml_predictor").exists()
            else []
        )
        assert (
            pkl_files == []
        ), f"No pkl files should be written in dry-run, found: {pkl_files}"

    def test_run_train_ml_saves_files_when_not_dry_run(self, tmp_model_dir):
        df = _make_ohlcv(200)
        lc_df = df.copy()
        lc_df.columns = [c.lower() for c in lc_df.columns]

        with mock.patch(
            "app.data_access.data_loader.load_data", return_value=df
        ), mock.patch(
            "app.data_access.data_cleaner.prepare_df", return_value=lc_df
        ), mock.patch(
            "app.config.build_settings.build_settings"
        ) as mock_cfg:
            cfg_obj = mock.MagicMock()
            cfg_obj.MODEL_DIR = str(tmp_model_dir)
            mock_cfg.return_value = cfg_obj
            result = run_train_ml(ticker="VOO", model_type="rf", dry_run=False)

        assert result["ok"] is True
        ml_dir = tmp_model_dir / "ml_predictor"
        assert (ml_dir / "ml_VOO_rf.pkl").exists(), "Model .pkl should be saved"
        assert (ml_dir / "ml_VOO_rf_scaler.pkl").exists(), "Scaler .pkl should be saved"

    def test_run_train_ml_no_data_returns_error(self, tmp_model_dir):
        with mock.patch(
            "app.data_access.data_loader.load_data", return_value=None
        ), mock.patch("app.config.build_settings.build_settings") as mock_cfg:
            cfg_obj = mock.MagicMock()
            cfg_obj.MODEL_DIR = str(tmp_model_dir)
            mock_cfg.return_value = cfg_obj
            result = run_train_ml(ticker="INVALID", dry_run=True)

        assert result["ok"] is False
        assert "error" in result

    def test_run_train_ml_insufficient_data_returns_error(self, tmp_model_dir):
        # Only 10 rows — below lookback=60
        df = _make_ohlcv(10)
        lc_df = df.copy()
        lc_df.columns = [c.lower() for c in lc_df.columns]

        with mock.patch(
            "app.data_access.data_loader.load_data", return_value=df
        ), mock.patch(
            "app.data_access.data_cleaner.prepare_df", return_value=lc_df
        ), mock.patch(
            "app.config.build_settings.build_settings"
        ) as mock_cfg:
            cfg_obj = mock.MagicMock()
            cfg_obj.MODEL_DIR = str(tmp_model_dir)
            mock_cfg.return_value = cfg_obj
            result = run_train_ml(ticker="VOO", dry_run=True)

        assert result["ok"] is False
        assert "Insufficient" in result["error"]

    def test_run_train_ml_custom_lookback(self, tmp_model_dir):
        df = _make_ohlcv(200)
        result = self._run("VOO", tmp_model_dir, df, lookback=30)
        assert result["ok"] is True
        assert result["samples"] > 0

    def test_run_train_ml_model_name_contains_ticker_and_type(self, tmp_model_dir):
        df = _make_ohlcv(200)
        result = self._run("SPY", tmp_model_dir, df, model_type="rf")
        assert result["ok"] is True
        assert "SPY" in result["model_name"]
        assert "rf" in result["model_name"]


# ---------------------------------------------------------------------------
# Integration: CLI argument parsing + dispatch
# ---------------------------------------------------------------------------


class TestTrainMlCLI:
    def _run_cli(self, argv: list) -> dict:
        """Run main() with given argv, capture stdout JSON output."""
        captured = {}

        def fake_emit(payload):
            captured["payload"] = payload

        with mock.patch.object(cli_main, "_emit", side_effect=fake_emit), mock.patch(
            "sys.argv", ["main.py"] + argv
        ), mock.patch("main.run_train_ml") as mock_train:
            mock_train.return_value = {
                "ok": True,
                "ticker": argv[1],
                "model_type": "rf",
                "model_name": f"ml_{argv[1]}_rf",
                "samples": 120,
                "dry_run": "--dry-run" in argv,
                "train_score": 0.80,
                "val_score": 0.72,
            }
            cli_main.main()

        return captured.get("payload", {})

    def test_cli_train_ml_basic(self):
        result = self._run_cli(["train-ml", "VOO", "--dry-run"])
        assert result.get("status") == "ok", f"Expected ok, got: {result}"
        assert result["use_case"] == "cli.train_ml"
        assert result["data"]["ok"] is True

    def test_cli_train_ml_model_type_gb(self):
        with mock.patch("main.run_train_ml") as mock_train, mock.patch.object(
            cli_main, "_emit"
        ):
            mock_train.return_value = {
                "ok": True,
                "ticker": "VOO",
                "model_type": "gb",
                "model_name": "ml_VOO_gb",
                "samples": 100,
                "dry_run": True,
                "train_score": 0.75,
                "val_score": 0.68,
            }
            with mock.patch(
                "sys.argv",
                ["main.py", "train-ml", "VOO", "--model-type", "gb", "--dry-run"],
            ):
                cli_main.main()
            mock_train.assert_called_once_with(
                ticker="VOO", model_type="gb", lookback=60, dry_run=True
            )

    def test_cli_train_ml_custom_lookback(self):
        with mock.patch("main.run_train_ml") as mock_train, mock.patch.object(
            cli_main, "_emit"
        ):
            mock_train.return_value = {
                "ok": True,
                "ticker": "QQQ",
                "model_type": "rf",
                "model_name": "ml_QQQ_rf",
                "samples": 80,
                "dry_run": True,
                "train_score": 0.7,
                "val_score": 0.65,
            }
            with mock.patch(
                "sys.argv",
                ["main.py", "train-ml", "QQQ", "--lookback", "30", "--dry-run"],
            ):
                cli_main.main()
            mock_train.assert_called_once_with(
                ticker="QQQ", model_type="rf", lookback=30, dry_run=True
            )

    def test_cli_train_ml_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["main.py", "train-ml", "--help"]):
                cli_main.main()
        assert exc_info.value.code == 0

    def test_cli_train_ml_default_model_type_is_rf(self):
        with mock.patch("main.run_train_ml") as mock_train, mock.patch.object(
            cli_main, "_emit"
        ):
            mock_train.return_value = {
                "ok": True,
                "ticker": "VOO",
                "model_type": "rf",
                "model_name": "ml_VOO_rf",
                "samples": 100,
                "dry_run": True,
                "train_score": 0.8,
                "val_score": 0.7,
            }
            with mock.patch("sys.argv", ["main.py", "train-ml", "VOO", "--dry-run"]):
                cli_main.main()
            _, kwargs = mock_train.call_args
            assert kwargs.get("model_type", "rf") == "rf"

    def test_cli_invalid_model_type_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch(
                "sys.argv", ["main.py", "train-ml", "VOO", "--model-type", "xgb"]
            ):
                cli_main.main()
        assert exc_info.value.code != 0
