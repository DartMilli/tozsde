"""
Tests for walk_forward execution paths.
"""

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from app.backtesting.walk_forward import WalkForwardOptimizer, run_walk_forward
from app.optimization.fitness import WalkForwardResult
from app.optimization.fitness import NEG_INF


def _make_df(rows=10):
    dates = pd.date_range(start="2025-01-01", periods=rows, freq="D")
    return pd.DataFrame({"Close": np.linspace(100, 110, rows)}, index=dates)


def test_run_returns_summary_when_valid():
    df = _make_df(10)

    with patch("app.backtesting.walk_forward.optimize_params") as mock_opt, patch(
        "app.backtesting.walk_forward.Backtester"
    ) as mock_bt, patch(
        "app.backtesting.walk_forward.fitness_single"
    ) as mock_fs, patch(
        "app.backtesting.walk_forward.compute_walk_forward_metrics"
    ) as mock_wf:

        mock_opt.return_value = {"p": 1}
        report = MagicMock()
        report.metrics = {
            "net_profit": 10.0,
            "max_drawdown": -0.1,
            "trade_count": 5,
        }
        mock_bt.return_value.run.return_value = report
        mock_fs.return_value = 1.23
        mock_wf.return_value = WalkForwardResult(
            avg_profit=1.0,
            avg_drawdown=-0.1,
            profit_std=0.1,
            dd_std=0.05,
            negative_fold_ratio=0.0,
            raw_fitness=0.5,
            normalized_score=0.2,
        )

        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=df,
            bounds={},
            train_window=3,
            test_window=2,
            step_size=2,
            verbose=False,
        )

        result = wf.run()

        assert result is not None
        assert result["best_params"] == {"p": 1}
        assert result["wf_summary"]["windows"] == 3
        assert result["wf_summary"]["win_rate"] == 1.0


def test_run_returns_none_on_neg_inf():
    df = _make_df(10)

    with patch("app.backtesting.walk_forward.optimize_params") as mock_opt, patch(
        "app.backtesting.walk_forward.Backtester"
    ) as mock_bt, patch(
        "app.backtesting.walk_forward.fitness_single"
    ) as mock_fs, patch(
        "app.backtesting.walk_forward.compute_walk_forward_metrics"
    ) as mock_wf:

        mock_opt.return_value = {"p": 1}
        report = MagicMock()
        report.metrics = {
            "net_profit": -10.0,
            "max_drawdown": -0.2,
            "trade_count": 2,
        }
        mock_bt.return_value.run.return_value = report
        mock_fs.return_value = -1.0
        mock_wf.return_value = WalkForwardResult(
            avg_profit=-1.0,
            avg_drawdown=-0.2,
            profit_std=0.2,
            dd_std=0.1,
            negative_fold_ratio=1.0,
            raw_fitness=NEG_INF,
            normalized_score=0.0,
        )

        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=df,
            bounds={},
            train_window=3,
            test_window=2,
            step_size=2,
            verbose=False,
        )

        assert wf.run() is None


def test_run_returns_none_when_too_few_windows():
    df = _make_df(8)

    with patch("app.backtesting.walk_forward.optimize_params") as mock_opt, patch(
        "app.backtesting.walk_forward.Backtester"
    ) as mock_bt, patch(
        "app.backtesting.walk_forward.fitness_single"
    ) as mock_fs, patch(
        "app.backtesting.walk_forward.compute_walk_forward_metrics"
    ) as mock_wf:

        mock_opt.return_value = {"p": 1}
        report = MagicMock()
        report.metrics = {
            "net_profit": 5.0,
            "max_drawdown": -0.05,
            "trade_count": 1,
        }
        mock_bt.return_value.run.return_value = report
        mock_fs.return_value = 0.2
        mock_wf.return_value = WalkForwardResult(
            avg_profit=0.5,
            avg_drawdown=-0.05,
            profit_std=0.1,
            dd_std=0.05,
            negative_fold_ratio=0.0,
            raw_fitness=0.3,
            normalized_score=0.2,
        )

        wf = WalkForwardOptimizer(
            ticker="TEST",
            df=df,
            bounds={},
            train_window=3,
            test_window=2,
            step_size=3,
            verbose=False,
        )

        assert wf.run() is None


def test_run_walk_forward_uses_config_windows():
    df = _make_df(30)

    with patch("app.backtesting.walk_forward.load_data") as mock_load, patch(
        "app.backtesting.walk_forward.WalkForwardOptimizer"
    ) as mock_wf, patch(
        "app.backtesting.walk_forward.ensure_data_cached"
    ) as mock_cache, patch(
        "app.backtesting.walk_forward.Config"
    ) as mock_cfg:

        mock_load.return_value = df
        mock_cfg.START_DATE = "2025-01-01"
        mock_cfg.END_DATE = "2025-02-01"
        mock_cfg.TRAIN_WINDOW_MONTHS = 2
        mock_cfg.TEST_WINDOW_MONTHS = 1
        mock_cfg.WINDOW_STEP_MONTHS = 1
        mock_cfg.OPTIMIZER_POPULATION = 10
        mock_cfg.OPTIMIZER_GENERATIONS = 5

        mock_wf.return_value.run.return_value = {"ok": True}
        mock_cache.return_value = True

        result = run_walk_forward("TEST")

        assert result == {"ok": True}

        _, kwargs = mock_wf.call_args
        assert kwargs["train_window"] == 42
        assert kwargs["test_window"] == 21
        assert kwargs["step_size"] == 21
