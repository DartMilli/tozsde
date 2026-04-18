"""
S18 – Coverage for backtesting/execution_utils.py (pure functions).
Target lines: 28-29, 34, 37-40, 43, 47, 55, 61, 64, 68, 71
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# seed_deterministic
# ---------------------------------------------------------------------------


class TestSeedDeterministic:
    def test_sets_numpy_seed(self):
        from app.backtesting.execution_utils import seed_deterministic

        seed_deterministic(7)
        # Check reproducibility
        a = np.random.rand()
        seed_deterministic(7)
        b = np.random.rand()
        assert a == b

    def test_default_seed(self):
        from app.backtesting.execution_utils import seed_deterministic

        seed_deterministic()  # should not raise

    def test_calls_torch_manual_seed_when_available(self):
        mock_torch = MagicMock()
        with patch("app.backtesting.execution_utils.torch", mock_torch):
            from app.backtesting.execution_utils import seed_deterministic

            seed_deterministic(42)
            mock_torch.manual_seed.assert_called_once_with(42)

    def test_handles_torch_exception(self):
        mock_torch = MagicMock()
        mock_torch.manual_seed.side_effect = RuntimeError("torch error")
        with patch("app.backtesting.execution_utils.torch", mock_torch):
            from app.backtesting.execution_utils import seed_deterministic

            seed_deterministic(42)  # should not raise


# ---------------------------------------------------------------------------
# normalize_action
# ---------------------------------------------------------------------------


class TestNormalizeAction:
    def test_none_returns_0(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(None) == 0

    def test_torch_tensor(self):
        """Test torch.Tensor path using real torch when available."""
        try:
            import torch as real_torch

            # Re-import the module to ensure torch is set correctly
            import importlib
            import app.backtesting.execution_utils as mod

            importlib.reload(mod)
            tensor = real_torch.tensor(2)
            result = mod.normalize_action(tensor)
            assert result == 2
        except ImportError:
            pytest.skip("torch not available")

    def test_torch_tensor_item_exception_fallback(self):
        """Test detach().cpu().item() fallback path."""
        try:
            import torch
            from app.backtesting.execution_utils import normalize_action

            tensor = torch.tensor(1)
            # Patch item to raise, ensuring fallback works
            with patch.object(tensor, "item", side_effect=Exception("err")):
                result = normalize_action(tensor)
                # fallback detach().cpu().item() should return 1
                assert result == 1
        except ImportError:
            pytest.skip("torch not available")

    def test_numpy_integer(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(np.int32(2)) == 2

    def test_numpy_integer_hold(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(np.int64(0)) == 0

    def test_plain_int(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(1) == 1

    def test_plain_int_unknown(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(99) == 0

    def test_float_integer_value(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action(1.0) == 1

    def test_float_non_integer(self):
        from app.backtesting.execution_utils import normalize_action

        # 1.5 is not .is_integer(), falls through to return 0
        assert normalize_action(1.5) == 0

    def test_string_buy(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action("BUY") == 1

    def test_string_sell(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action("SELL") == 2

    def test_string_hold(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action("HOLD") == 0

    def test_string_lowercase(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action("buy") == 1

    def test_string_unknown(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action("UNKNOWN") == 0

    def test_other_type_returns_0(self):
        from app.backtesting.execution_utils import normalize_action

        assert normalize_action([1, 2]) == 0


# ---------------------------------------------------------------------------
# resolve_execution_price
# ---------------------------------------------------------------------------


class TestResolveExecutionPrice:
    def _make_df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0],
                "Open": [99.0, 100.5, 101.5],
            }
        )

    def test_close_to_close(self):
        from app.backtesting.execution_utils import resolve_execution_price

        df = self._make_df()
        result = resolve_execution_price(df, 1, "close_to_close")
        assert result == 101.0

    def test_next_open(self):
        from app.backtesting.execution_utils import resolve_execution_price

        df = self._make_df()
        result = resolve_execution_price(df, 0, "next_open")
        assert result == 100.5

    def test_next_open_out_of_bounds(self):
        from app.backtesting.execution_utils import resolve_execution_price

        df = self._make_df()
        result = resolve_execution_price(df, 2, "next_open")  # last idx
        assert result is None

    def test_invalid_policy_defaults_to_next_open(self):
        from app.backtesting.execution_utils import resolve_execution_price

        df = self._make_df()
        result = resolve_execution_price(df, 0, "invalid_policy")
        assert result == 100.5

    def test_none_policy_defaults_to_next_open(self):
        from app.backtesting.execution_utils import resolve_execution_price

        df = self._make_df()
        result = resolve_execution_price(df, 0, None)
        assert result == 100.5

    def test_next_open_without_open_column(self):
        import pandas as pd
        from app.backtesting.execution_utils import resolve_execution_price

        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
        result = resolve_execution_price(df, 0, "next_open")
        assert result == 101.0


# ---------------------------------------------------------------------------
# go_live_metrics shim functions
# ---------------------------------------------------------------------------


class TestGoLiveMetricsShim:
    def test_compute_drawdown_summary(self):
        mock_settings = MagicMock()
        mock_dm = MagicMock()
        mock_dm.DB_PATH = ":memory:"
        mock_result = {"max_drawdown": -0.1}
        with (
            patch(
                "app.analysis.go_live_metrics.get_settings", return_value=mock_settings
            ),
            patch("app.analysis.go_live_metrics.DataManager", return_value=mock_dm),
            patch(
                "app.analysis.go_live_metrics.core_compute_drawdown_summary",
                return_value=mock_result,
            ),
        ):
            from app.analysis import go_live_metrics as mod

            result = mod.compute_drawdown_summary("VOO")
            assert result == mock_result

    def test_compute_loss_streak(self):
        mock_settings = MagicMock()
        mock_dm = MagicMock()
        mock_dm.DB_PATH = ":memory:"
        mock_result = {"max_streak": 3}
        with (
            patch(
                "app.analysis.go_live_metrics.get_settings", return_value=mock_settings
            ),
            patch("app.analysis.go_live_metrics.DataManager", return_value=mock_dm),
            patch(
                "app.analysis.go_live_metrics.core_compute_loss_streak",
                return_value=mock_result,
            ),
        ):
            from app.analysis import go_live_metrics as mod

            result = mod.compute_loss_streak("VOO")
            assert result == mock_result


# ---------------------------------------------------------------------------
# ExecutionCoordinator (2 uncovered lines: position sizing branch + execute_finalized)
# ---------------------------------------------------------------------------


class TestExecutionCoordinator:
    def _make_pipeline(self, *, enable_position_sizing=False):
        pipeline = MagicMock()
        cfg = MagicMock()
        cfg.ENABLE_POSITION_SIZING = enable_position_sizing
        cfg.INITIAL_CAPITAL = 10000.0
        pipeline._get_settings.return_value = cfg
        pipeline.allocate_capital.return_value = [
            {
                "decision": {"action": "BUY"},
                "payload": {"timestamp": "2025-01-15", "ticker": "VOO"},
            }
        ]
        pipeline.state_repo.fetch_latest_portfolio_state.return_value = {
            "equity": 20000.0
        }
        pipeline.settings = cfg
        return pipeline

    def test_split_and_finalize_with_position_sizing(self):
        from app.application.use_cases.execution_coordinator import ExecutionCoordinator

        pipeline = self._make_pipeline(enable_position_sizing=True)
        with patch(
            "app.application.use_cases.execution_coordinator.apply_position_sizing",
            return_value={"sized": True},
        ) as mock_sizer:
            coord = ExecutionCoordinator(pipeline)
            candidates = [
                {
                    "decision": {"no_trade": False, "action": "BUY"},
                    "payload": {"ticker": "VOO"},
                }
            ]
            no_trade, finalized = coord.split_and_finalize(candidates)
            mock_sizer.assert_called_once()

    def test_execute_finalized_with_decisions(self):
        from app.application.use_cases.execution_coordinator import ExecutionCoordinator

        pipeline = self._make_pipeline()
        coord = ExecutionCoordinator(pipeline)
        decisions = [{"decision": {}, "payload": {"timestamp": "2025-01-15"}}]
        coord.execute_finalized(decisions)
        pipeline.execute_trades.assert_called_once()

    def test_execute_finalized_empty_returns_early(self):
        from app.application.use_cases.execution_coordinator import ExecutionCoordinator

        pipeline = self._make_pipeline()
        coord = ExecutionCoordinator(pipeline)
        coord.execute_finalized(
            []
        )  # should return early without calling execute_trades
        pipeline.execute_trades.assert_not_called()
