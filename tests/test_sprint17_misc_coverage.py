"""S17E: Coverage boost for backtesting small modules, weighting, and position_sizer.

This test file targets small zero/low coverage files that are trivial to cover:
- app/backtesting/transaction_costs.py (0%)
- app/backtesting/training_dataset.py (0%)
- app/backtesting/reward_engine.py (0%)
- app/backtesting/dataset_builder.py (0%)
- app/core/decision/weighting.py (26%)
- app/core/decision/position_sizer.py (28%)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# TransactionCostModel
# ─────────────────────────────────────────────────────────────────────────────


class TestTransactionCostModel:
    def test_defaults(self):
        from app.backtesting.transaction_costs import TransactionCostModel

        m = TransactionCostModel()
        assert m.commission == 0.001
        assert m.slippage == 0.0005

    def test_apply_reduces_return(self):
        from app.backtesting.transaction_costs import TransactionCostModel

        m = TransactionCostModel()
        result = m.apply(0.05)
        assert result == pytest.approx(0.05 - 0.001 - 0.0005)

    def test_apply_custom_costs(self):
        from app.backtesting.transaction_costs import TransactionCostModel

        m = TransactionCostModel(commission=0.002, slippage=0.001)
        result = m.apply(0.0)
        assert result == pytest.approx(-0.003)

    def test_negative_return_worsened(self):
        from app.backtesting.transaction_costs import TransactionCostModel

        m = TransactionCostModel()
        result = m.apply(-0.01)
        assert result < -0.01


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset builder
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildTrainingRow:
    def _row(self, ts="2024-01-05T00:00:00", action_code=1, confidence=0.8):
        return {
            "timestamp": ts,
            "ticker": "AAPL",
            "decision": {
                "action_code": action_code,
                "confidence": confidence,
                "wf_score": 0.7,
                "ensemble_quality": "STABLE",
            },
            "audit": {"confidence_bucket": "HIGH", "decision_level": "STRONG"},
        }

    def test_basic_row(self):
        from app.backtesting.training_dataset import build_training_row

        replay = {"raw_return": 0.03, "reward": 0.024}
        row = build_training_row(
            replay_row=replay,
            record=self._row(),
            overconfidence_cases=[],
        )
        assert row["ticker"] == "AAPL"
        assert row["confidence"] == 0.8
        assert row["overconfident"] is False

    def test_overconfident_flagged(self):
        from app.backtesting.training_dataset import build_training_row

        ts = "2024-01-05T00:00:00"
        replay = {"raw_return": -0.02, "reward": -0.01}
        row = build_training_row(
            replay_row=replay,
            record=self._row(ts=ts),
            overconfidence_cases=[{"timestamp": ts}],
        )
        assert row["overconfident"] is True

    def test_action_code_preserved(self):
        from app.backtesting.training_dataset import build_training_row

        replay = {"raw_return": 0.0, "reward": 0.0}
        row = build_training_row(
            replay_row=replay,
            record=self._row(action_code=2),
            overconfidence_cases=[],
        )
        assert row["action_code"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Reward engine
# ─────────────────────────────────────────────────────────────────────────────


class TestRealizeReward:
    def test_hold_returns_zero(self):
        from app.backtesting.reward_engine import realize_reward

        result = realize_reward(0.05, {"action_code": 0}, {})
        assert result == 0.0

    def test_buy_positive_return_scaled(self):
        from app.backtesting.reward_engine import realize_reward

        result = realize_reward(
            0.05, {"action_code": 1, "confidence": 0.8}, {"decision_level": "STRONG"}
        )
        assert result > 0.0

    def test_weak_decision_level_lower_reward(self):
        from app.backtesting.reward_engine import realize_reward

        strong = realize_reward(
            0.05, {"action_code": 1, "confidence": 0.9}, {"decision_level": "STRONG"}
        )
        weak = realize_reward(
            0.05, {"action_code": 1, "confidence": 0.9}, {"decision_level": "WEAK"}
        )
        assert strong > weak

    def test_no_trade_level_returns_zero(self):
        from app.backtesting.reward_engine import realize_reward

        result = realize_reward(
            0.1, {"action_code": 1, "confidence": 0.9}, {"decision_level": "NO_TRADE"}
        )
        assert result == 0.0

    def test_unknown_decision_level(self):
        from app.backtesting.reward_engine import realize_reward

        result = realize_reward(0.05, {"action_code": 1}, {"decision_level": "UNKNOWN"})
        assert isinstance(result, float)

    def test_confidence_clamp_low(self):
        from app.backtesting.reward_engine import realize_reward

        # confidence 0.1 is clamped to 0.3
        result = realize_reward(
            1.0, {"action_code": 1, "confidence": 0.1}, {"decision_level": "NORMAL"}
        )
        assert result > 0.0

    def test_sell_positive_return(self):
        from app.backtesting.reward_engine import realize_reward

        result = realize_reward(
            0.03, {"action_code": 2, "confidence": 0.7}, {"decision_level": "NORMAL"}
        )
        assert result > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder (builds on training_dataset and replay_runner)
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildTrainingDataset:
    def test_empty_inputs(self):
        from app.backtesting.dataset_builder import build_training_dataset

        result = build_training_dataset([], [], {"overconfidence_cases": []})
        assert result == []

    def test_builds_rows(self):
        from app.backtesting.dataset_builder import build_training_dataset

        record = {
            "timestamp": "2024-01-05T00:00:00",
            "ticker": "AAPL",
            "decision": {"action_code": 1, "confidence": 0.8},
            "audit": {},
        }
        replay_row = {"raw_return": 0.02, "reward": 0.016}
        result = build_training_dataset([replay_row], [record], {})
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"


# ─────────────────────────────────────────────────────────────────────────────
# Decision weighting
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDecisionWeight:
    def test_no_model_votes_returns_zero(self):
        from app.core.decision.weighting import compute_decision_weight

        result = compute_decision_weight({"confidence": 0.8, "model_votes": []})
        assert result == 0.0

    def test_stable_quality_higher_weight(self):
        from app.core.decision.weighting import compute_decision_weight

        stable = compute_decision_weight(
            {
                "confidence": 0.9,
                "avg_wf_score": 0.8,
                "ensemble_quality": "STABLE",
                "model_votes": [{"model_path": "m1"}],
            }
        )
        chaotic = compute_decision_weight(
            {
                "confidence": 0.9,
                "avg_wf_score": 0.8,
                "ensemble_quality": "CHAOTIC",
                "model_votes": [{"model_path": "m1"}],
            }
        )
        assert stable > chaotic

    def test_result_between_zero_and_one(self):
        from app.core.decision.weighting import compute_decision_weight

        result = compute_decision_weight(
            {
                "confidence": 1.0,
                "avg_wf_score": 1.0,
                "ensemble_quality": "STABLE",
                "model_votes": [{"model_path": "m1"}],
            }
        )
        assert 0.0 <= result <= 1.0

    def test_no_model_path_in_votes(self):
        from app.core.decision.weighting import compute_decision_weight

        # votes have no model_path → reliabilities list empty → avg_reliability=0.5
        result = compute_decision_weight(
            {
                "confidence": 0.8,
                "avg_wf_score": 0.5,
                "ensemble_quality": "MIXED",
                "model_votes": [{}],
            }
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_known_quality_penalty(self):
        from app.core.decision.weighting import (
            compute_decision_weight,
            ENSEMBLE_QUALITY_PENALTY,
        )

        assert "STABLE" in ENSEMBLE_QUALITY_PENALTY
        assert ENSEMBLE_QUALITY_PENALTY["STABLE"] == 1.0

    def test_missing_confidence_defaults_to_zero(self):
        from app.core.decision.weighting import compute_decision_weight

        result = compute_decision_weight(
            {
                "model_votes": [{}],
                "ensemble_quality": "STABLE",
            }
        )
        # confidence=0.0 → weight should be ~0
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PositionSizer
# ─────────────────────────────────────────────────────────────────────────────


class TestPositionSizer:
    def test_basic_compute(self):
        from app.core.decision.position_sizer import PositionSizer

        sizer = PositionSizer()
        result = sizer.compute(
            base_position_size=1000.0,
            confidence=0.8,
            wf_score=0.9,
            safety_discount=0.0,
            equity=10000.0,
        )
        assert result.final_size > 0.0
        assert result.capped is False

    def test_cap_at_max_pct(self):
        from app.core.decision.position_sizer import PositionSizer

        sizer = PositionSizer(max_position_pct=0.05)
        result = sizer.compute(
            base_position_size=9000.0,
            confidence=1.0,
            wf_score=1.0,
            safety_discount=0.0,
            equity=1000.0,
        )
        assert result.capped is True
        assert result.final_size <= 1000.0 * 0.05 + 0.01  # ~50, with float tolerance

    def test_safety_discount_reduces_size(self):
        from app.core.decision.position_sizer import PositionSizer

        sizer = PositionSizer()
        no_discount = sizer.compute(1000.0, 0.8, 0.8, 0.0, 50000.0)
        with_discount = sizer.compute(1000.0, 0.8, 0.8, 0.5, 50000.0)
        assert with_discount.final_size < no_discount.final_size

    def test_custom_params_override_defaults(self):
        from app.core.decision.position_sizer import PositionSizer

        sizer = PositionSizer(params={"P6_POSITION_MAX_PCT": 0.20})
        result = sizer.compute(5000.0, 1.0, 1.0, 0.0, 10000.0)
        assert result.capped is False or result.final_size <= 2000.0 + 0.01

    def test_result_dataclass_fields(self):
        from app.core.decision.position_sizer import PositionSizer, PositionSizingResult

        sizer = PositionSizer()
        result = sizer.compute(500.0, 0.7, 0.6, 0.1, 5000.0)
        assert hasattr(result, "confidence_factor")
        assert hasattr(result, "wf_factor")
        assert hasattr(result, "safety_factor")


class TestApplyPositionSizing:
    def test_non_buy_not_modified(self):
        from app.core.decision.position_sizer import apply_position_sizing

        item = {"decision": {"action_code": 0}, "allocation_amount": 1000.0}
        result = apply_position_sizing(item, equity=10000.0)
        assert result["allocation_amount"] == 1000.0
        assert "position_sizing" not in result

    def test_zero_allocation_not_modified(self):
        from app.core.decision.position_sizer import apply_position_sizing

        item = {"decision": {"action_code": 1}, "allocation_amount": 0.0}
        result = apply_position_sizing(item, equity=10000.0)
        assert result["allocation_amount"] == 0.0

    def test_buy_gets_sized(self):
        from app.core.decision.position_sizer import apply_position_sizing

        item = {
            "decision": {"action_code": 1, "confidence": 0.8, "wf_score": 0.75},
            "allocation_amount": 1000.0,
        }
        result = apply_position_sizing(item, equity=10000.0)
        assert "position_sizing" in result
        assert result["allocation_amount"] > 0.0

    def test_safety_override_applies_discount(self):
        from app.core.decision.position_sizer import apply_position_sizing

        item_no_override = {
            "decision": {
                "action_code": 1,
                "confidence": 0.8,
                "wf_score": 0.8,
                "safety_override": False,
            },
            "allocation_amount": 1000.0,
        }
        item_with_override = {
            "decision": {
                "action_code": 1,
                "confidence": 0.8,
                "wf_score": 0.8,
                "safety_override": True,
            },
            "allocation_amount": 1000.0,
        }
        no_override = apply_position_sizing(item_no_override, equity=100000.0)
        with_override = apply_position_sizing(item_with_override, equity=100000.0)
        assert with_override["allocation_amount"] < no_override["allocation_amount"]
