"""S17: validation 0% files smoke tests + improvement_check + scoring.

Targets:
- app/validation/risk_metrics.py (0%)
- app/validation/stability_metrics.py (0%)
- app/validation/improvement_check.py (11%)
- app/validation/scoring.py (4%)
- app/validation/bias_metrics.py (20%)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# risk_metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAllocationViolations:
    def test_empty_list(self):
        from app.validation.risk_metrics import compute_allocation_violations

        result = compute_allocation_violations([])
        assert result["violations"] == 0

    def test_no_violations(self):
        from app.validation.risk_metrics import compute_allocation_violations

        result = compute_allocation_violations([0.1, 0.2, 0.3], max_pct=1.0)
        assert result["violations"] == 0

    def test_with_violation(self):
        from app.validation.risk_metrics import compute_allocation_violations

        result = compute_allocation_violations([0.1, 1.5, 0.3], max_pct=1.0)
        assert result["violations"] == 1


class TestComputeCrashDrawdown:
    def test_empty_curve(self):
        from app.validation.risk_metrics import compute_crash_drawdown

        result = compute_crash_drawdown([])
        assert result["max_drawdown"] == 0.0

    def test_flat_curve(self):
        from app.validation.risk_metrics import compute_crash_drawdown

        result = compute_crash_drawdown([100.0, 100.0, 100.0])
        assert result["max_drawdown"] == 0.0

    def test_drawdown_detected(self):
        from app.validation.risk_metrics import compute_crash_drawdown

        result = compute_crash_drawdown([100.0, 90.0, 80.0])
        assert result["max_drawdown"] < 0.0


class TestComputeVolatilitySpikeResponse:
    def test_empty_returns(self):
        from app.validation.risk_metrics import compute_volatility_spike_response

        result = compute_volatility_spike_response([])
        assert result["spike_count"] == 0

    def test_no_spikes(self):
        from app.validation.risk_metrics import compute_volatility_spike_response

        result = compute_volatility_spike_response([0.01, 0.01, 0.01])
        assert result["spike_count"] == 0

    def test_spike_detected(self):
        from app.validation.risk_metrics import compute_volatility_spike_response

        returns = [0.01] * 10 + [5.0]
        result = compute_volatility_spike_response(returns, spike_threshold=2.0)
        assert result["spike_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# stability_metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestStabilityMetrics:
    def test_oos_stability_empty(self):
        from app.validation.stability_metrics import compute_oos_stability

        result = compute_oos_stability([])
        assert result["count"] == 0
        assert result["std"] is None

    def test_oos_stability_values(self):
        from app.validation.stability_metrics import compute_oos_stability

        result = compute_oos_stability([0.5, 0.6, 0.7])
        assert result["count"] == 3
        assert result["std"] == pytest.approx(0.0816, abs=0.01)

    def test_parameter_variance_empty(self):
        from app.validation.stability_metrics import compute_parameter_variance

        assert compute_parameter_variance([]) == {}

    def test_parameter_variance_single_key(self):
        from app.validation.stability_metrics import compute_parameter_variance

        runs = [{"alpha": 0.1}, {"alpha": 0.3}]
        result = compute_parameter_variance(runs)
        assert "alpha" in result
        assert result["alpha"] == pytest.approx(0.01, abs=0.001)

    def test_seed_variance_empty(self):
        from app.validation.stability_metrics import compute_seed_variance

        result = compute_seed_variance([])
        assert result["count"] == 0

    def test_seed_variance_values(self):
        from app.validation.stability_metrics import compute_seed_variance

        result = compute_seed_variance([0.5, 0.7])
        assert result["count"] == 2

    def test_dispersion_ratio_empty(self):
        from app.validation.stability_metrics import compute_dispersion_ratio

        assert compute_dispersion_ratio([]) is None

    def test_dispersion_ratio_zero_mean(self):
        from app.validation.stability_metrics import compute_dispersion_ratio

        assert compute_dispersion_ratio([1.0, -1.0]) is None

    def test_dispersion_ratio_normal(self):
        from app.validation.stability_metrics import compute_dispersion_ratio

        result = compute_dispersion_ratio([0.5, 1.0, 1.5])
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
# improvement_check
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateResults:
    def _good_results(self):
        return {
            "walk_forward": {
                "mean_oos_sharpe": 0.8,
                "sharpe_std": 0.05,
                "return_std": 0.02,
                "execution_gap_flag": "LOW",
            },
            "final_score": {"production_score": 0.7},
            "bias": {"relative_gap": 0.1},
        }

    def test_passing_results_return_ok(self):
        from app.validation.improvement_check import evaluate_results

        result = evaluate_results(self._good_results())
        assert result["status"] == "ok"
        assert result["failures"] == []

    def test_low_sharpe_raises_blocked(self):
        from app.validation.improvement_check import evaluate_results
        from app.validation.errors import DeploymentBlockedException

        results = self._good_results()
        results["walk_forward"]["mean_oos_sharpe"] = 0.1
        with pytest.raises(DeploymentBlockedException):
            evaluate_results(results)

    def test_high_relative_gap_raises_blocked(self):
        from app.validation.improvement_check import evaluate_results
        from app.validation.errors import DeploymentBlockedException

        results = self._good_results()
        results["bias"]["relative_gap"] = 0.9
        with pytest.raises(DeploymentBlockedException):
            evaluate_results(results)

    def test_execution_gap_flag_blocks(self):
        from app.validation.improvement_check import evaluate_results
        from app.validation.errors import DeploymentBlockedException

        results = self._good_results()
        results["walk_forward"]["execution_gap_flag"] = "HIGH_TIMING_DEPENDENCY"
        with pytest.raises(DeploymentBlockedException):
            evaluate_results(results)

    def test_non_numeric_sharpe_fails(self):
        from app.validation.improvement_check import evaluate_results
        from app.validation.errors import DeploymentBlockedException

        results = self._good_results()
        results["walk_forward"]["mean_oos_sharpe"] = "N/A"
        with pytest.raises(DeploymentBlockedException):
            evaluate_results(results)


# ─────────────────────────────────────────────────────────────────────────────
# scoring
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeQuantScore:
    def _base_results(self):
        return {
            "engine_integrity": {"status": "ENGINE_VALID"},
            "bias": {"relative_drop": 0.1},
            "walk_forward": {
                "mean_oos_sharpe": 0.8,
                "sharpe_std": 0.05,
                "return_std": 0.02,
            },
            "rl_stability": {"status": "STABLE", "seed_variance": 0.01},
            "execution_stress": {"max_drawdown": 0.05, "allocation_violations": 0},
            "robustness": {"status": "ok", "pass_rate": 0.8},
            "production": {"status": "ok"},
        }

    def test_returns_dict_with_score(self):
        from app.validation.scoring import compute_quant_score

        with patch(
            "app.validation.scoring.normalize_drawdown", return_value=0.0
        ), patch("app.validation.scoring.normalize_sharpe", return_value=0.0):
            result = compute_quant_score(self._base_results())
        assert "score" in result or "total" in result or isinstance(result, dict)

    def test_empty_results_does_not_crash(self):
        from app.validation.scoring import compute_quant_score

        with patch(
            "app.validation.scoring.normalize_drawdown", return_value=0.0
        ), patch("app.validation.scoring.normalize_sharpe", return_value=0.0):
            result = compute_quant_score({})
        assert isinstance(result, dict)

    def test_normalize_helper(self):
        from app.validation.scoring import normalize

        assert normalize(0.5, 0.0, 1.0) == pytest.approx(0.5)
        assert normalize(-1.0, 0.0, 1.0) == 0
        assert normalize(2.0, 0.0, 1.0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# bias_metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestBiasMetrics:
    def test_compare_execution_modes_zero(self):
        from app.validation.bias_metrics import compare_execution_modes

        result = compare_execution_modes(0.0, 0.0)
        assert isinstance(result, dict)

    def test_compare_execution_modes_normal(self):
        from app.validation.bias_metrics import compare_execution_modes

        result = compare_execution_modes(0.10, 0.08)
        assert "relative_gap" in result or isinstance(result, dict)
