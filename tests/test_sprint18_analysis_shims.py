"""
S18 – Analysis shim module coverage.

Each shim in app/analysis/ exposes a wrapper class whose __init__ calls
_create_data_repository() and then super().__init__(...). This test file
instantiates every wrapper (with mocked dependencies) to cover those lines.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings():
    s = MagicMock()
    s.DB_PATH = ":memory:"
    return s


# ---------------------------------------------------------------------------
# confidence_calibrator.py
# ---------------------------------------------------------------------------


class TestConfidenceCalibratorShim:
    def test_instantiate_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.confidence_calibrator.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.confidence_calibrator.DataManager", return_value=mock_dm
            ),
        ):
            from app.analysis.confidence_calibrator import ConfidenceCalibrator

            obj = ConfidenceCalibrator()
            assert obj is not None

    def test_create_data_repository_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.confidence_calibrator.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.confidence_calibrator.DataManager", return_value=mock_dm
            ),
        ):
            import importlib
            import app.analysis.confidence_calibrator as mod

            importlib.reload(mod)
            result = mod._create_data_repository()
            assert result is not None

    def test_create_data_repository_type_error_fallback(self):
        """If DataManager(settings=...) raises TypeError, call DataManager() instead."""
        mock_settings = _mock_settings()
        mock_dm = MagicMock()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise TypeError("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.confidence_calibrator.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.confidence_calibrator.DataManager",
                side_effect=side_effect,
            ),
        ):
            import app.analysis.confidence_calibrator as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2  # first call raised, second succeeded


# ---------------------------------------------------------------------------
# decision_quality_analyzer.py
# ---------------------------------------------------------------------------


class TestDecisionQualityAnalyzerShim:
    def test_instantiate_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.decision_quality_analyzer.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.decision_quality_analyzer.DataManager",
                return_value=mock_dm,
            ),
        ):
            from app.analysis.decision_quality_analyzer import DecisionQualityAnalyzer

            obj = DecisionQualityAnalyzer()
            assert obj is not None

    def test_create_data_repository_exception_fallback(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise Exception("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.decision_quality_analyzer.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.decision_quality_analyzer.DataManager",
                side_effect=side_effect,
            ),
        ):
            import app.analysis.decision_quality_analyzer as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2


# ---------------------------------------------------------------------------
# safety_stress_tester.py
# ---------------------------------------------------------------------------


class TestSafetyStressTesterShim:
    def test_instantiate_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.safety_stress_tester.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.safety_stress_tester.DataManager", return_value=mock_dm
            ),
        ):
            from app.analysis.safety_stress_tester import SafetyStressTester

            obj = SafetyStressTester()
            assert obj is not None

    def test_create_data_repository_exception_fallback(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise Exception("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.safety_stress_tester.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.safety_stress_tester.DataManager", side_effect=side_effect
            ),
        ):
            import app.analysis.safety_stress_tester as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2


# ---------------------------------------------------------------------------
# validation_report_builder.py
# ---------------------------------------------------------------------------


class TestValidationReportBuilderShim:
    def test_instantiate_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.validation_report_builder.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.validation_report_builder.DataManager",
                return_value=mock_dm,
            ),
        ):
            from app.analysis.validation_report_builder import ValidationReportBuilder

            obj = ValidationReportBuilder()
            assert obj is not None

    def test_create_data_repository_exception_fallback(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise Exception("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.validation_report_builder.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.validation_report_builder.DataManager",
                side_effect=side_effect,
            ),
        ):
            import app.analysis.validation_report_builder as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2


# ---------------------------------------------------------------------------
# wf_stability_analyzer.py
# ---------------------------------------------------------------------------


class TestWfStabilityAnalyzerShim:
    def test_instantiate_success(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.wf_stability_analyzer.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.wf_stability_analyzer.DataManager", return_value=mock_dm
            ),
        ):
            from app.analysis.wf_stability_analyzer import WalkForwardStabilityAnalyzer

            obj = WalkForwardStabilityAnalyzer()
            assert obj is not None

    def test_create_data_repository_type_error_fallback(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise TypeError("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.wf_stability_analyzer.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "app.analysis.wf_stability_analyzer.DataManager",
                side_effect=side_effect,
            ),
        ):
            import app.analysis.wf_stability_analyzer as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2


# ---------------------------------------------------------------------------
# phase6_validator.py  (different __init__ signature)
# ---------------------------------------------------------------------------


class TestPhase6ValidatorShim:
    def test_instantiate_no_args(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.phase6_validator.get_settings", return_value=mock_settings
            ),
            patch("app.analysis.phase6_validator.DataManager", return_value=mock_dm),
        ):
            from app.analysis.phase6_validator import Phase6Validator

            obj = Phase6Validator()
            assert obj is not None

    def test_instantiate_with_dm_provided(self):
        """When dm is provided, _create_data_repository is not called."""
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch(
                "app.analysis.phase6_validator.get_settings", return_value=mock_settings
            ),
        ):
            from app.analysis.phase6_validator import Phase6Validator

            obj = Phase6Validator(dm=mock_dm, settings=mock_settings)
            assert obj is not None

    def test_instantiate_with_settings_provided(self):
        """When settings is passed, get_settings() is not called."""
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch("app.analysis.phase6_validator.DataManager", return_value=mock_dm),
        ):
            from app.analysis.phase6_validator import Phase6Validator

            obj = Phase6Validator(settings=mock_settings)
            assert obj is not None

    def test_create_data_repository_settings_arg(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        with (
            patch("app.analysis.phase6_validator.DataManager", return_value=mock_dm),
        ):
            import app.analysis.phase6_validator as mod

            result = mod._create_data_repository(settings=mock_settings)
            assert result is not None

    def test_create_data_repository_type_error_fallback(self):
        mock_settings = _mock_settings()
        mock_dm = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if "settings" in kwargs:
                raise TypeError("forced")
            return mock_dm

        with (
            patch(
                "app.analysis.phase6_validator.get_settings", return_value=mock_settings
            ),
            patch("app.analysis.phase6_validator.DataManager", side_effect=side_effect),
        ):
            import app.analysis.phase6_validator as mod

            result = mod._create_data_repository()
            assert call_count[0] == 2
