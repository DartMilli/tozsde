"""S17: Core analysis layer and infrastructure coverage boost.

Targets:
- app/core/analysis/explainability_linter.py (6% → 90%+)
- app/notifications/email_formatter.py (25% → 90%+)
- app/infrastructure/git_utils.py (29% → 100%)
- app/core/analysis/wf_stability_analyzer.py (19% → 60%+)
- app/core/analysis/decision_quality_analyzer.py (16% → 60%+)
- app/core/analysis/validation_report_builder.py (18% → 60%+)
- app/core/analysis/safety_stress_tester.py (22% → 60%+)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pandas as pd
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# explainability_linter.py — pure function
# ─────────────────────────────────────────────────────────────────────────────
class TestExplainabilityLinter:
    def _good_explanation(self):
        return {
            "hu": "Modellek szavazatai: BUY. BUY jel.",
            "en": "Model votes: BUY. BUY signal.",
            "meta": {
                "reasons_hu": ["ok1"],
                "reasons_en": ["ok2"],
            },
        }

    def test_valid_explanation_passes(self):
        from app.core.analysis.explainability_linter import lint_explanation

        result = lint_explanation(self._good_explanation(), {"action": "BUY"})
        assert result["ok"] is True
        assert result["issues"] == []

    def test_not_dict_fails(self):
        from app.core.analysis.explainability_linter import lint_explanation

        result = lint_explanation("not a dict", {})
        assert result["ok"] is False
        assert "explanation_not_dict" in result["issues"]

    def test_missing_hu_text(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        expl["hu"] = ""
        result = lint_explanation(expl, {})
        assert "missing_hu_text" in result["issues"]

    def test_missing_en_text(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        del expl["en"]
        result = lint_explanation(expl, {})
        assert "missing_en_text" in result["issues"]

    def test_action_missing_in_text(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        result = lint_explanation(expl, {"action": "SELL"})
        assert (
            "action_missing_hu" in result["issues"]
            or "action_missing_en" in result["issues"]
        )

    def test_empty_reasons_hu(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        expl["meta"]["reasons_hu"] = []
        result = lint_explanation(expl, {})
        assert "reasons_hu_empty" in result["issues"]

    def test_empty_reasons_en(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        expl["meta"]["reasons_en"] = []
        result = lint_explanation(expl, {})
        assert "reasons_en_empty" in result["issues"]

    def test_model_votes_missing(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = {
            "hu": "Valami szöveg.",
            "en": "Some text.",
            "meta": {"reasons_hu": ["r1"], "reasons_en": ["r2"]},
        }
        result = lint_explanation(expl, {})
        assert "model_votes_missing_en" in result["issues"]
        assert "model_votes_missing_hu" in result["issues"]

    def test_buy_action_without_size(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = self._good_explanation()
        result = lint_explanation(
            expl,
            {"action": "BUY", "action_code": 1},
            position_sizing={"final_size": 500},
        )
        # Size: missing from texts
        assert (
            "size_missing_en" in result["issues"]
            or "size_missing_hu" in result["issues"]
        )

    def test_buy_with_size_in_text(self):
        from app.core.analysis.explainability_linter import lint_explanation

        expl = {
            "hu": "Modellek szavazatai: BUY. Meret: 500.",
            "en": "Model votes: BUY. Size: 500.",
            "meta": {"reasons_hu": ["r1"], "reasons_en": ["r2"]},
        }
        result = lint_explanation(
            expl,
            {"action": "BUY", "action_code": 1},
            position_sizing={"final_size": 500},
        )
        assert "size_missing_en" not in result["issues"]
        assert "size_missing_hu" not in result["issues"]


# ─────────────────────────────────────────────────────────────────────────────
# email_formatter.py — pure functions
# ─────────────────────────────────────────────────────────────────────────────
class TestEmailFormatter:
    def _audit(self):
        return {
            "quality_score": 0.85,
            "confidence": 0.75,
            "wf_score": 0.90,
            "ensemble_quality": "GOOD",
            "model_count": 3,
        }

    def test_format_audit_line(self):
        from app.notifications.email_formatter import _format_audit_line

        line = _format_audit_line(self._audit())
        assert "Q=0.85" in line
        assert "Conf=0.75" in line
        assert "Models=3" in line

    def test_format_audit_line_none_wf(self):
        from app.notifications.email_formatter import _format_audit_line

        audit = self._audit()
        audit["wf_score"] = None
        line = _format_audit_line(audit)
        assert "n/a" in line

    def test_format_email_summary_hold(self):
        from app.notifications.email_formatter import format_email_summary

        result = format_email_summary(
            ticker="VOO",
            decision={"action": "HOLD", "action_code": 0},
            audit=self._audit(),
        )
        assert "VOO" in result
        assert "HOLD" in result

    def test_format_email_summary_buy_with_size(self):
        from app.notifications.email_formatter import format_email_summary

        result = format_email_summary(
            ticker="AAPL",
            decision={"action": "BUY", "action_code": 1},
            audit=self._audit(),
            position_sizing={"final_size": 1500.00},
        )
        assert "AAPL" in result
        assert "1,500.00" in result

    def test_format_email_summary_buy_no_size(self):
        from app.notifications.email_formatter import format_email_summary

        result = format_email_summary(
            ticker="AAPL",
            decision={"action": "BUY", "action_code": 1},
            audit=self._audit(),
            position_sizing={"final_size": None},
        )
        assert "BUY" in result

    def test_format_email_detail_hu(self):
        from app.notifications.email_formatter import format_email_detail

        explanation = {"hu": "Magyar szöveg", "en": "English text"}
        result = format_email_detail(explanation, self._audit(), lang="hu")
        assert "Magyar szöveg" in result
        assert "Q=0.85" in result

    def test_format_email_detail_en(self):
        from app.notifications.email_formatter import format_email_detail

        explanation = {"hu": "Magyar szöveg", "en": "English text"}
        result = format_email_detail(explanation, self._audit(), lang="en")
        assert "English text" in result

    def test_format_email_line_delegates(self):
        from app.notifications.email_formatter import format_email_line

        explanation = {"hu": "szöveg", "en": "text"}
        result = format_email_line(
            explanation, {"action": "HOLD"}, self._audit(), lang="en"
        )
        assert "text" in result


# ─────────────────────────────────────────────────────────────────────────────
# infrastructure/git_utils.py
# ─────────────────────────────────────────────────────────────────────────────
class TestGitUtils:
    def test_returns_commit_hash_on_success(self):
        from app.infrastructure.git_utils import get_git_commit

        with patch("app.infrastructure.git_utils.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "abc1234\n"
            mock_run.return_value = mock_result
            result = get_git_commit()
        assert result == "abc1234"

    def test_returns_unknown_on_exception(self):
        from app.infrastructure.git_utils import get_git_commit

        with patch(
            "app.infrastructure.git_utils.subprocess.run", side_effect=OSError("no git")
        ):
            result = get_git_commit()
        assert result == "unknown"

    def test_returns_unknown_on_empty_output(self):
        from app.infrastructure.git_utils import get_git_commit

        with patch("app.infrastructure.git_utils.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""
            mock_run.return_value = mock_result
            result = get_git_commit()
        assert result == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# WalkForwardStabilityAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class TestWFStabilityAnalyzer:
    def _make_analyzer(self):
        from app.core.analysis.wf_stability_analyzer import WalkForwardStabilityAnalyzer

        settings = MagicMock()
        dm = MagicMock()
        return WalkForwardStabilityAnalyzer(settings=settings, data_manager=dm), dm

    def test_no_results_returns_no_data(self):
        analyzer, dm = self._make_analyzer()
        with patch.object(analyzer, "_load_results", return_value=[]):
            result = analyzer.analyze("VOO")
        assert result["status"] == "no_data"
        dm.save_wf_stability_metrics.assert_called_once()

    def test_with_results_returns_metrics(self):
        analyzer, dm = self._make_analyzer()
        results = [
            {"best_params": {"sma_fast": 10, "sma_slow": 20}, "raw_fitness": 0.8},
            {"best_params": {"sma_fast": 12, "sma_slow": 22}, "raw_fitness": 0.75},
        ]
        with patch.object(analyzer, "_load_results", return_value=results):
            result = analyzer.analyze("VOO")
        assert result["ticker"] == "VOO"
        assert "wf_score_std" in result
        assert "param_variance" in result


# ─────────────────────────────────────────────────────────────────────────────
# DecisionQualityAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class TestDecisionQualityAnalyzer:
    def _make_analyzer(self):
        from app.core.analysis.decision_quality_analyzer import DecisionQualityAnalyzer

        settings = MagicMock()
        dm = MagicMock()
        return DecisionQualityAnalyzer(settings=settings, data_manager=dm), dm

    def test_empty_data_returns_no_data(self):
        analyzer, dm = self._make_analyzer()
        with patch.object(analyzer, "_load_data", return_value=pd.DataFrame()):
            result = analyzer.analyze("VOO")
        assert result["status"] == "no_data"

    def test_with_data_calls_save(self):
        analyzer, dm = self._make_analyzer()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "confidence": [0.5, 0.7, 0.8, 0.6],
                "success": [1, 0, 1, 1],
                "action_code": [1, 2, 1, 0],
                "pnl_pct": [0.02, -0.01, 0.03, 0.01],
                "safety_override": [0, 0, 1, 0],
            }
        )
        with patch.object(analyzer, "_load_data", return_value=df):
            result = analyzer.analyze("VOO")
        assert isinstance(result, dict)
        dm.save_decision_quality_metrics.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# ValidationReportBuilder
# ─────────────────────────────────────────────────────────────────────────────
class TestValidationReportBuilder:
    def _make_builder(self):
        from app.core.analysis.validation_report_builder import ValidationReportBuilder

        settings = MagicMock()
        dm = MagicMock()
        # _latest calls dm.connection().__enter__() etc.
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock())
        ctx.__exit__ = MagicMock(return_value=False)
        dm.connection.return_value = ctx
        dm.connection().__enter__().execute().fetchone.return_value = None
        return ValidationReportBuilder(settings=settings, data_manager=dm), dm

    def test_build_returns_report_dict(self):
        builder, dm = self._make_builder()
        with patch(
            "app.core.analysis.validation_report_builder.get_git_commit",
            return_value="abc",
        ):
            result = builder.build()
        assert "generated_at" in result
        assert "git_commit" in result
        dm.save_validation_report.assert_called_once()

    def test_fetch_latest_calls_dm(self):
        builder, dm = self._make_builder()
        dm.fetch_latest_validation_report.return_value = {}
        builder.fetch_latest()
        dm.fetch_latest_validation_report.assert_called_once()

    def test_export_json(self):
        builder, dm = self._make_builder()
        report = {"generated_at": "2024T00:00:00Z", "git_commit": "abc"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fpath = f.name
        try:
            builder.export(report, fpath, fmt="json")
            with open(fpath, "r") as r:
                loaded = json.load(r)
            assert "git_commit" in loaded
        finally:
            os.unlink(fpath)

    def test_export_md(self):
        builder, dm = self._make_builder()
        report = {
            "generated_at": "2024T00:00:00Z",
            "git_commit": "abc",
            "decision_quality": None,
            "decision_effectiveness": None,
            "confidence_calibration": None,
            "wf_stability": None,
            "safety_stress": None,
        }
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            fpath = f.name
        try:
            builder.export(report, fpath, fmt="md")
            with open(fpath, "r") as r:
                content = r.read()
            assert "# Validation Report" in content
        finally:
            os.unlink(fpath)

    def test_export_invalid_fmt_raises(self):
        builder, dm = self._make_builder()
        with pytest.raises(ValueError):
            builder.export({}, "/tmp/x.txt", fmt="csv")

    def test_to_markdown_structure(self):
        builder, dm = self._make_builder()
        report = {
            "generated_at": "2024T",
            "git_commit": "abc",
            "decision_quality": None,
            "decision_effectiveness": None,
            "confidence_calibration": None,
            "wf_stability": None,
            "safety_stress": None,
        }
        md = builder.to_markdown(report)
        assert "# Validation Report" in md
        assert "Git commit: abc" in md


# ─────────────────────────────────────────────────────────────────────────────
# SafetyStressTester
# ─────────────────────────────────────────────────────────────────────────────
class TestSafetyStressTester:
    def _make_tester(self):
        from app.core.analysis.safety_stress_tester import SafetyStressTester

        settings = MagicMock()
        dm = MagicMock()
        return SafetyStressTester(settings=settings, data_manager=dm), dm

    def test_no_decisions_returns_no_data(self):
        tester, dm = self._make_tester()
        with patch.object(tester, "_load_decisions", return_value=pd.DataFrame()):
            result = tester.run("VOO", "2022-01-01", "2023-12-31")
        assert result["status"] == "no_data"
        dm.save_safety_stress_results.assert_called_once()

    def test_no_ohlcv_returns_no_market_data(self):
        tester, dm = self._make_tester()
        dec_df = pd.DataFrame(
            {"id": [1], "timestamp": ["2022-01-02"], "action_code": [1]}
        )
        with patch.object(tester, "_load_decisions", return_value=dec_df):
            dm.load_ohlcv.return_value = pd.DataFrame()
            result = tester.run("VOO", "2022-01-01", "2022-12-31")
        assert result["status"] == "no_market_data"

    def test_run_returns_scenario_results(self):
        tester, dm = self._make_tester()
        dec_df = pd.DataFrame(
            {
                "id": range(5),
                "timestamp": pd.date_range("2022-01-03", periods=5, freq="B").astype(
                    str
                ),
                "action_code": [1, 0, 2, 1, 0],
            }
        )
        ohlcv_df = pd.DataFrame(
            {
                "Open": [100.0] * 50,
                "High": [101.0] * 50,
                "Low": [99.0] * 50,
                "Close": [100.0 + i * 0.1 for i in range(50)],
                "Volume": [1_000_000] * 50,
            },
            index=pd.date_range("2022-01-01", periods=50, freq="B"),
        )
        with patch.object(tester, "_load_decisions", return_value=dec_df):
            dm.load_ohlcv.return_value = ohlcv_df
            result = tester.run(
                "VOO", "2022-01-03", "2022-01-10", scenario="elevated_volatility"
            )
        assert "scenario" in result

    def test_run_scenario_gap_days(self):
        tester, dm = self._make_tester()
        dec_df = pd.DataFrame(
            {
                "id": range(3),
                "timestamp": pd.date_range("2022-01-03", periods=3, freq="B").astype(
                    str
                ),
                "action_code": [1, 0, 2],
            }
        )
        ohlcv_df = pd.DataFrame(
            {
                "Close": [100.0, 105.0, 90.0] * 10,
                "Open": [100.0] * 30,
                "High": [105.0] * 30,
                "Low": [90.0] * 30,
                "Volume": [1_000_000] * 30,
            },
            index=pd.date_range("2022-01-01", periods=30, freq="B"),
        )
        with patch.object(tester, "_load_decisions", return_value=dec_df):
            dm.load_ohlcv.return_value = ohlcv_df
            result = tester.run("VOO", "2022-01-03", "2022-01-05", scenario="gap_days")
        assert result["scenario"] == "gap_days"

    def test_run_scenario_drawdown_injection(self):
        tester, dm = self._make_tester()
        dec_df = pd.DataFrame(
            {
                "id": range(6),
                "timestamp": pd.date_range("2022-01-03", periods=6, freq="B").astype(
                    str
                ),
                "action_code": [1, 0, 2, 1, 0, 1],
            }
        )
        ohlcv_df = pd.DataFrame(
            {
                "Close": [100.0 + i for i in range(40)],
                "Open": [99.0 + i for i in range(40)],
                "High": [101.0 + i for i in range(40)],
                "Low": [98.0 + i for i in range(40)],
                "Volume": [1_000_000] * 40,
            },
            index=pd.date_range("2022-01-01", periods=40, freq="B"),
        )
        with patch.object(tester, "_load_decisions", return_value=dec_df):
            dm.load_ohlcv.return_value = ohlcv_df
            result = tester.run(
                "VOO", "2022-01-03", "2022-01-10", scenario="drawdown_injection"
            )
        assert result["scenario"] == "drawdown_injection"
