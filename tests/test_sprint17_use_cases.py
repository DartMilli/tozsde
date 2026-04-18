"""S17B: Use-case coverage boost.

Targets:
- RunMonthlyRetrainingUseCase (was 33%)
- RunWeeklyReliabilityUseCase (was 41%)
- DailyPipelineUseCase (selected branches, was 36%)
- RunGovernanceUseCase init / error path (was 10%)
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch, call
import pytest
from dataclasses import replace

from app.application.use_cases.run_monthly_retraining import RunMonthlyRetrainingUseCase
from app.application.use_cases.run_weekly_reliability import RunWeeklyReliabilityUseCase
from app.application.use_cases.daily_pipeline_use_case import DailyPipelineUseCase
from app.application.use_cases.result import UseCaseResult


def _ok(result) -> bool:
    return result["status"] == "ok"


def _data(result) -> dict:
    return result["data"]


def _code(result) -> str:
    return result.get("meta", {}).get(
        "code", result.get("error", {}).get("message", "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# RunMonthlyRetrainingUseCase
# ─────────────────────────────────────────────────────────────────────────────


class TestRunMonthlyRetrainingUseCase:
    def _settings(self, test_settings, enable_rl=False):
        return replace(test_settings, ENABLE_RL=enable_rl)

    def test_run_dry_run_skips_training(self, test_settings):
        """dry_run=True; RL enabled but training should NOT be called."""
        settings = self._settings(test_settings, enable_rl=True)
        train_fn = MagicMock()
        wf_fn = MagicMock(return_value={"raw_fitness": 0.6})

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA"],
            walk_forward_fn=wf_fn,
            train_rl_fn=train_fn,
        )
        result = uc.run(dry_run=True)

        assert _ok(result)
        assert _data(result)["processed"] == 1
        assert _data(result)["trained"] == 0
        train_fn.assert_not_called()

    def test_run_with_rl_enabled_trains(self, test_settings):
        """RL enabled, not dry_run → training called."""
        settings = self._settings(test_settings, enable_rl=True)
        train_fn = MagicMock()
        wf_fn = MagicMock(return_value={"raw_fitness": 0.7})

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA", "BBB"],
            walk_forward_fn=wf_fn,
            train_rl_fn=train_fn,
        )
        result = uc.run(dry_run=False)

        assert _ok(result)
        assert _data(result)["processed"] == 2
        assert _data(result)["trained"] == 2
        assert train_fn.call_count == 2

    def test_run_rl_disabled_skips_training(self, test_settings):
        """ENABLE_RL=False; training should never be called."""
        settings = self._settings(test_settings, enable_rl=False)
        train_fn = MagicMock()
        wf_fn = MagicMock(return_value={"raw_fitness": 0.5})

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA"],
            walk_forward_fn=wf_fn,
            train_rl_fn=train_fn,
        )
        result = uc.run(dry_run=False)

        assert _data(result)["trained"] == 0
        assert _data(result)["rl_enabled"] is False
        train_fn.assert_not_called()

    def test_run_walk_forward_returns_none_skips(self, test_settings):
        """walk_forward returns None/empty → ticker skipped."""
        settings = self._settings(test_settings, enable_rl=True)
        train_fn = MagicMock()
        wf_fn = MagicMock(return_value=None)

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA"],
            walk_forward_fn=wf_fn,
            train_rl_fn=train_fn,
        )
        result = uc.run(dry_run=False)

        assert _data(result)["skipped"] == 1
        assert _data(result)["processed"] == 0
        train_fn.assert_not_called()

    def test_run_wf_score_from_normalized_score(self, test_settings):
        """normalized_score already present in wf_summary → used directly."""
        settings = self._settings(test_settings, enable_rl=True)
        train_fn = MagicMock()
        wf_fn = MagicMock(return_value={"normalized_score": 0.8})

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA"],
            walk_forward_fn=wf_fn,
            train_rl_fn=train_fn,
        )
        result = uc.run(dry_run=False)

        assert _data(result)["processed"] == 1
        # Confirm train called with wf_score=0.8
        call_kwargs = train_fn.call_args.kwargs
        assert call_kwargs.get("wf_score") == 0.8

    def test_run_exception_in_ticker_counted_as_failed(self, test_settings):
        """Exception during processing → failed count incremented."""
        settings = self._settings(test_settings, enable_rl=False)
        wf_fn = MagicMock(side_effect=RuntimeError("network error"))

        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: ["AAA", "BBB"],
            walk_forward_fn=wf_fn,
            train_rl_fn=MagicMock(),
        )
        result = uc.run(dry_run=False)

        assert _data(result)["failed"] == 2
        assert _data(result)["processed"] == 0

    def test_run_empty_provider(self, test_settings):
        """No tickers → processed=0."""
        settings = self._settings(test_settings)
        uc = RunMonthlyRetrainingUseCase(
            settings=settings,
            ticker_provider=lambda: [],
            walk_forward_fn=MagicMock(),
            train_rl_fn=MagicMock(),
        )
        result = uc.run()
        assert _data(result)["processed"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# RunWeeklyReliabilityUseCase
# ─────────────────────────────────────────────────────────────────────────────


class TestRunWeeklyReliabilityUseCase:
    def _make_uc(
        self, test_settings, tickers=("AAA",), scores=None, save_fn=None, analyzer=None
    ):
        if scores is None:
            scores = {"reliability_score": 0.9}
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = scores
        analyzer_factory = MagicMock(return_value=mock_analyzer)
        save_scores_fn = save_fn or MagicMock()

        return (
            RunWeeklyReliabilityUseCase(
                settings=test_settings,
                ticker_provider=lambda: list(tickers),
                analyzer_factory=analyzer_factory,
                save_scores_fn=save_scores_fn,
            ),
            mock_analyzer,
            save_scores_fn,
        )

    def test_run_saves_scores_normally(self, test_settings):
        uc, analyzer, save_fn = self._make_uc(test_settings, tickers=["AAA", "BBB"])
        result = uc.run(dry_run=False)

        assert _ok(result)
        assert _data(result)["processed"] == 2
        assert _data(result)["saved"] == 2
        assert save_fn.call_count == 2

    def test_run_dry_run_does_not_save(self, test_settings):
        uc, analyzer, save_fn = self._make_uc(test_settings)
        result = uc.run(dry_run=True)

        assert _data(result)["saved"] == 0
        save_fn.assert_not_called()

    def test_run_analyzer_exception_counts_failed(self, test_settings):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = ValueError("no data")
        uc = RunWeeklyReliabilityUseCase(
            settings=test_settings,
            ticker_provider=lambda: ["AAA"],
            analyzer_factory=MagicMock(return_value=mock_analyzer),
            save_scores_fn=MagicMock(),
        )
        result = uc.run()

        assert _data(result)["failed"] == 1
        assert _data(result)["processed"] == 0

    def test_run_empty_tickers(self, test_settings):
        uc, _, _ = self._make_uc(test_settings, tickers=[])
        result = uc.run()
        assert _data(result)["processed"] == 0
        assert _data(result)["saved"] == 0

    def test_result_contains_period_dates(self, test_settings):
        uc, _, _ = self._make_uc(test_settings)
        result = uc.run(dry_run=True)
        assert "period_start" in _data(result)
        assert "period_end" in _data(result)


# ─────────────────────────────────────────────────────────────────────────────
# DailyPipelineUseCase (selected branches)
# ─────────────────────────────────────────────────────────────────────────────


class TestDailyPipelineUseCaseBranches:
    def _make_pipeline_mock(self, tickers=("AAA",), candidate=None, settings=None):
        pipeline = MagicMock()
        pipeline.get_tickers_to_process.return_value = list(tickers)
        if candidate is None:
            candidate = {
                "ticker": "AAA",
                "decision": {"action": "HOLD", "action_code": 0},
                "payload": {"current_price": 100.0},
                "allocation_amount": 0.0,
                "allocation_pct": 0.0,
            }
        pipeline.build_daily_candidate.return_value = candidate
        pipeline.state_repo.has_decision_for_date.return_value = False
        pipeline._get_settings.return_value = settings or MagicMock(
            ENABLE_REBALANCER=False
        )
        pipeline._log_go_live_metrics.return_value = None
        pipeline.check_portfolio_drift.return_value = {"should_rebalance": False}
        return pipeline

    def test_run_no_candidates_returns_ok(self, test_settings):
        pipeline = self._make_pipeline_mock()
        pipeline.build_daily_candidate.side_effect = RuntimeError("no data")
        uc = DailyPipelineUseCase(pipeline)
        result = uc.run(dry_run=True)
        assert _ok(result)
        assert _data(result)["processed"] == 0

    def test_run_with_hold_candidate(self, test_settings):
        candidate = {
            "ticker": "AAA",
            "decision": {"action": "HOLD", "action_code": 0},
            "payload": {"current_price": 100.0},
            "allocation_amount": 0.0,
            "allocation_pct": 0.0,
            "decision_source": None,
        }
        pipeline = self._make_pipeline_mock(candidate=candidate)
        uc = DailyPipelineUseCase(pipeline)
        result = uc.run(dry_run=True)
        assert _ok(result)
        assert _data(result)["processed"] == 1

    def test_run_skips_already_decided_ticker(self, test_settings):
        pipeline = self._make_pipeline_mock()
        pipeline.state_repo.has_decision_for_date.return_value = True
        uc = DailyPipelineUseCase(pipeline)
        result = uc.run(dry_run=True)
        assert _data(result)["processed"] == 0

    def test_run_rebalancer_disabled_skips_check(self, test_settings):
        cfg = MagicMock(ENABLE_REBALANCER=False)
        pipeline = self._make_pipeline_mock(settings=cfg)
        candidate = {
            "ticker": "AAA",
            "decision": {"action": "HOLD", "action_code": 0},
            "payload": {"current_price": 100.0},
            "allocation_amount": 0.0,
            "allocation_pct": 0.0,
            "decision_source": None,
        }
        pipeline.build_daily_candidate.return_value = candidate
        uc = DailyPipelineUseCase(pipeline)
        uc.run(dry_run=True)
        pipeline.check_portfolio_drift.assert_not_called()

    def test_collect_candidates_error_alerter_in_live_mode(self, test_settings):
        """Non-dry-run: ErrorAlerter.alert should be called on exception."""
        pipeline = self._make_pipeline_mock()
        pipeline.build_daily_candidate.side_effect = RuntimeError("bad data")
        uc = DailyPipelineUseCase(pipeline)
        with patch(
            "app.application.use_cases.daily_pipeline_use_case.ErrorAlerter"
        ) as mock_alerter:
            uc.run(dry_run=False)
            mock_alerter.alert.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# RunGovernanceUseCase (init and mode routing)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunGovernanceUseCaseInit:
    def test_init_stores_settings(self, test_settings):
        from app.application.use_cases.run_governance import RunGovernanceUseCase

        uc = RunGovernanceUseCase(settings=test_settings)
        assert uc.settings is test_settings

    def _patch_governance(self, fake_summary, exit_code=0):
        """Context manager that patches all heavy governance internals."""
        return (
            patch("app.governance.set_settings"),
            patch(
                "app.governance.quant_runner._run_tests",
                return_value={"all_passed": True},
            ),
            patch(
                "app.governance.quant_runner._run_diagnostics",
                return_value={"pipeline_audit": {}},
            ),
            patch("app.governance.quant_runner._run_validation", return_value={}),
            patch(
                "app.governance.quant_runner._build_summary", return_value=fake_summary
            ),
            patch("app.governance.quant_runner._apply_collapse_stage"),
            patch("app.governance.quant_runner._exit_code", return_value=exit_code),
            patch("app.governance.quant_runner._git_commit", return_value="abc123"),
            patch(
                "app.governance.quant_runner._configure_logging",
                return_value=MagicMock(),
            ),
            patch(
                "app.governance.checklist_runner.evaluate_checklist", return_value={}
            ),
            patch(
                "app.reporting.report_builder.prepare_report_dir",
                return_value=MagicMock(),
            ),
            patch("app.reporting.report_builder.write_report_bundle"),
            patch("app.validation.utils.get_validation_ticker", return_value="VOO"),
            patch(
                "app.reporting.report_schema.now_timestamp",
                return_value="20260101T000000Z",
            ),
        )

    def test_run_tests_mode_calls_run_tests(self, test_settings):
        """Mode 'tests' should call _run_tests and succeed."""
        from app.application.use_cases.run_governance import RunGovernanceUseCase

        uc = RunGovernanceUseCase(settings=test_settings)
        fake_summary = MagicMock()
        fake_summary.status = "PASS"
        fake_summary.to_dict.return_value = {}

        patches = self._patch_governance(fake_summary, exit_code=0)
        with patches[0], patches[1] as mock_tests, patches[2], patches[3], patches[
            4
        ], patches[5], patches[6], patches[7], patches[8], patches[9], patches[
            10
        ], patches[
            11
        ], patches[
            12
        ], patches[
            13
        ]:
            result = uc.run(mode="tests")

        assert _ok(result)

    def test_run_diagnostics_mode(self, test_settings):
        """Mode 'diagnostics' should succeed."""
        from app.application.use_cases.run_governance import RunGovernanceUseCase

        uc = RunGovernanceUseCase(settings=test_settings)
        fake_summary = MagicMock()
        fake_summary.status = "PASS"
        fake_summary.to_dict.return_value = {}

        patches = self._patch_governance(fake_summary, exit_code=0)
        with patches[0], patches[1], patches[2] as mock_diag, patches[
            3
        ] as mock_val, patches[4], patches[5], patches[6], patches[7], patches[
            8
        ], patches[
            9
        ], patches[
            10
        ], patches[
            11
        ], patches[
            12
        ], patches[
            13
        ]:
            result = uc.run(mode="diagnostics")

        assert _ok(result)

    def test_run_governance_failure_returns_error(self, test_settings):
        """Exit code != 0 → error result."""
        from app.application.use_cases.run_governance import RunGovernanceUseCase

        uc = RunGovernanceUseCase(settings=test_settings)
        fake_summary = MagicMock()
        fake_summary.status = "FAIL"
        fake_summary.to_dict.return_value = {}

        patches = self._patch_governance(fake_summary, exit_code=1)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6], patches[7], patches[8], patches[9], patches[10], patches[
            11
        ], patches[
            12
        ], patches[
            13
        ]:
            result = uc.run(mode="validation")

        assert not _ok(result)
        assert result.get("meta", {}).get("code") == "GOVERNANCE_FAILED"
