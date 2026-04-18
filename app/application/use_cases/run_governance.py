"""Governance use case – runs the quant validation pipeline in-process via DI."""

from app.application.use_cases.result import UseCaseResult, ok, error
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class RunGovernanceUseCase:
    """
    Execute the quant governance / validation runner inside the current process.

    Instead of spawning a subprocess for `quant_runner.py`, this use case calls
    the governance logic directly, receiving the same DI-injected settings as
    every other use case in the container.

    Args:
        settings: injected Settings instance
        mode: one of 'research', 'validation', 'diagnostics', 'predeploy', 'full', 'tests'
    """

    def __init__(self, settings):
        self.settings = settings

    def run(self, mode: str = "validation") -> UseCaseResult:
        from app.governance import set_settings
        from app.governance.quant_runner import (
            _run_tests,
            _run_diagnostics,
            _run_validation,
            _build_summary,
            _apply_collapse_stage,
            _exit_code,
            _git_commit,
            _configure_logging,
        )
        from app.reporting.report_builder import prepare_report_dir, write_report_bundle
        from app.reporting.report_schema import now_timestamp
        from app.governance.checklist_runner import evaluate_checklist
        from app.validation.utils import get_validation_ticker
        from app.backtesting.execution_utils import seed_deterministic
        import os
        import random
        import uuid
        from pathlib import Path

        # Inject settings into governance singleton so quant_runner helpers use same config
        set_settings(self.settings)

        run_id = str(uuid.uuid4())
        timestamp = now_timestamp().replace(":", "").replace("-", "")
        report_dir = prepare_report_dir(timestamp)
        commit = _git_commit()
        gov_logger = _configure_logging(report_dir / "run.log", run_id, mode, commit)
        gov_logger.info(
            "Starting governance use case (in-process, DI-injected settings)"
        )

        os.environ["EDGE_DIAGNOSTICS_MODE"] = "false"
        os.environ["PIPELINE_AUDIT_MODE"] = "false"

        if mode != "tests":
            os.environ["VALIDATION_MODE"] = mode
            os.environ["VALIDATION_DISABLE_SAFETY"] = "true"
            os.environ["VALIDATION_DISABLE_POLICY"] = "true"
            seed_deterministic(42)
            random.seed(42)

        diagnostics: dict = {}
        validation: dict = {}
        tests: dict = {}

        if mode == "research":
            os.environ["EDGE_DIAGNOSTICS_MODE"] = "true"
            os.environ["PIPELINE_AUDIT_MODE"] = "true"
            from app.backtesting.walk_forward import run_walk_forward
            from app.validation.utils import get_validation_ticker

            run_walk_forward(get_validation_ticker())
            diagnostics = _run_diagnostics(gov_logger)
            validation = _run_validation(
                gov_logger, include_shadow=False, include_risk=False, include_rl=False
            )

        elif mode == "diagnostics":
            diagnostics = _run_diagnostics(gov_logger)

        elif mode == "validation":
            validation = _run_validation(
                gov_logger, include_shadow=False, include_risk=False, include_rl=False
            )

        elif mode == "predeploy":
            diagnostics = _run_diagnostics(gov_logger)
            validation = _run_validation(
                gov_logger, include_shadow=True, include_risk=True, include_rl=True
            )

        elif mode == "tests":
            tests = _run_tests(gov_logger)

        elif mode == "full":
            tests = _run_tests(gov_logger)
            diagnostics = _run_diagnostics(gov_logger)
            validation = _run_validation(
                gov_logger, include_shadow=True, include_risk=True, include_rl=True
            )

        ticker = get_validation_ticker()
        _apply_collapse_stage(validation, diagnostics)

        checklist = evaluate_checklist(
            results=validation,
            diagnostics=diagnostics.get("pipeline_audit", diagnostics),
            tests=tests,
            report_files_present=True,
        )

        summary = _build_summary(
            mode, ticker, validation, diagnostics, tests, checklist
        )
        write_report_bundle(
            report_dir, summary, validation, diagnostics, tests, checklist
        )

        exit_code = _exit_code(summary)
        gov_logger.info("Governance use case completed (exit_code=%s)", exit_code)

        if exit_code == 0:
            return ok(
                "run_governance",
                data={
                    "summary": summary.to_dict(),
                    "exit_code": exit_code,
                    "report_dir": str(report_dir),
                },
                mode=mode,
            )
        return error(
            "run_governance",
            f"Governance {mode} finished with exit_code={exit_code} (status={summary.status})",
            code="GOVERNANCE_FAILED",
            exit_code=exit_code,
            mode=mode,
            report_dir=str(report_dir),
        )
