#!/usr/bin/env python
"""
Main Entry Point - Trading System Pipeline

Orchestrates daily, weekly, and monthly trading workflows:
  - Daily decision generation and allocation
  - Weekly reliability analysis
  - Monthly walk-forward optimization and RL retraining

CLI Usage:
  python main.py daily                     # Run daily recommendations
  python main.py daily --dry-run           # Dry-run (no emails)
  python main.py daily --ticker VOO        # Single ticker only
  python main.py weekly                    # Weekly reliability
  python main.py monthly                   # Monthly retraining
  python main.py walk-forward VOO          # WF optimization
  python main.py train-rl VOO              # Train RL for ticker
  python main.py --help                    # Show all options

Environment Variables:
  DRY_RUN=true/false                 # Skip side effects
  LOGGING_LEVEL=DEBUG/INFO           # Log verbosity
"""

import argparse
import sys
import os
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.infrastructure.logger import setup_logger
from app.bootstrap.bootstrap import build_application
from app.interfaces.compat import main_contract as _main_contract

# Build application container (settings + repos)
_APP_CONTAINER = build_application(ensure_dirs=False)
_SETTINGS = _APP_CONTAINER.settings
from app.application.use_cases.result import ok as result_ok
from app.application.use_cases.result import error as result_error

logger = setup_logger(__name__)


def _emit(payload):
    print(json.dumps(payload, indent=2, default=str))


#
# DAILY PIPELINE
#


def run_daily(dry_run: bool = False, ticker: str = None):
    """
    Run daily trading recommendation pipeline.

    Workflow:
      1. Generate signals for each ticker
      2. Apply policy rules (cooldown, safety)
      3. Allocate capital across positions
      4. Save decisions to database
      5. Send email notifications

    Args:
        dry_run: If True, skip email notifications
        ticker: If provided, analyze only this ticker (dev mode)
    """
    return _main_contract.run_daily(_APP_CONTAINER, dry_run=dry_run, ticker=ticker)


#
# WEEKLY PIPELINE
#


def run_weekly(dry_run: bool = False):
    """
    Run weekly model reliability analysis.

    Analyzes model accuracy over past RELIABILITY_PERIOD_DAYS.
    Computes reliability scores per model and saves to database.

    Args:
        dry_run: If True, don't save results
    """
    return _main_contract.run_weekly(_APP_CONTAINER, dry_run=dry_run)


#
# MONTHLY PIPELINE
#


def run_monthly(dry_run: bool = False):
    """
    Run monthly model retraining cycle.

    Workflow:
      1. Walk-forward optimization (30-day optimization window)
      2. RL agent training with optimized parameters
      3. Save trained models and reliability scores

    Args:
        dry_run: If True, don't save models
    """
    return _main_contract.run_monthly(_APP_CONTAINER, dry_run=dry_run)


#
# WALK-FORWARD OPTIMIZATION (Manual)
#


def run_walk_forward_manual(ticker: str, dry_run: bool = False):
    """
    Manually trigger walk-forward optimization for a ticker.

    Args:
        ticker: Asset ticker symbol
        dry_run: If True, don't save results
    """
    return _main_contract.run_walk_forward_manual(
        _APP_CONTAINER,
        ticker=ticker,
        dry_run=dry_run,
    )


#
# RL AGENT TRAINING (Manual)
#


def run_train_rl_manual(ticker: str, dry_run: bool = False):
    """
    Manually trigger RL agent training for a ticker.

    Args:
        ticker: Asset ticker symbol
        dry_run: If True, don't save models
    """
    return _main_contract.run_train_rl_manual(
        _APP_CONTAINER,
        ticker=ticker,
        dry_run=dry_run,
    )


def run_validation(
    ticker: str = None,
    start_date: str = None,
    end_date: str = None,
    scenario: str = "elevated_volatility",
    include_calibration: bool = True,
    repeat: int = 1,
    compare_repeat: bool = False,
):
    """
    Run Phase 5 validation analyzers and build a unified report.
    """
    return _main_contract.run_validation(
        _APP_CONTAINER,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        scenario=scenario,
        include_calibration=include_calibration,
        repeat=repeat,
        compare_repeat=compare_repeat,
    )


def run_paper_history(ticker: str, start_date: str, end_date: str):
    """
    Run deterministic historical paper trading over a date range.
    """
    return _main_contract.run_paper_history(
        _APP_CONTAINER,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )


#
# CLI ARGUMENT PARSER
#


def parse_arguments():
    """Parse command-line arguments."""

    parser = _build_parser()
    return parser.parse_args()


def _build_parser():
    """Build argparse parser."""

    parser = argparse.ArgumentParser(
        description="Trading system pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py daily                    # Run daily recommendations
  python main.py daily --dry-run          # Test without side effects
  python main.py daily --ticker VOO       # Single ticker test
  python main.py weekly                   # Weekly reliability
  python main.py monthly                  # Monthly retraining
  python main.py walk-forward VOO         # Manual WF optimization
  python main.py train-rl VOO             # Manual RL training
    python main.py run-paper-history --ticker VOO --start-date 2022-01-01 --end-date 2023-12-31
        """,
    )

    # Subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Pipeline command to execute"
    )

    # Daily pipeline
    daily_parser = subparsers.add_parser(
        "daily", help="Run daily trading recommendation pipeline"
    )
    daily_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without sending emails"
    )
    daily_parser.add_argument(
        "--ticker", type=str, help="Single ticker for development testing"
    )

    # Weekly pipeline
    weekly_parser = subparsers.add_parser(
        "weekly", help="Run weekly model reliability analysis"
    )
    weekly_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without saving results"
    )

    # Monthly pipeline
    monthly_parser = subparsers.add_parser(
        "monthly", help="Run monthly retraining cycle"
    )
    monthly_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without saving models"
    )

    # Walk-forward optimization
    wf_parser = subparsers.add_parser(
        "walk-forward", help="Run walk-forward optimization"
    )
    wf_parser.add_argument("ticker", type=str, help="Asset ticker symbol")
    wf_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without saving results"
    )

    # RL training
    rl_parser = subparsers.add_parser("train-rl", help="Train RL agent")
    rl_parser.add_argument("ticker", type=str, help="Asset ticker symbol")
    rl_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate without saving models"
    )

    # Phase 5 validation
    validation_parser = subparsers.add_parser(
        "validate", help="Run Phase 5 validation analyzers"
    )
    validation_parser.add_argument(
        "--ticker", type=str, help="Optional ticker for symbol-scoped analysis"
    )
    validation_parser.add_argument(
        "--start-date", type=str, help="Start date (YYYY-MM-DD)"
    )
    validation_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    validation_parser.add_argument(
        "--scenario",
        type=str,
        default="elevated_volatility",
        help="Stress scenario (elevated_volatility | gap_days | drawdown_injection)",
    )
    validation_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip confidence calibration",
    )
    validation_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run validation multiple times (default: 1)",
    )
    validation_parser.add_argument(
        "--compare-repeat",
        action="store_true",
        help="Compare last two runs for repeatability",
    )

    history_parser = subparsers.add_parser(
        "run-paper-history", help="Run historical paper trading"
    )
    history_parser.add_argument("--ticker", type=str, required=True)
    history_parser.add_argument("--start-date", type=str, required=True)
    history_parser.add_argument("--end-date", type=str, required=True)

    # Global options
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser


#
# MAIN ENTRY POINT
#


def main():
    """Main entry point."""

    args = parse_arguments()

    # If no command provided, show help
    if not args.command:
        parser = _build_parser()
        parser.print_help()
        _emit(result_error("cli", "no command provided", code="NO_COMMAND"))
        sys.exit(1)

    # Set logging level
    log_level = getattr(logging, args.loglevel, logging.INFO)
    logger.setLevel(log_level)

    # Dispatch to appropriate handler
    try:
        if args.command == "daily":
            result = run_daily(dry_run=args.dry_run, ticker=args.ticker)
            _emit(
                result_ok(
                    "cli.daily",
                    data=result,
                    command=args.command,
                    dry_run=args.dry_run,
                    ticker=args.ticker,
                )
            )

        elif args.command == "weekly":
            result = run_weekly(dry_run=args.dry_run)
            _emit(
                result_ok(
                    "cli.weekly",
                    data=result,
                    command=args.command,
                    dry_run=args.dry_run,
                )
            )

        elif args.command == "monthly":
            result = run_monthly(dry_run=args.dry_run)
            _emit(
                result_ok(
                    "cli.monthly",
                    data=result,
                    command=args.command,
                    dry_run=args.dry_run,
                )
            )

        elif args.command == "walk-forward":
            result = run_walk_forward_manual(ticker=args.ticker, dry_run=args.dry_run)
            _emit(
                result_ok(
                    "cli.walk_forward",
                    data=result,
                    command=args.command,
                    ticker=args.ticker,
                    dry_run=args.dry_run,
                )
            )

        elif args.command == "train-rl":
            result = run_train_rl_manual(ticker=args.ticker, dry_run=args.dry_run)
            _emit(
                result_ok(
                    "cli.train_rl",
                    data=result,
                    command=args.command,
                    ticker=args.ticker,
                    dry_run=args.dry_run,
                )
            )

        elif args.command == "validate":
            result = run_validation(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                scenario=args.scenario,
                include_calibration=not args.no_calibration,
                repeat=args.repeat,
                compare_repeat=args.compare_repeat,
            )
            _emit(
                result_ok(
                    "cli.validate",
                    data=result,
                    command=args.command,
                    ticker=args.ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    scenario=args.scenario,
                )
            )

        elif args.command == "run-paper-history":
            result = run_paper_history(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            _emit(
                result_ok(
                    "cli.run_paper_history",
                    data=result,
                    command=args.command,
                    ticker=args.ticker,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )

        elif args.command == "governance":
            # Call quant_runner orchestrator as subprocess for full isolation
            import subprocess

            quant_runner_path = os.path.join(
                os.path.dirname(__file__), "app", "governance", "quant_runner.py"
            )
            cmd = [sys.executable, quant_runner_path, "--mode", args.mode]
            exit_code = subprocess.call(cmd)
            if exit_code == 0:
                _emit(
                    result_ok(
                        "cli.governance", data={"exit_code": exit_code}, mode=args.mode
                    )
                )
            else:
                _emit(
                    result_error(
                        "cli.governance",
                        f"governance subprocess failed (exit_code={exit_code})",
                        mode=args.mode,
                        code="SUBPROCESS_FAILED",
                        exit_code=exit_code,
                    )
                )
            sys.exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        _emit(
            result_error(
                "cli",
                "interrupted by user",
                command=getattr(args, "command", None),
                code="INTERRUPTED",
            )
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        _emit(
            result_error(
                "cli",
                str(e),
                command=getattr(args, "command", None),
                code="UNHANDLED_EXCEPTION",
                exception_type=type(e).__name__,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
