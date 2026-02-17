#!/usr/bin/env python
"""
Main Entry Point — Trading System Pipeline

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
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.trading_pipeline import TradingPipelineService
from app.infrastructure.logger import setup_logger

from app.models.model_reliability import (
    ModelReliabilityAnalyzer,
    save_reliability_scores,
)
from app.backtesting.walk_forward import run_walk_forward
from app.models.model_trainer import train_rl_agent
from app.config.config import Config
from app.backtesting.history_store import HistoryStore
from app.data_access.data_manager import DataManager
from app.models.model_trainer import TradingEnv
from app.services.dependencies import (
    EmailNotifier,
    MarketDataFetcher,
    ModelEnsembleRunner,
)
from app.services.paper_execution import PaperExecutionEngine
from app.services.execution_engines import NoopExecutionEngine
from app.analysis.decision_quality_analyzer import DecisionQualityAnalyzer
from app.analysis.confidence_calibrator import ConfidenceCalibrator
from app.analysis.wf_stability_analyzer import WalkForwardStabilityAnalyzer
from app.analysis.safety_stress_tester import SafetyStressTester
from app.analysis.validation_report_builder import ValidationReportBuilder
from app.backtesting.historical_paper_runner import HistoricalPaperRunner
from app.decision.recommender import generate_daily_recommendation_payload
from app.decision.recommendation_builder import build_explanation, build_recommendation
from app.reporting.audit_builder import build_audit_metadata, build_audit_summary
from app.decision.decision_policy import apply_decision_policy
from app.decision.allocation import allocate_capital
from app.notifications.email_formatter import format_email_line
from app.notifications.mailer import send_email
from app.notifications.alerter import ErrorAlerter

logger = setup_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DAILY PIPELINE
# ═════════════════════════════════════════════════════════════════════════════


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
    use_legacy_env = os.getenv("USE_LEGACY_RUN_DAILY")
    use_legacy = (
        use_legacy_env.lower() == "true"
        if use_legacy_env is not None
        else getattr(generate_daily_recommendation_payload, "__module__", "")
        != "app.decision.recommender"
    )

    if use_legacy:
        history_store = HistoryStore()
        dm = DataManager()

        tickers_to_process = [ticker] if ticker else Config.get_supported_tickers()
        daily_candidates = []

        for ticker_symbol in tickers_to_process:
            try:
                payload = generate_daily_recommendation_payload(
                    ticker_symbol, history_store
                )

                if payload.get("error"):
                    raise ValueError(payload["error"])

                decision = payload.get("decision")
                if decision is None:
                    decision = build_recommendation(payload)

                explanation = payload.get("explanation") or build_explanation(
                    payload, decision
                )

                audit = build_audit_metadata(payload, decision)
                decision = apply_decision_policy(decision, audit)

                if explanation is None:
                    explanation = build_explanation(payload, decision)

                daily_candidates.append(
                    {
                        "ticker": ticker_symbol,
                        "payload": payload,
                        "decision": decision,
                        "explanation": explanation,
                        "audit": audit,
                    }
                )
            except Exception as e:
                logger.error(
                    f"ERR DAILY analysis failed for {ticker_symbol}: {e}",
                    exc_info=True,
                )
                if not dry_run:
                    ErrorAlerter.alert(
                        error_code="MISSING_TICKER_DATA",
                        message=f"Daily analysis failed for {ticker_symbol}: {e}",
                        details={"ticker": ticker_symbol},
                        severity="auto",
                    )

        if not daily_candidates:
            logger.warning("No candidates generated")
            return

        allocatable = [
            c for c in daily_candidates if not c["decision"].get("no_trade", False)
        ]
        finalized_decisions = allocate_capital(allocatable)

        email_lines = []

        for item in finalized_decisions:
            payload = item["payload"]
            decision = item["decision"]
            explanation = item["explanation"]
            audit = item["audit"]

            history_store.save_decision(
                payload=payload,
                decision=decision,
                explanation=explanation,
                audit=audit,
                model_votes=payload.get("model_votes", []),
                safety_overrides={
                    "safety_override": decision.get("safety_override"),
                    "no_trade_reason": decision.get("no_trade_reason"),
                    "reasons": decision.get("reasons", []),
                    "warnings": decision.get("warnings", []),
                },
                model_id=payload.get("model_id"),
                timestamp=payload.get("timestamp"),
            )

            dm.log_recommendation(
                ticker=payload.get("ticker"),
                signal=decision.get("action"),
                confidence=decision.get("confidence"),
                params={
                    "wf_score": decision.get("wf_score"),
                    "ensemble_quality": decision.get("ensemble_quality"),
                    "quality_score": decision.get("quality_score"),
                    "volatility": payload.get("volatility"),
                    "model_votes": payload.get("model_votes", []),
                },
            )

            email_lines.append(
                format_email_line(
                    explanation=explanation,
                    decision=decision,
                    audit=build_audit_summary(audit, payload, decision),
                )
            )

        if email_lines:
            subject = f"Napi ajánlások ({date.today().isoformat()})"
            body = "\n".join(email_lines)

            if dry_run:
                logger.info(f"[DRY-RUN] Email would be sent: {subject}")
            else:
                try:
                    send_email(subject, body, Config.NOTIFY_EMAIL)
                    logger.info(f"OK Email sent to {Config.NOTIFY_EMAIL}")
                except Exception as e:
                    logger.error(f"ERR Failed to send email: {e}")
                    ErrorAlerter.alert(
                        error_code="AUTHENTICATION_FAILED",
                        message=f"Failed to send notification email: {e}",
                        details={"recipient": Config.NOTIFY_EMAIL},
                        severity="auto",
                    )

        return

    if Config.EXECUTION_MODE == "paper":
        execution_engine = PaperExecutionEngine(DataManager(), logger)
    else:
        execution_engine = NoopExecutionEngine(logger)

    pipeline = TradingPipelineService(
        history_store=HistoryStore(),
        logger=logger,
        data_fetcher=MarketDataFetcher(),
        model_runner=ModelEnsembleRunner(
            model_dir=Config.MODEL_DIR, env_class=TradingEnv
        ),
        email_notifier=EmailNotifier(),
        execution_engine=execution_engine,
    )

    pipeline.run_daily(dry_run=dry_run, ticker=ticker)


# ═════════════════════════════════════════════════════════════════════════════
# WEEKLY PIPELINE
# ═════════════════════════════════════════════════════════════════════════════


def run_weekly(dry_run: bool = False):
    """
    Run weekly model reliability analysis.

    Analyzes model accuracy over past RELIABILITY_PERIOD_DAYS.
    Computes reliability scores per model and saves to database.

    Args:
        dry_run: If True, don't save results
    """
    logger.info("=" * 80)
    logger.info(f"WEEKLY reliability analysis started (dry_run={dry_run})")
    logger.info("=" * 80)

    try:
        analyzer = ModelReliabilityAnalyzer()

        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=Config.RELIABILITY_PERIOD_DAYS)

        logger.info(f"Analysis period: {start} to {end}")

        for ticker_symbol in Config.get_supported_tickers():
            try:
                scores = analyzer.analyze(
                    ticker=ticker_symbol,
                    start=start,
                    end=end,
                )

                if not dry_run:
                    save_reliability_scores(ticker_symbol, end.isoformat(), scores)
                    logger.info(f"OK {ticker_symbol}: {len(scores)} model scores saved")
                else:
                    logger.info(
                        f"[DRY-RUN] {ticker_symbol}: {len(scores)} model scores (not saved)"
                    )

            except Exception as e:
                logger.error(
                    f"ERR Reliability analysis failed for {ticker_symbol}: {e}"
                )

    except Exception as e:
        logger.error(f"ERR WEEKLY analysis failed: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("WEEKLY reliability analysis completed")
    logger.info("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# MONTHLY PIPELINE
# ═════════════════════════════════════════════════════════════════════════════


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
    logger.info("=" * 80)
    logger.info(f"MONTHLY retraining cycle started (dry_run={dry_run})")
    logger.info("=" * 80)

    for ticker_symbol in Config.get_supported_tickers():
        try:
            logger.info(f"\nProcessing {ticker_symbol}...")

            # 1. Walk-forward optimization
            logger.info(f"  Running walk-forward optimization for {ticker_symbol}...")
            wf_summary = run_walk_forward(ticker_symbol)
            if not wf_summary:
                logger.warning("  Walk-forward returned no summary; skipping RL")
                continue

            from app.optimization.fitness import normalize_wf_score

            wf_score = wf_summary.get("normalized_score")
            if wf_score is None:
                raw_fitness = wf_summary.get("raw_fitness")
                if raw_fitness is None:
                    raw_fitness = wf_summary.get("wf_fitness", 0.0)
                wf_score = normalize_wf_score(raw_fitness)
            logger.info(f"  OK Walk-forward score: {wf_score:.4f}")

            # 2. RL training
            if Config.ENABLE_RL:
                logger.info(f"  Training RL agent for {ticker_symbol}...")
                train_rl_agent(
                    ticker=ticker_symbol,
                    wf_score=wf_score,
                    wf_summary=wf_summary,
                )
                logger.info("  OK RL agent trained")
            else:
                logger.info(f"  ⊘ RL training disabled (ENABLE_RL=False)")

        except Exception as e:
            logger.error(
                f"ERR MONTHLY cycle failed for {ticker_symbol}: {e}", exc_info=True
            )

    logger.info("=" * 80)
    logger.info("MONTHLY retraining cycle completed")
    logger.info("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD OPTIMIZATION (Manual)
# ═════════════════════════════════════════════════════════════════════════════


def run_walk_forward_manual(ticker: str, dry_run: bool = False):
    """
    Manually trigger walk-forward optimization for a ticker.

    Args:
        ticker: Asset ticker symbol
        dry_run: If True, don't save results
    """
    logger.info("=" * 80)
    logger.info(f"Manual walk-forward: {ticker} (dry_run={dry_run})")
    logger.info("=" * 80)

    try:
        result = run_walk_forward(ticker)
        if not result:
            logger.warning("Walk-forward produced no result")
            return
        logger.info("Walk-forward completed")
        from app.optimization.fitness import normalize_wf_score

        wf_score = result.get("normalized_score")
        if wf_score is None:
            raw_fitness = result.get("raw_fitness")
            if raw_fitness is None:
                raw_fitness = result.get("wf_fitness", 0.0)
            wf_score = normalize_wf_score(raw_fitness)
        logger.info(f"  Normalized Score: {wf_score:.4f}")
        logger.info(f"  Return: {result.get('total_return', 'N/A')}")
        logger.info(f"  Sharpe: {result.get('sharpe_ratio', 'N/A')}")
    except Exception as e:
        logger.error(f"Walk-forward failed: {e}", exc_info=True)

    logger.info("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# RL AGENT TRAINING (Manual)
# ═════════════════════════════════════════════════════════════════════════════


def run_train_rl_manual(ticker: str, dry_run: bool = False):
    """
    Manually trigger RL agent training for a ticker.

    Args:
        ticker: Asset ticker symbol
        dry_run: If True, don't save models
    """
    logger.info("=" * 80)
    logger.info(f"Manual RL training: {ticker} (dry_run={dry_run})")
    logger.info(f"  Steps: {Config.RL_TIMESTEPS:,}")
    logger.info("=" * 80)

    try:
        train_rl_agent(ticker=ticker)
        logger.info(f"RL training completed for {ticker}")
    except Exception as e:
        logger.error(f"RL training failed: {e}", exc_info=True)

    logger.info("=" * 80)


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
    logger.info("=" * 80)
    logger.info("PHASE 5 validation started")
    logger.info("=" * 80)

    DataManager().initialize_tables()

    builder = ValidationReportBuilder()
    reports = []

    for i in range(repeat):
        DecisionQualityAnalyzer().analyze(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Decision quality metrics computed")

        if include_calibration:
            ConfidenceCalibrator().compute(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info("Confidence calibration computed")

        if ticker:
            WalkForwardStabilityAnalyzer().analyze(ticker=ticker)
            logger.info("Walk-forward stability computed")

            if start_date and end_date:
                SafetyStressTester().run(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    scenario=scenario,
                )
                logger.info("Safety stress test completed")
            else:
                logger.warning("Safety stress test skipped (missing start/end date)")

        report = builder.build()
        logger.info("Validation report built")
        reports.append(report)

        if repeat > 1:
            logger.info(f"Validation run {i + 1}/{repeat} completed")

    if compare_repeat and len(reports) >= 2:
        logger.info(f"Repeatable outputs match: {reports[-1] == reports[-2]}")

    logger.info("=" * 80)
    logger.info("PHASE 5 validation completed")
    logger.info("=" * 80)


def run_paper_history(ticker: str, start_date: str, end_date: str):
    """
    Run deterministic historical paper trading over a date range.
    """
    logger.info("=" * 80)
    logger.info(f"HISTORICAL paper run: {ticker} {start_date} -> {end_date}")
    logger.info("=" * 80)

    runner = HistoricalPaperRunner(logger=logger)
    runner.run(ticker=ticker, start_date=start_date, end_date=end_date)

    logger.info("=" * 80)
    logger.info("HISTORICAL paper run completed")
    logger.info("=" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════


def parse_arguments():
    """Parse command-line arguments."""

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

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════


def main():
    """Main entry point."""

    args = parse_arguments()

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set logging level
    log_level = getattr(logging, args.loglevel, logging.INFO)
    logger.setLevel(log_level)

    # Dispatch to appropriate handler
    try:
        if args.command == "daily":
            run_daily(dry_run=args.dry_run, ticker=args.ticker)

        elif args.command == "weekly":
            run_weekly(dry_run=args.dry_run)

        elif args.command == "monthly":
            run_monthly(dry_run=args.dry_run)

        elif args.command == "walk-forward":
            run_walk_forward_manual(ticker=args.ticker, dry_run=args.dry_run)

        elif args.command == "train-rl":
            run_train_rl_manual(ticker=args.ticker, dry_run=args.dry_run)

        elif args.command == "validate":
            run_validation(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                scenario=args.scenario,
                include_calibration=not args.no_calibration,
                repeat=args.repeat,
                compare_repeat=args.compare_repeat,
            )

        elif args.command == "run-paper-history":
            run_paper_history(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        elif args.command == "governance":
            # Call quant_runner orchestrator as subprocess for full isolation
            import subprocess
            import sys
            import os

            quant_runner_path = os.path.join(
                os.path.dirname(__file__), "app", "governance", "quant_runner.py"
            )
            cmd = [sys.executable, quant_runner_path, "--mode", args.mode]
            sys.exit(subprocess.call(cmd))

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
