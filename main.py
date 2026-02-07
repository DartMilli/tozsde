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

from app.decision.recommender import generate_daily_recommendation_payload
from app.decision.recommendation_builder import (
    build_recommendation,
    build_explanation,
)
from app.backtesting.history_store import HistoryStore
from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

from app.models.model_reliability import (
    ModelReliabilityAnalyzer,
    save_reliability_scores,
)
from app.backtesting.walk_forward import run_walk_forward
from app.models.model_trainer import train_rl_agent
from app.notifications.email_formatter import format_email_line
from app.reporting.audit_builder import build_audit_summary, build_audit_metadata
from app.decision.decision_policy import apply_decision_policy
from app.decision.decision_event import build_decision_event
from app.notifications.mailer import send_email
from app.notifications.alerter import ErrorAlerter
from app.config.config import Config
from app.decision.allocation import allocate_capital

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
    logger.info("=" * 80)
    logger.info(f"DAILY pipeline started (dry_run={dry_run})")
    if ticker:
        logger.info(f"DEV mode: Analyzing {ticker} only")
    logger.info("=" * 80)

    history = HistoryStore()
    dm = DataManager()

    # Select tickers to process
    tickers_to_process = [ticker] if ticker else Config.get_supported_tickers()

    # 1. GYŰJTÉS FÁZIS - Generate recommendations for each ticker
    daily_candidates = []

    for ticker_symbol in tickers_to_process:
        try:
            payload = generate_daily_recommendation_payload(ticker_symbol, history)

            if payload.get("error"):
                raise ValueError(payload["error"])

            if "decision" in payload:
                decision = payload["decision"]
                explanation = payload.get("explanation")
            else:
                decision = build_recommendation(payload)
                explanation = build_explanation(payload, decision)

            audit = build_audit_metadata(payload, decision)

            # Apply policy rules (cooldown, safety)
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

            logger.info(f"✓ Analyzed {ticker_symbol}: {decision['action']}")

        except Exception as e:
            logger.error(
                f"✗ DAILY analysis failed for {ticker_symbol}: {e}", exc_info=True
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

    # 2. ALLOKÁCIÓ FÁZIS (P7) - Determine capital allocation
    # Filters out no-trade signals
    allocatable = [
        c for c in daily_candidates if not c["decision"].get("no_trade", False)
    ]

    logger.info(f"Allocatable candidates: {len(allocatable)}/{len(daily_candidates)}")

    finalized_decisions = allocate_capital(allocatable)

    # 3. VÉGREHAJTÁS ÉS ÉRTESÍTÉS FÁZIS - Execute and notify
    email_lines = []

    for item in finalized_decisions:
        ticker_symbol = item["ticker"]
        decision = item["decision"]
        payload = item["payload"]
        audit = item["audit"]
        explanation = item["explanation"]

        # Log allocation
        if decision.get("action_code") == 1:
            amount = item.get("allocation_amount", 0)
            logger.info(f"  💰 {ticker_symbol}: ${amount:,.2f} allocated")

        # Save to history (audit trail)
        history.save_decision(
            payload=payload, decision=decision, explanation=explanation, audit=audit
        )

        # Save to database
        dm.log_recommendation(
            ticker=ticker_symbol,
            signal=decision["action"],
            confidence=decision["confidence"],
            params={
                "wf_score": decision["wf_score"],
                "strength": decision["strength"],
                "allocated_usd": item.get("allocation_amount", 0),
            },
        )

        # Format email notification
        email_lines.append(
            format_email_line(
                explanation=explanation,
                decision=decision,
                audit=build_audit_summary(audit, payload, decision),
            )
        )

    # Send notifications (unless dry-run)
    if email_lines:
        subject = f"Napi ajánlások ({date.today().isoformat()})"
        body = "\n".join(email_lines)

        if dry_run:
            logger.info(f"[DRY-RUN] Email would be sent: {subject}")
            logger.debug(f"Email body:\n{body}")
        else:
            try:
                send_email(subject, body, Config.NOTIFY_EMAIL)
                logger.info(f"✓ Email sent to {Config.NOTIFY_EMAIL}")
            except Exception as e:
                logger.error(f"✗ Failed to send email: {e}")
                ErrorAlerter.alert(
                    error_code="AUTHENTICATION_FAILED",
                    message=f"Failed to send notification email: {e}",
                    details={"recipient": Config.NOTIFY_EMAIL},
                    severity="auto",
                )

    logger.info("=" * 80)
    logger.info("DAILY pipeline completed")
    logger.info("=" * 80)


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
                    logger.info(f"✓ {ticker_symbol}: {len(scores)} model scores saved")
                else:
                    logger.info(
                        f"[DRY-RUN] {ticker_symbol}: {len(scores)} model scores (not saved)"
                    )

            except Exception as e:
                logger.error(f"✗ Reliability analysis failed for {ticker_symbol}: {e}")

    except Exception as e:
        logger.error(f"✗ WEEKLY analysis failed: {e}", exc_info=True)

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
            logger.info(f"  ✓ Walk-forward score: {wf_summary['normalized_score']:.4f}")

            # 2. RL training
            if Config.ENABLE_RL:
                logger.info(f"  Training RL agent for {ticker_symbol}...")
                train_rl_agent(
                    ticker=ticker_symbol,
                    wf_score=wf_summary["normalized_score"],
                    wf_summary=wf_summary,
                )
                logger.info(f"  ✓ RL agent trained")
            else:
                logger.info(f"  ⊘ RL training disabled (ENABLE_RL=False)")

        except Exception as e:
            logger.error(
                f"✗ MONTHLY cycle failed for {ticker_symbol}: {e}", exc_info=True
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
        logger.info(f"Walk-forward completed")
        logger.info(f"  Normalized Score: {result['normalized_score']:.4f}")
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

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
