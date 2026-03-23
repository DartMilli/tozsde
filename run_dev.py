#!/usr/bin/env python
"""
Development Mode Runner - Enhanced Version

Multi-mode development launcher supporting:
  - Flask API server (debug mode)
  - Daily trading pipeline
  - Walk-forward optimization
  - RL agent training
  - Combined Flask + Pipeline execution

Usage Examples:
  python run_dev.py                              # Flask only (default)
  python run_dev.py --mode flask --port 8000    # Flask on port 8000
  python run_dev.py --mode pipeline             # Daily pipeline
  python run_dev.py --mode walk-forward VOO     # Walk-forward for VOO
  python run_dev.py --mode train-rl VOO         # Train RL for VOO
  python run_dev.py --mode both                 # Flask + Pipeline (threads)
  python run_dev.py --mode pipeline --dry-run   # Dry-run (no emails)
  python run_dev.py --loglevel DEBUG            # Debug logging

Features:
   Multiple execution modes (Flask, Pipeline, Both)
   Dry-run support (test without side effects)
   Custom logging levels
   Auto hot-reload for Flask
   Graceful shutdown handling
   Development-specific configuration

Environment Variables:
  FLASK_ENV=development              # Enable dev features
  DRY_RUN=true/false                 # Simulate without side effects
  LOGGING_LEVEL=DEBUG/INFO/WARNING   # Log verbosity
"""

import argparse
import sys
import logging
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.compat.legacy_config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


#
# FLASK DEVELOPMENT SERVER
#


def run_flask_dev(port: int = 5000):
    """
    Run Flask development server with hot-reload.

    Args:
        port: Port to listen on (default: 5000)
    """
    logger.info(f"Starting Flask dev server on http://localhost:{port}")
    logger.info("Debug mode: ON (auto-reload enabled)")
    logger.info("Press Ctrl+C to stop")

    try:
        # Set environment
        os.environ["FLASK_ENV"] = "development"
        os.environ["FLASK_DEBUG"] = "1"

        from app.interfaces.web.ui_app import build_default_ui_app

        app = build_default_ui_app(ensure_dirs=False)

        app.run(
            debug=True, host="0.0.0.0", port=port, use_reloader=True, use_debugger=True
        )
    except KeyboardInterrupt:
        logger.info("Flask server stopped")
    except Exception as e:
        logger.error(f"Flask server error: {e}", exc_info=True)
        sys.exit(1)


#
# MAIN PIPELINE
#


def run_pipeline_dev(dry_run: bool = False, ticker: str = None):
    """
    Run main pipeline in development mode.

    Args:
        dry_run: If True, simulate without sending emails
        ticker: If provided, run only for this ticker (dev mode)
    """
    logger.info("Starting daily pipeline in DEV mode")

    if dry_run:
        os.environ["DRY_RUN"] = "true"
        logger.info("DRY-RUN mode: No emails will be sent")

    if ticker:
        logger.info(f"DEV mode: Running for {ticker} only")
        Config.TICKERS = [ticker]

    try:
        import main

        main.run_daily(dry_run=dry_run, ticker=ticker)
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


#
# WALK-FORWARD OPTIMIZATION
#


def run_walk_forward_dev(ticker: str):
    """
    Run walk-forward optimization for single ticker.

    Args:
        ticker: Asset ticker symbol
    """
    logger.info(f"Starting walk-forward optimization for {ticker}")

    try:
        import main

        main.run_walk_forward_manual(ticker=ticker)
        logger.info("Walk-forward completed")
    except Exception as e:
        logger.error(f"Walk-forward error: {e}", exc_info=True)
        sys.exit(1)


#
# RL AGENT TRAINING
#


def run_train_rl_dev(ticker: str):
    """
    Train RL agent for single ticker.

    Args:
        ticker: Asset ticker symbol
    """
    logger.info(f"Starting RL agent training for {ticker}")
    logger.info(f"Steps: {Config.RL_TIMESTEPS:,}")

    try:
        import main

        main.run_train_rl_manual(ticker=ticker)
        logger.info(f"RL training completed for {ticker}")
    except Exception as e:
        logger.error(f"RL training error: {e}", exc_info=True)
        sys.exit(1)


#
# COMBINED MODE (FLASK + PIPELINE)
#


def run_both_dev(port: int = 5000):
    """
    Run Flask and Pipeline in separate threads.

    Args:
        port: Flask port (default: 5000)
    """
    import threading

    logger.info("Starting both Flask API and Pipeline in separate threads")
    logger.info("Note: Press Ctrl+C to stop both services")

    # Create daemon threads
    flask_thread = threading.Thread(
        target=run_flask_dev, args=(port,), daemon=True, name="Flask-Server"
    )

    pipeline_thread = threading.Thread(
        target=run_pipeline_dev,
        args=(False, None),  # dry_run=False, ticker=None
        daemon=True,
        name="Pipeline-Runner",
    )

    try:
        # Start threads
        flask_thread.start()
        logger.info(f"v Flask server started on port {port}")

        # Small delay to prevent race conditions
        import time

        time.sleep(2)

        pipeline_thread.start()
        logger.info("v Pipeline runner started")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down both services...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in combined mode: {e}", exc_info=True)
        sys.exit(1)


#
# CLI ARGUMENT PARSER
#


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Development mode runner for tozsde_webapp trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dev.py                          # Flask server (default)
  python run_dev.py --mode pipeline          # Daily pipeline
  python run_dev.py --mode walk-forward VOO  # Walk-forward for VOO
  python run_dev.py --mode both --port 8000  # Flask + Pipeline on port 8000
  python run_dev.py --mode pipeline --dry-run --ticker VOO  # Test pipeline
        """,
    )

    # Main mode argument
    parser.add_argument(
        "--mode",
        choices=["flask", "pipeline", "both", "walk-forward", "train-rl"],
        default="flask",
        help="Execution mode (default: flask)",
    )

    # Ticker argument (for walk-forward and train-rl)
    parser.add_argument(
        "ticker",
        nargs="?",
        help="Ticker symbol (required for walk-forward and train-rl modes)",
    )

    # Flask options
    parser.add_argument(
        "--port", type=int, default=5000, help="Flask port (default: 5000)"
    )

    # Pipeline options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without side effects (pipeline mode only)",
    )

    # Single-ticker dev mode
    parser.add_argument(
        "--ticker",
        type=str,
        dest="dev_ticker",
        help="Run pipeline for single ticker only (dev mode)",
    )

    # Logging
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Environment
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default="development",
        help="Environment (default: development)",
    )

    return parser.parse_args()


#
# MAIN ENTRY POINT
#


def main():
    """Main entry point."""

    args = parse_arguments()

    # Set environment variables
    os.environ["FLASK_ENV"] = args.env

    # Configure logging
    Config.LOGGING_LEVEL = args.loglevel
    logger.setLevel(getattr(logging, args.loglevel))

    logger.info("=" * 80)
    logger.info(
        f"TOZSDE Development Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Environment: {args.env.upper()}")
    logger.info(f"Log Level: {args.loglevel}")
    logger.info("=" * 80)

    # Dispatch to appropriate handler
    try:
        if args.mode == "flask":
            run_flask_dev(port=args.port)

        elif args.mode == "pipeline":
            run_pipeline_dev(dry_run=args.dry_run, ticker=args.dev_ticker)

        elif args.mode == "walk-forward":
            if not args.ticker:
                logger.error("walk-forward mode requires TICKER argument")
                sys.exit(2)
            run_walk_forward_dev(args.ticker)

        elif args.mode == "train-rl":
            if not args.ticker:
                logger.error("train-rl mode requires TICKER argument")
                sys.exit(2)
            run_train_rl_dev(args.ticker)

        elif args.mode == "both":
            run_both_dev(port=args.port)

    except KeyboardInterrupt:
        logger.info("Development runner stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
