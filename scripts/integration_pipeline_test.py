"""
Integration Pipeline Test  (S15)
=================================
Sequential 7-step end-to-end test of the Tozsde trading pipeline.

Usage
-----
    python scripts/integration_pipeline_test.py
    python scripts/integration_pipeline_test.py --tickers VOO AAPL
    python scripts/integration_pipeline_test.py --skip 5,6
    python scripts/integration_pipeline_test.py --dry-run

Exit codes
----------
    0  – all executed steps passed
    1  – at least one step failed
"""

import argparse
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

# Ensure the project root is on sys.path when the script is invoked directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────
# Test step registry
# ──────────────────────────────────────────────────────────────

STEPS = []  # populated by @step decorator


def step(number: int, name: str):
    """Register an integration step function."""

    def decorator(fn):
        STEPS.append({"number": number, "name": name, "fn": fn})
        return fn

    return decorator


# ──────────────────────────────────────────────────────────────
# Individual steps
# ──────────────────────────────────────────────────────────────


@step(1, "Settings load")
def test_settings(tickers, dry_run):
    """Verify build_settings() returns a valid Settings object."""
    from app.config.build_settings import build_settings

    cfg = build_settings()
    assert cfg.INITIAL_CAPITAL > 0, "INITIAL_CAPITAL must be positive"
    assert cfg.TRANSACTION_FEE_PCT >= 0, "TRANSACTION_FEE_PCT must be non-negative"
    assert hasattr(cfg, "DRAWDOWN_HALT_PCT"), "DRAWDOWN_HALT_PCT missing from Settings"
    assert hasattr(cfg, "ATR_MULTIPLIER"), "ATR_MULTIPLIER missing from Settings"


@step(2, "Data load")
def test_data_load(tickers, dry_run):
    """Load 60 days of OHLCV data for each ticker from the SQLite cache."""
    from app.data_access.data_loader import load_data

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=60)).isoformat()
    for ticker in tickers:
        df = load_data(ticker, start=start, end=end)
        assert not df.empty, f"No OHLCV data for {ticker}"
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        assert required_cols.issubset(
            df.columns
        ), f"Missing columns for {ticker}: {required_cols - set(df.columns)}"


@step(3, "Indicator computation")
def test_indicators(tickers, dry_run):
    """Run compute_signals for each ticker and verify core indicators are present."""
    from app.data_access.data_loader import load_data
    from app.core.analysis.analyzer import compute_signals, get_params

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=180)).isoformat()
    for ticker in tickers:
        df = load_data(ticker, start=start, end=end)
        assert not df.empty, f"No data for {ticker}"
        _, indicators = compute_signals(df, ticker, params=get_params(ticker))
        for key in ("SMA", "EMA", "RSI", "MACD", "ATR", "ADX", "STOCH_K", "regime"):
            assert key in indicators, f"Indicator '{key}' missing for {ticker}"
        assert indicators["regime"] in (
            "TREND",
            "RANGE",
            "TRANSITION",
            "UNKNOWN",
        ), f"Unexpected regime value: {indicators['regime']}"


@step(4, "Model ensemble inference")
def test_model_inference(tickers, dry_run):
    """Run the model ensemble (skip if no trained models are found)."""
    from app.data_access.data_loader import load_data
    from app.data_access.data_cleaner import prepare_df
    from app.config.build_settings import build_settings
    from app.models.model_trainer import TradingEnv
    from app.models.rl_inference import RLModelEnsembleRunner

    cfg = build_settings()
    runner = RLModelEnsembleRunner(model_dir=cfg.MODEL_DIR, env_class=TradingEnv)

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=180)).isoformat()

    any_run = False
    for ticker in tickers:
        df_raw = load_data(ticker, start=start, end=end)
        if df_raw.empty:
            continue
        df = prepare_df(df_raw.copy(), ticker)
        if df.empty:
            continue
        votes, confidences, wf_scores, model_votes = runner.run_ensemble(
            df=df, ticker=ticker, top_n=3, debug=False
        )
        if not votes:
            continue  # no models available – acceptable
        any_run = True
        assert all(v in (0, 1, 2) for v in votes), "Votes must be 0/1/2"
        assert all(0.0 <= c <= 1.0 for c in confidences), "Confidences must be [0,1]"

    if not any_run:
        print("    [SKIP] No trained models found – inference step skipped.")


@step(5, "Paper execution (BUY simulation)")
def test_paper_execution(tickers, dry_run):
    """Simulate a BUY decision through PaperExecutionEngine."""
    from app.config.build_settings import build_settings
    from app.infrastructure.repositories import DataManagerRepository
    from app.services.paper_execution import PaperExecutionEngine
    from app.infrastructure.logger import setup_logger
    import datetime

    cfg = build_settings()
    dm = DataManagerRepository(settings=cfg)
    engine = PaperExecutionEngine(
        dm=dm, logger=setup_logger("integration_test"), settings=cfg
    )

    today = datetime.date.today()
    decisions = [
        {
            "ticker": tickers[0],
            "decision": {"action_code": 1},
            "payload": {"close": 100.0, "open": 101.0},
            "allocation_amount": 500.0,
            "decision_id": -1,
        }
    ]
    if not dry_run:
        # execute() can raise if DB is not populated; acceptable failure here.
        try:
            engine.execute(decisions, as_of=today)
        except Exception as exc:
            # DB empty is expected in CI – re-raise only for unexpected errors
            if (
                "insufficient" not in str(exc).lower()
                and "no data" not in str(exc).lower()
            ):
                raise


@step(6, "Daily pipeline (dry-run)")
def test_daily_pipeline(tickers, dry_run):
    """Run the daily pipeline end-to-end with dry_run=True."""
    from app.bootstrap.bootstrap import build_application

    container = build_application(ensure_dirs=False)
    result = container.daily_pipeline.run(dry_run=True, ticker=tickers[0])
    assert isinstance(result, dict), "Pipeline result must be a dict"
    assert result.get("status") != "error", f"Pipeline errored: {result.get('message')}"


@step(7, "Governance report")
def test_governance(tickers, dry_run):
    """Run governance in 'quick' mode to verify the quant runner executes."""
    from app.bootstrap.bootstrap import build_application

    container = build_application(ensure_dirs=False)
    result = container.governance.run(mode="quick")
    assert isinstance(result, dict), "Governance result must be a dict"


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────


def _bar(width=70):
    return "-" * width


def run_steps(
    tickers: List[str],
    dry_run: bool,
    skip: Optional[List[int]],
) -> bool:
    """Execute all registered steps and print a summary table.

    Returns True if all executed steps passed.
    """
    skip_set = set(skip or [])
    results = []

    print()
    print(_bar())
    print(f"  Tozsde Integration Pipeline Test  |  tickers: {', '.join(tickers)}")
    print(_bar())
    print()

    for entry in sorted(STEPS, key=lambda e: e["number"]):
        num = entry["number"]
        name = entry["name"]

        if num in skip_set:
            results.append((num, name, "SKIP", 0.0, None))
            print(f"  [{num:>2}] {name:<40}  SKIP")
            continue

        t0 = time.monotonic()
        err = None
        try:
            entry["fn"](tickers, dry_run)
            status = "PASS"
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            status = "FAIL"
            err = exc
        elapsed = time.monotonic() - t0
        results.append((num, name, status, elapsed, err))

        indicator = "OK" if status == "PASS" else ("--" if status == "SKIP" else "!!")
        print(f"  [{num:>2}] {name:<40}  {status}  ({elapsed:.2f}s)  {indicator}")
        if err:
            for line in traceback.format_exception(type(err), err, err.__traceback__):
                for sub in line.splitlines():
                    print(f"        {sub}")

    # Summary table
    print()
    print(_bar())
    print(f"  {'#':>3}  {'Step':<40}  {'Status':<6}  {'Time':>7}")
    print(_bar())
    for num, name, status, elapsed, _ in results:
        t = f"{elapsed:.2f}s" if status != "SKIP" else "-"
        print(f"  {num:>3}  {name:<40}  {status:<6}  {t:>7}")
    print(_bar())

    passed = sum(1 for _, _, s, _, _ in results if s == "PASS")
    failed = sum(1 for _, _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, _, s, _, _ in results if s == "SKIP")
    print(f"  PASSED: {passed}  FAILED: {failed}  SKIPPED: {skipped}")
    print(_bar())
    print()

    return failed == 0


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(description="Tozsde integration pipeline test")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["VOO"],
        help="Tickers to test (default: VOO)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated step numbers to skip (e.g. '5,6')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass dry_run=True to pipeline steps",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    skip_nums = [int(x.strip()) for x in args.skip.split(",") if x.strip().isdigit()]
    ok = run_steps(
        tickers=args.tickers,
        dry_run=args.dry_run,
        skip=skip_nums,
    )
    sys.exit(0 if ok else 1)
