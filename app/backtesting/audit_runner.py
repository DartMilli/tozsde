from app.backtesting.replay_runner import run_replay_for_ticker
from app.backtesting.backtest_audit import (
    audit_confidence_buckets,
    audit_decision_levels,
    detect_overconfidence,
)


def run_backtest_audit(
    ticker: str,
    horizon_days: int = 5,
) -> dict:
    rows = run_replay_for_ticker(
        ticker=ticker,
        horizon_days=horizon_days,
    )

    if not rows:
        return {}

    return {
        "confidence_buckets": audit_confidence_buckets(rows),
        "decision_levels": audit_decision_levels(rows),
        "overconfidence_cases": detect_overconfidence(rows),
    }
