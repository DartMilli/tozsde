import json
from dataclasses import dataclass
from datetime import date, timedelta

from app.infrastructure.logger import setup_logger
from app.infrastructure.repositories.data_manager_repository import (
    DataManagerRepository,
)


logger = setup_logger(__name__)


@dataclass(frozen=True)
class ExpectancyResult:
    expected_pnl: float
    expected_net_pnl: float
    sample_count: int
    gate_pass: bool
    reason: str


def bucket_confidence(confidence: float | None, settings=None) -> str:
    if confidence is None:
        return "UNKNOWN"

    strong_threshold = (
        getattr(settings, "STRONG_CONFIDENCE_THRESHOLD", 0.75)
        if settings is not None
        else 0.75
    )
    weak_threshold = (
        getattr(settings, "WEAK_CONFIDENCE_THRESHOLD", 0.4)
        if settings is not None
        else 0.4
    )

    if confidence >= strong_threshold:
        return "HIGH"
    if confidence >= weak_threshold:
        return "MEDIUM"
    return "LOW"


class ExpectancyGate:
    def __init__(self, settings=None, data_manager=None) -> None:
        self.settings = settings
        self.dm = data_manager or DataManagerRepository(settings=settings)

    def evaluate(
        self,
        ticker: str,
        action_code: int,
        confidence_bucket: str,
        regime: str,
        as_of_date: date,
    ) -> ExpectancyResult:
        min_samples = getattr(self.settings, "EXPECTANCY_MIN_SAMPLES", 10)
        lookback_days = getattr(self.settings, "EXPECTANCY_LOOKBACK_DAYS", 180)
        start_date = as_of_date - timedelta(days=lookback_days)

        matching_returns = []

        with self.dm.connection() as conn:
            rows = conn.execute(
                """
                SELECT dh.decision_blob, dh.audit_blob, o.pnl_pct
                FROM decision_history dh
                JOIN outcomes o ON o.decision_id = dh.id
                WHERE dh.ticker = ?
                  AND date(dh.timestamp) BETWEEN ? AND ?
                ORDER BY dh.timestamp DESC
                """,
                (ticker, start_date.isoformat(), as_of_date.isoformat()),
            ).fetchall()

        for decision_blob, audit_blob, pnl_pct in rows:
            decision = json.loads(decision_blob) if decision_blob else {}
            audit = json.loads(audit_blob) if audit_blob else {}

            historical_action = decision.get(
                "original_action", decision.get("action_code")
            )
            if historical_action != action_code:
                continue

            historical_regime = audit.get("regime", "UNKNOWN")
            if regime != "UNKNOWN" and historical_regime not in {regime, "UNKNOWN"}:
                continue

            historical_bucket = audit.get(
                "confidence_bucket",
                bucket_confidence(decision.get("confidence"), settings=self.settings),
            )
            if (
                confidence_bucket != "UNKNOWN"
                and historical_bucket != confidence_bucket
            ):
                continue

            matching_returns.append(float(pnl_pct or 0.0))

        if len(matching_returns) < min_samples:
            return ExpectancyResult(
                expected_pnl=0.0,
                expected_net_pnl=0.0,
                sample_count=len(matching_returns),
                gate_pass=True,
                reason="INSUFFICIENT_SAMPLES",
            )

        expected_pnl = sum(matching_returns) / len(matching_returns)
        total_cost = (
            getattr(self.settings, "TRANSACTION_FEE_PCT", 0.0)
            + getattr(self.settings, "MIN_SLIPPAGE_PCT", 0.0)
            + getattr(self.settings, "SPREAD_PCT", 0.0)
        )
        expected_net_pnl = expected_pnl - total_cost
        gate_pass = expected_net_pnl > 0.0
        reason = "POSITIVE_EXPECTANCY" if gate_pass else "NEGATIVE_EXPECTANCY"

        if not gate_pass:
            logger.info(
                "Expectancy gate blocked %s action=%s bucket=%s regime=%s net=%.5f samples=%d",
                ticker,
                action_code,
                confidence_bucket,
                regime,
                expected_net_pnl,
                len(matching_returns),
            )

        return ExpectancyResult(
            expected_pnl=expected_pnl,
            expected_net_pnl=expected_net_pnl,
            sample_count=len(matching_returns),
            gate_pass=gate_pass,
            reason=reason,
        )
