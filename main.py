import argparse
from datetime import date, timedelta

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
from app.config.config import Config
from app.decision.allocation import allocate_capital

logger = setup_logger(__name__)


def run_daily():
    logger.info("DAILY pipeline started")
    history = HistoryStore()
    dm = DataManager()

    # 1. GYŰJTÉS FÁZIS
    daily_candidates = []

    for ticker in Config.TICKERS:
        try:
            payload = generate_daily_recommendation_payload(ticker, history)
            decision = build_recommendation(payload)

            audit = build_audit_metadata(payload, decision)

            # Policy rules (cooldown, safety)
            decision = apply_decision_policy(decision, audit)

            explanation = build_explanation(payload, decision)

            daily_candidates.append(
                {
                    "ticker": ticker,
                    "payload": payload,
                    "decision": decision,
                    "explanation": explanation,
                    "audit": audit,
                }
            )

            logger.info(f"Analyzed {ticker}: {decision['action']}")

        except Exception as e:
            logger.error(f"DAILY analysis failed for {ticker}: {e}")

    # 2. ALLOKÁCIÓ FÁZIS (P7)
    # Itt dől el, hogy miből mennyit veszünk
    allocatable = [
        c for c in daily_candidates if not c["decision"].get("no_trade", False)
    ]

    finalized_decisions = allocate_capital(allocatable)

    # 3. VÉGREHAJTÁS ÉS ÉRTESÍTÉS FÁZIS
    email_lines = []

    for item in finalized_decisions:
        ticker = item["ticker"]
        decision = item["decision"]
        payload = item["payload"]
        audit = item["audit"]
        explanation = item["explanation"]

        # Ha van allokáció, jelezzük a döntésben
        if decision.get("action_code") == 1:
            amount = item.get("allocation_amount", 0)
            logger.info(f"--> {ticker} ALLOCATED: ${amount:.2f}")

        # Mentés History-ba
        history.save_decision(
            payload=payload, decision=decision, explanation=explanation, audit=audit
        )

        # Mentés DB-be
        dm.log_recommendation(
            ticker=ticker,
            signal=decision["action"],
            confidence=decision["confidence"],
            params={
                "wf_score": decision["wf_score"],
                "strength": decision["strength"],
                "allocated_usd": item.get("allocation_amount", 0),
            },
        )

        # Email sor formázása
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
        send_email(subject, body, Config.NOTIFY_EMAIL)

    logger.info("DAILY pipeline finished")


def run_weekly():
    logger.info("WEEKLY reliability started")
    try:
        analyzer = ModelReliabilityAnalyzer()

        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=Config.RELIABILITY_PERIOD_DAYS)

        tickers = Config.TICKERS

        for ticker in tickers:
            scores = analyzer.analyze(
                ticker=ticker,
                start=start,
                end=end,
            )

            save_reliability_scores(ticker, end.isoformat(), scores)

            logger.info(f"{ticker} reliability updated " f"(models={len(scores)})")

    except Exception as e:
        logger.exception("WEEKLY reliability failed")
    logger.info("WEEKLY reliability finished")


def run_monthly():
    logger.info("MONTHLY retraining started")

    for ticker in Config.TICKERS:
        try:
            wf_summary = run_walk_forward(ticker)

            train_rl_agent(
                ticker=ticker,
                wf_score=wf_summary["normalized_score"],
                wf_summary=wf_summary,
            )

            logger.info(f"{ticker} retrained")

        except Exception as e:
            logger.exception(f"MONTHLY failed for {ticker}")

    logger.info("MONTHLY retraining finished")


def run_train(ticker: str):
    logger.info(f"Manual train: {ticker}")
    train_rl_agent(ticker=ticker)


def run_optimize(ticker: str):
    logger.info(f"Manual optimize: {ticker}")
    run_walk_forward(ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["daily", "weekly", "monthly", "train", "optimize"],
    )
    parser.add_argument("--ticker")

    args = parser.parse_args()

    if args.mode == "daily":
        run_daily()
    elif args.mode == "weekly":
        #    run_weekly()
        logger.error("Not supprted yet!")
    elif args.mode == "monthly":
        run_monthly()
    elif args.mode == "train":
        run_train(args.ticker)
    elif args.mode == "optimize":
        run_optimize(args.ticker)


if __name__ == "__main__":
    main()
