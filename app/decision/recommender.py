import datetime
import hashlib
import os
from collections import Counter

from app.data_access.data_cleaner import prepare_df
from app.config.config import Config
from app.services.dependencies import MarketDataFetcher
from app.decision.decision_engine import DecisionEngine
from app.decision.safety_rules import SafetyRuleEngine
from app.backtesting.history_store import HistoryStore
from app.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.decision.volatility import (
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)
from app.analysis.confidence_calibrator import ConfidenceCalibrator
from app.decision.recommendation_builder import (
    build_recommendation,
    build_explanation,
)


def load_data(ticker: str, start: str, end: str):
    """Compatibility shim for tests that monkeypatch module-level load_data."""
    return MarketDataFetcher().load_data(ticker, start=start, end=end)


def _compute_features_hash(df) -> str | None:
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    pieces = [f"{col}={row[col]}" for col in df.columns]
    base = "|".join(pieces)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _extract_model_version(model_votes: list) -> str | None:
    if not model_votes:
        return None
    for vote in model_votes:
        if not isinstance(vote, dict):
            continue
        for key in ("model_version", "model_id", "model_path", "model"):
            value = vote.get(key)
            if value:
                return str(value)
    return None


def generate_daily_recommendation_payload(
    ticker: str,
    history_store: HistoryStore,
    top_n: int = 3,
    debug=True,
    data_fetcher=None,
    model_runner=None,
    as_of_date=None,
) -> dict:
    today = as_of_date or datetime.date.today()
    start = today - datetime.timedelta(days=180)

    if model_runner is None:
        from app.models.model_trainer import TradingEnv
        from app.models.rl_inference import RLModelEnsembleRunner

        model_runner = RLModelEnsembleRunner(
            model_dir=Config.MODEL_DIR,
            env_class=TradingEnv,
        )

    if data_fetcher is None:
        df_full = load_data(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
        )
    else:
        df_full = data_fetcher.load_data(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=today.strftime("%Y-%m-%d"),
        )
    if df_full.empty:
        return {"error": "NO_DATA"}

    df = prepare_df(df_full.copy(), ticker)
    latest_price = None
    if not df.empty and "Close" in df.columns:
        latest_price = float(df["Close"].iloc[-1])
    features_hash = _compute_features_hash(df)

    votes, confidences, wf_scores, model_votes, debug_rows = model_runner.run_ensemble(
        df=df,
        ticker=ticker,
        top_n=top_n,
        debug=debug,
    )

    if not votes:
        return {"error": "NO_MODELS"}

    # --- aggregate (P7.2 weighted ensemble) ---
    action_code, avg_confidence, ensemble_quality = aggregate_weighted_ensemble(
        votes=votes,
        confidences=confidences,
        wf_scores=wf_scores,
        model_votes=model_votes,
    )

    avg_wf_score = sum(wf_scores) / len(wf_scores) if wf_scores else 1.0

    volatility = compute_normalized_volatility(df)

    scaled_confidence = scale_confidence_by_volatility(
        avg_confidence,
        volatility,
    )

    if Config.ENABLE_CONFIDENCE_CALIBRATION:
        calibrator = ConfidenceCalibrator()
        params = calibrator.load_latest_params(
            ticker=ticker,
            as_of_date=today.isoformat(),
        )
        scaled_confidence = calibrator.apply(scaled_confidence, params)

    safety_engine = SafetyRuleEngine(history_store)
    enable_safety = os.getenv("VALIDATION_DISABLE_SAFETY", "false").lower() != "true"
    decision_engine = DecisionEngine(
        safety_engine=safety_engine,
        enable_safety=enable_safety,
        today=today,
    )

    # --- POLICY (build normalized decision with strength / labels) ---

    policy_payload = {
        "ticker": ticker,
        "avg_confidence": scaled_confidence,
        "avg_wf_score": avg_wf_score,
        "ensemble_quality": ensemble_quality,
        "action_code": action_code,
    }
    decision = build_recommendation(policy_payload)

    # --- SAFETY (cooldown / bear / VIX / chaos overrides) ---
    decision = decision_engine.run(ticker=ticker, decision=decision)

    # --- EXPLANATION (HU/EN) ---
    explanation = build_explanation(
        {
            "ticker": ticker,
            "avg_confidence": scaled_confidence,
            "avg_wf_score": avg_wf_score,
            "ensemble_quality": ensemble_quality,
            "model_votes": model_votes,
        },
        decision,
    )

    # --- PERSIST DECISION to DB ---
    history_store.save_decision(
        payload={
            "ticker": ticker,
            "timestamp": today.isoformat(),
            "as_of_date": today.isoformat(),
            "model_votes": model_votes,
            "features_hash": features_hash,
            "model_version": _extract_model_version(model_votes),
        },
        decision=decision,
        explanation=explanation,
        audit={},  # integrate audit_builder here if/when needed
        model_votes=model_votes,
        safety_overrides={
            "safety_override": decision.get("safety_override"),
            "no_trade_reason": decision.get("no_trade_reason"),
            "reasons": decision.get("reasons", []),
            "warnings": decision.get("warnings", []),
        },
        model_id=None,
    )

    return {
        "ticker": ticker,
        "date": today.isoformat(),
        "timestamp": today.isoformat(),
        "as_of_date": today.isoformat(),
        "latest_price": latest_price,
        "features_hash": features_hash,
        "model_version": _extract_model_version(model_votes),
        "decision": decision,
        "explanation": explanation,
        "votes": votes,
        "volatility": volatility,
        "confidences": confidences,
        "raw_confidence": avg_confidence,
        "wf_scores": wf_scores,
        "model_votes": model_votes,
        "debug": debug_rows,
    }
