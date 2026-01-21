import datetime
from collections import Counter

from app.data_access.data_loader import load_data
from app.data_access.data_cleaner import prepare_df
from app.config.config import Config
from app.models.rl_inference import RLModelEnsembleRunner
from app.models.model_trainer import TradingEnv  # ← env itt marad
from app.decision.decision_engine import DecisionEngine
from app.decision.safety_rules import SafetyRuleEngine
from app.backtesting.history_store import HistoryStore
from app.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.decision.volatility import (
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)
from app.decision.recommendation_builder import (
    build_recommendation,
    build_explanation,
)


def generate_daily_recommendation_payload(
    ticker: str, history_store: HistoryStore, top_n: int = 3, debug=True
) -> dict:
    today = datetime.date.today()
    start = today - datetime.timedelta(days=180)

    df_full = load_data(ticker, start=start.strftime("%Y-%m-%d"))
    if df_full.empty:
        return {"error": "NO_DATA"}

    df = prepare_df(df_full.copy(), ticker)

    runner = RLModelEnsembleRunner(
        model_dir=Config.MODEL_DIR,
        env_class=TradingEnv,
    )

    votes, confidences, wf_scores, model_votes, debug_rows = runner.run_ensemble(
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

    safety_engine = SafetyRuleEngine(history_store)

    decision_engine = DecisionEngine(
        safety_engine=safety_engine,
        enable_safety=True,
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
            "model_votes": model_votes,
        },
        decision=decision,
        explanation=explanation,
        audit={},  # integrate audit_builder here if/when needed
    )

    return {
        "ticker": ticker,
        "date": today.isoformat(),
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
