import datetime
import os

from app.backtesting.history_store import HistoryStore
from app.core.decision.decision_engine import DecisionEngine
from app.core.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.core.decision.recommender_helpers import (
    build_policy_payload,
    build_recommendation_response,
    compute_features_hash,
    extract_model_version,
)
from app.core.decision.safety_rules import SafetyRuleEngine
from app.core.decision.volatility import (
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)


def _build_recommendation_with_settings(payload, settings):
    from app.core.decision.recommendation_builder import (
        build_recommendation as core_build_recommendation,
    )

    action_labels = getattr(settings, "ACTION_LABELS")[getattr(settings, "LANG")]
    return core_build_recommendation(
        payload=payload,
        action_labels=action_labels,
        confidence_no_trade_threshold=getattr(
            settings, "CONFIDENCE_NO_TRADE_THRESHOLD"
        ),
        strong_confidence_threshold=getattr(settings, "STRONG_CONFIDENCE_THRESHOLD"),
        weak_confidence_threshold=getattr(settings, "WEAK_CONFIDENCE_THRESHOLD"),
        strong_wf_threshold=getattr(settings, "STRONG_WF_THRESHOLD"),
    )


def _build_explanation_with_settings(payload, decision, settings):
    from app.core.decision.recommendation_builder import (
        build_explanation as core_build_explanation,
    )

    action_labels = getattr(settings, "ACTION_LABELS")[getattr(settings, "LANG")]
    return core_build_explanation(
        payload=payload,
        decision=decision,
        action_labels=action_labels,
    )


def generate_daily_recommendation_payload(
    ticker: str,
    history_store: HistoryStore,
    top_n: int = 3,
    debug=True,
    data_fetcher=None,
    model_runner=None,
    as_of_date=None,
    settings=None,
    load_data_fn=None,
    prepare_df_fn=None,
    rl_runner_cls=None,
    safety_rule_engine_cls=SafetyRuleEngine,
    decision_engine_cls=DecisionEngine,
    build_recommendation_fn=None,
    build_explanation_fn=None,
) -> dict:
    if settings is None:
        from app.bootstrap.build_settings import build_settings

        settings = build_settings()

    if load_data_fn is None:
        from app.services.dependencies import MarketDataFetcher

        load_data_fn = lambda ticker, start, end: MarketDataFetcher().load_data(
            ticker, start=start, end=end
        )

    if prepare_df_fn is None:
        from app.data_access.data_cleaner import prepare_df

        prepare_df_fn = prepare_df

    if build_recommendation_fn is None:
        build_recommendation_fn = _build_recommendation_with_settings

    if build_explanation_fn is None:
        build_explanation_fn = _build_explanation_with_settings

    today = as_of_date or datetime.date.today()
    start = today - datetime.timedelta(days=180)

    cfg = settings

    if model_runner is None:
        if rl_runner_cls is None:
            from app.models.rl_inference import RLModelEnsembleRunner

            rl_runner_cls = RLModelEnsembleRunner

        from app.models.model_trainer import TradingEnv

        model_dir = getattr(cfg, "MODEL_DIR")
        model_runner = rl_runner_cls(model_dir=model_dir, env_class=TradingEnv)

    if data_fetcher is None:
        df_full = load_data_fn(
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

    df = prepare_df_fn(df_full.copy(), ticker)
    latest_price = None
    if not df.empty and "Close" in df.columns:
        latest_price = float(df["Close"].iloc[-1])
    features_hash = compute_features_hash(df)

    votes, confidences, wf_scores, model_votes, debug_rows = model_runner.run_ensemble(
        df=df,
        ticker=ticker,
        top_n=top_n,
        debug=debug,
    )

    if not votes:
        return {"error": "NO_MODELS"}

    action_code, avg_confidence, ensemble_quality = aggregate_weighted_ensemble(
        votes=votes,
        confidences=confidences,
        wf_scores=wf_scores,
        model_votes=model_votes,
    )

    avg_wf_score = sum(wf_scores) / len(wf_scores) if wf_scores else 1.0
    volatility = compute_normalized_volatility(df)
    scaled_confidence = scale_confidence_by_volatility(avg_confidence, volatility)

    if getattr(cfg, "ENABLE_CONFIDENCE_CALIBRATION"):
        from app.analysis.confidence_calibrator import ConfidenceCalibrator

        calibrator = ConfidenceCalibrator()
        params = calibrator.load_latest_params(
            ticker=ticker,
            as_of_date=today.isoformat(),
        )
        scaled_confidence = calibrator.apply(scaled_confidence, params)

    safety_engine = safety_rule_engine_cls(history_store, settings=settings)
    enable_safety = os.getenv("VALIDATION_DISABLE_SAFETY", "false").lower() != "true"
    decision_engine = decision_engine_cls(
        safety_engine=safety_engine,
        enable_safety=enable_safety,
        today=today,
    )

    policy_payload = build_policy_payload(
        ticker=ticker,
        action_code=action_code,
        scaled_confidence=scaled_confidence,
        avg_wf_score=avg_wf_score,
        ensemble_quality=ensemble_quality,
    )
    decision = build_recommendation_fn(policy_payload, settings=settings)
    decision = decision_engine.run(ticker=ticker, decision=decision)

    explanation = build_explanation_fn(
        {
            "ticker": ticker,
            "avg_confidence": scaled_confidence,
            "avg_wf_score": avg_wf_score,
            "ensemble_quality": ensemble_quality,
            "model_votes": model_votes,
        },
        decision,
        settings=settings,
    )

    history_store.save_decision(
        payload={
            "ticker": ticker,
            "timestamp": today.isoformat(),
            "as_of_date": today.isoformat(),
            "model_votes": model_votes,
            "features_hash": features_hash,
            "model_version": extract_model_version(model_votes),
        },
        decision=decision,
        explanation=explanation,
        audit={},
        model_votes=model_votes,
        safety_overrides={
            "safety_override": decision.get("safety_override"),
            "no_trade_reason": decision.get("no_trade_reason"),
            "reasons": decision.get("reasons", []),
            "warnings": decision.get("warnings", []),
        },
        model_id=None,
    )

    return build_recommendation_response(
        ticker=ticker,
        today_iso=today.isoformat(),
        latest_price=latest_price,
        features_hash=features_hash,
        model_version=extract_model_version(model_votes),
        decision=decision,
        explanation=explanation,
        votes=votes,
        volatility=volatility,
        confidences=confidences,
        raw_confidence=avg_confidence,
        wf_scores=wf_scores,
        model_votes=model_votes,
        debug_rows=debug_rows,
    )
