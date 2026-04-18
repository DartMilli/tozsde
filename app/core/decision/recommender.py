import datetime
from dataclasses import asdict

from app.backtesting.history_store import HistoryStore
from app.core.decision.decision_engine import DecisionEngine
from app.core.decision.ensemble_aggregator import aggregate_weighted_ensemble
from app.core.decision.expectancy_gate import ExpectancyGate, bucket_confidence
from app.core.decision.ensemble_quality import (
    EnsembleQualityBucket,
    bucket_ensemble_quality,
)
from app.core.decision.recommender_helpers import (
    build_policy_payload,
    build_recommendation_response,
    compute_features_hash,
    extract_model_version,
)
from app.core.decision.regime_policy import get_regime_policy
from app.core.decision.safety_rules import SafetyRuleEngine
from app.core.decision.volatility import (
    compute_normalized_volatility,
    scale_confidence_by_volatility,
)
from app.infrastructure.logger import setup_logger
from app.models.model_reliability import load_latest_reliability_scores

logger = setup_logger(__name__)


def _try_add_ml_vote(
    df,
    ticker: str,
    model_dir,
    votes: list,
    confidences: list,
    wf_scores: list,
    model_votes: list,
) -> None:
    """Optionally inject an ML predictor vote into the ensemble.

    Silently skips if no trained model is found or prediction fails.
    The ML vote gets a fixed wf_score of 0.5 (neutral walk-forward trust)
    since it is not walk-forward validated like RL models.
    """
    import os
    import pickle
    import warnings
    from pathlib import Path

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    try:
        ml_dir = Path(model_dir) / "ml_predictor"
        # Find any .pkl model file for this ticker (or generic)
        candidates = sorted(ml_dir.glob(f"*{ticker}*.pkl")) + sorted(
            ml_dir.glob("*.pkl")
        )
        model_path = next((p for p in candidates if "_scaler" not in p.name), None)
        if model_path is None:
            logger.debug("ML ensemble: no trained model found in %s", ml_dir)
            return

        from app.models.ml_predictor import MLMarketPredictor

        predictor = MLMarketPredictor(model_dir=str(ml_dir))

        scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"

        with warnings.catch_warnings():
            if InconsistentVersionWarning is not None:
                warnings.filterwarnings(
                    "ignore",
                    category=InconsistentVersionWarning,
                )
            with open(model_path, "rb") as f:
                predictor.model = pickle.load(f)
        # Scaler is optional — window-relative normalization doesn't require it
        if scaler_path.exists():
            with warnings.catch_warnings():
                if InconsistentVersionWarning is not None:
                    warnings.filterwarnings(
                        "ignore",
                        category=InconsistentVersionWarning,
                    )
                with open(scaler_path, "rb") as f:
                    predictor.scaler = pickle.load(f)
        predictor.is_trained = True

        # Load feature list from meta file so predict_price_change uses the
        # same features the model was trained with (e.g. with indicators)
        import json as _json

        meta_path = model_path.parent / f"{model_path.stem}_meta.json"
        if meta_path.exists():
            with open(meta_path) as _mf:
                _meta = _json.load(_mf)
            _feat = _meta.get("features")
            if _feat:
                predictor._features = _feat

        # Normalise column names to lowercase for the predictor
        df_lc = df.copy()
        df_lc.columns = [c.lower() for c in df_lc.columns]
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df_lc.columns for c in needed):
            logger.debug("ML ensemble: required OHLCV columns not present in df")
            return

        result = predictor.predict_price_change(df_lc)
        change_pct = result.get("price_change_pct", 0.0)
        confidence = min(max(abs(result.get("confidence", 0.5)), 0.0), 1.0)

        if change_pct > 0.5:
            action_code = 1  # BUY
        elif change_pct < -0.5:
            action_code = 2  # SELL
        else:
            action_code = 0  # HOLD

        votes.append(action_code)
        confidences.append(confidence)
        wf_scores.append(0.5)  # neutral WF trust for ML models
        model_votes.append(
            {
                "model_path": str(model_path),
                "model_type": "ML",
                "action": action_code,
                "action_label": {0: "HOLD", 1: "BUY", 2: "SELL"}.get(
                    action_code, "HOLD"
                ),
                "confidence": confidence,
                "wf_score": 0.5,
                "trust_weight": 0.5,
                "decision_level": "ML",
                "trade_allowed": True,
                "price_change_pct": change_pct,
            }
        )
        logger.info(
            "ML ensemble vote ticker=%s action=%s change_pct=%.2f confidence=%.3f",
            ticker,
            action_code,
            change_pct,
            confidence,
        )
    except Exception as exc:
        logger.debug("ML ensemble skipped for %s: %s", ticker, exc)


def _quality_label(eq_float: float, settings) -> EnsembleQualityBucket:
    """Convert aggregate_weighted_ensemble's float quality score to an EnsembleQualityBucket.

    Delegates to bucket_ensemble_quality() so that threshold logic is not duplicated
    and Settings-level ENSEMBLE_QUALITY_THRESHOLDS overrides are always respected.
    """
    thresholds = (
        getattr(settings, "ENSEMBLE_QUALITY_THRESHOLDS", None)
        if settings is not None
        else None
    )
    return bucket_ensemble_quality(eq_float, thresholds)


def _build_recommendation_with_settings(payload, settings):
    from app.core.decision.recommendation_builder import (
        build_recommendation as core_build_recommendation,
    )

    action_labels = getattr(settings, "ACTION_LABELS")[getattr(settings, "LANG")]
    regime_policy = payload.get("regime_policy") or {}
    return core_build_recommendation(
        payload=payload,
        action_labels=action_labels,
        confidence_no_trade_threshold=regime_policy.get(
            "confidence_floor",
            getattr(settings, "CONFIDENCE_NO_TRADE_THRESHOLD"),
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
    persist_decision: bool = True,
) -> dict:
    if settings is None:
        from app.config.build_settings import build_settings

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

    regime_info = None
    if getattr(cfg, "ENABLE_REGIME_POLICY", False):
        from app.core.decision.market_regime_detector import MarketRegimeDetector

        regime_detector = MarketRegimeDetector(settings=cfg)
        regime_info = regime_detector.detect_regime("SPY")
        regime = regime_info.regime_type
    else:
        latest_adx = None
        if "ADX" in df.columns:
            adx_col = df["ADX"].dropna()
            if not adx_col.empty:
                latest_adx = float(adx_col.iloc[-1])
        if latest_adx is None:
            regime = "UNKNOWN"
        elif latest_adx > 25:
            regime = "TREND"
        elif latest_adx < 20:
            regime = "RANGE"
        else:
            regime = "TRANSITION"

    regime_policy = asdict(get_regime_policy(regime, settings=cfg))

    # Optional ML predictor vote (ENABLE_ML_ENSEMBLE=true)
    if getattr(cfg, "ENABLE_ML_ENSEMBLE", False):
        _try_add_ml_vote(
            df=df,
            ticker=ticker,
            model_dir=getattr(cfg, "MODEL_DIR"),
            votes=votes,
            confidences=confidences,
            wf_scores=wf_scores,
            model_votes=model_votes,
        )

    if not votes:
        return {"error": "NO_MODELS"}

    reliability_scores = load_latest_reliability_scores(
        ticker,
        as_of_date=today.isoformat(),
        settings=cfg,
    )

    action_code, avg_confidence, ensemble_quality_raw = aggregate_weighted_ensemble(
        votes=votes,
        confidences=confidences,
        wf_scores=wf_scores,
        model_votes=model_votes,
        reliability_scores=reliability_scores,
        settings=cfg,
    )
    ensemble_quality = _quality_label(ensemble_quality_raw, cfg)
    ensemble_quality_value = getattr(ensemble_quality, "value", str(ensemble_quality))

    # S14.3 Model disagreement signal: flag when individual votes are not unanimous.
    model_disagreement = len(set(votes)) > 1 if votes else False

    avg_wf_score = sum(wf_scores) / len(wf_scores) if wf_scores else 1.0
    volatility = compute_normalized_volatility(df)
    scaled_confidence = scale_confidence_by_volatility(avg_confidence, volatility)
    confidence_bucket = bucket_confidence(scaled_confidence, settings=cfg)

    strategy_selection = None
    selected_strategy_name = None
    if getattr(cfg, "ENABLE_ADAPTIVE_STRATEGY", False):
        from app.core.decision.adaptive_strategy_selector import (
            AdaptiveStrategySelector,
        )

        selector = AdaptiveStrategySelector(settings=cfg)
        strategy_selection = selector.explore_or_exploit(market_regime=regime)
        selected_strategy_name = max(
            strategy_selection.selected_strategies,
            key=strategy_selection.selected_strategies.get,
        )
        adaptive_multiplier = 0.75 + 0.25 * strategy_selection.confidence_in_selection
        scaled_confidence = min(1.0, scaled_confidence * adaptive_multiplier)
        confidence_bucket = bucket_confidence(scaled_confidence, settings=cfg)

    if getattr(cfg, "ENABLE_CONFIDENCE_CALIBRATION"):
        from app.analysis.confidence_calibrator import ConfidenceCalibrator

        calibrator = ConfidenceCalibrator()
        params = calibrator.load_latest_params(
            ticker=ticker,
            as_of_date=today.isoformat(),
        )
        scaled_confidence = calibrator.apply(scaled_confidence, params)
        confidence_bucket = bucket_confidence(scaled_confidence, settings=cfg)

    safety_engine = safety_rule_engine_cls(history_store, settings=settings)
    enable_safety = not getattr(cfg, "VALIDATION_DISABLE_SAFETY", False)
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
        ensemble_quality=ensemble_quality_value,
        regime=regime,
        regime_policy=regime_policy,
    )
    decision = build_recommendation_fn(policy_payload, settings=settings)

    quality_order = {"CHAOTIC": 0, "WEAK": 1, "NORMAL": 2, "STRONG": 3}
    if quality_order.get(ensemble_quality_value, 0) < quality_order.get(
        regime_policy.get("ensemble_quality_floor", "WEAK"),
        0,
    ):
        decision["action_code"] = 0
        decision["action"] = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0]
        decision["strength"] = "NO_TRADE"
        decision["no_trade"] = True
        decision["no_trade_reason"] = "REGIME_QUALITY_FLOOR"

    if decision.get("original_action") == 1 and not regime_policy.get(
        "allow_new_buys", True
    ):
        decision["action_code"] = 0
        decision["action"] = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0]
        decision["strength"] = "NO_TRADE"
        decision["no_trade"] = True
        decision["no_trade_reason"] = "REGIME_BLOCK_NEW_BUYS"

    decision["regime"] = regime
    decision["regime_policy"] = regime_policy
    if selected_strategy_name is not None:
        decision["strategy_name"] = selected_strategy_name

    expectancy_result = None
    if getattr(cfg, "ENABLE_EXPECTANCY_GATE", False):
        expectancy_gate = ExpectancyGate(settings=cfg)
        expectancy_result = expectancy_gate.evaluate(
            ticker=ticker,
            action_code=decision.get("original_action", decision.get("action_code", 0)),
            confidence_bucket=confidence_bucket,
            regime=regime,
            as_of_date=today,
        )
        if not expectancy_result.gate_pass:
            decision["action_code"] = 0
            decision["action"] = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0]
            decision["strength"] = "NO_TRADE"
            decision["no_trade"] = True
            decision["no_trade_reason"] = (
                f"EXPECTANCY_NEGATIVE: {expectancy_result.reason}"
            )

    decision = decision_engine.run(ticker=ticker, decision=decision)

    # Attach disagreement flag; log it prominently so operators notice split votes.
    decision["model_disagreement"] = model_disagreement
    if model_disagreement:
        reasons = decision.get("reasons") or []
        if "MODEL_DISAGREEMENT" not in reasons:
            reasons.append("MODEL_DISAGREEMENT")
        decision["reasons"] = reasons

    explanation = build_explanation_fn(
        {
            "ticker": ticker,
            "avg_confidence": scaled_confidence,
            "avg_wf_score": avg_wf_score,
            "ensemble_quality": ensemble_quality_value,
            "model_votes": model_votes,
        },
        decision,
        settings=settings,
    )

    average_reliability = 0.0
    if model_votes:
        reliability_values = [
            reliability_scores.get(model_vote.get("model_path"), 0.5)
            for model_vote in model_votes
        ]
        average_reliability = sum(reliability_values) / len(reliability_values)

    if persist_decision:
        history_store.save_decision(
            payload={
                "ticker": ticker,
                "timestamp": today.isoformat(),
                "as_of_date": today.isoformat(),
                "model_votes": model_votes,
                "features_hash": features_hash,
                "model_version": extract_model_version(model_votes),
                "reliability_score": average_reliability,
            },
            decision=decision,
            explanation=explanation,
            audit={
                "regime": regime,
                "regime_info": asdict(regime_info) if regime_info is not None else {},
                "regime_policy": regime_policy,
                "confidence_bucket": confidence_bucket,
                "ensemble_quality": ensemble_quality_value,
                "reliability_scores": reliability_scores,
                "adaptive_strategy": (
                    {
                        "selected_strategies": strategy_selection.selected_strategies,
                        "selection_mode": strategy_selection.selection_mode,
                        "market_context": strategy_selection.market_context,
                        "confidence_in_selection": strategy_selection.confidence_in_selection,
                        "selected_strategy": selected_strategy_name,
                    }
                    if strategy_selection is not None
                    else {}
                ),
                "expectancy": (
                    {
                        "expected_pnl": expectancy_result.expected_pnl,
                        "expected_net_pnl": expectancy_result.expected_net_pnl,
                        "sample_count": expectancy_result.sample_count,
                        "gate_pass": expectancy_result.gate_pass,
                        "reason": expectancy_result.reason,
                    }
                    if expectancy_result is not None
                    else {}
                ),
            },
            model_votes=model_votes,
            safety_overrides={
                "safety_override": decision.get("safety_override"),
                "no_trade_reason": decision.get("no_trade_reason"),
                "reasons": decision.get("reasons", []),
                "warnings": decision.get("warnings", []),
            },
            model_id=None,
        )

    response = build_recommendation_response(
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
    response["reliability_scores"] = reliability_scores
    response["confidence_bucket"] = confidence_bucket
    response["regime"] = regime
    response["regime_policy"] = regime_policy
    if regime_info is not None:
        response["regime_info"] = asdict(regime_info)
    if strategy_selection is not None:
        response["adaptive_strategy"] = {
            "selected_strategies": strategy_selection.selected_strategies,
            "selection_mode": strategy_selection.selection_mode,
            "market_context": strategy_selection.market_context,
            "confidence_in_selection": strategy_selection.confidence_in_selection,
            "selected_strategy": selected_strategy_name,
        }
    if expectancy_result is not None:
        response["expectancy"] = {
            "expected_pnl": expectancy_result.expected_pnl,
            "expected_net_pnl": expectancy_result.expected_net_pnl,
            "sample_count": expectancy_result.sample_count,
            "gate_pass": expectancy_result.gate_pass,
            "reason": expectancy_result.reason,
        }
    return response
