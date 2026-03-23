import hashlib
from typing import Any, Dict, List, Optional


def compute_features_hash(df) -> Optional[str]:
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    pieces = [f"{col}={row[col]}" for col in df.columns]
    base = "|".join(pieces)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def extract_model_version(model_votes: List[dict]) -> Optional[str]:
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


def build_policy_payload(
    ticker: str,
    action_code: int,
    scaled_confidence: float,
    avg_wf_score: Optional[float],
    ensemble_quality: str,
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "avg_confidence": scaled_confidence,
        "avg_wf_score": avg_wf_score,
        "ensemble_quality": ensemble_quality,
        "action_code": action_code,
    }


def build_recommendation_response(
    ticker: str,
    today_iso: str,
    latest_price: Optional[float],
    features_hash: Optional[str],
    model_version: Optional[str],
    decision: Dict[str, Any],
    explanation: Dict[str, Any],
    votes: List[int],
    volatility: Optional[float],
    confidences: List[float],
    raw_confidence: float,
    wf_scores: List[float],
    model_votes: List[dict],
    debug_rows: List[dict],
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "date": today_iso,
        "timestamp": today_iso,
        "as_of_date": today_iso,
        "latest_price": latest_price,
        "features_hash": features_hash,
        "model_version": model_version,
        "decision": decision,
        "explanation": explanation,
        "votes": votes,
        "volatility": volatility,
        "confidences": confidences,
        "raw_confidence": raw_confidence,
        "wf_scores": wf_scores,
        "model_votes": model_votes,
        "debug": debug_rows,
    }
