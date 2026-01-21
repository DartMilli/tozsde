from collections import Counter
import datetime

from app.config.config import Config
from app.decision.ensemble_quality import bucket_ensemble_quality
from app.decision.volatility_bucket import bucket_volatility


def decision_reliability_level(
    confidence: float | None, wf_score: float | None
) -> tuple[str, bool]:
    """
    P7.1.1 – Descriptive decision reliability
    Returns: (decision_level, trade_allowed)
    """
    if confidence is None:
        return "UNKNOWN", False

    if confidence >= 0.7 and (wf_score is None or wf_score >= 0.6):
        return "STRONG", True

    if confidence >= 0.5:
        return "NORMAL", True

    if confidence >= 0.3:
        return "WEAK", False

    return "NO_TRADE", False


def compute_consistency_flags(payload: dict, decision: dict) -> dict:
    """
    P4.5 – Consistency & audit flags
    """

    votes = payload.get("model_votes", [])
    actions = [v["action"] for v in votes]

    vote_counter = Counter(actions)
    majority_action = vote_counter.most_common(1)[0][0] if vote_counter else None

    flags = {
        "majority_action": majority_action,
        "majority_action_label": (
            Config.ACTION_LABELS[Config.LANG][majority_action]
            if majority_action is not None
            else None
        ),
        "executed_action": decision["action_code"],
        "executed_action_label": Config.ACTION_LABELS[Config.LANG][
            decision["action_code"]
        ],
        "matches_majority": majority_action == decision["action_code"],
        "was_policy_override": decision.get("no_trade", False),
    }

    # --- Divergence detection ---
    if majority_action is not None:
        divergence = abs(vote_counter[majority_action] / len(actions) - 1.0)
        flags["vote_divergence"] = round(divergence, 3)
    else:
        flags["vote_divergence"] = None

    return flags


def confidence_bucket(confidence: float) -> str:
    if confidence is None:
        return "UNKNOWN"
    if confidence < 0.3:
        return "VERY_LOW"
    if confidence < 0.5:
        return "LOW"
    if confidence < 0.7:
        return "MEDIUM"
    return "HIGH"


def build_audit_metadata(payload: dict, decision: dict) -> dict:
    flags = compute_consistency_flags(payload, decision)

    decision_level, trade_allowed = decision_reliability_level(
        decision.get("confidence"),
        decision.get("wf_score"),
    )

    # --- P7.2.2 ensemble-chaos hard block ---
    if (
        bucket_ensemble_quality(decision.get("ensemble_quality", 0.0)).value
        == "CHAOTIC"
    ):
        decision_level = "NO_TRADE"
        trade_allowed = False

    raw_ensemble_quality = decision.get("ensemble_quality", 0.0)
    raw_volatility = payload.get("volatility")

    return {
        "consistency": flags,
        "confidence_bucket": confidence_bucket(decision.get("confidence")),
        "quality_score": decision.get("quality_score"),
        "ensemble_quality": bucket_ensemble_quality(raw_ensemble_quality).value,
        "volatility_bucket": bucket_volatility(raw_volatility).value,
        "decision_level": decision_level,  # P7.1.1
        "trade_allowed": trade_allowed,  # P7.1.1
        "timestamp_utc": datetime.utcnow().isoformat(),
    }


def build_audit_summary(
    audit: dict,
    payload: dict,
    decision: dict,
) -> dict:
    """
    Audit summary for email / logs

    Derived view from audit metadata
    """
    return {
        "quality_score": round(audit["quality_score"], 3),
        "confidence": round(decision["confidence"], 3),
        "wf_score": (
            round(decision["wf_score"], 3)
            if decision.get("wf_score") is not None
            else None
        ),
        "ensemble_quality": audit["ensemble_quality"],
        "confidence_bucket": audit["confidence_bucket"],
        "volatility": audit.get("volatility_bucket"),
        "decision_level": audit.get("decision_level"),
        "trade_allowed": audit.get("trade_allowed"),
        "model_count": len(payload.get("model_votes", [])),
    }
