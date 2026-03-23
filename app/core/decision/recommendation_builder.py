from typing import Dict, Mapping, Optional

from app.core.decision.decision_builder import compute_decision_quality


DEFAULT_ACTION_LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
}


def build_recommendation(
    payload: Dict,
    action_labels: Optional[Mapping[int, str]] = None,
    confidence_no_trade_threshold: float = 0.25,
    strong_confidence_threshold: float = 0.75,
    weak_confidence_threshold: float = 0.5,
    strong_wf_threshold: float = 0.6,
) -> Dict:
    avg_confidence = payload["avg_confidence"]
    avg_wf_score = payload["avg_wf_score"]
    action_code = payload["action_code"]
    ensemble_quality = payload["ensemble_quality"]

    labels = dict(DEFAULT_ACTION_LABELS)
    if action_labels:
        labels.update(dict(action_labels))

    original_action = action_code
    action_label = labels.get(action_code, str(action_code))
    strength = "NORMAL"

    no_trade = False
    no_trade_reason = None

    if avg_confidence < confidence_no_trade_threshold:
        no_trade = True
        no_trade_reason = "LOW_CONFIDENCE"
        action_code = 0
        action_label = labels.get(0, "HOLD")
        strength = "NO_TRADE"

    if not no_trade and action_code in (1, 2):
        if (
            avg_confidence >= strong_confidence_threshold
            and avg_wf_score is not None
            and avg_wf_score >= strong_wf_threshold
            and ensemble_quality == "STABLE"
        ):
            strength = "STRONG"
        elif avg_confidence < weak_confidence_threshold:
            strength = "WEAK"

    quality_score = compute_decision_quality(payload)

    return {
        "action_code": action_code,
        "action": action_label,
        "strength": strength,
        "confidence": avg_confidence,
        "wf_score": avg_wf_score,
        "ensemble_quality": ensemble_quality,
        "quality_score": quality_score,
        "no_trade": no_trade,
        "no_trade_reason": no_trade_reason,
        "original_action": original_action,
    }


def build_explanation(
    payload: Dict,
    decision: Dict,
    action_labels: Optional[Mapping[int, str]] = None,
) -> Dict:
    ticker = payload["ticker"]
    avg_conf = payload.get("avg_confidence")
    avg_wf = payload.get("avg_wf_score")
    ensemble_quality = payload.get("ensemble_quality")
    votes = payload.get("model_votes", [])

    action = decision["action"]
    strength = decision["strength"]
    quality_score = decision.get("quality_score", 0.0)

    labels = dict(DEFAULT_ACTION_LABELS)
    if action_labels:
        labels.update(dict(action_labels))

    reasons_hu = []
    reasons_en = []

    if decision.get("no_trade"):
        reason = decision.get("no_trade_reason", "POLICY_OVERRIDE")
        original_action = decision.get("original_action")

        reasons_hu.append(
            f"A rendszer nem engedte a kereskedest (policy ok: {reason})."
        )
        reasons_en.append(f"Trading was blocked by policy (reason: {reason}).")

        if original_action is not None:
            orig_label = labels.get(original_action, str(original_action))
            reasons_hu.append(f"Eredeti modell jelzes: {orig_label}.")
            reasons_en.append(f"Original model signal: {orig_label}.")

    if not decision.get("no_trade") and avg_wf is not None and avg_wf < 0.4:
        reasons_hu.append(f"A walk-forward stabilitas alacsony (WF={avg_wf:.2f}).")
        reasons_en.append(f"Walk-forward stability is low (WF={avg_wf:.2f}).")

    if ensemble_quality == "CHAOTIC":
        reasons_hu.append("A modellek jelzesei erosen elternek egymastol.")
        reasons_en.append("Model signals are highly divergent.")

    if not reasons_hu:
        reasons_hu.append("A modellek konzisztens es stabil jelzest adtak.")
        reasons_en.append("Models provided consistent and stable signals.")

    vote_lines = []
    for mv in votes:
        wf = mv.get("wf_score")
        wf_str = f"{wf:.2f}" if wf is not None else "n/a"

        vote_lines.append(
            f"{mv.get('model_name', mv.get('model_path', 'model'))} -> "
            f"{mv['action_label']} "
            f"(conf={mv['confidence']:.2f}, wf={wf_str})"
        )

    avg_conf_str = f"{avg_conf:.2f}" if avg_conf is not None else "n/a"
    avg_wf_str = f"{avg_wf:.2f}" if avg_wf is not None else "n/a"

    header_hu = (
        f"{ticker}: {strength} {action}\n"
        f"Osszbizalom: {avg_conf_str}, "
        f"WF: {avg_wf_str}, "
        f"Ensemble: {ensemble_quality}, "
        f"Minoseg pontszam: {quality_score:.2f}\n"
    )

    header_en = (
        f"{ticker}: {strength} {action}\n"
        f"Confidence: {avg_conf_str}, "
        f"WF: {avg_wf_str}, "
        f"Ensemble: {ensemble_quality}, "
        f"Decision quality score: {quality_score:.2f}\n"
    )

    explanation_hu = (
        header_hu
        + "\nIndoklas:\n- "
        + "\n- ".join(reasons_hu)
        + "\n\nModellek szavazatai:\n- "
        + "\n- ".join(vote_lines)
    )

    explanation_en = (
        header_en
        + "\nRationale:\n- "
        + "\n- ".join(reasons_en)
        + "\n\nModel votes:\n- "
        + "\n- ".join(vote_lines)
    )

    return {
        "hu": explanation_hu,
        "en": explanation_en,
        "meta": {
            "reasons_hu": reasons_hu,
            "reasons_en": reasons_en,
            "quality_score": quality_score,
        },
    }
