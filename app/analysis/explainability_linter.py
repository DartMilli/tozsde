from typing import Dict, List, Optional


def lint_explanation(
    explanation: Dict,
    decision: Dict,
    position_sizing: Optional[Dict] = None,
) -> Dict:
    issues: List[str] = []

    if not isinstance(explanation, dict):
        return {"ok": False, "issues": ["explanation_not_dict"]}

    for lang in ("hu", "en"):
        text = explanation.get(lang)
        if not isinstance(text, str) or not text.strip():
            issues.append(f"missing_{lang}_text")

    action = decision.get("action")
    if action:
        for lang in ("hu", "en"):
            text = explanation.get(lang, "")
            if action not in text:
                issues.append(f"action_missing_{lang}")

    meta = (
        explanation.get("meta", {}) if isinstance(explanation.get("meta"), dict) else {}
    )
    reasons_hu = meta.get("reasons_hu")
    reasons_en = meta.get("reasons_en")
    if isinstance(reasons_hu, list) and not reasons_hu:
        issues.append("reasons_hu_empty")
    if isinstance(reasons_en, list) and not reasons_en:
        issues.append("reasons_en_empty")

    if "Model votes" not in explanation.get("en", ""):
        issues.append("model_votes_missing_en")
    if "Modellek szavazatai" not in explanation.get("hu", ""):
        issues.append("model_votes_missing_hu")

    if decision.get("action_code") == 1 and position_sizing:
        if "Size:" not in explanation.get("en", ""):
            issues.append("size_missing_en")
        if "Meret:" not in explanation.get("hu", ""):
            issues.append("size_missing_hu")

    return {"ok": not issues, "issues": issues}
