def _format_audit_line(audit: dict) -> str:
    return (
        f"Q={audit['quality_score']:.2f} | "
        f"Conf={audit['confidence']:.2f} | "
        f"WF={audit['wf_score'] if audit['wf_score'] is not None else 'n/a'} | "
        f"{audit['ensemble_quality']} | "
        f"Models={audit['model_count']}"
    )


def format_email_summary(
    ticker: str,
    decision: dict,
    audit: dict,
    position_sizing: dict = None,
    lang: str = "hu",
) -> str:
    action = decision.get("action", "HOLD")
    size_text = ""
    if position_sizing and decision.get("action_code") == 1:
        final_size = position_sizing.get("final_size")
        if final_size is not None:
            size_text = f" | Size=${final_size:,.2f}"
    label = "Action" if lang == "en" else "Action"
    return f"{ticker}: {label}={action}{size_text}"


def format_email_detail(
    explanation: dict,
    audit: dict,
    lang: str = "hu",
) -> str:
    base = explanation[lang]
    return f"{base}\n[{_format_audit_line(audit)}]"


def format_email_line(
    explanation: dict,
    decision: dict,
    audit: dict,
    lang: str = "hu",
) -> str:
    return format_email_detail(explanation=explanation, audit=audit, lang=lang)
