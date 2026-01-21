def format_email_line(
    explanation: dict,
    decision: dict,
    audit: dict,
    lang: str = "hu",
) -> str:
    base = explanation[lang]

    audit_line = (
        f"Q={audit['quality_score']:.2f} | "
        f"Conf={audit['confidence']:.2f} | "
        f"WF={audit['wf_score'] if audit['wf_score'] is not None else 'n/a'} | "
        f"{audit['ensemble_quality']} | "
        f"Models={audit['model_count']}"
    )

    return f"{base}\n[{audit_line}]"
