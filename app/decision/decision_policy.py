from app.bootstrap.build_settings import build_settings
from app.core.decision.decision_policy import (
    apply_decision_policy as core_apply_decision_policy,
)


def apply_decision_policy(decision: dict, audit: dict, settings=None) -> dict:
    """
    P7.1.2 - P7.1.4
    Applies reliability-based policy, reason normalization and reward hint.
    Mutates and returns decision.
    """

    cfg = settings or build_settings()
    hold_action_label = getattr(cfg, "ACTION_LABELS")[getattr(cfg, "LANG")][0]
    return core_apply_decision_policy(
        decision=decision,
        audit=audit,
        hold_action_label=hold_action_label,
    )
