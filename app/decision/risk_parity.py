import numpy as np

from app.bootstrap.build_settings import build_settings
from app.core.decision.risk_parity import RiskParityAllocator as CoreRiskParityAllocator


class RiskParityAllocator(CoreRiskParityAllocator):
    def _get_settings(self):
        return self.settings or build_settings()


def apply_risk_parity(decisions, price_history):
    allocator = RiskParityAllocator(lookback_days=60)
    return allocator.allocate(decisions, price_history)
