from enum import Enum
from typing import Optional


class VolatilityBucket(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


def bucket_volatility(volatility: Optional[float]) -> VolatilityBucket:
    if volatility is None:
        return VolatilityBucket.NORMAL

    if volatility < 0.015:
        return VolatilityBucket.LOW
    if volatility < 0.03:
        return VolatilityBucket.NORMAL
    if volatility < 0.06:
        return VolatilityBucket.HIGH
    return VolatilityBucket.EXTREME
