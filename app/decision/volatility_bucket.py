from enum import Enum
from typing import Optional


class VolatilityBucket(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


def bucket_volatility(volatility):
    # type: (Optional[float]) -> VolatilityBucket
    """
    P7.4.1 – Volatility interpretation for audit / explain
    volatility is normalized ATR / close
    """

    if volatility is None:
        return VolatilityBucket.NORMAL

    if volatility < 0.015:
        return VolatilityBucket.LOW
    elif volatility < 0.03:
        return VolatilityBucket.NORMAL
    elif volatility < 0.06:
        return VolatilityBucket.HIGH
    else:
        return VolatilityBucket.EXTREME
