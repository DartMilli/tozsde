from enum import Enum
from typing import Mapping, Optional


class EnsembleQualityBucket(str, Enum):
    STRONG = "STRONG"
    NORMAL = "NORMAL"
    WEAK = "WEAK"
    CHAOTIC = "CHAOTIC"


DEFAULT_THRESHOLDS = {
    "STRONG": 0.6,
    "NORMAL": 0.3,
    "WEAK": 0.1,
}


def bucket_ensemble_quality(
    score: float, thresholds: Optional[Mapping[str, float]] = None
) -> EnsembleQualityBucket:
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update({k: float(v) for k, v in thresholds.items()})

    if score >= limits["STRONG"]:
        return EnsembleQualityBucket.STRONG
    if score >= limits["NORMAL"]:
        return EnsembleQualityBucket.NORMAL
    if score >= limits["WEAK"]:
        return EnsembleQualityBucket.WEAK
    return EnsembleQualityBucket.CHAOTIC
