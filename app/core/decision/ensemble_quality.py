from enum import Enum
from typing import Mapping, Optional, Union


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
    score: Union[float, "EnsembleQualityBucket", str],
    thresholds: Optional[Mapping[str, float]] = None,
) -> EnsembleQualityBucket:
    """Convert a quality score to an EnsembleQualityBucket.

    Accepts:
    - float: numeric quality score bucketed against thresholds
    - EnsembleQualityBucket: returned as-is (already bucketed)
    - str: parsed as enum value (e.g., "CHAOTIC" → EnsembleQualityBucket.CHAOTIC)
    """
    if isinstance(score, EnsembleQualityBucket):
        return score
    if isinstance(score, str):
        try:
            return EnsembleQualityBucket(score.upper())
        except ValueError:
            pass

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update({k: float(v) for k, v in thresholds.items()})

    numeric = float(score)
    if numeric >= limits["STRONG"]:
        return EnsembleQualityBucket.STRONG
    if numeric >= limits["NORMAL"]:
        return EnsembleQualityBucket.NORMAL
    if numeric >= limits["WEAK"]:
        return EnsembleQualityBucket.WEAK
    return EnsembleQualityBucket.CHAOTIC
