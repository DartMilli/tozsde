from enum import Enum
from app.config.config import Config


class EnsembleQualityBucket(str, Enum):
    STRONG = "STRONG"
    NORMAL = "NORMAL"
    WEAK = "WEAK"
    CHAOTIC = "CHAOTIC"


def bucket_ensemble_quality(score: float) -> EnsembleQualityBucket:
    """
    P7.2.1 – bucket numeric ensemble quality

    score: float >= 0
    """

    if score >= Config.ENSEMBLE_QUALITY_THRESHOLDS['STRONG']:
        return EnsembleQualityBucket.STRONG
    elif score >= Config.ENSEMBLE_QUALITY_THRESHOLDS['NORMAL']:
        return EnsembleQualityBucket.NORMAL
    elif score >= Config.ENSEMBLE_QUALITY_THRESHOLDS['WEAK']:
        return EnsembleQualityBucket.WEAK
    else:
        return EnsembleQualityBucket.CHAOTIC
