from app.bootstrap.build_settings import build_settings
from app.core.decision.ensemble_quality import (
    EnsembleQualityBucket,
    bucket_ensemble_quality as core_bucket_ensemble_quality,
)


def _get_settings(settings):
    return settings or build_settings()


def bucket_ensemble_quality(score: float, settings=None) -> EnsembleQualityBucket:
    """
    P7.2.1 - bucket numeric ensemble quality

    score: float >= 0
    """

    cfg = _get_settings(settings)
    thresholds = getattr(cfg, "ENSEMBLE_QUALITY_THRESHOLDS")
    return core_bucket_ensemble_quality(score, thresholds=thresholds)
