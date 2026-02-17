import numpy as np

from app.validation.edge_diagnostics import (
    classify_collapse_reason,
    compute_fold_edge_stats,
    compute_global_summary,
)


def _summary_from_edges(edges, threshold=0.01):
    thresholds = [threshold for _ in edges]
    fold = compute_fold_edge_stats(
        fold_id=0,
        raw_signal_count=len(edges),
        expected_edges=edges,
        thresholds=thresholds,
    )
    summary = compute_global_summary([fold])
    return summary, fold


def test_strong_edge_dataset():
    threshold = 0.01
    edges = [threshold * 2 for _ in range(10)]
    summary, fold = _summary_from_edges(edges, threshold=threshold)
    assert fold["threshold_sensitivity"]["1.00"] > 0
    assert classify_collapse_reason(summary) is None


def test_weak_edge_dataset():
    threshold = 0.01
    edges = [threshold * 0.85 for _ in range(10)]
    summary, fold = _summary_from_edges(edges, threshold=threshold)
    assert fold["threshold_sensitivity"]["0.85"] > 0
    assert classify_collapse_reason(summary) == "weak_edge_distribution"


def test_noise_edge_dataset():
    threshold = 0.01
    rng = np.random.default_rng(42)
    edges = (rng.random(50) * threshold * 0.5).tolist()
    summary, _ = _summary_from_edges(edges, threshold=threshold)
    assert classify_collapse_reason(summary) == "no_edge_present"
