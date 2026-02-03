"""Edge case tests for reporting.metrics BacktestReport."""

import pandas as pd

from app.reporting.metrics import BacktestReport


def test_backtest_report_serializes_series():
    diagnostics = {
        "equity": pd.Series([1, 2, 3])
    }
    report = BacktestReport(metrics={"a": 1.0}, diagnostics=diagnostics, meta={"x": 1})

    out = report.to_dict()

    assert out["diagnostics"]["equity"] == [1, 2, 3]
    assert out["metrics"]["a"] == 1.0
