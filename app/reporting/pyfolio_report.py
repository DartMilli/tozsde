"""
PyFolio Integration Module (P9 - Bonus Analysis)

Responsibility:
    - Generate comprehensive performance analysis using PyFolio
    - Compute rolling Sharpe ratio, max drawdown, Calmar ratio
    - Stability analysis and rolling volatility
    - Integration with reporting module

Features:
    - Rolling performance metrics (252-day windows)
    - Risk-adjusted returns
    - Drawdown analysis
    - Stability scoring

Usage:
    report = generate_pyfolio_report(returns_series, positions_df)
    print(f"Calmar ratio: {report['calmar_ratio']:.2f}")
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class PyFolioReportGenerator:
    """
    Comprehensive performance analysis using PyFolio.

    Requires: pyfolio package (optional dependency)
    Install: pip install pyfolio
    """

    def __init__(self):
        """Initialize PyFolio report generator."""
        self.pyfolio_available = self._check_pyfolio()

    def _check_pyfolio(self) -> bool:
        """Check if PyFolio is installed."""
        # TODO: Implement
        try:
            import pyfolio

            return True
        except ImportError:
            logger.warning("PyFolio not installed. Install with: pip install pyfolio")
            return False

    def generate_report(
        self,
        returns_series: pd.Series,
        positions_df: pd.DataFrame = None,
        benchmark: pd.Series = None,
    ) -> Dict:
        """
        Generate comprehensive PyFolio performance report.

        Args:
            returns_series: Daily returns (datetime index)
            positions_df: Position history (optional)
            benchmark: Benchmark returns for comparison (optional)

        Returns:
            {
                "sharpe_ratio": float,
                "calmar_ratio": float,
                "max_drawdown": float,
                "stability": float,
                "rolling_sharpe": Series,
                "rolling_drawdown": Series,
                "error": str or None
            }
        """
        # TODO: Implement

        if not self.pyfolio_available:
            return {
                "error": "PyFolio not installed",
                "sharpe_ratio": None,
                "calmar_ratio": None,
            }

        try:
            import pyfolio as pf

            # Compute metrics
            sharpe = pf.stats.sharpe_ratio(returns_series)
            calmar = pf.stats.calmar_ratio(returns_series)
            max_dd = pf.stats.max_drawdown(returns_series)
            stability = pf.stats.stability_of_timeseries(returns_series)

            # Rolling metrics
            rolling_sharpe = pf.stats.rolling_sharpe(returns_series, rolling_window=252)
            rolling_drawdown = pf.stats.rolling_max_drawdown(
                returns_series, rolling_window=252
            )

            return {
                "sharpe_ratio": float(sharpe),
                "calmar_ratio": float(calmar),
                "max_drawdown": float(max_dd),
                "stability": float(stability),
                "rolling_sharpe": rolling_sharpe.to_dict(),
                "rolling_drawdown": rolling_drawdown.to_dict(),
                "error": None,
            }

        except Exception as e:
            logger.error(f"PyFolio report generation failed: {e}")
            return {"error": str(e), "sharpe_ratio": None, "calmar_ratio": None}

    def generate_full_tearsheet(
        self, returns_series: pd.Series, live_start_date: str = None
    ) -> Dict:
        """
        Generate full PyFolio tearsheet with all analytics.

        Args:
            returns_series: Daily returns
            live_start_date: Date when live trading started (YYYY-MM-DD)

        Returns:
            Comprehensive performance analysis dictionary
        """
        # TODO: Implement

        if not self.pyfolio_available:
            return {"error": "PyFolio not available"}

        try:
            import pyfolio as pf

            # Generate full tearsheet metrics
            tearsheet_metrics = {
                "total_return": (1 + returns_series).prod() - 1,
                "annual_return": returns_series.mean() * 252,
                "annual_volatility": returns_series.std() * np.sqrt(252),
                "sharpe_ratio": pf.stats.sharpe_ratio(returns_series),
                "calmar_ratio": pf.stats.calmar_ratio(returns_series),
                "max_drawdown": pf.stats.max_drawdown(returns_series),
                "downside_risk": pf.stats.downside_risk(returns_series),
                "sortino_ratio": pf.stats.sortino_ratio(returns_series),
                "stability": pf.stats.stability_of_timeseries(returns_series),
                "omega_ratio": pf.stats.omega_ratio(returns_series),
            }

            return tearsheet_metrics

        except Exception as e:
            logger.error(f"Full tearsheet generation failed: {e}")
            return {"error": str(e)}


# Utility functions


def generate_pyfolio_report(
    returns_series: pd.Series, positions_df: pd.DataFrame = None
) -> Dict:
    """
    Convenience function: generate PyFolio report.

    Args:
        returns_series: Daily returns series
        positions_df: Position history (optional)

    Returns:
        Comprehensive performance metrics
    """
    # TODO: Implement
    generator = PyFolioReportGenerator()
    return generator.generate_report(returns_series, positions_df)


def analyze_drawdown_periods(
    returns_series: pd.Series, min_drawdown_pct: float = 0.10
) -> Dict:
    """
    Identify major drawdown periods.

    Args:
        returns_series: Daily returns
        min_drawdown_pct: Minimum drawdown to report (10% default)

    Returns:
        {
            "drawdown_periods": [{start_date, end_date, depth_pct}, ...],
            "total_drawdown_periods": int,
            "avg_drawdown_depth": float
        }
    """
    # TODO: Implement

    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    # Find drawdown periods
    periods = []
    in_drawdown = False
    start_date = None
    max_dd = 0.0

    for date, dd in drawdown.items():
        if dd < -min_drawdown_pct and not in_drawdown:
            in_drawdown = True
            start_date = date
            max_dd = dd
        elif dd < max_dd:
            max_dd = dd
        elif dd >= 0 and in_drawdown:
            periods.append(
                {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": date.strftime("%Y-%m-%d"),
                    "depth_pct": abs(max_dd),
                }
            )
            in_drawdown = False
            max_dd = 0.0

    return {
        "drawdown_periods": periods,
        "total_drawdown_periods": len(periods),
        "avg_drawdown_depth": (
            np.mean([p["depth_pct"] for p in periods]) if periods else 0.0
        ),
    }
