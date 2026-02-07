"""
Admin Dashboard - Enhanced monitoring and analytics UI endpoints.

This module provides Flask API endpoints for the admin dashboard, including
strategy performance monitoring, capital utilization tracking, decision quality
metrics, and real-time system health monitoring.

Key Features:
- Strategy performance charts (win rate, Sharpe, drawdown)
- Decision quality metrics
- Capital utilization dashboard
- Real-time drift monitoring
- Error rate tracking
- System health indicators
"""

from flask import Blueprint, jsonify, request
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from app.config.config import Config

logger = logging.getLogger(__name__)

# Create Flask blueprint
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _check_admin_auth():
    """Verify admin API key from request header."""
    api_key = request.headers.get("X-Admin-Key")
    return api_key == Config.ADMIN_API_KEY


def _parse_positive_int(value, default: int, field_name: str):
    if value is None:
        return default, None
    try:
        parsed = int(value)
        if parsed <= 0:
            return None, f"{field_name} must be positive"
        return parsed, None
    except (TypeError, ValueError):
        return None, f"Invalid {field_name}"


@admin_bp.before_request
def _require_admin_auth():
    """Enforce admin authentication on all admin endpoints."""
    if not _check_admin_auth():
        return jsonify({"error": "Unauthorized"}), 401


@admin_bp.route("/health", methods=["GET"])
def health_check():
    """
    System health check endpoint.

    Returns:
        JSON: System health status
    """
    try:
        from app.infrastructure.metrics import get_metrics

        metrics = get_metrics()
        health_status = metrics.get_health_status()
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@admin_bp.route("/performance/summary", methods=["GET"])
def get_performance_summary():
    """
    Get overall performance summary.

    Query params:
        days: Number of days to analyze (default: 30)

    Returns:
        JSON: Performance metrics summary
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 30, "days")
        if err:
            return jsonify({"error": err}), 400

        # Import analytics module
        from app.reporting.performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        returns, dates = analytics.load_returns_from_db(days_back=days)

        if not returns:
            return (
                jsonify(
                    {"message": "No performance data available", "period_days": days}
                ),
                200,
            )

        metrics = analytics.calculate_performance_metrics(returns, dates)

        response = {
            "total_return": round(metrics.total_return * 100, 2),
            "annualized_return": round(metrics.annualized_return * 100, 2),
            "volatility": round(metrics.volatility * 100, 2),
            "sharpe_ratio": round(metrics.sharpe_ratio, 2),
            "sortino_ratio": round(metrics.sortino_ratio, 2),
            "calmar_ratio": round(metrics.calmar_ratio, 2),
            "max_drawdown": round(metrics.max_drawdown * 100, 2),
            "win_rate": round(metrics.win_rate * 100, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "total_trades": metrics.total_trades,
            "period_start": metrics.period_start.isoformat(),
            "period_end": metrics.period_end.isoformat(),
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/performance/drawdown", methods=["GET"])
def get_drawdown_analysis():
    """
    Get detailed drawdown analysis.

    Query params:
        days: Number of days to analyze (default: 90)

    Returns:
        JSON: Drawdown statistics
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 90, "days")
        if err:
            return jsonify({"error": err}), 400

        from app.reporting.performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        returns, dates = analytics.load_returns_from_db(days_back=days)

        if not returns:
            return (
                jsonify({"message": "No drawdown data available", "period_days": days}),
                200,
            )

        dd_analysis = analytics.analyze_drawdowns(returns, dates)

        response = {
            "max_drawdown": round(dd_analysis.max_drawdown * 100, 2),
            "max_drawdown_duration_days": dd_analysis.max_drawdown_duration_days,
            "current_drawdown": round(dd_analysis.current_drawdown * 100, 2),
            "drawdown_start": (
                dd_analysis.drawdown_start.isoformat()
                if dd_analysis.drawdown_start
                else None
            ),
            "drawdown_end": (
                dd_analysis.drawdown_end.isoformat()
                if dd_analysis.drawdown_end
                else None
            ),
            "recovery_date": (
                dd_analysis.recovery_date.isoformat()
                if dd_analysis.recovery_date
                else None
            ),
            "time_to_recovery_days": dd_analysis.time_to_recovery_days,
            "total_drawdowns": len(dd_analysis.drawdowns),
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting drawdown analysis: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/performance/rolling", methods=["GET"])
def get_rolling_performance():
    """
    Get rolling window performance metrics.

    Query params:
        days: Total days to analyze (default: 90)
        window: Rolling window size in days (default: 30)

    Returns:
        JSON: Rolling performance data
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 90, "days")
        if err:
            return jsonify({"error": err}), 400
        window, err = _parse_positive_int(request.args.get("window"), 30, "window")
        if err:
            return jsonify({"error": err}), 400

        from app.reporting.performance_analytics import PerformanceAnalytics

        analytics = PerformanceAnalytics()
        returns, dates = analytics.load_returns_from_db(days_back=days)

        if not returns:
            return (
                jsonify(
                    {"message": "No performance data available", "period_days": days}
                ),
                200,
            )

        rolling = analytics.calculate_rolling_metrics(
            returns, dates, window_days=window
        )

        response = {
            "window_size_days": rolling.window_size_days,
            "data_points": len(rolling.returns),
            "rolling_returns": [round(r * 100, 2) for r in rolling.returns],
            "rolling_volatilities": [round(v * 100, 2) for v in rolling.volatilities],
            "rolling_sharpe_ratios": [round(s, 2) for s in rolling.sharpe_ratios],
            "dates": [d.isoformat() for d in rolling.dates],
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting rolling performance: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/performance/pyfolio", methods=["GET"])
def get_pyfolio_report():
    """
    Get PyFolio performance report (optional dependency).

    Query params:
        days: Number of days to analyze (default: 252)

    Returns:
        JSON: PyFolio metrics and rolling stats
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 252, "days")
        if err:
            return jsonify({"error": err}), 400

        from app.reporting.performance_analytics import PerformanceAnalytics
        from app.reporting.pyfolio_report import PyFolioReportGenerator
        import pandas as pd

        try:
            analytics = PerformanceAnalytics(db_path=str(Config.DB_PATH))
        except TypeError:
            analytics = PerformanceAnalytics()
        returns, dates = analytics.load_returns_from_db(days_back=days)

        if not returns:
            return (
                jsonify(
                    {"message": "No performance data available", "period_days": days}
                ),
                200,
            )

        series = pd.Series(returns, index=pd.to_datetime(dates))
        generator = PyFolioReportGenerator()
        report = generator.generate_report(series)

        return jsonify(report), 200
    except Exception as e:
        logger.error(f"Error getting PyFolio report: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/errors/summary", methods=["GET"])
def get_error_summary():
    """
    Get error statistics summary.

    Query params:
        hours: Number of hours to analyze (default: 24)

    Returns:
        JSON: Error statistics
    """
    try:
        hours, err = _parse_positive_int(request.args.get("hours"), 24, "hours")
        if err:
            return jsonify({"error": err}), 400

        from app.infrastructure.error_reporter import ErrorReporter

        reporter = ErrorReporter()
        stats = reporter.get_error_statistics(hours_back=hours)

        response = {
            "total_errors": stats.total_errors,
            "errors_by_severity": stats.errors_by_severity,
            "errors_by_type": stats.errors_by_type,
            "errors_by_module": stats.errors_by_module,
            "error_rate_per_hour": round(stats.error_rate_per_hour, 2),
            "most_common_error": stats.most_common_error,
            "critical_errors": stats.critical_errors,
            "period_hours": hours,
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting error summary: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/errors/recent", methods=["GET"])
def get_recent_errors():
    """
    Get recent error records.

    Query params:
        limit: Maximum number of errors to return (default: 50)
        severity: Filter by severity (optional)

    Returns:
        JSON: List of recent errors
    """
    try:
        limit, err = _parse_positive_int(request.args.get("limit"), 50, "limit")
        if err:
            return jsonify({"error": err}), 400
        severity_str = request.args.get("severity")

        from app.infrastructure.error_reporter import ErrorReporter, ErrorSeverity

        severity = None
        if severity_str:
            try:
                severity = ErrorSeverity[severity_str.upper()]
            except KeyError:
                return jsonify({"error": f"Invalid severity: {severity_str}"}), 400

        reporter = ErrorReporter()
        errors = reporter.get_recent_errors(limit=limit, severity=severity)

        return jsonify({"errors": errors, "count": len(errors)}), 200
    except Exception as e:
        logger.error(f"Error getting recent errors: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/errors/trends", methods=["GET"])
def get_error_trends():
    """
    Get error trends over time.

    Query params:
        days: Number of days to analyze (default: 7)

    Returns:
        JSON: Error trend data
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 7, "days")
        if err:
            return jsonify({"error": err}), 400

        from app.infrastructure.error_reporter import ErrorReporter

        reporter = ErrorReporter()
        trends = reporter.get_error_trends(days=days)

        return jsonify(trends), 200
    except Exception as e:
        logger.error(f"Error getting error trends: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/capital/utilization", methods=["GET"])
def get_capital_utilization():
    """
    Get current capital utilization metrics.

    Returns:
        JSON: Capital utilization data
    """
    try:
        from app.decision.capital_optimizer import CapitalUtilizationOptimizer

        optimizer = CapitalUtilizationOptimizer()

        # Get position history
        history = optimizer.get_position_history()

        if not history:
            return jsonify({"message": "No capital utilization data available"}), 200

        # Calculate summary metrics
        recent_positions = history[:10] if len(history) >= 10 else history

        avg_utilization = sum(p["portfolio_weight"] for p in recent_positions) / len(
            recent_positions
        )
        avg_kelly = sum(p["kelly_fraction"] for p in recent_positions) / len(
            recent_positions
        )

        response = {
            "average_utilization": round(avg_utilization * 100, 2),
            "average_kelly_fraction": round(avg_kelly, 3),
            "total_positions": len(history),
            "recent_positions": recent_positions[:5],
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting capital utilization: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/decisions/no-trades", methods=["GET"])
def get_no_trade_decisions():
    """
    Get no-trade decision analysis.

    Query params:
        days: Number of days to analyze (default: 7)

    Returns:
        JSON: No-trade decision statistics
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 7, "days")
        if err:
            return jsonify({"error": err}), 400

        from app.infrastructure.decision_logger import NoTradeDecisionLogger

        logger_inst = NoTradeDecisionLogger()
        analysis = logger_inst.get_no_trade_analysis(days_back=days)

        return jsonify(analysis), 200
    except Exception as e:
        logger.error(f"Error getting no-trade decisions: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/strategies/performance", methods=["GET"])
def get_strategy_performance():
    """
    Get performance by strategy.

    Query params:
        days: Number of days to analyze (default: 30)

    Returns:
        JSON: Strategy performance breakdown
    """
    try:
        days, err = _parse_positive_int(request.args.get("days"), 30, "days")
        if err:
            return jsonify({"error": err}), 400

        from app.decision.decision_history_analyzer import DecisionHistoryAnalyzer

        analyzer = DecisionHistoryAnalyzer()

        # Get all strategies
        strategies = ["momentum", "mean_reversion", "breakout"]  # Example strategies

        performance = {}
        for strategy in strategies:
            stats = analyzer.analyze_strategy_performance(strategy, days=days)
            if stats and stats.trades_analyzed > 0:
                performance[strategy] = {
                    "win_rate": round(stats.win_rate * 100, 2),
                    "sharpe": round(stats.sharpe_ratio, 2),
                    "max_drawdown": round(stats.max_drawdown * 100, 2),
                    "total_trades": stats.trades_analyzed,
                }

        return jsonify({"strategies": performance, "period_days": days}), 200
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/confidence/distribution", methods=["GET"])
def get_confidence_distribution():
    """
    Get confidence score distribution.

    Returns:
        JSON: Confidence bucket statistics
    """
    try:
        from app.decision.confidence_allocator import ConfidenceBucketAllocator

        allocator = ConfidenceBucketAllocator()
        bucket_stats = allocator.get_bucket_statistics()

        distribution = {}
        for bucket, stats in bucket_stats.items():
            distribution[bucket.value] = {
                "count": stats.count,
                "avg_confidence": round(stats.avg_confidence, 3),
                "total_capital": round(stats.total_capital, 2),
                "avg_multiplier": round(stats.avg_multiplier, 2),
            }

        return jsonify({"distribution": distribution}), 200
    except Exception as e:
        logger.error(f"Error getting confidence distribution: {e}")
        return jsonify({"error": str(e)}), 500


def register_admin_routes(app):
    """
    Register admin dashboard routes with Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(admin_bp)
    logger.info("Admin dashboard routes registered")
