from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask import send_file
from datetime import datetime, timedelta
import logging
import json

from app.data_access.data_loader import get_supported_tickers, load_data
from app.data_access.data_cleaner import sanitize_dataframe
from app.analysis.analyzer import compute_signals, get_params
from app.indicators.technical import get_indicator_description
from app.reporting.plotter import (
    get_candle_img_buffer,
    get_equity_curve_buffer,
    get_drawdown_curve_buffer,
)
from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.backtesting.backtester import Backtester
from app.reporting.metrics import BacktestReport

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# ═════════════════════════════════════════════════════════════════════════════
# CORS & DEVELOPMENT MIDDLEWARE
# ═════════════════════════════════════════════════════════════════════════════

# Enable CORS for frontend development
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:3000",
                "http://localhost:5000",
                "http://localhost:8000",
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


# Request/response logging middleware for development
@app.before_request
def log_request():
    """Log incoming requests in development mode."""
    if app.config.get("ENV") == "development":
        app.logger.info(
            f"[REQUEST] {request.method} {request.path} | "
            f"Remote: {request.remote_addr}"
        )


@app.after_request
def log_response(response):
    """Log outgoing responses in development mode."""
    if app.config.get("ENV") == "development":
        app.logger.info(
            f"[RESPONSE] {response.status_code} | "
            f"Size: {response.content_length or 0} bytes"
        )
    return response


# ═════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT ENDPOINTS (only in development mode)
# ═════════════════════════════════════════════════════════════════════════════


@app.route("/dev/status")
def dev_status():
    """
    Development endpoint: System status and health check.

    Returns:
      - Environment, version, and configuration info
      - Database connection status
      - Last run timestamps

    Only available in development mode (FLASK_ENV=development).
    """
    # TODO: Implement
    if app.config.get("ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403

    dm = DataManager()

    return jsonify(
        {
            "status": "OK",
            "environment": app.config.get("ENV", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "TICKERS": Config.TICKERS,
                "ENABLE_NOTIFICATIONS": Config.ENABLE_NOTIFICATIONS,
                "ENABLE_RL": Config.ENABLE_RL,
            },
            "database": {
                "connected": True,  # TODO: Implement actual DB check
                "tables": ["ohlcv", "recommendations", "trades"],  # TODO: Get from DB
            },
        }
    )


@app.route("/dev/config")
def dev_config():
    """
    Development endpoint: Display current configuration.

    Returns all Config parameters (safe, no secrets exposed).

    Only available in development mode.
    """
    # TODO: Implement
    if app.config.get("ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403

    config_dict = {
        k: v
        for k, v in Config.__dict__.items()
        if not k.startswith("_") and not k.startswith("SECRET")
    }

    return jsonify(config_dict)


@app.route("/dev/db-init", methods=["POST"])
def dev_db_init():
    """
    Development endpoint: Reinitialize database schema.

    WARNING: Deletes all data! Use only for testing.

    Only available in development mode.
    """
    # TODO: Implement
    if app.config.get("ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403

    try:
        dm = DataManager()
        dm.initialize_tables()  # Drops and recreates all tables
        return jsonify({"status": "OK", "message": "Database reinitialized"})
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


@app.route("/dev/clear-recs", methods=["POST"])
def dev_clear_recs():
    """
    Development endpoint: Clear today's recommendations.

    Useful for retesting daily pipeline without full DB reset.

    Only available in development mode.
    """
    # TODO: Implement
    if app.config.get("ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403

    try:
        dm = DataManager()
        today = datetime.today().date()
        # TODO: Implement clearing of recommendations for today
        return jsonify(
            {"status": "OK", "message": f"Cleared recommendations for {today}"}
        )
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


@app.route("/dev/metrics")
def dev_metrics():
    """
    Development endpoint: System metrics for monitoring.

    Returns:
      - CPU usage, memory usage
      - Active tickers, data age
      - Last pipeline run, next scheduled run

    Only available in development mode.
    """
    # TODO: Implement
    if app.config.get("ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        return jsonify(
            {
                "system": {
                    "cpu_percent": process.cpu_percent(interval=1),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                },
                "data": {
                    "tickers": Config.TICKERS,
                    "last_update": None,  # TODO: Get from DB
                },
                "pipeline": {
                    "last_run": None,  # TODO: Get from history
                    "next_run": None,  # TODO: Calculate from schedule
                },
            }
        )
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTION ROUTES
# ═════════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    tickers = get_supported_tickers()
    today = datetime.today().date()
    dm = DataManager()
    recommendations = dm.get_today_recommendations()
    return render_template(
        "dashboard.html", tickers=tickers, today=today, recommendations=recommendations
    )


@app.route("/chart")
def chart():
    ticker = request.args.get("ticker", "VOO")

    format = "%Y-%m-%d"
    default_end_date = datetime.today()
    end = request.args.get("end", default_end_date.strftime(format))
    try:
        datetime.strptime(end, format)
    except ValueError:
        end = default_end_date.strftime(format)

    default_start_date = datetime.strptime(end, format) - timedelta(days=183)
    start = request.args.get("start", default_start_date.strftime(format))
    try:
        datetime.strptime(start, format)
    except ValueError:
        start = default_start_date.strftime(format)

    df = load_data(
        ticker,
        datetime.strptime(start, format).date(),
        datetime.strptime(end, format).date(),
    )
    if df is None or df.empty:
        return f"Nincs elérhető adat a {ticker} tickerhez {start} és {end} között.", 404
    df = sanitize_dataframe(df)
    df.index.name = "Date"

    signals, indicators = compute_signals(df, ticker)

    buffer = get_candle_img_buffer(df, indicators, signals)

    return send_file(buffer, mimetype="image/png")


@app.route("/history")
def ticker_history():
    ticker = request.args.get("ticker")
    start = request.args.get("start")
    end = request.args.get("end")
    if not ticker:
        return "Hiányzó ticker", 400

    dm = DataManager()
    history = dm.get_ticker_historical_recommendations(ticker, start, end)
    return {"history": history}


@app.route("/indicators")
def indicator_description():
    return get_indicator_description()


@app.route("/report")
def report():
    ticker = request.args.get("ticker", "VOO")
    start = request.args.get("start", "2022-01-01")
    end = request.args.get("end", "2025-01-01")

    df = load_data(ticker, start, end)
    if df is None or df.empty:
        return f"Nincs adat {ticker} tickerhez.", 404

    df = sanitize_dataframe(df)
    df.index.name = "Date"

    params = get_params()

    bt = Backtester(df, ticker)
    report: BacktestReport = bt.run(params)

    # equity curve → PNG
    equity_curve = report.diagnostics["equity_curve"]
    equity_img = get_equity_curve_buffer(ticker, equity_curve)
    drawdown_img = get_drawdown_curve_buffer(equity_curve)

    return render_template(
        "report.html",
        ticker=ticker,
        report=report,
        equity_image=equity_img,
        drawdown_image=drawdown_img,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═════════════════════════════════════════════════════════════════════════════


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    app.logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


# ═════════════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIG (NOT debug mode in production!)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Never use debug=True in production!
    debug_mode = app.config.get("ENV") == "development"
    app.run(debug=debug_mode)
