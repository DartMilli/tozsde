from datetime import datetime, timedelta
import json

from flask import Blueprint, current_app, jsonify


def create_dev_blueprint(*, settings_provider, ticker_provider, repository_factory):
    dev_bp = Blueprint("dev_tools", __name__)

    @dev_bp.route("/dev/status")
    def dev_status():
        if current_app.config.get("ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403

        tables = []
        db_connected = False
        db_error = None

        try:
            with repository_factory().connection() as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                tables = [row[0] for row in rows]
                db_connected = True
        except Exception as exc:
            db_error = str(exc)
            current_app.logger.error(f"Dev status DB check failed: {exc}")

        settings_obj = settings_provider()
        return jsonify(
            {
                "status": "OK",
                "environment": current_app.config.get("ENV", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "TICKERS": ticker_provider(),
                    "ENABLE_NOTIFICATIONS": getattr(
                        settings_obj, "enable_notifications", False
                    ),
                    "ENABLE_RL": getattr(settings_obj, "enable_rl", False),
                },
                "database": {
                    "connected": db_connected,
                    "tables": tables,
                    "error": db_error,
                },
            }
        )

    @dev_bp.route("/dev/config")
    def dev_config():
        if current_app.config.get("ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403

        settings_obj = settings_provider()
        config_dict = {
            key: value
            for key, value in settings_obj.__dict__.items()
            if not key.startswith("_") and not key.startswith("secret")
        }

        safe_config = {}
        for key, value in config_dict.items():
            try:
                safe_config[key] = json.loads(json.dumps(value, default=str))
            except Exception:
                safe_config[key] = str(value)

        safe_config["TICKERS"] = ticker_provider()
        return jsonify(safe_config)

    @dev_bp.route("/dev/db-init", methods=["POST"])
    def dev_db_init():
        if current_app.config.get("ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403

        try:
            data_repo = repository_factory()
            if hasattr(data_repo, "initialize_tables") and callable(
                getattr(data_repo, "initialize_tables")
            ):
                data_repo.initialize_tables()
            return jsonify({"status": "OK", "message": "Database reinitialized"})
        except Exception as exc:
            return jsonify({"status": "ERROR", "error": str(exc)}), 500

    @dev_bp.route("/dev/clear-recs", methods=["POST"])
    def dev_clear_recs():
        if current_app.config.get("ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403

        try:
            today = datetime.today().date()
            deleted = 0
            try:
                data_repo = repository_factory()
                if hasattr(data_repo, "connection") and callable(
                    getattr(data_repo, "connection")
                ):
                    with data_repo.connection() as conn:
                        cursor = conn.execute(
                            "DELETE FROM recommendations WHERE date = ?",
                            (today.isoformat(),),
                        )
                        conn.commit()
                        deleted = cursor.rowcount if cursor.rowcount is not None else 0
            except Exception as exc:
                current_app.logger.error(f"Dev clear recs DB error: {exc}")

            return jsonify(
                {
                    "status": "OK",
                    "message": f"Cleared recommendations for {today}",
                    "deleted": deleted,
                }
            )
        except Exception as exc:
            return jsonify({"status": "ERROR", "error": str(exc)}), 500

    @dev_bp.route("/dev/metrics")
    def dev_metrics():
        if current_app.config.get("ENV") != "development":
            return jsonify({"error": "Not available in production"}), 403

        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            last_update = None
            last_run = None

            try:
                with repository_factory().connection() as conn:
                    last_update_row = conn.execute(
                        "SELECT MAX(date) FROM ohlcv"
                    ).fetchone()
                    last_update = last_update_row[0] if last_update_row else None

                    last_run_row = conn.execute(
                        "SELECT MAX(timestamp) FROM pipeline_metrics"
                    ).fetchone()
                    last_run = last_run_row[0] if last_run_row else None
            except Exception as exc:
                current_app.logger.error(f"Dev metrics DB query failed: {exc}")

            now = datetime.now()
            next_run = now.replace(hour=6, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run + timedelta(days=1)

            return jsonify(
                {
                    "system": {
                        "cpu_percent": process.cpu_percent(interval=1),
                        "memory_mb": process.memory_info().rss / 1024 / 1024,
                    },
                    "data": {
                        "tickers": ticker_provider(),
                        "last_update": last_update,
                    },
                    "pipeline": {
                        "last_run": last_run,
                        "next_run": next_run.isoformat() if next_run else None,
                    },
                }
            )
        except Exception as exc:
            current_app.logger.error(f"Dev metrics error: {exc}")
            return jsonify(
                {
                    "system": {"cpu_percent": 0.0, "memory_mb": 0.0},
                    "data": {
                        "tickers": ticker_provider(),
                        "last_update": None,
                    },
                    "pipeline": {"last_run": None, "next_run": None},
                    "error": str(exc),
                }
            )

    return dev_bp
