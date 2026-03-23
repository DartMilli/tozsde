from flask import Blueprint, Flask, jsonify, request

from app.application.use_cases.result import error as result_error
from app.bootstrap.bootstrap import build_application


def create_api_blueprint(container=None) -> Blueprint:
    web_container = container or build_application()
    web_api = Blueprint("web_api", __name__)

    @web_api.route("/walk-forward", methods=["GET"])
    def walk_forward():
        try:
            ticker = request.args.get("ticker")
            result = web_container.walk_forward.run(ticker=ticker)
            return jsonify(result)
        except Exception as exc:
            return (
                jsonify(
                    result_error(
                        "run_walk_forward",
                        str(exc),
                        ticker=request.args.get("ticker"),
                        code="REQUEST_FAILED",
                    )
                ),
                500,
            )

    @web_api.route("/daily-pipeline", methods=["POST"])
    def daily_pipeline():
        payload = request.get_json(silent=True) or {}
        try:
            result = web_container.daily_pipeline.run(
                dry_run=bool(payload.get("dry_run", False)),
                ticker=payload.get("ticker"),
            )
            return jsonify(result)
        except Exception as exc:
            return (
                jsonify(
                    result_error(
                        "run_daily_pipeline",
                        str(exc),
                        dry_run=bool(payload.get("dry_run", False)),
                        ticker=payload.get("ticker"),
                        code="REQUEST_FAILED",
                    )
                ),
                500,
            )

    @web_api.route("/train-rl", methods=["POST"])
    def train_rl():
        payload = request.get_json(silent=True) or {}
        ticker = payload.get("ticker")
        if not ticker:
            return (
                jsonify(
                    result_error(
                        "train_rl_model",
                        "ticker is required",
                        code="MISSING_TICKER",
                    )
                ),
                400,
            )

        kwargs = {k: v for k, v in payload.items() if k != "ticker"}
        try:
            result = web_container.train_rl.run(ticker=ticker, **kwargs)
            return jsonify(result)
        except Exception as exc:
            return (
                jsonify(
                    result_error(
                        "train_rl_model",
                        str(exc),
                        ticker=ticker,
                        code="REQUEST_FAILED",
                    )
                ),
                500,
            )

    @web_api.route("/validate-model", methods=["POST"])
    def validate_model():
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode", "quick")
        try:
            result = web_container.validate_model.run(mode=mode)
            return jsonify(result)
        except Exception as exc:
            return (
                jsonify(
                    result_error(
                        "validate_model",
                        str(exc),
                        mode=mode,
                        code="VALIDATION_FAILED",
                    )
                ),
                500,
            )

    return web_api


app = Flask(__name__)
app.register_blueprint(create_api_blueprint())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
