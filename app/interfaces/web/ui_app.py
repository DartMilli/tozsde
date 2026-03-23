from flask import Flask, jsonify, request
from flask_cors import CORS


def create_ui_app(
    *,
    settings,
    admin_bp,
    market_blueprint_factory,
    api_blueprint_factory,
    dev_blueprint_factory,
    ticker_provider,
    repository_factory,
    api_container=None,
):
    app = Flask(__name__)
    app.secret_key = (
        getattr(settings, "secret_key", None)
        or getattr(settings, "SECRET_KEY", None)
        or "dev-secret-key"
    )

    app.register_blueprint(admin_bp)
    app.register_blueprint(market_blueprint_factory())
    app.register_blueprint(
        api_blueprint_factory(container=api_container),
        url_prefix="/api",
    )
    app.register_blueprint(
        dev_blueprint_factory(
            settings_provider=lambda: settings,
            ticker_provider=ticker_provider,
            repository_factory=repository_factory,
        )
    )

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

    @app.before_request
    def log_request():
        if app.config.get("ENV") == "development":
            app.logger.info(
                f"[REQUEST] {request.method} {request.path} | "
                f"Remote: {request.remote_addr}"
            )

    @app.after_request
    def log_response(response):
        if app.config.get("ENV") == "development":
            app.logger.info(
                f"[RESPONSE] {response.status_code} | "
                f"Size: {response.content_length or 0} bytes"
            )
        return response

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error(f"Internal server error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

    return app


def build_default_ui_app(ensure_dirs: bool = False):
    from app.bootstrap.bootstrap import build_application
    from app.infrastructure.repositories import DataManagerRepository
    from app.interfaces.web.app import create_api_blueprint
    from app.ui.admin_dashboard import admin_bp
    from app.ui.dev_tools import create_dev_blueprint
    from app.ui.market_routes import create_market_blueprint

    container = build_application(ensure_dirs=ensure_dirs)
    settings = container.settings

    def _ticker_provider() -> list[str]:
        try:
            return container.get_supported_tickers()
        except Exception:
            tickers = getattr(settings, "tickers", None) or getattr(
                settings, "TICKERS", []
            )
            return list(tickers) if tickers else []

    def _repository_factory():
        if isinstance(container.data_manager, DataManagerRepository):
            return container.data_manager
        return DataManagerRepository(settings=settings)

    return create_ui_app(
        settings=settings,
        admin_bp=admin_bp,
        market_blueprint_factory=create_market_blueprint,
        api_blueprint_factory=create_api_blueprint,
        dev_blueprint_factory=create_dev_blueprint,
        ticker_provider=_ticker_provider,
        repository_factory=_repository_factory,
        api_container=container,
    )
