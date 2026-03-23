from flask import jsonify, render_template, request, send_file

from app.bootstrap.bootstrap import build_application
from app.infrastructure.repositories import DataManagerRepository
from app.interfaces.compat import ui_contract as _ui_contract
from app.interfaces.web.app import create_api_blueprint
from app.interfaces.web.ui_app import create_ui_app
from app.ui.admin_dashboard import admin_bp
from app.ui.dev_tools import create_dev_blueprint
from app.ui.market_routes import create_market_blueprint

# Build app container once and reuse its wired dependencies.
_APP_CONTAINER = build_application(ensure_dirs=False)
settings = _APP_CONTAINER.settings

get_supported_tickers = _ui_contract.get_supported_tickers
load_data = _ui_contract.load_data
sanitize_dataframe = _ui_contract.sanitize_dataframe
compute_signals = _ui_contract.compute_signals
get_params = _ui_contract.get_params
get_indicator_description = _ui_contract.get_indicator_description
get_candle_img_buffer = _ui_contract.get_candle_img_buffer
get_equity_curve_buffer = _ui_contract.get_equity_curve_buffer
get_drawdown_curve_buffer = _ui_contract.get_drawdown_curve_buffer
Backtester = _ui_contract.Backtester


def _parse_date(value: str, field_name: str):
    return _ui_contract.parse_date(value)


def _get_supported_ticker_list() -> list[str]:
    try:
        return get_supported_tickers()
    except Exception:
        tickers = getattr(settings, "tickers", None) or getattr(settings, "TICKERS", [])
        return list(tickers) if tickers else []


def _validate_ticker(ticker: str):
    return _ui_contract.validate_ticker(ticker, _get_supported_ticker_list)


DataManager = DataManagerRepository


def _create_data_repository() -> DataManagerRepository:
    if DataManager is DataManagerRepository:
        return _APP_CONTAINER.data_manager
    return DataManager()


app = create_ui_app(
    settings=settings,
    admin_bp=admin_bp,
    market_blueprint_factory=create_market_blueprint,
    api_blueprint_factory=create_api_blueprint,
    dev_blueprint_factory=create_dev_blueprint,
    ticker_provider=_get_supported_ticker_list,
    repository_factory=_create_data_repository,
    api_container=_APP_CONTAINER,
)


if __name__ == "__main__":
    debug_mode = app.config.get("ENV") == "development"
    app.run(debug=debug_mode)
