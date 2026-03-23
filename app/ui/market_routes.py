from datetime import datetime, timedelta

from flask import Blueprint, jsonify


def create_market_blueprint() -> Blueprint:
    market_bp = Blueprint("market_routes", __name__)

    @market_bp.route("/")
    def index():
        from app.ui import app as ui_app

        tickers = ui_app.get_supported_tickers()
        today = datetime.today().date()
        try:
            recommendations = (
                ui_app._create_data_repository().get_today_recommendations()
            )
        except Exception:
            recommendations = []
        return ui_app.render_template(
            "dashboard.html",
            tickers=tickers,
            today=today,
            recommendations=recommendations,
        )

    @market_bp.route("/chart")
    def chart():
        from app.ui import app as ui_app

        ticker = ui_app.request.args.get("ticker", "VOO").upper()
        if not ui_app._validate_ticker(ticker):
            return jsonify({"error": "Invalid ticker"}), 400

        default_end_date = datetime.today().date()
        end_str = ui_app.request.args.get("end", default_end_date.strftime("%Y-%m-%d"))
        end_date = ui_app._parse_date(end_str, "end")
        if not end_date:
            return jsonify({"error": "Invalid end date. Use YYYY-MM-DD."}), 400

        default_start_date = end_date - timedelta(days=183)
        start_str = ui_app.request.args.get(
            "start", default_start_date.strftime("%Y-%m-%d")
        )
        start_date = ui_app._parse_date(start_str, "start")
        if not start_date:
            return jsonify({"error": "Invalid start date. Use YYYY-MM-DD."}), 400

        if start_date > end_date:
            return jsonify({"error": "Start date must be <= end date."}), 400

        dataframe = ui_app.load_data(ticker, start_date, end_date)
        if dataframe is None or dataframe.empty:
            return (
                f"Nincs elerheto adat a {ticker} tickerhez {start_date} es {end_date} kozott.",
                404,
            )
        dataframe = ui_app.sanitize_dataframe(dataframe)
        dataframe.index.name = "Date"

        signals, indicators = ui_app.compute_signals(dataframe, ticker)
        buffer = ui_app.get_candle_img_buffer(dataframe, indicators, signals)
        return ui_app.send_file(buffer, mimetype="image/png")

    @market_bp.route("/history")
    def ticker_history():
        from app.ui import app as ui_app

        ticker = ui_app.request.args.get("ticker", "").upper()
        start = ui_app.request.args.get("start")
        end = ui_app.request.args.get("end")
        if not ui_app._validate_ticker(ticker):
            return jsonify({"error": "Invalid ticker"}), 400

        if start and not ui_app._parse_date(start, "start"):
            return jsonify({"error": "Invalid start date. Use YYYY-MM-DD."}), 400
        if end and not ui_app._parse_date(end, "end"):
            return jsonify({"error": "Invalid end date. Use YYYY-MM-DD."}), 400

        try:
            history = (
                ui_app._create_data_repository().get_ticker_historical_recommendations(
                    ticker, start, end
                )
            )
        except Exception:
            history = []
        return {"history": history}

    @market_bp.route("/indicators")
    def indicator_description():
        from app.ui import app as ui_app

        return ui_app.get_indicator_description()

    @market_bp.route("/report")
    def report():
        from app.ui import app as ui_app

        ticker = ui_app.request.args.get("ticker", "VOO").upper()
        start = ui_app.request.args.get("start", "2022-01-01")
        end = ui_app.request.args.get("end", "2025-01-01")

        if not ui_app._validate_ticker(ticker):
            return jsonify({"error": "Invalid ticker"}), 400
        if not ui_app._parse_date(start, "start"):
            return jsonify({"error": "Invalid start date. Use YYYY-MM-DD."}), 400
        if not ui_app._parse_date(end, "end"):
            return jsonify({"error": "Invalid end date. Use YYYY-MM-DD."}), 400

        dataframe = ui_app.load_data(ticker, start, end)
        if dataframe is None or dataframe.empty:
            return f"Nincs adat {ticker} tickerhez.", 404

        dataframe = ui_app.sanitize_dataframe(dataframe)
        dataframe.index.name = "Date"

        params = ui_app.get_params(ticker)
        backtester = ui_app.Backtester(dataframe, ticker)
        report_obj = backtester.run(params)

        equity_curve = report_obj.diagnostics["equity_curve"]
        equity_img = ui_app.get_equity_curve_buffer(ticker, equity_curve)
        drawdown_img = ui_app.get_drawdown_curve_buffer(equity_curve)

        return ui_app.render_template(
            "report.html",
            ticker=ticker,
            report=report_obj,
            equity_image=equity_img,
            drawdown_image=drawdown_img,
        )

    @market_bp.route("/params")
    def params_view():
        from app.ui import app as ui_app

        ticker = ui_app.request.args.get("ticker", "VOO").upper()
        if not ui_app._validate_ticker(ticker):
            return jsonify({"error": "Invalid ticker"}), 400

        params = ui_app.get_params(ticker)
        try:
            tickers = ui_app.get_supported_tickers()
        except Exception:
            tickers = getattr(ui_app.settings, "TICKERS", [])

        return ui_app.render_template(
            "params.html",
            ticker=ticker,
            params=params,
            tickers=tickers,
        )

    return market_bp
