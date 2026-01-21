from flask import Flask, render_template, request
from flask import send_file
from datetime import datetime, timedelta

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
