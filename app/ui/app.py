from flask import Flask, render_template, request
from flask import send_file
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

from app.utils.recommendation_logger import (
    load_today_recommendations,
    load_recommendations_by_ticker_and_range,
)
from app.core.data_loader import get_supported_tickers, load_data
from app.utils.data_cleaner import sanitize_dataframe
from app.utils.analizer import compute_signals, run_signal_backtest
from app.utils.technical_analizer import get_indicator_description
from app.utils.plotter import get_candle_img_buffer, get_equity_curve_buffer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_key")


@app.route("/")
def index():
    tickers = get_supported_tickers()
    today = datetime.today().date()
    recommendations = load_today_recommendations()
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

    signals, indicators = compute_signals(df)

    buffer = get_candle_img_buffer(df, indicators, signals)

    return send_file(buffer, mimetype="image/png")


@app.route("/history")
def ticker_history():
    ticker = request.args.get("ticker")
    start = request.args.get("start")
    end = request.args.get("end")
    if not ticker:
        return "Hiányzó ticker", 400

    history = load_recommendations_by_ticker_and_range(ticker, start, end)
    return {"history": history}


@app.route("/indicators")
def indicator_description():
    return get_indicator_description()


@app.route("/report")
def report():
    ticker = request.args.get("ticker", "VOO")
    start = request.args.get("start", "2022-01-01")
    end = request.args.get("end", "2025-01-01")

    equity_curve, performance_metrics = run_signal_backtest(ticker, start, end)

    image_b64 = get_equity_curve_buffer(ticker, equity_curve)

    return render_template(
        "report.html", ticker=ticker, metrics=performance_metrics, chart_image=image_b64
    )
