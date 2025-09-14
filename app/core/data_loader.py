import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json

import app.utils.router as rtr

DB_PATH = rtr.PRICE_DB_PATH
DATA_DIR = rtr.DATA_DIR
FAILED_DAYS_FILE = rtr.FAILED_DAYS_FILE_PATH


TICKERS = {
    "VOO": {
        "description": "Vanguard S&P 500 ETF: alacsony költség, USA top 500 részvény lefedése",
        "why": "Alapozó stabilitást, széles piaci diverzifikáció",
        "currency": "$",
    },
    "VTI": {
        "description": "Vanguard Total Stock Market ETF: akár 3600 amerikai részvény, széles piaci lefedés",
        "why": "Alapozó stabilitást: széles piaci diverzifikáció",
        "currency": "$",
    },
    "QQQ": {
        "description": "Invesco QQQ (Nasdaq‑100): tech fókusz, magas hozam, de magasabb volatilitás",
        "why": "Növekedési lehetőségek: tech + AI terület",
        "currency": "$",
    },
    "SPY": {
        "description": "SPDR S&P 500 ETF: likvid, széles körben elfogadott benchmark",
        "why": "Alapozó stabilitást: széles piaci diverzifikáció",
        "currency": "$",
    },
    "SCHD": {
        "description": "Schwab U.S. Dividend Equity ETF: magas, stabil osztalékhozam",
        "why": "Jövedelmező kiegészítők: osztalék, súlyozás",
        "currency": "$",
    },
    "RSP": {
        "description": "Invesco S&P 500 Equal Weight ETF: tech túlkoncentráció csökkentéséhez, kiegyensúlyozottabb súlyozás",
        "why": "Jövedelmező kiegészítők: osztalék, súlyozás",
        "currency": "$",
    },
    "SMH": {
        "description": "VanEck Semiconductor ETF: félvezető szektor – AI és technológiai növekedés kitűnő indikátora",
        "why": "Növekedési lehetőségek: tech + AI terület",
        "currency": "$",
    },
    "GLD": {
        "description": "SPDR Gold Shares / MiniShares: arany alapú fedezeti ETF infláció ellen",
        "why": "Kockázatkezelés: arany fedezeti szerep",
        "currency": "$",
    },
    "OTP.BD": {"description": "OTP Bank Nyrt.", "why": "magyar", "currency": "Ft"},
    "MOL.BD": {
        "description": "MOL Magyar Olaj- és Gázipari Nyilvánosan Működő Részvénytársaság",
        "why": "magyar",
        "currency": "Ft",
    },
    "RICHTER.BD": {
        "description": "Richter Gedeon Vegyészeti Gyár Nyilvánosan Működő Rt.",
        "why": "magyar",
        "currency": "Ft",
    },
}


def init_db():
    os.makedirs("app/data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """
    )
    conn.commit()
    conn.close()


def load_from_db(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM prices WHERE ticker = ?", conn, params=(ticker,)
    )
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
    # print(df)
    return df


def save_to_db(ticker, df):
    conn = sqlite3.connect(DB_PATH)
    df = df.copy()
    df.index.name = "date"
    df.reset_index(inplace=True)

    # lapítsuk le az oszlopokat, ha MultiIndex lenne
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # átnevezés: egységesítjük az oszlopneveket az SQLite táblához
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    df["ticker"] = ticker

    # csak az elvárt oszlopokat tartsuk meg, hogy ne legyen konfliktus
    # df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    # df.to_sql("prices", conn, if_exists="append", index=False, method="multi")

    for index, row in df.iterrows():
        try:
            cursor = conn.cursor()
            sql = f"INSERT OR IGNORE INTO prices(ticker,date,open,high,low,close,volume) values('{row['ticker']}','{row['date']}',{row['open']},{row['high']},{row['low']},{row['close']},{row['volume']})"
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"[ERROR] Nem sikerült az adatbázis művelet\n{sql}\n{e}")

    conn.close()


def get_supported_tickers():
    return TICKERS


def get_supported_ticker_list():
    return TICKERS.keys()


def load_data(ticker, start=None, end=None):
    if start == None or end == None:
        end = datetime.today().date()
        start = end - timedelta(days=183)

    format = "%Y-%m-%d"
    if type(end) == str:
        try:
            end = datetime.strptime(end, format).date()
        except:
            end = datetime.today().date()

    if type(start) == str:
        try:
            start = datetime.strptime(start, format).date()
        except:
            start = end - timedelta(days=183)

    init_db()
    df = load_from_db(ticker)

    if not df.empty:
        if df.index[0].date() > start:
            download_and_save_data(ticker, start, df.index[0].date())
        if df.index[-1].date() + timedelta(days=1) < end:
            download_and_save_data(ticker, df.index[-1].date() + timedelta(days=1), end)
    else:
        download_and_save_data(ticker, start, end)

    df = load_from_db(ticker)

    mask = (df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))
    return df[mask]


def download_and_save_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)
    except Exception as e:
        print(f"[ERROR] Nem sikerült letölteni {ticker}: {e}")

    if df.empty:
        print(f"[WARN] Üres adat: {ticker}")
        return False
    else:
        df.index = pd.to_datetime(df.index)
        save_to_db(ticker, df)
        return True


def run_full_download(config):
    tickers_to_download = config["TICKERS"]
    start = config["START_DATE"]
    end = config["END_DATE"]

    apple = yf.download("AAPL", start=start, end=end, interval="1d", auto_adjust=True)

    # Kinyerjük a kereskedési napokat az adatokból
    # Normalizáljuk a dátumokat, hogy csak a dátumot tartalmazza idő nélkül
    trading_days = set(apple.index.normalize().to_list())

    fails = []
    fail_counter = {}
    for ticker in tickers_to_download:
        print(f"{ticker:10s}", start, "-", end)
        df = load_data(ticker, start, end)
        available_days = set(df.index.normalize().to_list())
        hianyzo_elemek = trading_days.difference(available_days)
        for h in hianyzo_elemek:
            success = download_and_save_data(ticker, h, h + timedelta(days=1))
            if not success:
                fails.append({ticker: str(h)})
                if ticker not in fail_counter:
                    fail_counter[ticker] = 0
                fail_counter[ticker] += 1

    with open(FAILED_DAYS_FILE, "w") as f:
        json.dump(fails, f)

    print("Fails:")
    for f in fail_counter:
        print(
            f"{f:10s}->{fail_counter[f]:3d}:{str([[f[k] for k in f][0] for f in fails])}"
        )


if __name__ == "__main__":
    config = {
        "START_DATE": "2020-01-01",
        "END_DATE": datetime.today().strftime("%Y-%m-%d"),
        "TICKERS": get_supported_ticker_list(),
    }
    run_full_download(config)
