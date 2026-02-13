import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json

from app.config.config import Config
from app.data_access.data_manager import DataManager
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)

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


def get_supported_tickers():
    return TICKERS


def get_supported_ticker_list():
    return TICKERS.keys()


def load_data(ticker, start=None, end=None):
    """
    Szigorú adatbetöltés:
    1. Megnézzük, megvan-e a DB-ben a kért időszak.
    2. Ha nincs (vagy lyukas), letöltjük és ELSŐNEK lementjük DB-be.
    3. Kizárólag a DB-ből olvasunk vissza az alkalmazásnak.
    """
    dm = DataManager()

    # Alapértelmezett dátumok
    if end is None:
        end = datetime.today().date()
    if start is None:
        start = end - timedelta(days=365 * 2)  # Alapból 2 év

    # String konverzió biztosítása
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()

    # Warmup időszak (hogy az indikátoroknak legyen előzménye)
    warmup_start = start - timedelta(days=Config.WARMUP_DAYS)

    # 1. Próbáljuk betölteni DB-ből a teljes szükséges időszakot
    df_db = dm.load_ohlcv(ticker=ticker, start_date=warmup_start.strftime("%Y-%m-%d"))

    # 1. Ellenőrzés: Megvan az adat a végdátumig?
    data_missing = False
    last_date_in_db = None

    if df_db.empty:
        data_missing = True
    else:
        last_date_in_db = df_db.index[-1].date()
        # Ha a kért végdátum messzebb van, mint az adatbázis vége
        if (end - last_date_in_db).days > 3:
            data_missing = True

    # 2. Ha hiányzik, frissítés (CSAK A HIÁNYZÓ RÉSZRE)
    if data_missing:
        # Ha van adatunk, onnan folytatjuk, ha nincs, akkor a warmup elejétől
        download_start = (
            (last_date_in_db + timedelta(days=1)) if last_date_in_db else warmup_start
        )

        # Biztonsági ellenőrzés: ne töltsünk le a jövőből
        if download_start < end:
            logger.info(f"{ticker}: Data update required {download_start} -> {end}")
            success = download_and_save_data(ticker, download_start, end)
            if not success:
                logger.warning(f"{ticker}: Download failed.")

    # 3. Végső olvasás kizárólag a DB-ből (Single Source of Truth)
    df_final = dm.load_ohlcv(
        ticker=ticker, start_date=warmup_start.strftime("%Y-%m-%d")
    )

    # Vágás a kért időszakra (de a memóriában maradhat az eleje indikátor számításhoz)
    # A hívó fél felelőssége a `prepare_df` hívása, ami kezeli a warmupot.
    return df_final


def ensure_data_cached(ticker, start=None, end=None) -> bool:
    """
    Ensure OHLCV data exists for the full requested interval.

    Returns True when coverage is adequate, otherwise False.
    """
    dm = DataManager()

    # Normalize dates
    if end is None:
        end = datetime.today().date()
    if start is None:
        start = end - timedelta(days=365 * 2)

    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()

    load_data(ticker, start=start, end=end)

    df_check = dm.load_ohlcv(ticker=ticker, start_date=start.strftime("%Y-%m-%d"))
    if df_check.empty:
        logger.info(f"{ticker}: cache empty, downloading full range")
        if not download_and_save_data(ticker, start, end):
            logger.warning(f"{ticker}: cache check failed (no data)")
            return False
        df_check = dm.load_ohlcv(ticker=ticker, start_date=start.strftime("%Y-%m-%d"))
        if df_check.empty:
            logger.warning(f"{ticker}: cache check failed (no rows in DB)")
            return False

    min_date = df_check.index.min().date()
    max_date = df_check.index.max().date()

    start_gap_days = (min_date - start).days
    if start_gap_days > 3:
        logger.info(f"{ticker}: backfilling start gap {start} -> {min_date}")
        if not download_and_save_data(ticker, start, min_date):
            logger.warning(
                f"{ticker}: cache start gap (min={min_date}, expected_start={start})"
            )
            return False

    if (end - max_date).days > 3:
        download_start = max_date + timedelta(days=1)
        if download_start < end:
            logger.info(f"{ticker}: backfilling end gap {download_start} -> {end}")
            if not download_and_save_data(ticker, download_start, end):
                logger.warning(
                    f"{ticker}: cache end gap (max={max_date}, expected_end={end})"
                )
                return False

    df_check = dm.load_ohlcv(ticker=ticker, start_date=start.strftime("%Y-%m-%d"))
    if df_check.empty:
        logger.warning(f"{ticker}: cache check failed after backfill")
        return False

    min_date = df_check.index.min().date()
    max_date = df_check.index.max().date()

    start_gap_days = (min_date - start).days
    if start_gap_days > 3:
        logger.warning(
            f"{ticker}: cache start gap (min={min_date}, expected_start={start})"
        )
        return False

    if (end - max_date).days > 3:
        logger.warning(f"{ticker}: cache end gap (max={max_date}, expected_end={end})")
        return False

    return True


def download_and_save_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)
    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}", exc_info=True)

    if df.empty:
        logger.warning(f"Empty data received for {ticker}")
        return False
    else:
        df.index = pd.to_datetime(df.index)
        # save_to_db(ticker, df)

        dm = DataManager()
        dm.save_ohlcv(ticker=ticker, df=df)

        return True


def run_full_download(tickers=None, start_date=None, end_date=None):
    tickers_to_download = Config.TICKERS if tickers == None else tickers
    start = Config.START_DATE if start_date == None else start_date
    end = Config.END_DATE if end_date == None else end_date

    apple = yf.download("AAPL", start=start, end=end, interval="1d", auto_adjust=True)

    # Kinyerjük a kereskedési napokat az adatokból
    # Normalizáljuk a dátumokat, hogy csak a dátumot tartalmazza idő nélkül
    trading_days = set(apple.index.normalize().to_list())

    fails = []
    fail_counter = {}
    for ticker in tickers_to_download:
        logger.info(f"Downloading {ticker:10s} {start}-{end}")
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

    with open(Config.FAILED_DAYS_FILE_PATH, "w") as f:
        json.dump(fails, f)

    for f in fail_counter:
        logger.warning(
            f"Failed {f:10s}->{fail_counter[f]:3d}:{str([[f[k] for k in f][0] for f in fails])}"
        )


def get_market_volatility_index():
    """
    Visszaadja a legfrissebb VIX (félelem index) értéket.
    Ha nem sikerül letölteni, None-t ad (nem blokkol).
    """
    dm = DataManager()
    symbol = "^VIX"

    # 1. Megnézzük a DB-ben a mai adatot
    today_str = datetime.now().strftime("%Y-%m-%d")
    db_data = dm.get_market_data(symbol, days=1)

    if db_data and db_data[0][0] == today_str:
        return db_data[0][1]

    # 2. Ha nincs meg mára, letöltjük az utolsó 5 napot (hétvége miatt)
    try:
        logger.info(f"Downloading {symbol} data...")
        vix_ticker = yf.Ticker(symbol)
        df = vix_ticker.history(period="5d")
        if not df.empty:
            dm.save_market_data(symbol, df)
            return df["Close"].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")

    # Fallback: ha a letöltés nem sikerült, próbáljuk a legfrissebb DB adatot
    db_data = dm.get_market_data(symbol, days=10)
    return db_data[0][1] if db_data else None


if __name__ == "__main__":
    run_full_download(
        tickers=get_supported_ticker_list(),
        start_date="2020-01-01",
        end_date=datetime.today().strftime("%Y-%m-%d"),
    )
