import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import json

from app.data_access import get_settings
from app.infrastructure.repositories import DataManagerRepository
from app.infrastructure.repositories.sqlite_ohlcv_repository import (
    SqliteOhlcvRepository,
)
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)
DataManager = DataManagerRepository


def _create_data_repository(settings=None):
    try:
        return DataManager(settings=settings)
    except TypeError:
        return DataManager()


# LEGACY PLACEHOLDERS REMOVED: prefer explicit `settings` or package `get_settings()`.


def _conf(settings):
    """Return a config-like object: prefer provided `settings`, else call package `get_settings()` at runtime.

    This avoids resolving settings at import time.
    """
    # 1) If explicit settings passed, use them
    if settings is not None:
        return settings

    # 2) Prefer package-injected settings
    try:
        return get_settings()
    except Exception:
        pass

    return None


TICKERS = {
    "VOO": {
        "description": "Vanguard S&P 500 ETF: alacsony koltseg, USA top 500 reszveny lefedese",
        "why": "Alapozo stabilitast, szeles piaci diverzifikacio",
        "currency": "$",
    },
    "VTI": {
        "description": "Vanguard Total Stock Market ETF: akar 3600 amerikai reszveny, szeles piaci lefedes",
        "why": "Alapozo stabilitast: szeles piaci diverzifikacio",
        "currency": "$",
    },
    "QQQ": {
        "description": "Invesco QQQ (Nasdaq100): tech fokusz, magas hozam, de magasabb volatilitas",
        "why": "Novekedesi lehetosegek: tech + AI terulet",
        "currency": "$",
    },
    "SPY": {
        "description": "SPDR S&P 500 ETF: likvid, szeles korben elfogadott benchmark",
        "why": "Alapozo stabilitast: szeles piaci diverzifikacio",
        "currency": "$",
    },
    "SCHD": {
        "description": "Schwab U.S. Dividend Equity ETF: magas, stabil osztalekhozam",
        "why": "Jovedelmezo kiegeszitok: osztalek, sulyozas",
        "currency": "$",
    },
    "RSP": {
        "description": "Invesco S&P 500 Equal Weight ETF: tech tulkoncentracio csokkentesehez, kiegyensulyozottabb sulyozas",
        "why": "Jovedelmezo kiegeszitok: osztalek, sulyozas",
        "currency": "$",
    },
    "SMH": {
        "description": "VanEck Semiconductor ETF: felvezeto szektor - AI es technologiai novekedes kituno indikatora",
        "why": "Novekedesi lehetosegek: tech + AI terulet",
        "currency": "$",
    },
    "GLD": {
        "description": "SPDR Gold Shares / MiniShares: arany alapu fedezeti ETF inflacio ellen",
        "why": "Kockazatkezeles: arany fedezeti szerep",
        "currency": "$",
    },
    "OTP.BD": {"description": "OTP Bank Nyrt.", "why": "magyar", "currency": "Ft"},
    "MOL.BD": {
        "description": "MOL Magyar Olaj- es Gazipari Nyilvanosan Mukodo Reszvenytarsasag",
        "why": "magyar",
        "currency": "Ft",
    },
    "RICHTER.BD": {
        "description": "Richter Gedeon Vegyeszeti Gyar Nyilvanosan Mukodo Rt.",
        "why": "magyar",
        "currency": "Ft",
    },
}


def get_supported_tickers():
    return TICKERS


def get_supported_ticker_list():
    return TICKERS.keys()


def load_data(ticker, start=None, end=None, data_manager=None, settings=None):
    """
    Szigoru adatbetoltes:
    1. Megnezzuk, megvan-e a DB-ben a kert idoszak.
    2. Ha nincs (vagy lyukas), letoltjuk es ELSONEK lementjuk DB-be.
    3. Kizarolag a DB-bol olvasunk vissza az alkalmazasnak.
    """
    # Ensure the repository always has a working DataManager instance so tests
    # can monkeypatch `data_loader.DataManager` separately.
    if data_manager is not None:
        dm_instance = data_manager
    else:
        dm_instance = _create_data_repository(settings=settings)
    ohlcv_repo = SqliteOhlcvRepository(data_manager=dm_instance)

    # Alapertelmezett datumok
    if end is None:
        end = datetime.today().date()
    if start is None:
        start = end - timedelta(days=365 * 2)  # Alapbol 2 ev

    # String konverzio biztositasa
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()

    # Warmup idoszak (hogy az indikatoroknak legyen elozmenye)
    conf = _conf(settings)
    warmup_days = getattr(conf, "WARMUP_DAYS", 30)
    warmup_start = start - timedelta(days=warmup_days)

    # 1. Probaljuk betolteni DB-bol a teljes szukseges idoszakot
    df_db = ohlcv_repo.load_ohlcv(
        ticker=ticker, start_date=warmup_start.strftime("%Y-%m-%d")
    )

    # 1. Ellenorzes: Megvan az adat a vegdatumig?
    data_missing = False
    last_date_in_db = None

    if df_db.empty:
        data_missing = True
    else:
        last_date_in_db = df_db.index[-1].date()
        # Ha a kert vegdatum messzebb van, mint az adatbazis vege
        if (end - last_date_in_db).days > 3:
            data_missing = True

    # 2. Ha hianyzik, frissites (CSAK A HIANYZO RESZRE)
    if data_missing:
        # Ha van adatunk, onnan folytatjuk, ha nincs, akkor a warmup elejetol
        download_start = (
            (last_date_in_db + timedelta(days=1)) if last_date_in_db else warmup_start
        )

        # Biztonsagi ellenorzes: ne toltsunk le a jovobol
        if download_start < end:
            logger.info(f"{ticker}: Data update required {download_start} -> {end}")
            success = download_and_save_data(ticker, download_start, end)
            if not success:
                logger.warning(f"{ticker}: Download failed.")

    # 3. Vegso olvasas kizarolag a DB-bol (Single Source of Truth)
    df_final = ohlcv_repo.load_ohlcv(
        ticker=ticker, start_date=warmup_start.strftime("%Y-%m-%d")
    )

    # Vagas a kert idoszakra (de a memoriaban maradhat az eleje indikator szamitashoz)
    # A hivo fel felelossege a `prepare_df` hivasa, ami kezeli a warmupot.
    return df_final


def ensure_data_cached(
    ticker, start=None, end=None, data_manager=None, settings=None
) -> bool:
    """
    Ensure OHLCV data exists for the full requested interval.

    Returns True when coverage is adequate, otherwise False.
    """
    ohlcv_repo = SqliteOhlcvRepository(data_manager=data_manager)

    # Normalize dates
    if end is None:
        end = datetime.today().date()
    if start is None:
        start = end - timedelta(days=365 * 2)

    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()

    # Maintain compatibility: load_data may accept a settings kwarg in some
    # call-sites; pass through if present.
    try:
        load_data(
            ticker,
            start=start,
            end=end,
            data_manager=data_manager,
            settings=settings,
        )
    except TypeError:
        load_data(ticker, start=start, end=end, data_manager=data_manager)

    df_check = ohlcv_repo.fetch_ohlcv(
        ticker=ticker, start_date=start.strftime("%Y-%m-%d"), end_date=None
    )
    if df_check.empty:
        logger.info(f"{ticker}: cache empty, downloading full range")
        if not download_and_save_data(ticker, start, end, settings=settings):
            logger.warning(f"{ticker}: cache check failed (no data)")
            return False
        df_check = ohlcv_repo.fetch_ohlcv(
            ticker=ticker, start_date=start.strftime("%Y-%m-%d"), end_date=None
        )
        if df_check.empty:
            logger.warning(f"{ticker}: cache check failed (no rows in DB)")
            return False

    min_date = df_check.index.min().date()
    max_date = df_check.index.max().date()

    start_gap_days = (min_date - start).days
    if start_gap_days > 3:
        logger.info(f"{ticker}: backfilling start gap {start} -> {min_date}")
        if not download_and_save_data(ticker, start, min_date, settings=settings):
            logger.warning(
                f"{ticker}: cache start gap (min={min_date}, expected_start={start})"
            )
            return False

    if (end - max_date).days > 3:
        download_start = max_date + timedelta(days=1)
        if download_start < end:
            logger.info(f"{ticker}: backfilling end gap {download_start} -> {end}")
            if not download_and_save_data(
                ticker, download_start, end, settings=settings
            ):
                logger.warning(
                    f"{ticker}: cache end gap (max={max_date}, expected_end={end})"
                )
                return False

    df_check = ohlcv_repo.fetch_ohlcv(
        ticker=ticker, start_date=start.strftime("%Y-%m-%d"), end_date=None
    )
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


def download_and_save_data(ticker, start, end, settings=None):
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

        try:
            # Use module-level DataManager (tests monkeypatch `data_loader.DataManager`)
            dm = _create_data_repository(settings=settings)
            dm.save_ohlcv(ticker=ticker, df=df)
        except Exception:
            logger.error(f"Failed to save ohlcv for {ticker}", exc_info=True)
            return False

        return True


def run_full_download(tickers=None, start_date=None, end_date=None, settings=None):
    conf = _conf(settings)
    # Resolve tickers in a robust way: support class-properties, lists, or callables
    if tickers is None:
        raw = getattr(conf, "TICKERS", None)
        if raw is None:
            tickers_to_download = []
        else:
            try:
                # If TICKERS is a property object on a class, prefer calling get_supported_tickers()
                if isinstance(raw, property) or hasattr(conf, "get_supported_tickers"):
                    try:
                        tickers_to_download = list(
                            conf.get_supported_tickers()
                            if hasattr(conf, "get_supported_tickers")
                            else raw()
                        )
                    except Exception:
                        tickers_to_download = list(raw) if raw is not None else []
                elif callable(raw):
                    try:
                        tickers_to_download = list(raw())
                    except Exception:
                        tickers_to_download = list(raw)
                else:
                    tickers_to_download = list(raw)
            except Exception:
                tickers_to_download = []
    else:
        tickers_to_download = tickers

    start = getattr(conf, "START_DATE", None) if start_date is None else start_date
    end = getattr(conf, "END_DATE", None) if end_date is None else end_date

    apple = yf.download("AAPL", start=start, end=end, interval="1d", auto_adjust=True)

    # Kinyerjuk a kereskedesi napokat az adatokbol
    # Normalizaljuk a datumokat, hogy csak a datumot tartalmazza ido nelkul
    trading_days = set(apple.index.normalize().to_list())

    fails = []
    fail_counter = {}
    for ticker in tickers_to_download:
        logger.info(f"Downloading {ticker:10s} {start}-{end}")
        try:
            df = load_data(ticker, start, end, settings=settings)
        except TypeError:
            # Support tests that monkeypatch load_data without a settings param
            df = load_data(ticker, start, end)
        available_days = set()
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            idx = idx[~pd.isna(idx)]
            available_days = set(idx.normalize().to_list())
        except Exception:
            available_days = set()
        hianyzo_elemek = trading_days.difference(available_days)
        for h in hianyzo_elemek:
            success = download_and_save_data(
                ticker, h, h + timedelta(days=1), settings=settings
            )
            if not success:
                fails.append({ticker: str(h)})
                if ticker not in fail_counter:
                    fail_counter[ticker] = 0
                fail_counter[ticker] += 1

    failed_days_path = getattr(conf, "FAILED_DAYS_FILE_PATH", "failed_days.json")
    with open(failed_days_path, "w") as f:
        json.dump(fails, f)

    for f in fail_counter:
        logger.warning(
            f"Failed {f:10s}->{fail_counter[f]:3d}:{str([[f[k] for k in f][0] for f in fails])}"
        )


def get_market_volatility_index(settings=None):
    """
    Visszaadja a legfrissebb VIX (felelem index) erteket.
    Ha nem sikerul letolteni, None-t ad (nem blokkol).
    """
    # Prefer a DataManager provided by the module (tests may monkeypatch
    # `data_loader.DataManager`). Fall back to attempting to construct one.
    try:
        dm = _create_data_repository(settings=settings)
    except Exception as e:
        logger.error(f"Failed to initialize DataManager for VIX lookup: {e}")
        return None

    symbol = "^VIX"

    # 1. Megnezzuk a DB-ben a mai adatot
    today_str = datetime.now().strftime("%Y-%m-%d")
    db_data = dm.get_market_data(symbol, days=1)

    if db_data and db_data[0][0] == today_str:
        try:
            return float(db_data[0][1])
        except Exception:
            return db_data[0][1]

    # 2. Ha nincs meg mara, letoltjuk az utolso 5 napot (hetvege miatt)
    try:
        logger.info(f"Downloading {symbol} data...")
        vix_ticker = yf.Ticker(symbol)
        df = vix_ticker.history(period="5d")
        if not df.empty:
            try:
                dm.save_market_data(symbol, df)
            except Exception:
                # Don't fail the VIX lookup if saving fails; continue to return
                # the downloaded value.
                logger.debug("Failed to save VIX data to DM, continuing")
            try:
                return float(df["Close"].iloc[-1])
            except Exception:
                return df["Close"].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")

    # Fallback: ha a letoltes nem sikerult, probaljuk a legfrissebb DB adatot
    db_data = dm.get_market_data(symbol, days=10)
    if db_data:
        try:
            return float(db_data[0][1])
        except Exception:
            return db_data[0][1]
    return None


if __name__ == "__main__":
    run_full_download(
        tickers=get_supported_ticker_list(),
        start_date="2020-01-01",
        end_date=datetime.today().strftime("%Y-%m-%d"),
    )
