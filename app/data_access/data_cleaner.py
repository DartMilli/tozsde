import pandas as pd

from app.analysis.analyzer import compute_signals, get_params


def sanitize_dataframe(df: pd.DataFrame, index_col: str = "date") -> pd.DataFrame:
    """
    Megtisztitja a DataFrame-et:
    - Biztositja, hogy csak az elvart oszlopok maradjanak
    - Minden erteket float/int tipusra konvertal
    - Eltavolitja az ervenytelen (NaN) sorokat
    - Datumindexet allit be, ha szukseges
    """

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Csak a vart oszlopokat tartjuk meg
    df = df.copy()
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
        df.set_index(index_col, inplace=True)

    df = df[[col for col in expected_cols if col in df.columns]]

    # Konvertalas szamokra
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # df.dropna(inplace=True)
    df.sort_index(inplace=True)

    return df


def prepare_df(df, ticker, params=None):
    """Indikatorokat hozzaad a df-hez a RL kornyezethez."""
    if params is None:
        params_to_use = get_params(ticker)
    else:
        params_to_use = params
    existing_cols = set(df.columns)
    indicator_keys = {
        "SMA",
        "EMA",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "ATR",
        "ADX",
        "PLUS_DI",
        "MINUS_DI",
        "STOCH_K",
        "STOCH_D",
    }
    if not indicator_keys.issubset(existing_cols):
        _, indicators = compute_signals(df, ticker, params_to_use)
        for key, series in indicators.items():
            df[key] = series
        df.dropna(inplace=True)
    return df
