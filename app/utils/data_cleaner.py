import pandas as pd

from app.utils.analizer import compute_signals, get_params


def sanitize_dataframe(df: pd.DataFrame, index_col: str = "date") -> pd.DataFrame:
    """
    Megtisztítja a DataFrame-et:
    - Biztosítja, hogy csak az elvárt oszlopok maradjanak
    - Minden értéket float/int típusra konvertál
    - Eltávolítja az érvénytelen (NaN) sorokat
    - Dátumindexet állít be, ha szükséges
    """

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]

    # Csak a várt oszlopokat tartjuk meg
    df = df.copy()
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col], errors="coerce")
        df.set_index(index_col, inplace=True)

    df = df[[col for col in expected_cols if col in df.columns]]

    # Konvertálás számokra
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    return df


def prepare_df(df, params=None):
    """Indikátorokat hozzáad a df-hez a RL környezethez."""
    if params is None:
        params_to_use = get_params()
    else:
        params_to_use = params
    _, indicators = compute_signals(df, params_to_use)
    for key, series in indicators.items():
        df[key] = series
    df.dropna(inplace=True)
    return df
