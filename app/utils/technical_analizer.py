import numpy as np

"""
Támogatott indikátorok:
Indikátor	Cél	            Megjegyzés
SMA	        trend	        paraméterezhető periódus
EMA	        trend	        paraméterezhető periódus
RSI	        momentum	    szintek: 30, 70
MACD	    trend/momentum	3 paraméter: fast, slow, signal
BBANDS	    volatilitás	    periódus + szórás
ATR	        volatilitás	    figyelmeztetésre
ADX	        trend erősség	trend szűrésre
STOCH	    momentum	    %K, %D keresztezés

Minden tömb kimenet pontosan megegyezik az input hosszával,
hiányzó értékek az elején NaN-nal töltve.
"""

technical_indicators_summary = {
    "SMA": {
        "name": "Simple Moving Average",
        "shows": "Átlagos záróár az elmúlt periódus alapján.",
        "used_for": [
            "Trend irányának megállapítása",
            "Mozgóátlag keresztezési stratégiák (pl. SMA20 vs. SMA50)",
        ],
        "signals": [
            "Ár felfelé áttöri az SMA-t → vételi jel",
            "Ár lefelé áttöri az SMA-t → eladási jel",
        ],
        "strengths": ["Egyszerű, megbízható", "Könnyen értelmezhető"],
        "weaknesses": [
            "Lassú reagálás gyors piaci mozgásra",
            "Minden nap azonos súllyal szerepel",
        ],
    },
    "EMA": {
        "name": "Exponential Moving Average",
        "shows": "Az árak exponenciálisan súlyozott mozgóátlaga – a frissebb adatok nagyobb súlyt kapnak.",
        "used_for": [
            "Rövid távú trendváltozás detektálása",
            "Mozgóátlag crossover stratégiák",
        ],
        "signals": [
            "Rövidebb EMA felfelé áttöri a hosszabb EMA-t → vétel",
            "EMA lefelé áttöri a másikat → eladás",
        ],
        "strengths": ["Gyorsabb reagálás, mint SMA", "Jobb rövid távú kereskedéshez"],
        "weaknesses": [
            "Túlérzékeny lehet oldalazó piacokon",
            "Nagyobb a fals jelzések esélye",
        ],
    },
    "RSI": {
        "name": "Relative Strength Index",
        "shows": "Áremelkedések és árcsökkenések aránya alapján kiszámított oszcillátor, 0–100 skálán.",
        "used_for": [
            "Túlvett / túladott állapotok felismerése",
            "Divergenciák azonosítása",
        ],
        "signals": [
            "RSI < 30 és felfelé áttöri → vétel (túladott)",
            "RSI > 70 és lefelé áttöri → eladás (túlvett)",
        ],
        "strengths": [
            "Jól jelzi extrém piaci állapotokat",
            "Egyszerű értelmezésű, vizuális",
        ],
        "weaknesses": [
            "Oldalazó piacon gyakori fals jelzések",
            "Nem trendindikátor – csak momentumot jelez",
        ],
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "shows": "Két különböző EMA közötti különbség, a trend és momentum kombinált mutatója.",
        "used_for": [
            "Trendváltások azonosítása",
            "Momentum megerősítése",
            "Divergencia figyelés az árhoz képest",
        ],
        "signals": [
            "MACD vonal felfelé áttöri a szignálvonalat → vétel",
            "MACD lefelé áttöri → eladás",
        ],
        "strengths": [
            "Két indikátor erejét egyesíti (trend + momentum)",
            "Vizualizálható szignálkeresztezés",
        ],
        "weaknesses": ["Késleltetett indikátor", "Paraméterfüggő és érzékeny"],
    },
    "BBANDS": {
        "name": "Bollinger Bands",
        "shows": "Sávok az SMA körül, ± adott számú szórás. A sávszélesség a volatilitás függvénye.",
        "used_for": [
            "Volatilitás mérése",
            "Ár extrém kilengéseinek detektálása",
            "Közelgő kitörések előrejelzése",
        ],
        "signals": [
            "Ár kilép az alsó sávból → túladott → potenciális vétel",
            "Ár kilép a felső sávból → túlvett → potenciális eladás",
            "Sávok szűkülnek → volatilitás növekedése várható",
        ],
        "strengths": ["Vizualisan intuitív", "Alkalmazkodik a volatilitáshoz"],
        "weaknesses": [
            "Nincs irányjelzés, csak szórás-alapú",
            "Kilépés nem mindig jelez trendváltást",
        ],
    },
    "ATR": {
        "name": "Average True Range",
        "shows": "Az átlagos napi mozgástartomány – volatilitás indikátor, nem irányfüggő.",
        "used_for": [
            "Volatilitás növekedés figyelése",
            "Stop-loss és take-profit szintek meghatározása",
        ],
        "signals": [
            "ATR hirtelen növekedése → kitörés vagy piaci feszültség",
            "Alacsony ATR → konszolidáció, trendhiány",
        ],
        "strengths": ["Jól működik bármilyen irányban", "Kitörések előjele lehet"],
        "weaknesses": ["Nem mutat irányt", "Nem ad konkrét belépési jelzést"],
    },
    "ADX": {
        "name": "Average Directional Index",
        "shows": "A trend erősségét, függetlenül annak irányától (értéktartomány: 0–100).",
        "used_for": ["Trendkövető stratégia szűrése", "Oldalazó piacok kiszűrése"],
        "signals": [
            "ADX > 25 → erős trend",
            "ADX < 20 → gyenge vagy oldalazó piac",
            "Használható +DI és -DI értékekkel kombinálva",
        ],
        "strengths": [
            "Trenddetektálásra kiváló",
            "Szűrőként hatékony kereskedési rendszerekben",
        ],
        "weaknesses": ["Nem ad konkrét vétel/eladás szignált", "Erősen késleltetett"],
    },
    "STOCH": {
        "name": "Stochastic Oscillator",
        "shows": "A záróár pozícióját mutatja a periódus legmagasabb és legalacsonyabb értékei között (0–100 skálán).",
        "used_for": [
            "Túlvett (80 felett) és túladott (20 alatt) szintek felismerése",
            "Momentum-alapú fordulópontok",
            "Divergencia keresés",
        ],
        "signals": [
            "%K felfelé keresztezi %D-t túladott zónában → vétel",
            "%K lefelé keresztezi %D-t túlvett zónában → eladás",
            "Divergencia ármozgással → figyelmeztetés",
        ],
        "strengths": [
            "Gyors, jól reagál ciklikus mozgásokra",
            "Vizualizálható és kombinálható más indikátorokkal",
        ],
        "weaknesses": [
            "Túl sok fals jelzés gyors piacon",
            "Szükséges a zónahatárok kombinálása más indikátorral",
        ],
    },
}


def get_indicator_description():
    return technical_indicators_summary


def sma(data, period):
    """
    SMA (Simple Moving Average)
        Mit mutat?
            - Átlagos záróár a kiválasztott időszakban.
        Mire használható?
            - Trendkövetés
            - Mozgóátlag keresztezéses stratégiák
        Hogyan jelzi a feladatát?
            - Ár felfelé áttöri az SMA-t → vételi jelzés
            - Ár lefelé áttöri az SMA-t → eladási jelzés
            - Hosszabb SMA fölött való tartózkodás = trend megerősítése
        Erősség: Egyszerű, megbízható trendindikátor
        Gyengeség: Késik a piaci mozgásokhoz képest
    """
    data = np.asarray(data)
    sma_vals = np.convolve(data, np.ones(period) / period, mode="valid")
    return np.concatenate([np.full(period - 1, np.nan), sma_vals])


def ema(data, period):
    """
    EMA (Exponential Moving Average)
        Mit mutat?
            - Frissebb adatokra érzékenyebb átlag, mint az SMA.
        Mire használható?
            - Rövid távú trendváltozások detektálására
            - Crossover stratégiák (pl. EMA10 vs. EMA50)
        Hogyan jelzi a feladatát?
            - Rövidebb EMA felfelé áttöri a hosszabbat → vételi jel
            - EMA lefelé áttöri → eladási jel
            - EMA iránya is trendet mutat (emelkedő = bikapiac)
        Erősség: Gyors reakció, kiváló rövid távra
        Gyengeség: Zajos piacon fals jeleket adhat
    """
    data = np.asarray(data)
    alpha = 2 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def rsi(data, period=14):
    """
    RSI (Relative Strength Index)
        Mit mutat?
            - Árak relatív erőssége 0–100 között, alapértelmezett határok: 30 (túladott), 70 (túlvett)
        Mire használható?
            - Extrém piaci állapotok detektálása
            - Fordulópontok figyelése
        Hogyan jelzi a feladatát?
            - RSI < 30 és felfelé áttöri → vétel (túladott)
            - RSI > 70 és lefelé áttöri → eladás (túlvett)
            - Divergencia az ár és RSI között = figyelmeztető jel
        Erősség: Jó azonosító extrém zónákban
        Gyengeség: Oldalazásnál sok fals szignál
    """
    data = np.asarray(data)
    delta = np.diff(data, prepend=data[0])
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.full_like(data, np.nan)
    avg_loss = np.full_like(data, np.nan)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_vals = 100 - (100 / (1 + rs))
    rsi_vals[:period] = np.nan
    return rsi_vals


def macd(data, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence)
        Mit mutat?
            - Két EMA különbségéből származtatott indikátor. Momentum és trend együtt.
        Mire használható?
            - Trendváltások megerősítése
            - Divergencia figyelése
        Hogyan jelzi a feladatát?
            - MACD vonal felfelé áttöri a szignálvonalat → vételi jel
            - MACD lefelé áttöri → eladási jel
            - MACD / ár divergencia = figyelmeztetés
        Erősség: Kombinált trend és momentum indikátor
        Gyengeség: Késleltetett reakció, beállításérzékeny
    """
    data = np.asarray(data)
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def bbands(data, period=20, std_dev=2):
    """
    Bollinger Bands
        Mit mutat?
            - Ár volatilitása köré épülő sáv. A középvonal SMA.
        Mire használható?
            - Kitörések előrejelzése
            - Túlvett/túladott zónák figyelése
        Hogyan jelzi a feladatát?
            - Ár kilép az alsó sávból → túladott → vétel lehetőség
            - Ár kilép a felső sávból → túlvett → eladás lehetőség
            - Szalag szűkülése → közelgő volatilitásnövekedés
        Erősség: Jól mutatja a piac feszültségét
        Gyengeség: Nincs konkrét irányjelzés
    """
    data = np.asarray(data)
    ma = sma(data, period)
    rolling_std = np.array(
        [
            np.std(data[i - period + 1 : i + 1]) if i >= period - 1 else np.nan
            for i in range(len(data))
        ]
    )
    upper = ma + std_dev * rolling_std
    lower = ma - std_dev * rolling_std
    return upper, ma, lower


def atr(high, low, close, period=14):
    """
    ATR (Average True Range)
        Mit mutat?
            - Ár napi mozgásának átlagos nagysága – volatilitás mértéke.
        Mire használható?
            - Stop-loss/take-profit meghatározás
            - Volatilitás kitörések azonosítása
        Hogyan jelzi a feladatát?
            - ATR megugrik → nő a piaci bizonytalanság
            - Alacsony ATR → piac konszolidál, trendmentes
        Erősség: Robusztus volatilitásmérő
        Gyengeség: Nem ad irányt, nem szignálgenerátor önmagában
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    atr_vals = np.full_like(close, np.nan)
    atr_vals[period] = np.mean(tr[:period])
    for i in range(period + 1, len(close)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr_vals


def adx(high, low, close, period=14):
    """
    ADX (Average Directional Index)
        Mit mutat?
            - Trend erősségét 0–100 között (nem az irányt!).
        Mire használható?
            - Trendkövető stratégia alkalmazhatóságának eldöntése
        Hogyan jelzi a feladatát?
            - ADX > 25 → erős trend (trendkövetés működhet)
            - ADX < 20 → oldalazó piac, trendstratégiák kerülendők
        Erősség: Jól mutatja, mikor érdemes egyáltalán belépni
        Gyengeség: Nem ad konkrét vételi/eladási jelet

        Visszatér: (adx, plus_di, minus_di) numpy tömbökkel. Ha túl rövid az adatsor, akkor NaN-okkal.
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    min_len = period * 2 + 1
    length = len(close)

    # Ha túl rövid az adatsor, térjünk vissza NaN-okkal
    if length < min_len:
        nan_arr = np.full(length, np.nan)
        return nan_arr, nan_arr, nan_arr

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])

    plus_dm_smooth = np.full(length, np.nan)
    minus_dm_smooth = np.full(length, np.nan)
    tr_smooth = np.full(length, np.nan)

    plus_dm_smooth[period] = np.sum(plus_dm[:period])
    minus_dm_smooth[period] = np.sum(minus_dm[:period])
    tr_smooth[period] = np.sum(tr[:period])

    for i in range(period + 1, length):
        plus_dm_smooth[i] = (
            plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i - 1]
        )
        minus_dm_smooth[i] = (
            minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i - 1]
        )
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i - 1]

    plus_di = 100 * np.divide(
        plus_dm_smooth,
        tr_smooth,
        out=np.full_like(plus_dm_smooth, np.nan),
        where=tr_smooth != 0,
    )
    minus_di = 100 * np.divide(
        minus_dm_smooth,
        tr_smooth,
        out=np.full_like(minus_dm_smooth, np.nan),
        where=tr_smooth != 0,
    )

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    adx_vals = np.full(length, np.nan)
    adx_vals[period * 2] = np.nanmean(dx[period : period * 2])

    for i in range(period * 2 + 1, length):
        adx_vals[i] = (adx_vals[i - 1] * (period - 1) + dx[i]) / period

    return adx_vals, plus_di, minus_di


def stoch(high, low, close, k_period=14, d_period=3):
    """
    STOCH (Stochastic Oscillator)
        Mit mutat?
            - A záróár helyzetét az elmúlt periódus sávján belül.
        Mire használható?
            - Túlvett (80 fölött), túladott (20 alatt) állapot
            - Ciklikus fordulók keresése
        Hogyan jelzi a feladatát?
            - %K felfelé keresztezi %D-t túladott zónában → vétel
            - %K lefelé keresztezi %D-t túlvett zónában → eladás
            - Divergencia = figyelmeztető jelzés
        Erősség: Gyorsan jelez, jól működik ciklikus piacokon
        Gyengeség: Zajos piacon sok fals jel
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    lowest_low = np.array(
        [
            np.min(low[i - k_period + 1 : i + 1]) if i >= k_period - 1 else np.nan
            for i in range(len(low))
        ]
    )
    highest_high = np.array(
        [
            np.max(high[i - k_period + 1 : i + 1]) if i >= k_period - 1 else np.nan
            for i in range(len(high))
        ]
    )
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = np.convolve(k[~np.isnan(k)], np.ones(d_period) / d_period, mode="valid")
    d_full = np.concatenate([np.full(len(k) - len(d), np.nan), d])
    return k, d_full
