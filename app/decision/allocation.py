from app.config.config import Config
from app.data_access.data_manager import DataManager


def allocate_capital(decisions: list) -> list:
    """
    P7 - Allocation with Inverse Volatility AND Correlation check.
    """
    buy_candidates = [d for d in decisions if d["decision"]["action_code"] == 1]

    if not buy_candidates:
        return decisions

    total_capital = Config.INITIAL_CAPITAL
    tradeable_capital = total_capital

    # Dátum kinyerése az első döntésből
    current_date = None
    if decisions:
        # A Recommender.construct_decision 'date' kulcsot rak a root szintre vagy a payloadba?
        # A snapshot alapján a Recommender.construct_decision eredménye:
        # { "ticker":..., "date": date_str, "decision": {...} }
        current_date = decisions[0].get("date")

    # 1. Inverz volatilitás súlyok (Alapréteg)
    raw_weights = []
    tickers = []

    for item in buy_candidates:
        vol = item["payload"].get("volatility", 0.02)
        if vol <= 0:
            vol = 0.01
        raw_weights.append(1.0 / vol)
        tickers.append(item["ticker"])

    # 2. Korrelációs büntetés (ÚJ)
    # Ha több eszköz van, megnézzük, mennyire mozognak együtt
    if len(tickers) > 1:
        correlation_matrix = _get_correlation_matrix(tickers, current_date)

        # Minden eszközre kiszámoljuk az "átlagos korrelációját" a többivel
        # Ha valaki nagyon hasonlít a többire, csökkentjük a súlyát (hogy diverzifikáljunk)
        corr_penalty = []
        for i, ticker in enumerate(tickers):
            # Az i. sor átlaga a mátrixban (kivéve önmagát, ami 1.0)
            avg_corr = (correlation_matrix.iloc[i].sum() - 1.0) / (len(tickers) - 1)
            # Büntető szorzó: minél nagyobb a korreláció, annál kisebb szorzó (pl. 0.5 - 1.0 között)
            # Ha avg_corr 1.0 -> penalty 0.5. Ha avg_corr 0.0 -> penalty 1.0
            penalty = 1.0 - (max(0, avg_corr) * 0.5)
            corr_penalty.append(penalty)

        # Súlyok módosítása a büntetéssel
        for i in range(len(raw_weights)):
            raw_weights[i] *= corr_penalty[i]

    # 3. Normalizálás és kiosztás
    total_weight = sum(raw_weights)
    buy_indices = [
        i for i, d in enumerate(decisions) if d["decision"]["action_code"] == 1
    ]

    for i, list_idx in enumerate(buy_indices):
        normalized_weight = raw_weights[i] / total_weight
        allocation_amount = tradeable_capital * normalized_weight

        decisions[list_idx]["allocation_amount"] = round(allocation_amount, 2)
        decisions[list_idx]["allocation_pct"] = round(normalized_weight, 4)

        # Metaadatba beírjuk, hogy történt korrelációs súlyozás
        if len(tickers) > 1:
            decisions[list_idx]["decision"]["allocation_note"] = "Correlation adjusted"

    # Reset others
    for item in decisions:
        if item["decision"]["action_code"] != 1:
            item["allocation_amount"] = 0.0
            item["allocation_pct"] = 0.0

    return decisions


def _get_correlation_matrix(tickers, ref_date=None):
    """
    Segédfüggvény: letölti az árakat és korrelációs mátrixot számol.
    """
    dm = DataManager()
    return dm.get_correlation_matrix(tickers, ref_date=ref_date)
