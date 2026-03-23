from app.bootstrap.build_settings import build_settings
from app.infrastructure.repositories import DataManagerRepository


def allocate_capital(
    decisions: list,
    settings=None,
    get_correlation_matrix_fn=None,
) -> list:
    buy_candidates = [d for d in decisions if d["decision"]["action_code"] == 1]

    if not buy_candidates:
        return decisions

    cfg = settings or build_settings()
    total_capital = getattr(cfg, "INITIAL_CAPITAL")
    tradeable_capital = total_capital

    current_date = None
    if decisions:
        current_date = decisions[0].get("date")

    raw_weights = []
    tickers = []

    for item in buy_candidates:
        vol = item["payload"].get("volatility", 0.02)
        if vol <= 0:
            vol = 0.01
        raw_weights.append(1.0 / vol)
        tickers.append(item["ticker"])

    corr_matrix_loader = get_correlation_matrix_fn or _get_correlation_matrix

    if len(tickers) > 1:
        correlation_matrix = corr_matrix_loader(
            tickers,
            current_date,
            settings=settings,
        )

        corr_penalty = []
        for i, ticker in enumerate(tickers):
            avg_corr = (correlation_matrix.iloc[i].sum() - 1.0) / (len(tickers) - 1)
            penalty = 1.0 - (max(0, avg_corr) * 0.5)
            corr_penalty.append(penalty)

        for i in range(len(raw_weights)):
            raw_weights[i] *= corr_penalty[i]

    total_weight = sum(raw_weights)
    buy_indices = [
        i for i, d in enumerate(decisions) if d["decision"]["action_code"] == 1
    ]

    for i, list_idx in enumerate(buy_indices):
        normalized_weight = raw_weights[i] / total_weight
        allocation_amount = tradeable_capital * normalized_weight

        decisions[list_idx]["allocation_amount"] = round(allocation_amount, 2)
        decisions[list_idx]["allocation_pct"] = round(normalized_weight, 4)

        if len(tickers) > 1:
            decisions[list_idx]["decision"]["allocation_note"] = "Correlation adjusted"

    for item in decisions:
        if item["decision"]["action_code"] != 1:
            item["allocation_amount"] = 0.0
            item["allocation_pct"] = 0.0

    return decisions


def _get_correlation_matrix(tickers, ref_date=None, settings=None):
    try:
        dm = DataManagerRepository(settings=settings)
    except TypeError:
        dm = DataManagerRepository()
    return dm.get_correlation_matrix(tickers, ref_date=ref_date)


def enforce_correlation_limits(
    decisions: list,
    max_correlation: float = 0.7,
    settings=None,
    get_correlation_matrix_fn=None,
) -> list:
    import logging

    logger = logging.getLogger(__name__)

    tickers = [d["ticker"] for d in decisions if d.get("allocation_amount", 0) > 0]
    current_date = decisions[0].get("date") if decisions else None

    if len(tickers) < 2:
        return decisions

    corr_matrix_loader = get_correlation_matrix_fn or _get_correlation_matrix

    try:
        correlation_matrix = corr_matrix_loader(
            tickers,
            current_date,
            settings=settings,
        )

        high_corr_pairs = []
        for i, ticker1 in enumerate(correlation_matrix.columns):
            for j, ticker2 in enumerate(correlation_matrix.columns):
                if i < j and correlation_matrix.iloc[i, j] > max_correlation:
                    high_corr_pairs.append(
                        (ticker1, ticker2, correlation_matrix.iloc[i, j])
                    )

        if not high_corr_pairs:
            return decisions

        for ticker1, ticker2, corr in high_corr_pairs:
            d1 = next((d for d in decisions if d["ticker"] == ticker1), None)
            d2 = next((d for d in decisions if d["ticker"] == ticker2), None)

            if d1 and d2:
                conf1 = d1.get("decision", {}).get("confidence", 0)
                conf2 = d2.get("decision", {}).get("confidence", 0)

                weaker = d2 if conf1 > conf2 else d1
                stronger_ticker = ticker1 if conf1 > conf2 else ticker2

                if weaker.get("allocation_amount", 0) > 0:
                    weaker["allocation_amount"] *= 0.5
                    weaker["allocation_pct"] *= 0.5

                    logger.warning(
                        f"High correlation {ticker1}-{ticker2}: {corr:.2%}. "
                        f"Reduced {weaker['ticker']} allocation by 50% "
                        f"(kept {stronger_ticker} at full allocation due to higher confidence)"
                    )

                    if "decision" in weaker:
                        weaker["decision"]["correlation_adjustment"] = True

        return decisions

    except Exception as e:
        logger.error(f"Error in enforce_correlation_limits: {e}")
        return decisions
