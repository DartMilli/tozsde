import math


def normalize_sharpe(sharpe: float) -> float:
    """
    Normalizes Sharpe ratio into [0,1] using tanh.
    """
    if sharpe is None:
        return 0.5
    return max(0.0, min(1.0, math.tanh(sharpe / 2)))


def normalize_drawdown(max_dd: float) -> float:
    """
    max_dd expected as NEGATIVE value (e.g. -0.25)
    """
    if max_dd is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 + max_dd))


def normalize_profit(total_return: float) -> float:
    """
    total_return: e.g. 0.35 = +35%
    """
    if total_return is None:
        return 0.5
    return max(0.0, min(1.0, math.tanh(total_return)))


def compute_ga_wf_score(
    win_rate: float,
    avg_return: float,
    sharpe: float,
    max_drawdown: float,
):
    """
    Normalize GA + Walk-Forward performance into [0,1]
    """

    win_score = min(max(win_rate, 0), 1)

    return_score = min(max(avg_return / 0.5, 0), 1)  # 50% cap
    sharpe_score = min(max(sharpe / 2.0, 0), 1)
    dd_penalty = min(max(max_drawdown / 0.5, 0), 1)

    score = (
        0.35 * win_score + 0.30 * sharpe_score + 0.25 * return_score - 0.20 * dd_penalty
    )

    return max(0.0, min(score, 1.0))
