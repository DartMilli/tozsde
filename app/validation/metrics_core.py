"""
metrics_core.py
Core pénzügyi metrika számítások.

Helye:
app/validation/metrics_core.py
"""

import numpy as np
import pandas as pd


# -----------------------------
# Return számítások
# -----------------------------


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    returns = equity_curve.pct_change().dropna()
    return returns


# -----------------------------
# Sharpe
# -----------------------------


def compute_sharpe(
    returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252
) -> float:
    excess = returns - risk_free_rate / annualization
    if excess.std() == 0:
        return 0.0
    return np.sqrt(annualization) * excess.mean() / excess.std()


# -----------------------------
# Sortino
# -----------------------------


def compute_sortino(returns: pd.Series, annualization: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    return np.sqrt(annualization) * returns.mean() / downside.std()


# -----------------------------
# Max Drawdown
# -----------------------------


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


# -----------------------------
# Profit Factor
# -----------------------------


def compute_profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0:
        return np.inf
    return gains / losses


# -----------------------------
# Calmar Ratio
# -----------------------------


def compute_calmar(equity_curve: pd.Series, annualization: int = 252) -> float:
    returns = compute_returns(equity_curve)
    annual_return = returns.mean() * annualization
    max_dd = abs(compute_max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return annual_return / max_dd


# -----------------------------
# Volatility
# -----------------------------


def compute_volatility(returns: pd.Series, annualization: int = 252) -> float:
    return returns.std() * np.sqrt(annualization)


# -----------------------------
# Összesített metrika wrapper
# -----------------------------


def compute_all_metrics(
    equity_curve: pd.Series, trade_returns: pd.Series | None = None
) -> dict:
    returns = compute_returns(equity_curve)

    metrics = {
        "sharpe": compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "volatility": compute_volatility(returns),
        "calmar": compute_calmar(equity_curve),
    }

    if trade_returns is not None:
        metrics["profit_factor"] = compute_profit_factor(trade_returns)

    return metrics


def compute_trade_statistics(trade_returns: pd.Series) -> dict:
    if trade_returns is None or trade_returns.empty:
        return {
            "trade_count": 0,
            "winrate": 0.0,
            "avg_trade_return": 0.0,
            "profit_factor": 0.0,
        }

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    winrate = float(len(wins)) / float(len(trade_returns))
    avg_trade_return = float(trade_returns.mean())
    profit_factor = compute_profit_factor(trade_returns)

    return {
        "trade_count": int(len(trade_returns)),
        "winrate": winrate,
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
    }
