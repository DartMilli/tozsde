"""
BACKTESTER – SINGLE SOURCE OF TRUTH

ROLE:
    - Execute strategy with given parameters
    - Produce performance metrics
    - Compute FITNESS value

MUST NOT:
    - Optimize parameters
    - Know about walk-forward
    - Use ML models
    - Make recommendations

OUTPUT METRICS:
    - net_profit
    - max_drawdown
    - winrate
    - trade_count
    - fitness
"""

import pandas as pd
import numpy as np
from app.config.config import Config
from app.analysis.analyzer import compute_signals  # Innen vesszük a jeleket
from app.reporting.metrics import BacktestReport


class Backtester:
    def __init__(self, df: pd.DataFrame, ticker: str):
        self.df = df.copy()
        self.initial_capital = Config.INITIAL_CAPITAL
        self.fee_pct = Config.TRANSACTION_FEE_PCT
        self.slippage_pct = Config.MIN_SLIPPAGE_PCT
        self.spread_pct = Config.SPREAD_PCT
        self.ticker = ticker

    def run(self, params: dict) -> dict:
        """
        Lefuttatja a szimulációt a megadott paraméterekkel.
        Ez váltja ki a genetic_optimizer 'backtest_signal_strategy' függvényét is!
        """
        MIN_REQUIRED_BARS = max(
            [
                params.get("sma_period", 0),
                params.get("ema_period", 0),
                params.get("rsi_period", 0),
                params.get("macd_slow", 0) + params.get("macd_signal", 0),
                params.get("bbands_period", 0),
                params.get("atr_period", 0),
                params.get("adx_period", 0) * 2 + 1,
                params.get("stoch_k", 0) + params.get("stoch_d", 0),
            ]
        )

        if len(self.df) < MIN_REQUIRED_BARS + 1:
            return BacktestReport(
                metrics={
                    "net_profit": 0,
                    "max_drawdown": 0,
                    "winrate": 0,
                    "trade_count": 0,
                    "avg_trade_return": 0,
                    "profit_factor": 0,
                },
                diagnostics={
                    "equity_curve": pd.Series(dtype=float),
                    "final_value": self.initial_capital,
                    "sharpe": 0,
                    "total_cost": 0,
                },
                meta={
                    "ticker": self.ticker,
                    "bars": len(self.df),
                    "params": params,
                },
            )

        # 1. Jelek generálása (A közös analizer.py-ból)
        signals, _ = compute_signals(self.df, self.ticker, params, return_series=True)

        # Ha a signals nem lista, hanem Series/DataFrame, konvertáljuk
        if hasattr(signals, "tolist"):
            signals = signals.tolist()

        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        trade_count = 0
        trades = []  # minden lezárt trade ide kerül
        open_trade = None  # aktuális nyitott pozíció
        total_cost = 0.0

        closes = self.df["Close"].values
        dates = self.df.index

        # --- A TE EREDETI LOGIKÁD OPTIMALIZÁLVA ---
        for i in range(len(self.df)):
            price = closes[i]
            # Mai jel
            signal_today = signals[i]

            # Portfólió érték rögzítése (tranzakció előtt)
            current_val = cash + (shares * price)
            portfolio_values.append(current_val)

            # VÉTEL
            if "BUY" in signal_today and cash > price:
                buy_price = price * (1 + self.slippage_pct + self.spread_pct / 2)
                max_shares = int(cash / buy_price)
                if max_shares > 0:
                    cost = max_shares * buy_price
                    commission = cost * self.fee_pct
                    if cash >= (cost + commission):
                        cash -= cost + commission
                        shares += max_shares
                        trade_count += 1
                        open_trade = {
                            "entry_price": buy_price,
                            "shares": max_shares,
                            "entry_value": cost + commission,
                        }

            # ELADÁS
            elif "SELL" in signal_today and shares > 0:
                sell_price = price * (1 - self.slippage_pct - self.spread_pct / 2)
                revenue = shares * sell_price
                commission = revenue * self.fee_pct
                cash += revenue - commission
                shares = 0
                trade_count += 1

                pnl = revenue - commission - open_trade["entry_value"]
                pnl_pct = pnl / open_trade["entry_value"] * 100

                trades.append(
                    {
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    }
                )

                open_trade = None

        # --- Eredmények számítása ---
        equity_curve = pd.Series(portfolio_values, index=dates)
        final_value = equity_curve.iloc[-1]

        # Hozam
        returns = equity_curve.pct_change().dropna()
        total_return_pct = ((final_value / self.initial_capital) - 1) * 100

        # Sharpe
        sharpe_ratio = 0
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown_pct = drawdown.min() * 100

        # Fitness Score (A te logikád: büntetjük a nagy visszaesést)
        fitness = sharpe_ratio
        if max_drawdown_pct < -20:
            fitness *= 0.5

        total_cost = commission + (max_shares * price * self.spread_pct)

        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            losses = [t for t in trades if t["pnl"] < 0]

            winrate = len(wins) / len(trades)

            avg_trade_return = np.mean([t["pnl_pct"] for t in trades])

            gross_profit = sum(t["pnl"] for t in wins)
            gross_loss = abs(sum(t["pnl"] for t in losses))

            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )
        else:
            winrate = 0.0
            avg_trade_return = 0.0
            profit_factor = 0.0

        report = BacktestReport(
            metrics={
                "net_profit": total_return_pct,
                "max_drawdown": max_drawdown_pct,
                "winrate": winrate,
                "trade_count": trade_count,
                "avg_trade_return": avg_trade_return,
                "profit_factor": profit_factor,
            },
            diagnostics={
                "equity_curve": equity_curve,
                "final_value": final_value,
                "sharpe": sharpe_ratio,
                "total_cost": total_cost,
            },
            meta={
                "ticker": self.ticker,
                "bars": len(self.df),
                "params": params,
            },
        )

        return report
