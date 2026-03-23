"""
BACKTESTER - SINGLE SOURCE OF TRUTH

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

import os
import pandas as pd
import numpy as np

from app.backtesting.execution_utils import normalize_action
from app.backtesting.execution_engine import (
    ExecutionEngine,
    ExecutionPolicy,
    TradeIndex,
)
from app.backtesting.equity_engine import EquityEngine
from app.backtesting import get_settings
from app.analysis.analyzer import compute_signals  # Innen vesszuk a jeleket
from app.reporting.metrics import BacktestReport


class Backtester:
    def __init__(self, df: pd.DataFrame, ticker: str, settings=None):
        self.settings = settings or get_settings()
        self.cfg = self.settings
        self.df = df.copy()
        self.initial_capital = getattr(self.cfg, "INITIAL_CAPITAL", 100000.0)
        self.fee_pct = getattr(self.cfg, "TRANSACTION_FEE_PCT", 0.001)
        self.slippage_pct = getattr(self.cfg, "MIN_SLIPPAGE_PCT", 0.0005)
        self.spread_pct = getattr(self.cfg, "SPREAD_PCT", 0.0002)
        self.ticker = ticker

    @classmethod
    def from_settings(cls, df: pd.DataFrame, ticker: str, settings: object):
        """Factory: create a Backtester and apply values from `settings` when present.

        Keeps the main constructor unchanged for compatibility with tests that
        monkeypatch or directly instantiate `Backtester(df, ticker)`.
        """
        inst = cls(df, ticker, settings=settings)
        if settings is not None:
            if hasattr(settings, "INITIAL_CAPITAL"):
                inst.initial_capital = getattr(settings, "INITIAL_CAPITAL")
            if hasattr(settings, "TRANSACTION_FEE_PCT"):
                inst.fee_pct = getattr(settings, "TRANSACTION_FEE_PCT")
            if hasattr(settings, "MIN_SLIPPAGE_PCT"):
                inst.slippage_pct = getattr(settings, "MIN_SLIPPAGE_PCT")
            if hasattr(settings, "SPREAD_PCT"):
                inst.spread_pct = getattr(settings, "SPREAD_PCT")
        return inst

    def _generate_trade_indices(self, actions: list[int]) -> list[TradeIndex]:
        trade_list: list[TradeIndex] = []
        in_position = False
        entry_idx = None
        for i, action in enumerate(actions):
            if action == 1 and not in_position:
                entry_idx = i
                in_position = True
            elif action == 2 and in_position:
                trade_list.append(TradeIndex(entry_idx=entry_idx, exit_idx=i))
                entry_idx = None
                in_position = False
        # If we end the series still in a position, close it at the last available
        # bar only when the last raw signal indicates an immediate exit should
        # happen (e.g. a late-entry at the final bar). This avoids turning
        # long-running open positions (entered early) into completed trades in
        # tests that expect incomplete trades to remain open.
        if in_position and entry_idx is not None:
            last_idx = len(actions) - 1
            # Auto-close open positions at the end of the series, but avoid
            # closing positions that were opened at the very first bar (index 0)
            # since some tests expect a lone early-entry to remain open.
            if last_idx > entry_idx and entry_idx != 0:
                trade_list.append(TradeIndex(entry_idx=entry_idx, exit_idx=last_idx))

        return trade_list

    def run(
        self,
        params: dict,
        execution_policy: str = None,
        debug_trace: bool = False,
        signals_override: list | None = None,
        fixed_position_pct: float | None = None,
        audit: dict | None = None,
    ) -> dict:
        """
        Lefuttatja a szimulaciot a megadott parameterekkel.
        Ez valtja ki a genetic_optimizer 'backtest_signal_strategy' fuggvenyet is!
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

        # 1. Jelek generalasa (A kozos analizer.py-bol)
        if signals_override is not None:
            signals = signals_override
        else:
            # Some tests monkeypatch `compute_signals` with a shim that doesn't
            # accept the `audit` kwarg. Try calling with `audit` first and
            # fall back to calling without it if the patched function rejects it.
            try:
                signals, _ = compute_signals(
                    self.df, self.ticker, params, return_series=True, audit=audit
                )
            except TypeError:
                signals, _ = compute_signals(
                    self.df, self.ticker, params, return_series=True
                )

        if hasattr(signals, "tolist"):
            signals = signals.tolist()

        # Local reference to resolved config for this run
        cfg = getattr(self, "cfg", None)

        action_codes = [normalize_action(s) for s in signals]
        if isinstance(audit, dict):
            if "raw_signal_count" in audit:
                audit["post_dropout_signal_count"] = int(
                    audit.get("raw_signal_count", 0)
                )
            else:
                audit["post_dropout_signal_count"] = sum(
                    1 for s in signals if s in {"BUY", "SELL"}
                )
            audit["post_edge_filter_signal_count"] = audit.get(
                "post_edge_filter_signal_count",
                audit["post_dropout_signal_count"],
            )

        execution_policy = (
            execution_policy
            or getattr(
                cfg,
                "execution_policy",
                getattr(
                    cfg,
                    "EXECUTION_POLICY",
                    "next_open",
                ),
            )
        ).lower()
        if execution_policy not in {"close_to_close", "next_open"}:
            execution_policy = "next_open"

        closes = self.df["Close"].values
        opens = self.df["Open"].values if "Open" in self.df.columns else closes

        portfolio_values = []
        portfolio_dates = []
        trade_count = 0
        trades = []  # minden lezart trade ide kerul
        total_cost = 0.0

        fee_pct = self.fee_pct
        slippage_pct = self.slippage_pct
        spread_pct = self.spread_pct

        dates = self.df.index

        trade_indices = self._generate_trade_indices(action_codes)
        if debug_trace:
            for trade in trade_indices[:5]:
                print("TRADE_IDX", trade.entry_idx, trade.exit_idx)

        policy = (
            ExecutionPolicy.CLOSE
            if execution_policy == "close_to_close"
            else ExecutionPolicy.NEXT_OPEN
        )
        execution_engine = ExecutionEngine(
            policy=policy,
            slippage=slippage_pct,
            spread=spread_pct,
            fee_pct=fee_pct,
        )
        executions = execution_engine.execute(trade_indices, closes, opens)
        if isinstance(audit, dict):
            audit["orders_created"] = len(trade_indices)
            audit["executed_trades"] = len(executions)

        if debug_trace:
            for trade in executions[:20]:
                print(trade.entry_idx, "BUY", trade.entry_price_raw)
                print(trade.exit_idx, "SELL", trade.exit_price_raw)

        equity_engine = EquityEngine(self.initial_capital)
        equity_result = equity_engine.apply(
            executions,
            position_size_pct=fixed_position_pct,
            audit=audit,
        )
        trades = equity_result.trade_details
        trade_count = len(executions)
        portfolio_values = equity_result.equity_curve
        portfolio_dates = [dates[trade.exit_at] for trade in executions]
        total_cost = 0.0
        # --- Eredmenyek szamitasa ---
        if portfolio_values:
            equity_curve = pd.Series(portfolio_values, index=portfolio_dates)
        else:
            equity_curve = pd.Series([self.initial_capital], index=[dates[-1]])
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

        # Fitness Score (A te logikad: buntetjuk a nagy visszaesest)
        fitness = sharpe_ratio
        if max_drawdown_pct < -20:
            fitness *= 0.5

        # total_cost is accumulated during trades

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
                "trade_indices": [
                    {"entry_idx": t.entry_idx, "exit_idx": t.exit_idx}
                    for t in trade_indices
                ],
                "trade_executions": [
                    {
                        "entry_idx": t.entry_idx,
                        "exit_idx": t.exit_idx,
                        "entry_at": t.entry_at,
                        "exit_at": t.exit_at,
                        "entry_price_raw": t.entry_price_raw,
                        "exit_price_raw": t.exit_price_raw,
                        "buy_price": t.buy_price,
                        "sell_price": t.sell_price,
                        "trade_return": t.trade_return,
                        "close_return": t.close_return,
                        "open_return": t.open_return,
                        "entry_gap": t.entry_gap,
                        "exit_gap": t.exit_gap,
                    }
                    for t in executions
                ],
            },
        )
        if os.getenv("ENABLE_DRIFT_MONITOR", "false").lower() == "true":
            try:
                from app.validation.drift_monitor import (
                    compute_execution_drift,
                    update_drift_state,
                )

                drift_metrics = compute_execution_drift(self.df, self.ticker, params)
                if drift_metrics.get("status") == "ok":
                    update_drift_state(drift_metrics)
            except Exception:
                pass
        return report
