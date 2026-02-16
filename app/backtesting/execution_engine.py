"""Execution engine that maps trade indices to prices and returns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class ExecutionPolicy(str, Enum):
    CLOSE = "close_to_close"
    NEXT_OPEN = "next_open"


@dataclass(frozen=True)
class TradeIndex:
    entry_idx: int
    exit_idx: int


@dataclass(frozen=True)
class TradeExecution:
    entry_idx: int
    exit_idx: int
    entry_at: int
    exit_at: int
    entry_price_raw: float
    exit_price_raw: float
    buy_price: float
    sell_price: float
    trade_return: float
    close_return: float
    open_return: float
    entry_gap: float
    exit_gap: float


class ExecutionEngine:
    def __init__(
        self, policy: ExecutionPolicy, slippage: float, spread: float, fee_pct: float
    ):
        self.policy = policy
        self.slippage = float(slippage)
        self.spread = float(spread)
        self.fee_pct = float(fee_pct)

    def _raw_prices(
        self, trade: TradeIndex, closes, opens
    ) -> tuple[float, float, int, int]:
        entry_idx = trade.entry_idx
        exit_idx = trade.exit_idx

        if self.policy == ExecutionPolicy.CLOSE:
            return (
                float(closes[entry_idx]),
                float(closes[exit_idx]),
                entry_idx,
                exit_idx,
            )

        entry_at = entry_idx + 1
        exit_at = exit_idx + 1
        if entry_at >= len(opens) or exit_at >= len(opens):
            return None, None, entry_at, exit_at

        return float(opens[entry_at]), float(opens[exit_at]), entry_at, exit_at

    def execute(self, trades: List[TradeIndex], closes, opens) -> List[TradeExecution]:
        executions: List[TradeExecution] = []
        for trade in trades:
            entry_price_raw, exit_price_raw, entry_at, exit_at = self._raw_prices(
                trade, closes, opens
            )
            if entry_price_raw is None or exit_price_raw is None:
                continue

            buy_price = entry_price_raw * (
                1 + self.slippage + self.spread / 2 + self.fee_pct
            )
            sell_price = exit_price_raw * (
                1 - self.slippage - self.spread / 2 - self.fee_pct
            )
            trade_return = sell_price / buy_price - 1

            close_entry = float(closes[trade.entry_idx])
            close_exit = float(closes[trade.exit_idx])
            open_entry = (
                float(opens[trade.entry_idx + 1])
                if trade.entry_idx + 1 < len(opens)
                else None
            )
            open_exit = (
                float(opens[trade.exit_idx + 1])
                if trade.exit_idx + 1 < len(opens)
                else None
            )
            close_return = close_exit / close_entry - 1 if close_entry else 0.0
            open_return = (
                open_exit / open_entry - 1
                if open_entry is not None and open_exit is not None and open_entry
                else 0.0
            )
            entry_gap = (
                open_entry / close_entry - 1
                if open_entry is not None and close_entry
                else 0.0
            )
            exit_gap = (
                open_exit / close_exit - 1
                if open_exit is not None and close_exit
                else 0.0
            )

            executions.append(
                TradeExecution(
                    entry_idx=trade.entry_idx,
                    exit_idx=trade.exit_idx,
                    entry_at=entry_at,
                    exit_at=exit_at,
                    entry_price_raw=entry_price_raw,
                    exit_price_raw=exit_price_raw,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    trade_return=trade_return,
                    close_return=close_return,
                    open_return=open_return,
                    entry_gap=entry_gap,
                    exit_gap=exit_gap,
                )
            )

        return executions
