"""Equity engine for applying trade returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EquityResult:
    equity_curve: List[float]
    trade_details: List[dict]


class EquityEngine:
    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)

    def apply(
        self,
        trade_executions: list,
        position_size_pct: float | None = None,
        audit: dict | None = None,
    ) -> EquityResult:
        equity = self.initial_capital
        curve = []
        trade_details = []
        size_zero_count = 0

        if isinstance(audit, dict):
            audit["capital_start"] = float(self.initial_capital)

        fixed_position_value = None
        if isinstance(position_size_pct, (int, float)) and 0 < position_size_pct <= 1:
            fixed_position_value = self.initial_capital * float(position_size_pct)

        for trade in trade_executions:
            position_value = (
                fixed_position_value if fixed_position_value is not None else equity
            )
            if position_value <= 0:
                size_zero_count += 1
                continue
            pnl = position_value * trade.trade_return
            equity += pnl
            curve.append(equity)
            trade_details.append(
                {
                    "pnl": pnl,
                    "pnl_pct": trade.trade_return * 100,
                }
            )

        if isinstance(audit, dict):
            audit["capital_end"] = float(equity)
            audit["size_zero_count"] = int(size_zero_count)

        return EquityResult(equity_curve=curve, trade_details=trade_details)
