from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any, Optional


@dataclass
class BacktestReport:
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "diagnostics": self._serialize_diagnostics(),
            "meta": self.meta,
        }

    def _serialize_diagnostics(self):
        out = {}
        for k, v in self.diagnostics.items():
            if isinstance(v, pd.Series):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out


# old implementation (commented out, not used)
# @dataclass
# class BacktestMetrics:
#     net_profit: float
#     max_drawdown: float  # %
#     winrate: float  # 0-1
#     trade_count: int
#     sharpe: Optional[float]
#
#
# @dataclass
# class BacktestDiagnostics:
#     sharpe_ratio: float
#     equity_curve: list
#     final_value: float
#     total_cost: float


@dataclass
class WalkForwardMetrics:
    avg_profit: float
    avg_dd: float
    profit_std: float
    dd_std: float
    negative_fold_ratio: float
