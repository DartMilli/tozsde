import sqlite3

import app.core.decision.decision_history_analyzer as _core_dha
from app.core.decision.decision_history_analyzer import (  # noqa: F401
    StrategyStats,
    TickerReliability,
)
from app.decision import get_settings


_core_dha.sqlite3 = sqlite3


class DecisionHistoryAnalyzer(_core_dha.DecisionHistoryAnalyzer):
    def __init__(self, settings=None):
        super().__init__(settings=settings or get_settings())
