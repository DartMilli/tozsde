"""
WALK FORWARD OPTIMIZER – RESPONSIBILITY CONTRACT

ROLE:
    - Control TIME
    - Slice data into train / test folds
    - Call genetic_optimizer.optimize_params
    - Evaluate OOS performance
    - Select stable parameter sets

MUST NOT:
    - Implement GA logic
    - Define parameter bounds
    - Train ML models

INPUT:
    - Full historical DataFrame (single ticker)
    - Bounds (from analyzer)
    - GA config

OUTPUT:
    - Best parameter set + stability metrics
"""

import numpy as np
from typing import Optional

from app.optimization.genetic_optimizer import optimize_params
from app.optimization.fitness import (
    compute_walk_forward_metrics,
    fitness_single,
    NEG_INF,
)
from app.backtesting.backtester import Backtester
from app.data_access.data_loader import ensure_data_cached, load_data
from app.analysis.analyzer import param_bounds, save_params_for_ticker
from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class WalkForwardOptimizer:

    def __init__(
        self,
        ticker,  # type: str
        df,
        bounds,  # type: dict
        train_window,  # type: int
        test_window,  # type: int
        step_size,  # type: int
        verbose=True,  # type: bool
        ga_config=None,  # type: Optional[dict]
    ):
        self.ticker = ticker
        self.df = df
        self.bounds = bounds
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.verbose = verbose
        self.ga_config = ga_config or {}

    def run(self):
        results = []

        start = self.train_window
        end = len(self.df) - self.test_window

        oos_profits = []
        oos_drawdowns = []

        while start <= end:
            train_df = self.df.iloc[start - self.train_window : start]
            test_df = self.df.iloc[start : start + self.test_window]

            # -------------------------
            # 🔥 GA HÍVÁS ITT
            # -------------------------
            best_params = optimize_params(
                dataframes={self.ticker: train_df},
                bounds=self.bounds,
                population_size=self.ga_config.get("population_size", 50),
                ngen=self.ga_config.get("ngen", 30),
                cxpb=self.ga_config.get("cxpb", 0.7),
                mutpb=self.ga_config.get("mutpb", 0.2),
            )

            # -------------------------
            # OOS validáció
            # -------------------------

            bt = Backtester(test_df, self.ticker)
            riport = bt.run(best_params)  # BacktestReport
            oos_fitness = fitness_single(riport.metrics)

            oos_return = riport.metrics["net_profit"]
            oos_drawdown = riport.metrics["max_drawdown"]
            oos_trades = riport.metrics["trade_count"]
            is_profitable = oos_return > 0

            oos_profits.append(oos_return)
            oos_drawdowns.append(oos_drawdown)

            results.append(
                {
                    "start": start,
                    "params": best_params,
                    "oos_fitness": oos_fitness,
                    "oos_return": oos_return,
                    "oos_drawdown": oos_drawdown,
                    "oos_trades": oos_trades,
                    "profitable": is_profitable,
                }
            )

            start += self.step_size

        wf_result = compute_walk_forward_metrics(oos_profits, oos_drawdowns)

        logger.info(
            "Walk Forward done, raw_fitness=%.4f normalized_score=%.4f",
            wf_result.raw_fitness,
            wf_result.normalized_score,
        )

        if wf_result.raw_fitness == NEG_INF:
            logger.warning("Invalid WF fitness – unstable strategy")
            return None

        if len(results) < 3:
            logger.warning("Too few WF windows – WF is not interpretable")
            return None

        total_windows = len(results)
        profitable_windows = sum(r["profitable"] for r in results)

        win_rate = profitable_windows / total_windows

        avg_return = np.mean([r["oos_return"] for r in results])
        avg_drawdown = np.mean([r["oos_drawdown"] for r in results])
        avg_fitness = np.mean([r["oos_fitness"] for r in results])

        best_window = max(results, key=lambda r: r["oos_fitness"])

        return {
            "best_params": best_window["params"],
            "raw_fitness": wf_result.raw_fitness,
            "wf_fitness": wf_result.raw_fitness,
            "normalized_score": wf_result.normalized_score,
            "wf_summary": {
                "windows": total_windows,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "avg_drawdown": avg_drawdown,
            },
        }


def run_walk_forward(ticker: str):
    """Cron convenience wrapper.

    - Betölti a teljes historikus adatot
    - Config hónap-alapú ablakait ~21 trading day/hó közelítéssel bar-számra váltja
    - Meghívja a WalkForwardOptimizer-t és visszaadja a WF összefoglalót
    """

    if not ensure_data_cached(ticker, start=Config.START_DATE, end=Config.END_DATE):
        logger.error(
            f"{ticker}: data cache incomplete for {Config.START_DATE} -> {Config.END_DATE}"
        )
        return None

    df = load_data(ticker, start=Config.START_DATE, end=Config.END_DATE)

    days_per_month = 21
    train_window = int(Config.TRAIN_WINDOW_MONTHS * days_per_month)
    test_window = int(Config.TEST_WINDOW_MONTHS * days_per_month)
    step_size = int(Config.WINDOW_STEP_MONTHS * days_per_month)

    wf = WalkForwardOptimizer(
        ticker=ticker,
        df=df,
        bounds=param_bounds,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        verbose=False,
        ga_config={
            "population_size": Config.OPTIMIZER_POPULATION,
            "ngen": Config.OPTIMIZER_GENERATIONS,
            "cxpb": 0.7,
            "mutpb": 0.2,
        },
    )
    result = wf.run()
    if result:
        try:
            from app.data_access.data_manager import DataManager
            import json

            DataManager().save_walk_forward_result(
                ticker=ticker,
                result_json=json.dumps(result, default=str),
            )
        except Exception:
            pass
        try:
            best_params = result.get("best_params")
            if best_params:
                save_params_for_ticker(ticker, best_params)
        except Exception:
            pass
    return result
