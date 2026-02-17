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

import math
import uuid
import numpy as np
from typing import Optional
from datetime import datetime, timezone

from app.optimization.genetic_optimizer import optimize_params
from app.optimization.fitness import (
    compute_walk_forward_metrics,
    fitness_single,
    NEG_INF,
)
from app.optimization.ga_wf_normalizer import normalize_drawdown, normalize_sharpe
from app.validation.execution_stress import evaluate_execution_stress
from app.validation.ga_robustness import run_ga_robustness_tests
from app.analysis.analyzer import get_params
from app.backtesting.backtester import Backtester
from app.data_access.data_loader import ensure_data_cached, load_data
from app.analysis.analyzer import param_bounds, save_params_for_ticker
from app.config.config import Config
from app.infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def _narrow_bounds(bounds: dict, center_params: dict, cv_flags: list[str]) -> dict:
    if not cv_flags:
        return bounds
    narrowed = dict(bounds)
    for key in cv_flags:
        if key not in bounds:
            continue
        min_v, max_v = bounds[key]
        center = center_params.get(key, (min_v + max_v) / 2)
        new_min = max(min_v, center * 0.75)
        new_max = min(max_v, center * 1.25)
        if isinstance(min_v, int) and isinstance(max_v, int):
            new_min = int(round(new_min))
            new_max = int(round(new_max))
        if new_min >= new_max:
            continue
        narrowed[key] = (new_min, new_max)
    return narrowed


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

        cv_flags = self.ga_config.get("param_cv_flags")
        if cv_flags is None:
            cv_flags = run_ga_robustness_tests().get("param_cv_flags", [])
        base_bounds = dict(self.bounds)
        current_center = get_params(self.ticker)

        start = self.train_window
        end = len(self.df) - self.test_window

        oos_profits = []
        oos_drawdowns = []
        fold_relative_gaps = []
        fold_sharpe_stds = []
        fold_worst_case = []
        discarded_folds = 0

        while start <= end:
            train_df = self.df.iloc[start - self.train_window : start]
            test_df = self.df.iloc[start : start + self.test_window]

            # -------------------------
            # 🔥 GA HÍVÁS ITT
            # -------------------------
            narrowed_bounds = _narrow_bounds(base_bounds, current_center, cv_flags)

            best_params = optimize_params(
                dataframes={self.ticker: train_df},
                bounds=narrowed_bounds,
                population_size=self.ga_config.get("population_size", 50),
                ngen=self.ga_config.get("ngen", 30),
                cxpb=self.ga_config.get("cxpb", 0.7),
                mutpb=self.ga_config.get("mutpb", 0.2),
                param_cv_flags=self.ga_config.get("param_cv_flags"),
            )

            current_center = best_params

            stress = evaluate_execution_stress(train_df, self.ticker, best_params)
            relative_gap = stress.get("relative_gap_baseline")
            sharpe_std = stress.get("sharpe_std")
            worst_case = stress.get("worst_case_sharpe")
            if isinstance(relative_gap, (int, float)):
                fold_relative_gaps.append(float(relative_gap))
            if isinstance(sharpe_std, (int, float)):
                fold_sharpe_stds.append(float(sharpe_std))
            if isinstance(worst_case, (int, float)):
                fold_worst_case.append(float(worst_case))

            # -------------------------
            # OOS validáció
            # -------------------------

            bt = Backtester(test_df, self.ticker)
            riport = bt.run(best_params, execution_policy="next_open")  # BacktestReport
            oos_fitness = fitness_single(riport.metrics)
            oos_sharpe = float(riport.diagnostics.get("sharpe", 0.0))

            oos_return = riport.metrics["net_profit"]
            oos_drawdown = riport.metrics["max_drawdown"]
            oos_trades = riport.metrics["trade_count"]
            is_profitable = oos_return > 0

            gap_ratio = None
            if isinstance(relative_gap, (int, float)) and isinstance(
                oos_return, (int, float)
            ):
                if oos_return != 0:
                    gap_ratio = abs(relative_gap / oos_return)

            if isinstance(gap_ratio, (int, float)) and gap_ratio > 0.5:
                discarded_folds += 1
                start += self.step_size
                continue

            oos_profits.append(oos_return)
            oos_drawdowns.append(oos_drawdown)

            results.append(
                {
                    "start": start,
                    "params": best_params,
                    "oos_fitness": oos_fitness,
                    "oos_return": oos_return,
                    "oos_drawdown": oos_drawdown,
                    "oos_sharpe": oos_sharpe,
                    "oos_trades": oos_trades,
                    "profitable": is_profitable,
                    "fold_relative_gap": relative_gap,
                    "stress": {
                        "baseline_sharpe": stress.get("baseline_sharpe"),
                        "robustness_score": stress.get("robustness_score"),
                        "sharpe_std": sharpe_std,
                        "worst_case_sharpe": worst_case,
                        "relative_gap_baseline": relative_gap,
                        "constraint_passed": stress.get("constraint_passed"),
                        "stress_tested": stress.get("stress_tested"),
                    },
                    "gap_ratio": gap_ratio,
                }
            )

            start += self.step_size

        total_folds = max(1, len(results) + discarded_folds)
        if discarded_folds / total_folds > 0.3:
            logger.warning("Too many folds discarded due to execution gap ratio")
            return None

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

        best_gap = best_window.get("fold_relative_gap")
        execution_penalty = (
            min(abs(best_gap), 1.0) if isinstance(best_gap, (int, float)) else 0.0
        )
        robustness_factor = math.exp(-2 * execution_penalty)
        stability_score = float(wf_result.normalized_score)
        normalized_oos_sharpe = normalize_sharpe(best_window.get("oos_sharpe"))
        max_dd_norm = normalize_drawdown(best_window.get("oos_drawdown"))
        production_score = (
            0.4 * normalized_oos_sharpe
            + 0.2 * stability_score
            + 0.2 * robustness_factor
            + 0.2 * (1 - max_dd_norm)
        )
        production_score = max(0.0, min(1.0, production_score))

        mean_execution_gap = (
            float(np.mean(fold_relative_gaps)) if fold_relative_gaps else None
        )
        max_execution_gap = (
            float(np.max(fold_relative_gaps)) if fold_relative_gaps else None
        )
        gap_std = float(np.std(fold_relative_gaps)) if fold_relative_gaps else None
        execution_flag = None
        if isinstance(mean_execution_gap, (int, float)) and mean_execution_gap > 0.7:
            execution_flag = "HIGH_TIMING_DEPENDENCY"

        mean_sharpe_std = float(np.mean(fold_sharpe_stds)) if fold_sharpe_stds else None
        worst_case_sharpe_global = min(fold_worst_case) if fold_worst_case else None

        fold_sharpe_distribution = [r.get("oos_sharpe") for r in results]
        rolling_sharpe_trend = []
        window = 3
        for idx in range(len(fold_sharpe_distribution)):
            start_idx = max(0, idx - window + 1)
            segment = [
                v
                for v in fold_sharpe_distribution[start_idx : idx + 1]
                if v is not None
            ]
            rolling_sharpe_trend.append(float(np.mean(segment)) if segment else None)

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
                "mean_execution_gap": mean_execution_gap,
                "max_execution_gap": max_execution_gap,
                "gap_std": gap_std,
                "execution_gap_flag": execution_flag,
                "mean_sharpe_std": mean_sharpe_std,
                "worst_case_sharpe_global": worst_case_sharpe_global,
                "discarded_folds": discarded_folds,
                "discarded_ratio": discarded_folds / total_folds,
                "fold_sharpe_distribution": fold_sharpe_distribution,
                "rolling_sharpe_trend": rolling_sharpe_trend,
            },
            "robustness_factor": robustness_factor,
            "production_score": production_score,
            "execution_gaps": {
                "fold_relative_gaps": fold_relative_gaps,
                "mean_execution_gap": mean_execution_gap,
                "max_execution_gap": max_execution_gap,
                "gap_std": gap_std,
                "flag": execution_flag,
            },
            "execution_stress": best_window.get("stress"),
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
    # If optimizer returned a full WF summary (has best_params), attach metadata
    # and persist; otherwise return the raw result unchanged (tests expect this).
    if result and result.get("best_params") is not None:
        result["wf_run_id"] = str(uuid.uuid4())
        result["wf_run_at"] = datetime.now(timezone.utc).isoformat()
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
