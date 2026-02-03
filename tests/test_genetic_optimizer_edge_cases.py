"""
genetic_optimizer edge case tests
Focus: evaluate_individual(), parsimony_penalty(), optimize_params() edge cases
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from deap import base, creator, tools

from app.optimization.genetic_optimizer import (
    individual_to_params,
    evaluate_individual,
    evaluate_individual_with_log,
    evaluate_individual_without_log,
    _params_to_key,
    parsimony_penalty,
    optimize_params
)


class TestIndividualToParams:
    """Edge cases for individual_to_params()"""

    def test_empty_individual(self):
        """Empty individual list"""
        individual = []
        param_keys = []
        result = individual_to_params(individual, param_keys)
        assert result == {}

    def test_single_param(self):
        """Single parameter"""
        individual = [50]
        param_keys = ["sma_period"]
        result = individual_to_params(individual, param_keys)
        assert result == {"sma_period": 50}

    def test_multiple_params(self):
        """Multiple parameters in order"""
        individual = [20, 14, 70, 30]
        param_keys = ["sma_period", "ema_period", "rsi_period", "macd_fast"]
        result = individual_to_params(individual, param_keys)
        assert result == {
            "sma_period": 20,
            "ema_period": 14,
            "rsi_period": 70,
            "macd_fast": 30
        }

    def test_order_matters(self):
        """Order of param_keys determines mapping"""
        individual = [1, 2, 3]
        keys1 = ["a", "b", "c"]
        keys2 = ["c", "b", "a"]
        
        result1 = individual_to_params(individual, keys1)
        result2 = individual_to_params(individual, keys2)
        
        assert result1 != result2
        assert result1 == {"a": 1, "b": 2, "c": 3}
        assert result2 == {"c": 1, "b": 2, "a": 3}


class TestEvaluateIndividual:
    """Edge cases for evaluate_individual()"""

    @patch("app.optimization.genetic_optimizer.Backtester")
    @patch("app.optimization.genetic_optimizer.fitness_single")
    def test_empty_dataframes(self, mock_fitness, mock_backtester):
        """No tickers provided"""
        individual = [20, 14, 70]
        dataframes = {}
        keys = ["sma_period", "ema_period", "rsi_period"]
        
        # Returns consistent 2-element tuple
        fitness, valid_count = evaluate_individual(individual, dataframes, keys)
        assert fitness == -1000.0
        assert valid_count == 0
        mock_backtester.assert_not_called()

    @patch("app.optimization.genetic_optimizer.Backtester")
    @patch("app.optimization.genetic_optimizer.fitness_single")
    def test_single_ticker_success(self, mock_fitness, mock_backtester):
        """Single ticker successful evaluation"""
        individual = [20, 14, 70]
        df = pd.DataFrame({"Close": [100] * 100})  # 100 bars
        dataframes = {"AAPL": df}
        keys = ["sma_period", "ema_period", "rsi_period"]
        
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = MagicMock(metrics={"sharpe": 1.5})
        mock_backtester.return_value = mock_bt_instance
        mock_fitness.return_value = 1.5
        
        fitness, valid_count = evaluate_individual(individual, dataframes, keys)
        
        assert valid_count == 1
        assert fitness > 0
        mock_backtester.assert_called_once_with(df, "AAPL")

    @patch("app.optimization.genetic_optimizer.Backtester")
    def test_insufficient_bars(self, mock_backtester):
        """DataFrame with too few bars for indicators"""
        individual = [200, 200, 200]  # Very long periods
        df = pd.DataFrame({"Close": [100] * 10})  # Only 10 bars
        dataframes = {"AAPL": df}
        keys = ["sma_period", "ema_period", "rsi_period"]
        
        fitness, valid_count = evaluate_individual(individual, dataframes, keys)
        assert fitness == -1000.0
        assert valid_count == 0
        mock_backtester.assert_not_called()

    @patch("app.optimization.genetic_optimizer._FITNESS_CACHE", {})
    @patch("app.optimization.genetic_optimizer.Backtester")
    @patch("app.optimization.genetic_optimizer.fitness_single")
    def test_multiple_tickers_averaging(self, mock_fitness, mock_backtester):
        """Multiple tickers should average fitness"""
        individual = [20, 14, 70]
        df1 = pd.DataFrame({"Close": [100] * 100})
        df2 = pd.DataFrame({"Close": [200] * 100})
        dataframes = {"AAPL": df1, "MSFT": df2}
        keys = ["sma_period", "ema_period", "rsi_period"]
        
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = MagicMock(metrics={"sharpe": 1.5})
        mock_backtester.return_value = mock_bt_instance
        mock_fitness.side_effect = [2.0, 3.0]  # Different fitness for each ticker
        
        result = evaluate_individual(individual, dataframes, keys)
        fitness, valid_count = result
        
        assert valid_count == 2
        # Average is 2.5, but parsimony penalty applies (3 active params * 0.05 = 0.15 penalty)
        # 2.5 * 0.85 = 2.125, but extreme values also penalized
        assert abs(fitness - 2.5) < 1.5  # Approximate check after parsimony
        assert mock_backtester.call_count == 2

    @patch("app.optimization.genetic_optimizer._FITNESS_CACHE", {})
    @patch("app.optimization.genetic_optimizer.Backtester")
    @patch("app.optimization.genetic_optimizer.fitness_single")
    def test_exception_handling(self, mock_fitness, mock_backtester):
        """Exception in backtesting should skip that ticker"""
        individual = [20, 14, 70]
        df1 = pd.DataFrame({"Close": [100] * 100})
        df2 = pd.DataFrame({"Close": [200] * 100})
        dataframes = {"AAPL": df1, "MSFT": df2}
        keys = ["sma_period", "ema_period", "rsi_period"]
        
        # Mock first ticker succeeds
        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = MagicMock(metrics={"sharpe": 1.5})
        
        call_count = [0]
        def bt_side_effect(df, ticker):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (AAPL)
                return mock_bt_instance
            else:  # Second call (MSFT)
                raise Exception("Backtest failed")
        
        mock_backtester.side_effect = bt_side_effect
        mock_fitness.return_value = 2.0
        
        result = evaluate_individual(individual, dataframes, keys)
        fitness, valid_count = result
        
        # Exception is caught with continue, so only AAPL succeeds
        assert valid_count == 1
        assert fitness > 0  # AAPL fitness with parsimony


class TestParsimonyPenalty:
    """Edge cases for parsimony_penalty()"""

    def test_empty_params(self):
        """Empty params dict"""
        result = parsimony_penalty({})
        assert result == 1.0  # No penalty

    def test_single_param(self):
        """Single active parameter"""
        params = {"sma_period": 50}
        result = parsimony_penalty(params, weight=0.05)
        assert result == 0.95  # 1.0 - (1 * 0.05)

    def test_many_params_high_penalty(self):
        """Many parameters increase penalty"""
        params = {f"param_{i}": 50 for i in range(10)}
        result = parsimony_penalty(params, weight=0.05)
        assert result == 0.5  # 1.0 - (10 * 0.05)

    def test_none_values_ignored(self):
        """None values should not count"""
        params = {"sma_period": 50, "ema_period": None, "rsi_period": 14}
        result = parsimony_penalty(params, weight=0.05)
        assert result == 0.9  # Only 2 active params

    def test_zero_values_ignored(self):
        """Zero values should not count"""
        params = {"sma_period": 50, "ema_period": 0, "rsi_period": 14}
        result = parsimony_penalty(params, weight=0.05)
        assert result == 0.9  # Only 2 active params

    def test_extreme_values_extra_penalty(self):
        """Extreme values far from mid should add penalty"""
        param_limits = {"sma_period": (10, 200)}
        
        # Extreme value
        params_extreme = {"sma_period": 200}  # At max
        penalty_extreme = parsimony_penalty(params_extreme, weight=0.05, param_limits=param_limits)
        
        # Mid value
        params_mid = {"sma_period": 105}  # Near mid (105)
        penalty_mid = parsimony_penalty(params_mid, weight=0.05, param_limits=param_limits)
        
        assert penalty_extreme < penalty_mid  # Extreme gets more penalty

    def test_minimum_penalty_floor(self):
        """Penalty should not go below 0.3"""
        params = {f"param_{i}": 200 for i in range(100)}  # Huge penalty
        result = parsimony_penalty(params, weight=0.05)
        assert result >= 0.3


class TestOptimizeParams:
    """Edge cases for optimize_params()"""

    @patch("app.optimization.genetic_optimizer.custom_ea_simple")
    @patch("app.optimization.genetic_optimizer.evaluate_individual_without_log")
    def test_minimal_config(self, mock_eval, mock_ea):
        """Minimal population and generations"""
        df = pd.DataFrame({"Close": [100] * 100})
        dataframes = {"AAPL": df}
        bounds = {"sma_period": (10, 50)}
        
        # Create proper mock individual that can be unpacked
        mock_individual = [30]  # Single value for sma_period
        mock_ind_obj = MagicMock()
        mock_ind_obj.__getitem__ = lambda self, i: mock_individual[i]
        mock_ind_obj.__iter__ = lambda self: iter(mock_individual)
        mock_ind_obj.fitness = MagicMock(values=(1.5,))
        
        mock_population = [mock_ind_obj]
        mock_logbook = MagicMock(stream="log")
        
        # Mock EA and halloffame update  
        def ea_side_effect(population, toolbox, cxpb, mutpb, ngen, stats, halloffame):
            # Manually update halloffame with mock individual
            if halloffame is not None:
                halloffame.update([mock_ind_obj])
            return (mock_population, mock_logbook)
        
        mock_ea.side_effect = ea_side_effect
        
        result = optimize_params(
            dataframes=dataframes,
            bounds=bounds,
            population_size=2,
            ngen=1
        )
        
        assert isinstance(result, dict)
        assert "sma_period" in result
        mock_ea.assert_called_once()

    @patch("app.optimization.genetic_optimizer.custom_ea_simple")
    def test_multiple_params(self, mock_ea):
        """Multiple parameters in bounds"""
        df = pd.DataFrame({"Close": [100] * 200})
        dataframes = {"AAPL": df}
        bounds = {
            "sma_period": (10, 200),
            "ema_period": (5, 100),
            "rsi_period": (10, 30)
        }
        
        # Create mock individual with values matching bounds keys
        mock_individual = [50, 30, 20]  # Values for sma, ema, rsi
        mock_individual_obj = MagicMock()
        mock_individual_obj.__getitem__ = lambda self, i: mock_individual[i]
        mock_individual_obj.__iter__ = lambda self: iter(mock_individual)
        mock_individual_obj.fitness = MagicMock(values=(2.0,))
        
        mock_population = [mock_individual_obj]
        mock_logbook = MagicMock(stream="log")
        
        # Mock EA with halloffame update
        def ea_side_effect(population, toolbox, cxpb, mutpb, ngen, stats, halloffame):
            if halloffame is not None:
                halloffame.update([mock_individual_obj])
            return (mock_population, mock_logbook)
        
        mock_ea.side_effect = ea_side_effect
        
        result = optimize_params(
            dataframes=dataframes,
            bounds=bounds,
            population_size=5,
            ngen=2
        )
        
        assert len(result) == 3
        assert all(k in result for k in bounds.keys())


class TestGeneticOptimizerAdditional:
    """Additional tests for cache and helpers."""

    def test_params_to_key_deterministic(self):
        params1 = {"b": 2, "a": 1}
        params2 = {"a": 1, "b": 2}

        assert _params_to_key(params1) == _params_to_key(params2)

    @patch("app.optimization.genetic_optimizer._FITNESS_CACHE", {})
    @patch("app.optimization.genetic_optimizer.Backtester")
    @patch("app.optimization.genetic_optimizer.fitness_single")
    def test_evaluate_individual_uses_cache(self, mock_fitness, mock_backtester):
        individual = [20, 14, 70]
        df = pd.DataFrame({"Close": [100] * 100})
        dataframes = {"AAPL": df}
        keys = ["sma_period", "ema_period", "rsi_period"]

        mock_bt_instance = MagicMock()
        mock_bt_instance.run.return_value = MagicMock(metrics={"sharpe": 1.5})
        mock_backtester.return_value = mock_bt_instance
        mock_fitness.return_value = 2.0

        evaluate_individual(individual, dataframes, keys)
        evaluate_individual(individual, dataframes, keys)

        # Backtester should be called only once due to cache
        assert mock_backtester.call_count == 1

    def test_evaluate_wrappers_return_tuple(self):
        with patch("app.optimization.genetic_optimizer.evaluate_individual") as mock_eval:
            mock_eval.return_value = (1.23, 2)

            assert evaluate_individual_with_log([1], {}, ["p"]) == (1.23,)
            assert evaluate_individual_without_log([1], {}, ["p"]) == (1.23,)

    @patch("app.optimization.genetic_optimizer.custom_ea_simple")
    @patch("app.optimization.genetic_optimizer.is_logger_debug", return_value=True)
    def test_optimize_params_uses_debug_evaluate(self, mock_debug, mock_ea):
        df = pd.DataFrame({"Close": [100] * 100})
        dataframes = {"AAPL": df}
        bounds = {"sma_period": (10, 50)}

        def ea_side_effect(population, toolbox, cxpb, mutpb, ngen, stats, halloffame):
            # Ensure evaluate is callable
            assert callable(toolbox.evaluate)
            # Call evaluate once to ensure it works
            fitness = toolbox.evaluate(population[0])
            population[0].fitness.values = fitness
            if halloffame is not None:
                halloffame.update(population)
            mock_logbook = MagicMock(stream="log")
            return (population, mock_logbook)

        mock_ea.side_effect = ea_side_effect

        result = optimize_params(
            dataframes=dataframes,
            bounds=bounds,
            population_size=2,
            ngen=1
        )

        assert "sma_period" in result
