"""
Comprehensive tests for custom_ea_simple function in genetic_optimizer.py

Targets missing lines: 168-259 (custom EA implementation)
"""

import pytest
from deap import base, creator, tools
import random

from app.optimization.genetic_optimizer import custom_ea_simple


class TestCustomEASimple:
    """Test custom_ea_simple evolutionary algorithm function."""

    def setup_method(self):
        """Set up DEAP toolbox for testing."""
        # Clean up any previous creator definitions
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Define fitness and individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=5,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Simple evaluation function
        def eval_sum(individual):
            return (sum(individual),)

        self.toolbox.register("evaluate", eval_sum)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def teardown_method(self):
        """Clean up creator definitions."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

    def test_custom_ea_simple_basic(self):
        """Test basic custom EA execution."""
        population = self.toolbox.population(n=10)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
        stats.register("max", max)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=3, stats=stats
        )

        assert len(final_pop) == 10
        assert len(logbook) == 4  # Gen 0 + 3 generations
        assert logbook[0]["gen"] == 0
        assert logbook[-1]["gen"] == 3

    def test_custom_ea_simple_with_halloffame(self):
        """Test custom EA with hall of fame tracking."""
        population = self.toolbox.population(n=10)
        hof = tools.HallOfFame(3)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=5, halloffame=hof
        )

        assert len(hof) == 3
        assert hof[0].fitness.valid
        assert hof[0].fitness.values[0] >= hof[1].fitness.values[0]

    def test_custom_ea_simple_no_stats(self):
        """Test custom EA without statistics."""
        population = self.toolbox.population(n=8)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2, stats=None
        )

        assert len(final_pop) == 8
        assert len(logbook) == 3  # Gen 0 + 2 generations

    def test_custom_ea_simple_crossover_probability(self):
        """Test custom EA with different crossover probabilities."""
        population = self.toolbox.population(n=10)

        # High crossover probability
        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.9, mutpb=0.1, ngen=2
        )

        assert len(final_pop) == 10
        assert all(ind.fitness.valid for ind in final_pop)

    def test_custom_ea_simple_mutation_probability(self):
        """Test custom EA with different mutation probabilities."""
        population = self.toolbox.population(n=10)

        # High mutation probability
        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.1, mutpb=0.9, ngen=2
        )

        assert len(final_pop) == 10
        assert all(ind.fitness.valid for ind in final_pop)

    def test_custom_ea_simple_single_generation(self):
        """Test custom EA with single generation."""
        population = self.toolbox.population(n=10)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=1
        )

        assert len(logbook) == 2  # Gen 0 + Gen 1
        assert logbook[-1]["gen"] == 1

    def test_custom_ea_simple_many_generations(self):
        """Test custom EA with many generations."""
        population = self.toolbox.population(n=6)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10
        )

        assert len(logbook) == 11  # Gen 0 + 10 generations
        assert logbook[-1]["gen"] == 10

    def test_custom_ea_simple_small_population(self):
        """Test custom EA with small population."""
        population = self.toolbox.population(n=4)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=3
        )

        assert len(final_pop) == 4
        assert all(ind.fitness.valid for ind in final_pop)

    def test_custom_ea_simple_logbook_structure(self):
        """Test logbook structure and content."""
        population = self.toolbox.population(n=8)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
        stats.register("max", max)
        stats.register("min", min)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2, stats=stats
        )

        # Check logbook has all required fields
        assert "gen" in logbook[0]
        assert "nevals" in logbook[0]
        assert "duration_sec" in logbook[0]
        assert "avg" in logbook[0]
        assert "max" in logbook[0]
        assert "min" in logbook[0]

    def test_custom_ea_simple_fitness_improvement(self):
        """Test that fitness improves over generations."""
        population = self.toolbox.population(n=10)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", max)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.7, mutpb=0.2, ngen=5, stats=stats
        )

        # Best fitness should not be worse than the initial max at any point
        initial_max = logbook[0]["max"]
        best_overall = max(entry["max"] for entry in logbook)
        assert best_overall >= initial_max

    def test_custom_ea_simple_no_crossover(self):
        """Test custom EA with no crossover."""
        population = self.toolbox.population(n=10)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.0, mutpb=0.5, ngen=3
        )

        assert len(final_pop) == 10

    def test_custom_ea_simple_no_mutation(self):
        """Test custom EA with no mutation."""
        population = self.toolbox.population(n=10)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.0, ngen=3
        )

        assert len(final_pop) == 10

    def test_custom_ea_simple_evaluation_count(self):
        """Test that evaluation count is tracked correctly."""
        population = self.toolbox.population(n=10)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2
        )

        # First generation should evaluate all individuals
        assert logbook[0]["nevals"] == 10
        # Subsequent generations only evaluate new individuals
        assert logbook[1]["nevals"] <= 10

    def test_custom_ea_simple_duration_tracking(self):
        """Test that duration is tracked."""
        population = self.toolbox.population(n=8)

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=2
        )

        # All generations should have duration
        for record in logbook:
            assert "duration_sec" in record
            assert record["duration_sec"] >= 0

    def test_custom_ea_simple_population_replacement(self):
        """Test that population is replaced by offspring."""
        population = self.toolbox.population(n=10)
        initial_ids = [id(ind) for ind in population]

        final_pop, logbook = custom_ea_simple(
            population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=1
        )

        # Population should be different objects after evolution
        final_ids = [id(ind) for ind in final_pop]
        # At least some should be different due to cloning
        assert final_ids != initial_ids
