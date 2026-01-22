"""
GENETIC OPTIMIZER – RESPONSIBILITY CONTRACT

ROLE:
    - Search optimal strategy PARAMETERS using GA
    - Evaluate individuals ONLY via Backtester

MUST NOT:
    - Know anything about time or walk-forward
    - Slice data
    - Make trading decisions
    - Use ML models

INPUT:
    - dataframes: Dict[ticker -> DataFrame]
    - bounds: Dict[param -> (min, max)]

OUTPUT:
    - Dict[param -> value]

THIS MODULE IS:
    - Stateless
    - Deterministic (given seed)
"""

from deap import tools, base, creator
import numpy as np
import datetime
import random
import time
import datetime
from typing import Dict, List
import json
from typing import Optional

from app.backtesting.backtester import Backtester
from app.optimization.fitness import fitness_single
from app.analysis.analyzer import param_bounds
from app.infrastructure.logger import setup_logger, is_logger_debug

logger = setup_logger(__name__)

_FITNESS_CACHE = {}


def _params_to_key(params: dict) -> str:
    """
    Deterministic hashable key from params dict.
    """
    return json.dumps(params, sort_keys=True)


def individual_to_params(
    individual: List[int], param_keys: List[str]
) -> Dict[str, int]:
    """
    GA individual -> param dict
    A sorrend KRITIKUS, kívülről jön (analyze.py)
    """
    return dict(zip(param_keys, individual))


def evaluate_individual(individual, dataframes, keys):
    """
    Kiértékeli az egyént (paramétercsomagot) a megadott részvények listáján.
    """
    params = individual_to_params(individual, keys)
    total_fitness = 0
    valid_count = 0

    min_required_bars = max(
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

    # Végigmegyünk az összes részvényen
    for ticker, df in dataframes.items():
        if len(df) < min_required_bars + 1:
            continue  # túl kevés adat
        try:
            cache_key = (ticker, _params_to_key(params))

            if cache_key in _FITNESS_CACHE:
                fitness = _FITNESS_CACHE[cache_key]
            else:
                # Lefuttatjuk a visszatesztelést az adott részvényen
                bt = Backtester(df, ticker)
                report = bt.run(params)  # BacktestRiport dataclass
                base_fitness = fitness_single(report.metrics)

                parsimony = parsimony_penalty(
                    params=params, weight=0.05, param_limits=param_bounds
                )

                fitness = base_fitness * parsimony
                _FITNESS_CACHE[cache_key] = fitness

            total_fitness += fitness
            valid_count += 1
        except Exception as e:
            continue  # Hibás részvényadat: átugorjuk

    if valid_count == 0:
        return (-1000.0,)  # Extrém negatív fitness, ha semmin nem működött

    return total_fitness / valid_count, valid_count


def evaluate_individual_with_log(indivdual, dataframes, keys):
    fitness, valid_count = evaluate_individual(indivdual, dataframes, keys)
    logger.debug(
        f"(részvények: {valid_count}/{len(dataframes)} | fitnesz: {fitness:.2f}) "
    )
    return (fitness,)


def evaluate_individual_without_log(indivdual, dataframes, keys):
    fitness, _ = evaluate_individual(indivdual, dataframes, keys)
    return (fitness,)


def parsimony_penalty(
    params,  # type: dict
    weight=0.05,  # type: float
    param_limits=None,  # type: Optional[dict]
):  # type: (...) -> float
    """
    Bünteti a túl komplex paraméterhalmazokat.

    Returns:
        multiplier (<= 1.0)
    """
    if not params:
        return 1.0

    active_params = {k: v for k, v in params.items() if v is not None and v != 0}

    param_count = len(active_params)

    # alap büntetés: minél több paraméter, annál rosszabb
    penalty = 1.0 - (param_count * weight)

    # extra büntetés szélsőséges értékekre
    if param_limits:
        for k, v in active_params.items():
            if k in param_limits:
                min_v, max_v = param_limits[k]
                mid = (min_v + max_v) / 2
                distance = abs(v - mid) / (max_v - min_v)
                penalty -= distance * weight

    return max(0.3, penalty)  # ne ölje meg teljesen


def custom_ea_simple(
    population, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, stats=None, halloffame=None
):
    """Saját EA ciklus DEAP mintájára sorszámozással és időméréssel"""

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "duration_sec"] + (stats.fields if stats else [])

    # Elitizmus (legjobbak nyilvántartása)
    if halloffame is not None:
        halloffame.update(population)

    # ➤ Első generáció: értékelés
    logger.info("Generáció 0 értékelése...")
    start_time = time.time()

    fitnesses = []
    for i, ind in enumerate(population, 1):
        start = time.time()
        logger.debug(f"  [0] Egyén {i}/{len(population)} értékelése...")
        fit = toolbox.evaluate(ind)
        fitnesses.append(fit)
        ind.fitness.values = fit
        elapsed = time.time() - start
        logger.debug(
            f"  [0] Egyén {i}/{len(population)} kész {datetime.timedelta(seconds=elapsed)} alatt"
        )

    duration = time.time() - start_time
    record = stats.compile(population) if stats else {}
    logbook.record(
        gen=0, nevals=len(population), duration_sec=round(duration, 2), **record
    )

    # logger.debug(logbook.stream[-1])
    top = tools.selBest(population, 1)[0]
    logger.info(
        f"\n [{0}] Legjobb egyén: {top}, fitnesz: {top.fitness.values[0]:.2f}, idő: {datetime.timedelta(seconds=duration)}"
    )

    # ➤ További generációk
    for gen in range(1, ngen + 1):
        logger.info(f"\nGeneráció {gen}/{ngen}", flush=True)
        start_time = time.time()

        # ➤ Szelekció és másolatok
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # ➤ Keresztezés és mutáció
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # ➤ Értékelés csak az új egyedekre
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.debug(f"  {len(invalid_ind)} új egyén értékelése...")

        for i, ind in enumerate(invalid_ind, 1):
            start = time.time()
            logger.debug(
                f"    [{gen}] Egyén {i}/{len(invalid_ind)} értékelése...",
            )
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
            elapsed = time.time() - start
            logger.debug(
                f"    [{gen}] Egyén {i}/{len(population)} kész {datetime.timedelta(seconds=elapsed)} alatt"
            )

        # ➤ Hall of Fame frissítés
        if halloffame is not None:
            halloffame.update(offspring)

        # ➤ Populáció frissítése
        population[:] = offspring

        # ➤ Naplózás
        duration = time.time() - start_time
        record = stats.compile(population) if stats else {}
        logbook.record(
            gen=gen, nevals=len(invalid_ind), duration_sec=round(duration, 2), **record
        )
        # logger.debug(logbook.stream[-1])
        top = tools.selBest(population, 1)[0]
        logger.info(
            f"\n [{gen}] Legjobb egyén: {top}, fitnesz: {top.fitness.values[0]:.2f}, idő: {datetime.timedelta(seconds=duration)}"
        )

    return population, logbook


def optimize_params(
    dataframes: Dict[str, object],
    bounds: Dict[str, tuple],
    population_size: int = 50,
    ngen: int = 30,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
) -> Dict[str, int]:
    """
    bounds:
        {
            "sma_period": (5, 200),
            "ema_period": (5, 200),
            ...
        }
    """

    tickers = list(dataframes.keys())
    param_keys = list(bounds.keys())

    # --------------------------------------------------------
    # DEAP creator (csak egyszer!)
    # --------------------------------------------------------
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # --------------------------------------------------------
    # PARAM TÉR – KÍVÜLRŐL JÖN (analyze.py)
    # --------------------------------------------------------
    for key, (low, high) in bounds.items():
        toolbox.register(f"attr_{key}", random.randint, low, high)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [toolbox.__getattribute__(f"attr_{k}") for k in param_keys],
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --------------------------------------------------------
    # FITNESS REGISZTRÁCIÓ
    # --------------------------------------------------------
    if is_logger_debug(logger):
        toolbox.register(
            "evaluate",
            evaluate_individual_with_log,
            dataframes=dataframes,
            keys=param_keys,
        )
    else:
        toolbox.register(
            "evaluate",
            evaluate_individual_without_log,
            dataframes=dataframes,
            keys=param_keys,
        )

    toolbox.register("mate", tools.cxTwoPoint)

    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=[bounds[k][0] for k in param_keys],
        up=[bounds[k][1] for k in param_keys],
        indpb=0.2,
    )

    toolbox.register("select", tools.selTournament, tournsize=3)

    # --------------------------------------------------------
    # POPULÁCIÓ + STAT
    # --------------------------------------------------------
    population = toolbox.population(n=population_size)

    halloffame = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logger.debug(
        f"[GA] start | pop={population_size}, gen={ngen}, params={len(param_keys)}"
    )

    # --------------------------------------------------------
    # 🔥 A TE GA MOTOROD
    # --------------------------------------------------------
    _, log = custom_ea_simple(
        population=population,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=halloffame,
    )

    logger.info(log.stream)

    best_individual = halloffame[0]
    best_params = individual_to_params(best_individual, param_keys)

    logger.info(f"GA done | fitness={best_individual.fitness.values[0]:.6f}")
    logger.info(f"GA best_params={best_params}")

    return best_params


assert "walk_forward" not in globals()
assert "model_trainer" not in globals()
