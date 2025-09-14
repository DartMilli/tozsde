from deap import tools, algorithms
import numpy as np
import datetime
import random
import time
import datetime
import json
import pandas as pd
from pandas.tseries.offsets import DateOffset

from app.utils.analizer import compute_signals, get_params, get_default_params
import app.utils.router as rtr
from app.core.genetic_optimizer import get_toolbox
from app.core.data_loader import load_data, get_supported_ticker_list

toolbox, keys = get_toolbox()


def backtest_signal_strategy(df, params, monthly_savings=100):
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

    # Ellenőrzés: van-e elég adat
    if len(df) < MIN_REQUIRED_BARS + 1:
        return {"final_value": 0, "capital_invested": 0, "return_pct": 0, "history": []}

    capital = 0
    cash = 0
    shares = 0
    portfolio_values = []
    last_month = df.index[MIN_REQUIRED_BARS].month

    for i in range(MIN_REQUIRED_BARS, len(df)):
        current_month = df.index[i].month
        if current_month != last_month:
            cash += monthly_savings
            capital += monthly_savings
            last_month = current_month

        sub_df = df.iloc[: i + 1]
        try:
            signals, _ = compute_signals(sub_df, params)
        except Exception as e:
            # Ha valami hiba történik (pl. invalid adat), hagyjuk ki
            continue

        # print(f"\n{df.index[i].date()} | szignálok: {signals}\n")

        price = df["Close"].iloc[i]

        if any("BUY" in s for s in signals) and cash > 0:
            shares += cash / price
            cash = 0
        elif any("SELL" in s for s in signals) and shares > 0:
            cash += shares * price
            shares = 0

        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    if capital == 0 or not portfolio_values:
        return {
            "final_value": 0,
            "capital_invested": capital,
            "return_pct": 0,
            "history": portfolio_values,
        }

    return {
        "final_value": portfolio_values[-1],
        "capital_invested": capital,
        "return_pct": (portfolio_values[-1] - capital) / capital * 100,
        "history": portfolio_values,
    }


def set_df_for_eval(df):
    toolbox.register(
        "evaluate",
        lambda ind: (
            backtest_signal_strategy(df, {k: v for k, v in zip(keys, ind)})[
                "final_value"
            ],
        ),
    )


def evaluate_individual(individual, dataframes, keys):
    """
    Kiértékeli az egyént (paramétercsomagot) a megadott részvények listáján.
    A fitnesz érték a részvényeken elért átlagos százalékos hozam lesz.
    """
    params = dict(zip(keys, individual))
    total_return = 0
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
            # Lefuttatjuk a visszatesztelést az adott részvényen
            result = backtest_signal_strategy(df, params)
            if result["capital_invested"] > 0:
                total_return += result["return_pct"]
                valid_count += 1
        except Exception as e:
            continue  # Hibás részvényadat: átugorjuk

    if valid_count == 0:
        return (-1000.0,)  # Extrém negatív fitness, ha semmin nem működött

    # A fitnesz az átlagos hozam
    average_return = total_return / valid_count

    return average_return, valid_count


def evaluate_individual_with_log(indivdual, dataframes, keys):
    fitness, valid_count = evaluate_individual(indivdual, dataframes, keys)
    print(
        f"(részvények: {valid_count}/{len(dataframes)} | fitnesz: {fitness:.2f}) ",
        end="",
        flush=True,
    )
    return (fitness,)


def evaluate_individual_without_log(indivdual, dataframes, keys):
    fitness, _ = evaluate_individual(indivdual, dataframes, keys)
    return (fitness,)


def optimize_params(dataframes, generations=20, population_size=30):
    """
    Genetikus algoritmus a paraméterek optimalizálására.
    """
    toolbox, keys = get_toolbox()

    # Célfüggvény regisztrálása a több DataFrame-mel
    verbose = True
    if verbose:
        toolbox.register(
            "evaluate", evaluate_individual_with_log, dataframes=dataframes, keys=keys
        )
    else:
        toolbox.register(
            "evaluate",
            evaluate_individual_without_log,
            dataframes=dataframes,
            keys=keys,
        )

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = algorithms.eaSimple(pop, toolbox,
    #                           cxpb=0.5,
    #                           mutpb=0.2,
    #                           ngen=generations,
    #                           stats=stats,
    #                           halloffame=hof,
    #                           verbose=verbose)
    pop, log = custom_ea_simple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )
    print(log.stream)

    best_params = dict(zip(keys, hof[0]))
    return best_params


def custom_ea_simple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """Saját EA ciklus DEAP mintájára sorszámozással és időméréssel"""

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "duration_sec"] + (stats.fields if stats else [])

    # Elitizmus (legjobbak nyilvántartása)
    if halloffame is not None:
        halloffame.update(population)

    # ➤ Első generáció: értékelés
    if verbose:
        print("Generáció 0 értékelése...", flush=True)
    start_time = time.time()

    fitnesses = []
    for i, ind in enumerate(population, 1):
        start = time.time()
        if verbose:
            print(
                f"  [0] Egyén {i}/{len(population)} értékelése...", end="", flush=True
            )
        fit = toolbox.evaluate(ind)
        fitnesses.append(fit)
        ind.fitness.values = fit
        elapsed = time.time() - start
        if verbose:
            print(f"kész ({datetime.timedelta(seconds=elapsed)})", flush=True)

    duration = time.time() - start_time
    record = stats.compile(population) if stats else {}
    logbook.record(
        gen=0, nevals=len(population), duration_sec=round(duration, 2), **record
    )
    if verbose:
        # print(logbook.stream[-1])
        top = tools.selBest(population, 1)[0]
        print(
            f"\n [{0}] Legjobb egyén: {top}, fitnesz: {top.fitness.values[0]:.2f}, idő: {datetime.timedelta(seconds=duration)}"
        )

    # ➤ További generációk
    for gen in range(1, ngen + 1):
        print(f"\nGeneráció {gen}/{ngen}", flush=True)
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
        print(f"  {len(invalid_ind)} új egyén értékelése...", flush=True)

        for i, ind in enumerate(invalid_ind, 1):
            start = time.time()
            if verbose:
                print(
                    f"    [{gen}] Egyén {i}/{len(invalid_ind)} értékelése...",
                    end="",
                    flush=True,
                )
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
            elapsed = time.time() - start
            if verbose:
                print(f"kész ({datetime.timedelta(seconds=elapsed)})", flush=True)

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
        if verbose:
            # print(logbook.stream[-1])
            top = tools.selBest(population, 1)[0]
            print(
                f"\n [{gen}] Legjobb egyén: {top}, fitnesz: {top.fitness.values[0]:.2f}, idő: {datetime.timedelta(seconds=duration)}"
            )

    return population, logbook


# A meglévő optimize_params-t átalakítjuk, hogy csak a GA logikát tartalmazza egy adathalmazon
def run_ga_on_slice(dataframes, generations, population_size):
    """Lefuttatja a genetikus algoritmust a megadott DataFrame szeleten."""
    toolbox, keys = get_toolbox()

    # Célfüggvény regisztrálása a több DataFrame-mel (a verbose logolást most kikapcsoljuk a tisztább kimenetért)
    toolbox.register(
        "evaluate", evaluate_individual_without_log, dataframes=dataframes, keys=keys
    )

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    # A custom_ea_simple verbose flag-jét False-ra állítjuk, hogy ne árassza el a konzolt
    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations,
        stats=None,
        halloffame=hof,
        verbose=False,
    )

    best_params = dict(zip(keys, hof[0]))
    return best_params


def run_optimization_walk_forward(config, save_to_file=True):
    """
    Walk-Forward Optimalizációt végrehajtó vezérlő függvény.

    Args:
        config (dict): A központi konfigurációs szótár a main.py-ból.
    """
    print("\n🚶‍♂️ Walk-Forward Optimalizáció Indítása...")

    # Walk-Forward paraméterek
    in_sample_months = 12  # Tanulási periódus hossza (hónap)
    out_of_sample_months = 3  # Tesztelési (validációs) periódus hossza
    step_months = 3  # Ennyit lépünk előre minden ciklusban

    tickers = config.get("TICKERS", [])
    start_date = pd.to_datetime(config.get("START_DATE"))
    end_date = pd.to_datetime(config.get("END_DATE"))

    # Teljes adathalmaz betöltése egyszer
    print("Teljes adathalmaz betöltése...")
    full_dataframes = {}
    for ticker in tickers:
        df = load_data(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty:
            full_dataframes[ticker] = df

    if not full_dataframes:
        print("Hiba: Nincs adat az optimalizációhoz.")
        return

    # Ciklusok (folds) inicializálása
    fold_start_date = start_date
    final_params = None
    fold_number = 1

    while (
        fold_start_date + DateOffset(months=in_sample_months + out_of_sample_months)
        <= end_date
    ):
        # ---- Periódusok kijelölése ----
        in_sample_start = fold_start_date
        in_sample_end = fold_start_date + DateOffset(months=in_sample_months)
        out_of_sample_end = in_sample_end + DateOffset(months=out_of_sample_months)

        print(f"\n--- FOLD {fold_number} ---")
        print(
            f"  In-Sample (Tanuló):  {in_sample_start.date()} -> {in_sample_end.date()}"
        )
        print(
            f"  Out-of-Sample (Teszt): {in_sample_end.date()} -> {out_of_sample_end.date()}"
        )

        # ---- Adatok szeletelése az aktuális 'fold'-hoz ----
        in_sample_dfs = {
            t: df.loc[in_sample_start:in_sample_end]
            for t, df in full_dataframes.items()
        }
        out_of_sample_dfs = {
            t: df.loc[in_sample_end:out_of_sample_end]
            for t, df in full_dataframes.items()
        }

        # Üres DataFrame-ek kiszűrése
        in_sample_dfs = {t: df for t, df in in_sample_dfs.items() if not df.empty}
        out_of_sample_dfs = {
            t: df for t, df in out_of_sample_dfs.items() if not df.empty
        }

        if not in_sample_dfs:
            print("  Hiba: Nincs elég adat az In-Sample periódushoz.")
            fold_start_date += DateOffset(months=step_months)
            fold_number += 1
            continue

        # ---- Genetikus Optimalizáció az In-Sample adatokon ----
        print("  Optimalizáció az In-Sample adatokon...")
        best_in_sample_params = run_ga_on_slice(
            in_sample_dfs,
            generations=config.get("OPTIMIZER_GENERATIONS", 20),
            population_size=config.get("OPTIMIZER_POPULATION", 30),
        )

        # ---- Validáció az Out-of-Sample adatokon ----
        print("  Validáció az Out-of-Sample adatokon...")
        if out_of_sample_dfs:
            # Itt most csak kiértékeljük a teljesítményt, de komplexebb logikával
            # el is menthetnénk és a végén a legjobb átlagot adó paramétert választanánk.
            # Most az egyszerűség kedvéért mindig a legutolsó fold paramétereit fogadjuk el.
            fitness, _ = evaluate_individual(
                list(best_in_sample_params.values()),
                out_of_sample_dfs,
                list(best_in_sample_params.keys()),
            )
            print(f"  Out-of-Sample Fitness (átlag hozam %): {fitness:.2f}")

        final_params = best_in_sample_params

        # Ablak csúsztatása
        fold_start_date += DateOffset(months=step_months)
        fold_number += 1

    # ---- Végső Paraméterek Mentése ----
    if final_params:
        default_params = get_default_params()
        for p in final_params:
            print(f"{p:25s}: original:{default_params[p]:2d} best:{final_params[p]:2d}")
        if save_to_file:
            with open(rtr.PARAMS_FILE_PATH, "w") as f:
                json.dump(final_params, f, indent=4)
        print(
            f"\nWalk-Forward optimalizáció befejeződött. Végső paraméterek mentve ide: {rtr.PARAMS_FILE_PATH}"
        )
    else:
        print(
            "\nNem sikerült a Walk-Forward optimalizáció, nem találtunk paramétereket."
        )


def run_optimization_simple(config, save_to_file=True):
    tickers = config["TICKERS"]
    start = config["START_DATE"]
    end = config["END_DATE"]
    gen = config["OPTIMIZER_GENERATIONS"]
    pop = config["OPTIMIZER_POPULATION"]

    dataframes = {}
    for ticker in tickers:
        print(f"Adatok betöltése: {ticker}...")
        dataframes[ticker] = load_data(ticker, start=start, end=end)

    print("\nParaméterek optimalizálása a teljes portfólióra...")
    best_params = optimize_params(dataframes, generations=gen, population_size=pop)

    print("\nOptimális paraméterek a teljes portfólióra:")
    default_params = get_default_params()
    for p in best_params:
        print(f"{p:25s}: original:{default_params[p]:2d} best:{best_params[p]:2d}")

    if save_to_file:
        print(f"\nOptimális paraméterek mentése a(z) {rtr.PARAMS_FILE_PATH} fájlba...")
        with open(rtr.PARAMS_FILE_PATH, "w") as f:
            json.dump(best_params, f, indent=4)


def run_optimization(config, save_to_file=True, walk_forward=True):
    if walk_forward:
        run_optimization_walk_forward(config, save_to_file)
    else:
        run_optimization_simple(config, save_to_file)


if __name__ == "__main__":
    config = {
        "START_DATE": "2020-01-01",
        "END_DATE": datetime.today().strftime("%Y-%m-%d"),
        "TICKERS": [
            t
            for t in get_supported_ticker_list()
            if t not in ["OTP.BD", "MOL.BD", "RICHTER.BD"]
        ],
        "OPTIMIZER_GENERATIONS": 2,
        "OPTIMIZER_POPULATION": 30,
    }

    run_optimization(config, save_to_file=False)
