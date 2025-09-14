import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import pandas as pd
import shutil
import os

from app.utils.data_cleaner import prepare_df
from app.core.data_loader import load_data
from app.core.data_loader import load_data, get_supported_ticker_list
from app.utils.analizer import get_params
from app.utils.plotter import (
    plot_bar_chart,
    plot_gradient_scatter,
    plot_strategy_colored_scatter,
)
import app.utils.router as rtr


class TradingEnv_old(gym.Env):
    """
    Egyedi reinforcement learning környezet a tőzsdei döntésekhez.
    """

    def __init__(self, df, initial_cash=10000, reward_strategy="portfolio_value"):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reward_strategy = reward_strategy
        self.current_step = 0

        # Akciótér: 0 = tartás, 1 = vétel, 2 = eladás
        self.action_space = spaces.Discrete(3)

        # Állapottér: ár + indikátorok (14) + portfólió állapota (3) = 17
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # A reset() hívása itt már nem szükséges, mert a hívó oldalon megteszik
        # de ha közvetlenül példányosítjuk, akkor kellhet.
        # A gymnasium standard szerint a reset()-et a felhasználónak kell hívnia az első használat előtt.

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [
                row["Close"],
                row.get("RSI", 0),
                row.get("MACD", 0),
                row.get("MACD_SIGNAL", 0),
                row.get("BB_upper", row["Close"]),
                row.get("BB_lower", row["Close"]),
                row.get("SMA", row["Close"]),
                row.get("EMA", row["Close"]),
                row.get("ATR", 0),
                row.get("ADX", 0),
                row.get("PLUS_DI", 0),
                row.get("MINUS_DI", 0),
                row.get("STOCH_K", 0),
                row.get("STOCH_D", 0),
                self.cash,
                self.shares_held,
                self.portfolio_value,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "shares_held": self.shares_held,
            "cash": self.cash,
        }

    def _calculate_reward(self, prev_portfolio_value):
        if self.reward_strategy == "portfolio_value":
            return self.portfolio_value - prev_portfolio_value
        elif self.reward_strategy == "log_return":
            return np.log(self.portfolio_value / (prev_portfolio_value + 1e-8))
        elif self.reward_strategy == "sharpe_like":
            risk_penalty = self.df.loc[self.current_step, "ATR"]
            return (self.portfolio_value - prev_portfolio_value) / (risk_penalty + 1e-8)
        elif self.reward_strategy == "drawdown_penalty":
            drawdown = max(0, self.prev_max_value - self.portfolio_value)
            return (self.portfolio_value - prev_portfolio_value) - 0.1 * drawdown
        elif self.reward_strategy == "profit_with_hold_penalty":
            hold_penalty = 0.01 if self.last_action == 0 else 0.0
            return (self.portfolio_value - prev_portfolio_value) - hold_penalty
        elif self.reward_strategy == "action_based_profit":
            reward = self.portfolio_value - prev_portfolio_value
            if self.last_action == 1 and self.shares_held > 0:
                return reward * 1.2
            elif self.last_action == 2 and self.shares_held == 0:
                return reward * 0.5
            else:
                return reward
        elif self.reward_strategy == "baseline_compare":
            buy_and_hold_value = self.initial_cash * (
                self.df.loc[self.current_step, "Close"] / self.df.loc[0, "Close"]
            )
            return (self.portfolio_value - buy_and_hold_value) - prev_portfolio_value
        else:
            return 0.0  # biztonsági fallback

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        current_price = self.df.loc[self.current_step, "Close"]

        self.last_action = action
        self.prev_max_value = max(self.prev_max_value, self.portfolio_value)

        # Akció végrehajtása
        if action == 1:  # Vétel
            if self.cash > current_price:  # Van elég pénz egy részvényre
                self.shares_held += 1
                self.cash -= current_price
        elif action == 2:  # Eladás
            if self.shares_held > 0:  # Van mit eladni
                self.shares_held -= 1
                self.cash += current_price
        # action == 0 (tartás) esetén nem csinálunk semmit

        self.current_step += 1

        # Portfólió értékének frissítése az új (következő lépés) árfolyamon
        next_price = (
            self.df.loc[self.current_step, "Close"]
            if self.current_step < len(self.df)
            else current_price
        )
        self.portfolio_value = self.cash + self.shares_held * next_price

        # Jutalom számítása
        reward = self._calculate_reward(prev_portfolio_value)

        # Epizód vége ellenőrzés
        terminated = self.current_step >= len(self.df) - 1

        obs = self._get_obs()
        info = self._get_info()

        # Gymnasium API-nak megfelelő visszatérés
        # A `truncated` itt False, mert az epizód végét csak a `terminated` jelzi.
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        # A kezdeti portfólió érték a készpénz
        self.portfolio_value = self.initial_cash
        self.prev_max_value = self.initial_cash
        self.last_action = 0

        obs = self._get_obs()
        info = self._get_info()

        # Gymnasium API-nak megfelelő visszatérés
        return obs, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Shares: {self.shares_held}, Cash: {self.cash:.2f}"
        )

    def get_reward_strategies():
        """
        Elérhető reward stratégiák.

        Return:
            List[str]: Az elérhető stratégia nevek.
        """
        return [
            "portfolio_value",  # portfólió értékének abszolút változása
            "log_return",  # logaritmikus hozam (skálázott)
            "sharpe_like",  # nyereség / kockázat (pl. ATR alapján)
            "drawdown_penalty",  # drawdown büntetéssel csökkentett profit
            "profit_with_hold_penalty",  # inaktivitás büntetése
            "action_based_profit",  # akció-specifikus nyereségsúlyozás
            "baseline_compare",  # buy-and-hold stratégiához viszonyított teljesítmény
        ]


class TradingEnv(gym.Env):
    """
    Fejlett reinforcement learning környezet kockázatkezeléssel.
    - Dinamikus pozícióméretezés a portfólió egy százalékának kockáztatásával.
    - ATR alapú stop-loss a veszteségek korlátozására.
    """

    def __init__(
        self,
        df,
        initial_cash=10000,
        reward_strategy="portfolio_value",
        risk_pct=0.02,  # A portfólió 2%-át kockáztatjuk egy trade-en
        stop_loss_atr_multiplier=2.0,
    ):  # Stop-loss = 2 * ATR

        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reward_strategy = reward_strategy
        self.risk_pct = risk_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier

        self.current_step = 0
        self.stop_loss_price = None  # Aktuális stop-loss ár

        # Akciótér: 0 = tartás, 1 = vétel, 2 = eladás
        self.action_space = spaces.Discrete(3)

        # Állapottér: ár + indikátorok (14) + portfólió állapota (3) = 17
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        # A reset()-et a felhasználó hívja meg az első használat előtt.

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [
                row["Close"],
                row.get("RSI", 0),
                row.get("MACD", 0),
                row.get("MACD_SIGNAL", 0),
                row.get("BB_upper", row["Close"]),
                row.get("BB_lower", row["Close"]),
                row.get("SMA", row["Close"]),
                row.get("EMA", row["Close"]),
                row.get("ATR", 0),
                row.get("ADX", 0),
                row.get("PLUS_DI", 0),
                row.get("MINUS_DI", 0),
                row.get("STOCH_K", 0),
                row.get("STOCH_D", 0),
                self.cash,
                self.shares_held,
                self.portfolio_value,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "shares_held": self.shares_held,
            "cash": self.cash,
            "stop_loss": self.stop_loss_price,
        }

    def _calculate_reward(self, prev_portfolio_value):
        if self.reward_strategy == "portfolio_value":
            return self.portfolio_value - prev_portfolio_value
        elif self.reward_strategy == "log_return":
            return np.log(self.portfolio_value / (prev_portfolio_value + 1e-8))
        elif self.reward_strategy == "sharpe_like":
            risk_penalty = self.df.loc[self.current_step, "ATR"]
            return (self.portfolio_value - prev_portfolio_value) / (risk_penalty + 1e-8)
        elif self.reward_strategy == "drawdown_penalty":
            drawdown = max(0, self.prev_max_value - self.portfolio_value)
            return (self.portfolio_value - prev_portfolio_value) - 0.1 * drawdown
        elif self.reward_strategy == "profit_with_hold_penalty":
            hold_penalty = 0.01 if self.last_action == 0 else 0.0
            return (self.portfolio_value - prev_portfolio_value) - hold_penalty
        elif self.reward_strategy == "action_based_profit":
            reward = self.portfolio_value - prev_portfolio_value
            if self.last_action == 1 and self.shares_held > 0:
                return reward * 1.2
            elif self.last_action == 2 and self.shares_held == 0:
                return reward * 0.5
            else:
                return reward
        elif self.reward_strategy == "baseline_compare":
            buy_and_hold_value = self.initial_cash * (
                self.df.loc[self.current_step, "Close"] / self.df.loc[0, "Close"]
            )
            return (self.portfolio_value - buy_and_hold_value) - prev_portfolio_value
        else:
            return 0.0  # biztonsági fallback

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.initial_cash
        self.prev_max_value = self.initial_cash
        self.last_action = 0
        self.stop_loss_price = None

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        current_price = self.df.loc[self.current_step, "Close"]
        current_atr = self.df.loc[self.current_step, "ATR"]

        # 1. STOP-LOSS ELLENŐRZÉS (ha van nyitott pozíció)
        if self.shares_held > 0 and current_price < self.stop_loss_price:
            # A stop-loss aktiválódott, kényszerített eladás
            self.cash += self.shares_held * self.stop_loss_price  # Eladás a stop áron
            self.shares_held = 0
            self.stop_loss_price = None
            action = 2  # Eladás történt

        # 2. AKCIÓ VÉGREHAJTÁSA
        self.last_action = action
        if action == 1 and self.shares_held == 0:  # Vétel (csak ha nincs már pozíciónk)
            # ---- Pozícióméretezés ----
            stop_price = current_price - (self.stop_loss_atr_multiplier * current_atr)
            risk_per_share = current_price - stop_price

            if risk_per_share > 0:
                # Mennyit kockáztatunk ezen a trade-en?
                risk_amount = self.portfolio_value * self.risk_pct
                # Hány részvényt vehetünk ekkora kockázattal?
                num_shares_to_buy = risk_amount / risk_per_share

                cost = num_shares_to_buy * current_price
                if self.cash >= cost:
                    self.shares_held = num_shares_to_buy
                    self.cash -= cost
                    self.stop_loss_price = stop_price  # Új stop-loss beállítása

        elif action == 2 and self.shares_held > 0:  # Eladás (ha van mit)
            self.cash += self.shares_held * current_price
            self.shares_held = 0
            self.stop_loss_price = None  # Pozíció lezárva, nincs stop-loss

        # 3. KÖVETKEZŐ LÉPÉS ÉS KIÉRTÉKELÉS
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        next_price = (
            self.df.loc[self.current_step, "Close"] if not terminated else current_price
        )
        self.portfolio_value = self.cash + self.shares_held * next_price

        reward = self._calculate_reward(prev_portfolio_value)
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Shares: {self.shares_held}, Cash: {self.cash:.2f}"
        )

    def get_reward_strategies():
        """
        Elérhető reward stratégiák.

        Return:
            List[str]: Az elérhető stratégia nevek.
        """
        return [
            "portfolio_value",  # portfólió értékének abszolút változása
            "log_return",  # logaritmikus hozam (skálázott)
            "sharpe_like",  # nyereség / kockázat (pl. ATR alapján)
            "drawdown_penalty",  # drawdown büntetéssel csökkentett profit
            "profit_with_hold_penalty",  # inaktivitás büntetése
            "action_based_profit",  # akció-specifikus nyereségsúlyozás
            "baseline_compare",  # buy-and-hold stratégiához viszonyított teljesítmény
        ]


# tanulás követése:
# bash: tensorboard --logdir ./tensorboard
# -> http://localhost:6006
def train_rl_agent(
    ticker,
    model_type="DQN",  # vagy "PPO"
    start="2020-01-01",
    end="2025-06-30",
    params=None,
    timesteps=100_000,
    model_path=None,
    reward_strategy="price_diff",  # Ezt a javaslatom alapján javíthatod "portfolio_value"-ra
):
    df = load_data(ticker, start=start, end=end)
    df = prepare_df(df, params)
    env = DummyVecEnv([lambda: TradingEnv(df, reward_strategy=reward_strategy)])

    policy_kwargs = dict(net_arch=[64, 32])
    learning_rate = 0.0005

    # Közös paraméterek egy szótárban
    common_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 0,
        "learning_rate": learning_rate,
        "policy_kwargs": policy_kwargs,
    }

    # Modell-specifikus paraméterek hozzáadása
    if model_type.upper() == "PPO":
        model_cls = PPO
        # A PPO-nak nincsenek szüksége a DQN paramétereire.
        # Itt adhatnál meg PPO-specifikus paramétereket, ha szeretnél, pl:
        # common_params['n_steps'] = 2048
        common_params.update(
            {
                "tensorboard_log": f"{rtr.TENSORBOARD_DIR}/{ticker}_PPO_{reward_strategy}",
            }
        )
    else:  # Feltételezzük, hogy a másik opció a DQN
        model_cls = DQN
        # DQN-specifikus paraméterek hozzáadása a közösökhöz
        common_params.update(
            {
                "train_freq": 32,
                "gradient_steps": 32,
                "buffer_size": 100_000,  # Fontos paraméter a DQN-hez!
                "learning_starts": 1000,  # Hány lépés után kezdjen tanulni
                "tensorboard_log": f"{rtr.TENSORBOARD_DIR}/{ticker}_DQN_{reward_strategy}",
            }
        )

    # A modell példányosítása a megfelelő paraméterekkel
    model = model_cls(**common_params)

    # model.learn(total_timesteps=timesteps)
    tqdm_steps = 1000
    tqdm_iters = timesteps // tqdm_steps
    tqdm_desc = f"Progress {ticker} [{model_type} | {reward_strategy}]"
    for _ in tqdm(range(tqdm_iters), desc=tqdm_desc):
        model.learn(total_timesteps=tqdm_steps, reset_num_timesteps=False)

    if model_path is None:
        model_path = (
            f"{rtr.MODEL_DIR}/{model_type.lower()}_model_{ticker}_{reward_strategy}.zip"
        )

    model.save(model_path)
    print(f"{model_type.upper()} modell mentve: {model_path}")


def backtest_rl_model(
    ticker: str,
    model_path: str,
    params=None,
    days: int = 183,  # fél év
    verbose: bool = True,
    reward_strategy="portfolio_value",
):
    # Adatok betöltése
    df = load_data(ticker)
    df = df.tail(days)
    df = prepare_df(df, params=params)

    env = DummyVecEnv([lambda: TradingEnv(df, reward_strategy=reward_strategy)])

    # Modell típusának meghatározása
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
        model_type = "PPO"
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
        model_type = "DQN"
    else:
        raise ValueError(
            "Ismeretlen modell típus. A fájlnév tartalmazza: 'ppo' vagy 'dqn'"
        )

    obs = env.reset()  # csak 1 érték tér vissza DummyVecEnv esetén
    done = False
    rewards = []
    portfolio_values = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        rewards.append(float(reward[0]))
        portfolio_values.append(infos[0].get("portfolio_value", 0))

    total_reward = sum(rewards)
    final_value = portfolio_values[-1] if portfolio_values else 0

    # Sharpe-ráta számítása (egyszerűsítve, RF = 0, napi hozamokon)
    returns = np.diff(portfolio_values)  # napi hozamok
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Kompozit score: egyszerű példában 70% final_value + 30% sharpe
    composite_score = 0.7 * final_value + 0.3 * sharpe_ratio

    if verbose:
        print(
            f"{ticker}: Reward: {total_reward:.2f}, Portfolio: {final_value:.2f}, Sharpe: {sharpe_ratio:.2f}, Score: {composite_score:.2f}"
        )

    return {
        "ticker": ticker,
        "reward_strategy": reward_strategy,
        "model_type": model_type,
        "total_reward": total_reward,
        "final_portfolio_value": final_value,
        "sharpe_ratio": sharpe_ratio,
        "composite_score": composite_score,
        "steps": len(rewards),
    }


def run_training(config, force_retrain=False):
    tickers = config["TICKERS"]
    timesteps = config["RL_TIMESTEPS"]
    start = config["START_DATE"]
    end = config["END_DATE"]
    model_types = ["PPO", "DQN"]
    reward_strategys = TradingEnv.get_reward_strategies()

    print("\n----TANULÁS------")
    for rs in reward_strategys:
        for mt in model_types:
            for ticker in tickers:
                print(f"Tanítás: {ticker} | modell: {mt} | reward: {rs}")
                model_path = f"{rtr.MODEL_DIR}/{mt}_model_{ticker}_{rs}.zip"

                if os.path.exists(model_path) and not force_retrain:
                    print(f"Kihagyva (már létezik): {model_path}")
                    continue  # Ugrás a következő ciklus iterációra

                train_rl_agent(
                    ticker,
                    start=start,
                    end=end,
                    model_type=mt,
                    params=get_params(),
                    model_path=model_path,
                    reward_strategy=rs,
                    timesteps=timesteps,
                )


def run_backtest(config, save_to_file=True, create_plots=True, promote_top_n=3):
    tickers = config["TICKERS"]
    model_types = ["PPO", "DQN"]
    reward_strategys = TradingEnv.get_reward_strategies()
    results = []
    print("\n----TESZTELÉS------")
    for rs in reward_strategys:
        for mt in model_types:
            for ticker in tickers:
                print(f"Teszt: {ticker} | modell: {mt} | reward: {rs}")
                model_path = f"{rtr.MODEL_DIR}/{mt}_model_{ticker}_{rs}.zip"
                res = backtest_rl_model(
                    ticker,
                    model_path=model_path,
                    params=get_params(),
                    reward_strategy=rs,
                )
                results.append(res)

    # Eredmények DataFrame-be
    df_results = pd.DataFrame(results)

    # Ticker + model kombinációk szerinti legjobb reward stratégia
    best_by_group = (
        df_results.sort_values("composite_score", ascending=False)
        .groupby(["ticker", "model_type"])
        .first()
        .reset_index()
    )

    print("\nLegjobb stratégiák tickerenként:")
    print(
        best_by_group[
            ["ticker", "model_type", "reward_strategy", "final_portfolio_value"]
        ]
    )

    if save_to_file:
        df_results.to_json(rtr.MODEL_TEST_RESULT_FILE_PATH, index=False)

    if create_plots:
        for mt in model_types:
            for ticker in tickers:
                subset = df_results[
                    (df_results["model_type"] == mt) & (df_results["ticker"] == ticker)
                ]

                if subset.empty:
                    continue

                # 1. Reward stratégia bar chart
                plot_bar_chart(subset, ticker, mt)

                # 2. Scatter plot – gradient színezéssel (kompozit score alapján)
                plot_gradient_scatter(subset, ticker, mt)

                # 3. Scatter plot – reward stratégia szerinti színezéssel
                plot_strategy_colored_scatter(subset, ticker, mt)

    if promote_top_n > 0:
        promote_best_models(df_results, promote_top_n)


def promote_best_models(results_df, top_n=3):
    """
    Kiválasztja és átmásolja a legjobb top_n modellt minden tickerhez.
    """
    print(f"\n---- LEGJOBB {top_n} MODELL KIHELYEZÉSE ----")

    tickers = results_df["ticker"].unique()

    for ticker in tickers:
        print(f"  Ticker: {ticker}")
        # A tickerhez tartozó legjobb N modell kiválasztása
        best_models_for_ticker = results_df[results_df["ticker"] == ticker].nlargest(
            top_n, "composite_score"
        )

        if best_models_for_ticker.empty:
            print(f"    - Nincs menthető modell.")
            continue

        # A legjobb modellek másolása rangsorolt névvel
        for i, (_, row) in enumerate(best_models_for_ticker.iterrows()):
            rank = i + 1
            reward_strategy = row["reward_strategy"]
            model_type = row["model_type"].upper()

            source_model_name = f"{model_type}_model_{ticker}_{reward_strategy}.zip"
            source_path = f"{rtr.MODEL_DIR}/{source_model_name}"

            # Célfájl rangsorolt névvel
            target_model_name = f"top{rank}_{model_type}_{ticker}.zip"
            target_path = f"{rtr.MODEL_DIR}/{target_model_name}"

            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                print(f"    - Top {rank}: {source_model_name} -> {target_model_name}")
            else:
                print(f"    - HIBA: A forrásmodell nem található: {source_path}")


if __name__ == "__main__":
    _ = os.system("cls")

    config = {
        "START_DATE": "2020-01-01",
        "END_DATE": "2025-06-30",
        "TICKERS": [
            t
            for t in get_supported_ticker_list()
            if t not in ["OTP.BD", "MOL.BD", "RICHTER.BD"]
        ],
        "RL_TIMESTEPS": 100_000,
    }

    run_training(config)

    run_backtest(config, save_to_file=True, create_plots=False)

    # Plotok generálása
