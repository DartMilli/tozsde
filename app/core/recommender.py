import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import traceback
import numpy as np
from collections import Counter
import glob

from app.core.data_loader import load_data, get_supported_ticker_list
from app.utils.recommendation_logger import log_recommendation
from app.core.model_trainer import TradingEnv
from app.utils.data_cleaner import prepare_df
from app.email.mailer import send_email
import app.utils.router as rtr


def generate_daily_recommendation_ensemble(ticker: str, top_n: int = 3) -> str:
    """
    Generál napi ajánlást a top_n modell szavazata alapján.
    Dinamikusan kezeli a PPO és DQN modelleket is.
    """
    today = datetime.date.today()
    start = today - datetime.timedelta(days=180)
    df_full = load_data(ticker, start=start.strftime("%Y-%m-%d"))

    if df_full.empty:
        return f"{ticker}: HIBA (nincs adat)"

    df = prepare_df(df_full.copy())

    votes = []
    confidences = []

    for i in range(1, top_n + 1):
        # VÁLTOZTATÁS: Fájl keresése 'glob'-bal, a típus joker karakter
        search_pattern = f"{rtr.MODEL_DIR}/top{i}_*_{ticker}.zip"
        model_files = glob.glob(search_pattern)

        if not model_files:
            continue  # Nincs ilyen rangú modell, ugorjunk

        model_path = model_files[0]
        model_class = None
        model_type = ""

        # VÁLTOZTATÁS: Modell típusának felismerése a fájlnévből
        if "PPO" in model_path.upper():
            model_class = PPO
            model_type = "PPO"
        elif "DQN" in model_path.upper():
            model_class = DQN
            model_type = "DQN"
        else:
            print(f"Ismeretlen modell típus: {model_path}")
            continue

        try:
            env = DummyVecEnv([lambda: TradingEnv(df)])
            # VÁLTOZTATÁS: A megfelelő osztályt használjuk a betöltéshez
            model = model_class.load(model_path)

            obs = env.envs[0].reset()[0]
            for step in range(len(df) - 1):
                obs, _, _, _, _ = env.envs[0].step(0)

            action, _ = model.predict(obs, deterministic=True)
            votes.append(int(action))

            # VÁLTOZTATÁS: A bizalmi szint kezelése típustól függően
            confidence = 0.0
            if model_type == "DQN":
                # A DQN modelleknél ki tudjuk nyerni a Q-értéket
                q_values = model.policy.q_net(model.policy.obs_to_tensor(obs)[0])
                confidence = float(q_values[0][int(action)])
            else:
                # A PPO modellek nem adnak közvetlen 'bizalmi' értéket.
                # Itt használhatnánk egy alapértelmezett értéket, vagy bonyolultabb
                # logikával az action probability-t, de most az egyszerűség kedvéért
                # egy fix, magas értéket adunk neki, jelezve, hogy a modell döntött.
                confidence = 1.0
            confidences.append(confidence)

        except Exception as e:
            print(f"Hiba a(z) {model_path} betöltésekor: {e}")
            continue

    if not votes:
        return f"{ticker}: HIBA (nem sikerült egy modellt sem betölteni)"

    # --- A szavazásos logika innentől változatlan ---
    vote_counts = Counter(votes)
    final_action = vote_counts.most_common(1)[0][0]

    winning_confidences = [
        conf for vote, conf in zip(votes, confidences) if vote == final_action
    ]
    avg_confidence = (
        sum(winning_confidences) / len(winning_confidences)
        if winning_confidences
        else 0
    )

    action_map = {0: "TARTÁS", 1: "VÉTEL", 2: "ELADÁS"}
    recommendation = action_map.get(final_action, "ISMERETLEN")

    log_recommendation(
        ticker=ticker,
        recommendation=recommendation,
        confidence=avg_confidence,
        day=today,
    )

    return f"{ticker}: {recommendation} (bizalom: {avg_confidence:.2f}, szavazatok: {dict(vote_counts)})"


def run_daily_recommendations(to_email: str):
    tickers = get_supported_ticker_list()
    today = datetime.date.today()
    recommendations = []

    for ticker in tickers:
        try:
            rec = generate_daily_recommendation_ensemble(ticker)
        except Exception as e:
            tb = traceback.format_exc()
            rec = f"{ticker}: HIBA ({str(e)})\n{tb}"
        recommendations.append(rec)

    subject = f"Napi Ajánlások ({today})"
    body = "\n".join(recommendations)
    send_email(subject, body, to_email)
    print("Ajánlások elküldve e-mailben.")

    with open(f"logs/recommendations_{today}.txt", "w") as f:
        f.write(body)
