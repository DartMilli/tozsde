import os
from dotenv import load_dotenv

from app.core.recommender import run_daily_recommendations

load_dotenv()
target_email = os.getenv("RECOMMENDER_EMAIL")

if target_email:
    run_daily_recommendations(target_email)
else:
    print("Hiba: RECOMMENDER_EMAIL nincs beállítva a .env fájlban.")