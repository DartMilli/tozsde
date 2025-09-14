#!/bin/bash

echo "🚀 Python Kereskedő Applikáció Teljes Inicializálása Indul..."
set -e # Azonnal lépjen ki, ha egy parancs hibát dob

# === Konfiguráció ===
# Abszolút útvonal meghatározása a megbízhatóságért
BASE_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_CMD="python3"
VENV_DIR="$BASE_DIR/venv"
LOG_DIR="$BASE_DIR/logs"
CHART_DIR="$BASE_DIR/charts"
MODEL_DIR="$BASE_DIR/models"

echo "Alap könyvtár: $BASE_DIR"

# === Előkészületek ===
echo "📁 Szükséges mappák létrehozása..."
mkdir -p "$LOG_DIR"
mkdir -p "$CHART_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$BASE_DIR/app/data" # Adatbázisoknak

# === Telepítés ===
echo "🐍 Virtuális környezet létrehozása: $VENV_DIR"
$PYTHON_CMD -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "📦 Függőségek telepítése..."
# Külön telepítjük a torch-ot a requirements.txt-ben lévő megjegyzés alapján
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# A többi csomag telepítése
pip install -r "$BASE_DIR/requirements.txt"

# === Adatok és Paraméterek ===
echo "🗃️ Adatbázis inicializálása és historikus adatok letöltése (ez sokáig tarthat)..."
$PYTHON_CMD "$BASE_DIR/app/core/data_loader.py"

echo "🧬 Stratégia paramétereinek optimalizálása a genetikus algoritmussal..."
$PYTHON_CMD "$BASE_DIR/genetic_optimizer.py"

# === Modellek Tanítása ===
echo "🧠 Reinforcement Learning modellek tanítása és a legjobbak kihelyezése..."
# Ez a szkript most már automatikusan kiválasztja és átmásolja a legjobb modelleket
$PYTHON_CMD "$BASE_DIR/model_trainer.py"

echo "✅ Inicializálás Sikeresen Befejeződött!"
echo "---"

# === Cron Job Beállítása ===
echo "🕒 Cron Job Beállítási Útmutató:"
echo "A napi ajánlások automatikus küldéséhez add hozzá a következő sort a crontab-hoz."
echo "Nyisd meg a szerkesztőt a 'crontab -e' paranccsal, majd illeszd be az alábbi sort:"
echo ""
# Példa: Minden hétköznap reggel 8:05-kor fut
echo "5 8 * * 1-5 cd $BASE_DIR && $VENV_DIR/bin/python $BASE_DIR/cron_job.py >> $LOG_DIR/cron.log 2>&1"
echo ""
echo "Magyarázat:"
echo "  - '5 8 * * 1-5': Minden hétköznap (1-5) reggel 8:05-kor."
echo "  - 'cd $BASE_DIR': Belép a projekt főkönyvtárába a futtatás előtt."
echo "  - '$VENV_DIR/bin/python': A virtuális környezetben lévő Pythont használja."
echo "  - '>> $LOG_DIR/cron.log 2>&1': A szkript kimenetét és hibáit a logs/cron.log fájlba írja."
echo ""
echo "❗️ FONTOS: Mielőtt élesítenéd, győződj meg róla, hogy a .env fájlban a 'RECOMMENDER_EMAIL' helyesen van beállítva a NAS-on!"

# Deaktiváljuk a virtuális környezetet a szkript végén
deactivate

echo "🚀 Folyamat kész."