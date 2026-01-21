# main.py használata

A terminálból, a projekt gyökérkönyvtárában:

Minden frissítése és újratanítása (verziófrissítésnél):

```bash
python main.py update
python main.py train --force
```

Csak a webszerver indítása (normál használat):

```bash
python main.py runserver --host 0.0.0.0 
```
(A 0.0.0.0 host kell, hogy a NAS hálózatán más eszközről is elérd.)



# 📦 NAS Telepítési és Biztonsági Útmutató (Synology DS118 + FastAPI App)

Ez a leírás lépésről lépésre bemutatja, hogyan telepítsd és futtasd a tőzsdei MI alkalmazásodat egy **Synology NAS-on (DSM 7+)**, biztonságosan, **külső elérés nélkül**.

---

## 📁 Projekt struktúra
A következő könyvtárszerkezet legyen a NAS-on (például: `/volume1/docker/trading_ai/`):

```
/trading_ai
├── venv/                 # Python virtuális környezet
├── .env                  # Beállítások (EMAIL, jelszó, stb.)
├── requirements.txt      # Python csomagok
├── main.py               # Belépési pont
├── ui/app.py             # FastAPI app (ha szükséges)
└── app/...               # A teljes projekt többi része
```

---

## 🛠️ 1. Virtuális környezet és csomagok telepítése

SSH-val jelentkezz be a NAS-ra:

```bash
cd /volume1/docker/trading_ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ 2. `.env` fájl beállítása

Hozz létre egy `.env` fájlt a projekt gyökerében:

```
EMAIL_ADDRESS=te_cimed@domain.hu
EMAIL_PASSWORD=app_jelszo
```

---

## 🔒 3. FastAPI biztonságos futtatása (opcionális)

Ha szeretnéd használni a webes felületet (pl. `/recommend` végpont):

```bash
# Csak helyi hálózatra:
uvicorn ui.app:app --host 192.168.1.50 --port 8000
```

- **NE** állíts be port forwardingot a routeren!
- NE használd a QuickConnectet!
- Tűzfalban engedélyezd csak a 192.168.x.x IP-tartományból az elérést.

---

## 📅 4. Időzített futtatás (napi ajánlások)

A cron job lefuttatja a napi ajánló rendszert (emailen):

```bash
crontab -e
```

Add hozzá (például minden nap 8:00-kor, vagy minden negyedévben 1.-én 0:00-kor):

```
0 6 * * * /volume1/docker/trading_ai/venv/bin/python /volume1/docker/trading_ai/scheduler/cron_job.py --mode daily
0 0 1 */3 * /volume1/docker/trading_ai/venv/bin/python /volume1/docker/trading_ai/scheduler/cron_job.py --mode optimize
0 4 * * MON /volume1/docker/trading_ai/venv/bin/python /volume1/docker/trading_ai/scheduler/cron_job.py --mode reliability
```

---

⚠️ Fontos: adatbázis séma életciklus

A rendszer szándékosan szétválasztja:

adatbázis séma kezelését (DDL, migrációk)

üzemi futtatást (napi ajánlások, optimalizáció, reliability)

🧱 Sémakezelés (INFRA lépés)

Az adatbázis sémát NEM a main vagy a cron jobok hozzák létre.

A sémát külön, manuálisan vagy deploy során kell alkalmazni:

```
python scripts/apply_schema.py
```

---

## ✅ Eredmény

- A NAS minden nap reggel emailben küldi a tőzsdei javaslatokat.
- A rendszer teljesen biztonságos: csak belső hálózatról elérhető.
- A FastAPI API opcionálisan használható, de nem szükséges a működéshez.

---

## 🧠 Tipp

Ha csak az emailes ajánlás kell, nem szükséges futtatni a webes API-t. A cron job önállóan elvégzi a napi elemzést és küldi az értesítést.

---

Készítette: [OpenAI ChatGPT + Te](https://chat.openai.com)