# Raspberry Pi Setup Utmutato - Teljes Lepesrol-Lepesre

English summary: this guide follows the current deploy_rpi.sh workflow and uses the admin health endpoint with X-Admin-Key.

**Cel Hardver:** Raspberry Pi 4B vagy 5 (64-bites ARM)  
**Operacios Rendszer:** Raspberry Pi OS Lite (64-bites, Debian alapu)  
**Becsult Ido:** 1,5-2 ora osszesen (nagyobb resze: letoltesek/inditasok varakozasa)  
**Szukseges:** SD-kartya (64GB+), USB-C tapegyseg, HDMI kabel, billentyuzet/eger (kezdeti setup-hoz)

---

## FAZIS 1: HARDVER SETUP (15 perc)

### 1.1 Kicsomagolas es Csatlakoztatas

**Felszereles ellenorzolista:**
- [ ] Raspberry Pi 4B vagy 5 (hutobordakkal ajanlott)
- [ ] Micro SD-kartya (64GB+ ajanlott)
- [ ] USB-C tapegyseg (27W Pi 5-hoz, 15W minimum Pi 4-hez)
- [ ] Micro HDMI kabel(ek) - **2x Pi 4-hez, 1x Pi 5-hoz**
- [ ] USB billentyuzet
- [ ] USB eger
- [ ] Opcionalis: Halozati kabel (gyorsabb, mint WiFi a setup soran)
- [ ] Opcionalis: Hutesi ventillator (Pi 5 meleg lehet, kulonosen backtest alatt)

### 1.2 SD-kartya Formazasa Operacios Rendszerrel

**A laptopod/asztali gepedrol:**

**A lehetoseg: Raspberry Pi Imager (Legegyszerubb)**
1. Letoltes: https://www.raspberrypi.com/software/
2. SD-kartya behelyezese kartyaolvasoba
3. Imager futtatasa:
   - Eszkoz valasztasa: **Raspberry Pi 4** vagy **Raspberry Pi 5**
   - OS valasztasa: **Raspberry Pi OS (Egyeb) -> Raspberry Pi OS Lite (64-bites)**
   - Storage valasztasa: Az SD-kartyad
   - Kattints **TOVABB**
   - Valaszd az **BEALLITASOK SZERKESZTESE** (opcionalis SSH/WiFi setuphoz)
   - Kattints **MENTES** -> **IGEN** (torli a kartyat)
   - Varakozz az irasra es ellenorzesre (~5-10 perc)

**B lehetoseg: Parancssor (macOS/Linux)**
```bash
# Legujabb Pi OS Lite 64-bites letoltese
curl -L https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2024-01-15/2024-01-15-raspios-bookworm-arm64-lite.img.zip -o ~/Downloads/rpi-os.zip
unzip ~/Downloads/rpi-os.zip

# SD-kartya keresese
diskutil list
# Keress "external, physical" jelolessel az SD-kartyadra

# Rendszerkep irasa (csereld diskX-et a sajat diskre, pl. disk4)
sudo dd if=2024-01-15-raspios-bookworm-arm64-lite.img of=/dev/rdiskX bs=4m
# Varakozz... (2-5 perc)

# Kartya kiadasa
diskutil eject /dev/diskX
```

### 1.3 Kezdeti Inditas

1. Formazott SD-kartya behelyezese Pi-be
2. Csatlakoztatas:
   - HDMI -> Monitor/TV
   - USB -> Billentyuzet + Eger
   - Halozati kabel (opcionalis)
   - USB-C tapellatas UTOLSONAK (ez inditja meg a Pi-t)
3. Varakozz ~30 masodpercet (a boot-hoz)
4. Bejelentkezes:
   ```
   felhasznalonev: pi
   jelszo: raspberry
   ```

---

## FAZIS 2: KEZDETI KONFIGURACIO (20 perc)

### 2.1 Raspi-config Futtatasa

```bash
sudo raspi-config
```

**Konfiguralj ezeket az opciokat:**

1. **System Options -> Hostname**
   - Valtoztass: `tozsde-pi` (vagy a preferalt nev)
   - Ujrainditas szukseges

2. **Interface Options -> SSH**
   - SSH engedelyezese (igy SSH-zdhetsz a laptopodrol kesobb)

3. **Advanced Options -> Expand Filesystem**
   - Kiterjeszkedj a teljes SD-kartya meretere

4. **Localization Options -> Timezone**
   - Allitsd be az idozonadat

5. **Kilepes** es valaszd **IGEN az ujrainditashoz**

### 2.2 IP-cim Megkeresese

Ujrainditas utan bejelentkezz ujra es futtasd:
```bash
hostname -I
# Kimenet: 192.168.x.x
```

**Jegyezd fel ezt!** Ezt fogod hasznalni SSH-hoz a laptopodrol.

### 2.3 Rendszer Frissitese

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

Ez 5-10 percet vesz igenybe. Kave time! 

---

## FAZIS 3: APP KLONOZASA & DEPLOY (45 perc)

### 3.1 SSH a Laptopodrol (Nincs Tobb Billentyuzet/Eger!)

```bash
# A laptopod/asztali gep terminalabol
ssh pi@192.168.x.x
# Vagy hasznald a hostname-t:
ssh pi@tozsde-pi.local

# Elso alkalom: valaszolj "yes" az ujjlenyomat kerdesre
# Jelszo: raspberry (vagy az uj jelszo ha megvaltoztattad)
```

### 3.2 Repository Klonozasa

```bash
# Klonozd az app repo-t (URL-t igazitsd az igenyhez)
git clone https://github.com/your-repo/tozsde_webapp.git ~/tozsde_webapp
cd ~/tozsde_webapp

# Ellenorizd a strukturat
ls -la
# Latnod kell: app/, tests/, requirements.txt, deploy_rpi.sh, stb.
```

### 3.3 EGYKATTINTASOS Deploy Futtatasa

```bash
# Ez az EGYETLEN parancs mindent megtesz:
bash deploy_rpi.sh

# Varhato kimenet folyama:
# [v] System dependencies installed
# [v] Virtual environment created
# [v] Python dependencies installed
# [v] systemd service created and enabled
# [v] Cron jobs configured
# [v] Health check script created
# [v] Log rotation configured
# [v] Flask API service started successfully
# [v] API responding on port 5000
# DEPLOYMENT COMPLETE! 

# Teljes ido: 10-15 perc (az internet sebessegetol fugg)
```

**Mit telepit a script:**
-  System csomagok (Python 3.11, pip, cron, curl)
-  Python virtualis kornyezet
-  requirements.txt fuggosegek
-  systemd service (Flask API)
-  3 cron job (napi, heti, havi)
-  Opcionalis RL training hook (env varokkal bekapcsolhato)
-  Health check script (5 percenkent)
-  Log rotation (7 nap megorzes)
-  Osszes service inditasa
-  Teljes ellenorzes

---

## FAZIS 4: ELLENORZES (5 perc)

### 4.1 API Health Test

```bash
# A Pi-bol vagy a laptopodrol
curl http://tozsde-pi.local:5000/admin/health -H "X-Admin-Key: <key>"

# Varhato valasz:
# {"status": "healthy", "timestamp": "2026-01-23T10:30:45", ...}
```

Megjegyzes: admin health checkhez mindig `/admin/health` endpointot es `X-Admin-Key` headert hasznalj.

### Opcionalis RL training a deploy alatt (opt-in)
Allitsd be ezeket az env varokat a `deploy_rpi.sh` futtatasa elott:
- `TRAIN_RL_ON_DEPLOY=true`
- `TRAIN_RL_FORCE=true` (kenyszeritett training akkor is, ha van modell es fingerprint OK)
- `TRAIN_RL_TICKER=VOO`
- `TRAIN_RL_REWARD_STRATEGY=portfolio_value`
- A walk-forward (GA) automatikusan fut RL elott.

Viselkedes: a deploy script ugyanazt a logikat koveti, mint a CI. Ha vannak modellek es a fingerprint nem valtozott, a training kimarad (kiveve ha `TRAIN_RL_FORCE=true`).
Megjegyzes: az RL/GA futas ellenorzi a cache-t a Config.START_DATE/END_DATE idoszakra, es leall, ha a letoltes hianyos.

### Opcionalis RL training cron (opt-in)
Allitsd be ezeket az env varokat a `deploy_rpi.sh` futtatasa elott:
- `ENABLE_RL_CRON=true`
- `RL_CRON_MODE=minimal|full` (full: `main.py monthly`)
- `RL_CRON_TICKER=VOO` (minimal mod)
- `RL_CRON_REWARD_STRATEGY=portfolio_value` (minimal mod)
- A walk-forward (GA) automatikusan fut RL elott.

### 4.2 Utemezett Feladatok Megtekintese

```bash
# Nezd meg az osszes cron jobot
crontab -l

# Kimenet kell legyen:
# 0 6 * * * ...daily pipeline...
# 0 4 * * 1 ...weekly audit...
# 0 1 1 * * ...monthly optimize...
```

### 4.3 Service Statusz

```bash
# Flask API fut?
sudo systemctl status tozsde-api.service

# Kell latnod: Active (running)
```

### 4.4 Live Naplok Megtekintese

```bash
# Valos ideju Flask naplok
sudo journalctl -u tozsde-api.service -f

# Osszes elerheto naplo
ls -lh ~/tozsde_webapp/logs/
```

### 4.5 Manualis Futtatas Tesztelese

```bash
# Ma pipeline-jet probald ki (szaraz futas)
source ~/tozsde_webapp/venv/bin/activate
cd ~/tozsde_webapp
python -m app.infrastructure.cron_tasks --daily --dry-run 2>&1 | head -20

# Ellenorizd a naplot
cat ~/tozsde_webapp/logs/cron_daily.log | tail -10
```

---

## FAZIS 5: POST-DEPLOYMENT (Folyamatos)

### 5.1 Napi Monitoring

```bash
# Minden reggel ellenorizd az egeszseget
curl http://tozsde-pi.local:5000/admin/health -H "X-Admin-Key: <key>"

# Figyeld a rendszer eroforrasokat (Pi 5 meleg lehet)
df -h /home
free -h
/opt/vc/bin/vcgencmd measure_temp  # <80C kell legyen
```

### 5.2 Cron Vegrehajtas Ellenorzese

```bash
# 6:00 AM utan ellenorizd, hogy a napi pipeline futott
cat ~/tozsde_webapp/logs/cron_daily.log | tail -5

# Hetfo 4:00 AM utan ellenorizd a heti auditot
cat ~/tozsde_webapp/logs/cron_weekly.log | tail -5
```

### 5.3 Hibaelharitas

| Tunet | Diagnozis | Megoldas |
|-------|-----------|----------|
| **API nem indul** | `sudo journalctl -u tozsde-api.service -n 50` | Ellenorizd a hibat, restart: `sudo systemctl restart tozsde-api.service` |
| **Cron nem futott** | Ellenorizd: `crontab -l` | Ellenorizd az idozonat: `date`, az ido egyezik-e |
| **Lemez megtelt (>85%)** | `df -h /home` | Torold a regi naplokat: `rm ~/tozsde_webapp/logs/*.gz` |
| **Pi tulmelegszik (>85C)** | `/opt/vc/bin/vcgencmd measure_temp` | Adjunk ventilatort/hutobordakat, csokkentsd a backtest periodusokat |
| **Halozat timeout** | `ping 8.8.8.8` | Restart: `sudo systemctl restart networking` |
| **Magas CPU** | `top` | Ellenorizd, ha backtest fut, fontold meg rovidebb periodusokat |
| **Import hibak** | `python -m app` | Ellenorizd a venv-et: `which python` kell mutasson: /home/pi/... |

### 5.4 Tavololi Hozzaferes Barhonnan

SSH tunnel a biztonsagos tavololi hozzafereshez:

```bash
# Option 1: SSH alagut (biztonsagos)
ssh -R 5000:localhost:5000 user@your-vps.com
# Majd a VPS-rol: curl http://localhost:5000/admin/health -H "X-Admin-Key: <key>"

# Option 2: VPN (mint Tailscale, ingyenes, konnyu)
curl https://tailscale.com/install.sh | sh
sudo tailscale up
# Majd hozzafer Tailscale IP-n barhonnan
```

---

## TECHNIKAI RESZLETEK

### Fajl Helyek

| Utvonal | Cel |
|--------|-----|
| `/home/pi/tozsde_webapp/` | Alkalmazas gyokere |
| `/home/pi/tozsde_webapp/venv/` | Python virtualis kornyezet |
| `/home/pi/tozsde_webapp/logs/` | Alkalmazas naplok (cron_*.log, health_check.log) |
| `/home/pi/tozsde_webapp/app/data/` | SQLite adatbazis (market_data.db) |
| `/etc/systemd/system/tozsde-api.service` | Flask service definicio |

### Servicek & Timers

```bash
# Osszes service megtekintese
systemctl list-units --type service

# Rendszer naplok
journalctl -n 50 --follow

# Service restart ha szukseges
sudo systemctl restart tozsde-api.service

# Auto-start letiltasa (ha szukseges)
sudo systemctl disable tozsde-api.service
```

### Cron Job Formatum

```bash
# Szerkeszd a crontab-ot
crontab -e

# Formatum: perc ora nap honap nap_hete parancs
# Peldak:
0 6 * * * parancs  # Naponta 6:00-kor
0 4 * * 1 parancs  # Hetfonkent 4:00-kor
0 1 1 * * parancs  # Ho 1. napjan 1:00-kor
*/5 * * * * parancs  # Minden 5 percben
```

### Eroforras Korlatok (Fontos!)

**Raspberry Pi 4B (4GB):**
- Max parhuzamos backtest periodus: 2-3
- Ajanlott: Rovidebb torteneti periodusok (1-2 ev)
- Memoria: Figyelj `free -h`, legyen >200MB szabad

**Raspberry Pi 5 (8GB):**
- Max parhuzamos backtest periodus: 4-5
- Elbir: 3+ ev tortenetet
- Jobb hokezeles: Tartsd ventilatort engedelyezve

**Lemez Terulet:**
- Napi naplok: ~5-20MB nap
- SQLite adatbazis: Idovel novekszik
- Log rotation: Utolso 7 napot tartja (lasd `/etc/logrotate.d/tozsde`)

---

## FONTOS: Egyszeri Setup Ujrainditas Utan

Ha valaha ujrainditod a Pi-t (aramkimaradas, manualis restart):

```bash
# Engedelyezd a serviceket ujra
sudo systemctl enable tozsde-api.service
sudo systemctl start tozsde-api.service

# Ellenorizd, hogy cron meg aktiv
crontab -l  # Kell mutatnia az elozo jobokat

# Ha nem, ujra add hozza:
crontab -e  # Olvasd hozza a 3 sort a deploy_rpi.sh-bol
```

---

## KOVETKEZO LEPESEK

1.  Hardver setup kesz
2.  OS flash-elve es konfiguralva
3.  App telepitve EGYETLEN scripttel
4.  Servicek futnak es ellenorzottek
5.  **KOVETKEZO:** Figyeld az elso napi futast (holnap 6:00 AM)
6.  **MAJD:** E-mail riasztasok konfiguralasa vagy dashboard

---

## KERDESEK?

- **API nem valaszol?** -> Ellenorizd: `curl http://tozsde-pi.local:5000/admin/health -H "X-Admin-Key: <key>"`
- **Cron nem fut?** -> Ellenorizd: `crontab -l` es `/home/pi/tozsde_webapp/logs/cron_*.log`
- **Engedely problemak?** -> Mindent a `pi` felhasznalonak kell lennie. Ellenorizd: `ls -l ~/tozsde_webapp/`
- **Memoria/lemez problemak?** -> Ellenorizd: `df -h /home` es `free -h`

**Boldog kereskedest! **
