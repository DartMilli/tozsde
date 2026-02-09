# Raspberry Pi Setup Útmutató - Teljes Lépésről-Lépésre

English summary: this guide follows the current deploy_rpi.sh workflow and uses the admin health endpoint with X-Admin-Key.

**Cél Hardver:** Raspberry Pi 4B vagy 5 (64-bites ARM)  
**Operációs Rendszer:** Raspberry Pi OS Lite (64-bites, Debian alapú)  
**Becsült Idő:** 1,5-2 óra összesen (nagyobb része: letöltések/indítások várakozása)  
**Szükséges:** SD-kártya (64GB+), USB-C tápegység, HDMI kábel, billentyűzet/egér (kezdeti setup-hoz)

---

## FÁZIS 1: HARDVER SETUP (15 perc)

### 1.1 Kicsomagolás és Csatlakoztatás

**Felszerelés ellenőrzőlista:**
- [ ] Raspberry Pi 4B vagy 5 (hűtőbordákkal ajánlott)
- [ ] Micro SD-kártya (64GB+ ajánlott)
- [ ] USB-C tápegység (27W Pi 5-höz, 15W minimum Pi 4-hez)
- [ ] Micro HDMI kábel(ek) - **2x Pi 4-hez, 1x Pi 5-höz**
- [ ] USB billentyűzet
- [ ] USB egér
- [ ] Opcionális: Hálózati kábel (gyorsabb, mint WiFi a setup során)
- [ ] Opcionális: Hűtési ventillátor (Pi 5 meleg lehet, különösen backtest alatt)

### 1.2 SD-kártya Formázása Operációs Rendszerrel

**A laptopod/asztali gépedről:**

**A lehetőség: Raspberry Pi Imager (Legegyszerűbb)**
1. Letöltés: https://www.raspberrypi.com/software/
2. SD-kártya behelyezése kártyaolvasóba
3. Imager futtatása:
   - Eszköz választása: **Raspberry Pi 4** vagy **Raspberry Pi 5**
   - OS választása: **Raspberry Pi OS (Egyéb) → Raspberry Pi OS Lite (64-bites)**
   - Storage választása: Az SD-kártyád
   - Kattints **TOVÁBB**
   - Válaszd az **BEÁLLÍTÁSOK SZERKESZTÉSE** (opcionális SSH/WiFi setuphoz)
   - Kattints **MENTÉS** → **IGEN** (törli a kártyát)
   - Várakozz az írásra és ellenőrzésre (~5-10 perc)

**B lehetőség: Parancssor (macOS/Linux)**
```bash
# Legújabb Pi OS Lite 64-bites letöltése
curl -L https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2024-01-15/2024-01-15-raspios-bookworm-arm64-lite.img.zip -o ~/Downloads/rpi-os.zip
unzip ~/Downloads/rpi-os.zip

# SD-kártya keresése
diskutil list
# Keress "external, physical" jelöléssel az SD-kártyádra

# Rendszerkép írása (cseréld diskX-et a saját diskre, pl. disk4)
sudo dd if=2024-01-15-raspios-bookworm-arm64-lite.img of=/dev/rdiskX bs=4m
# Várakozz... (2-5 perc)

# Kártya kiadása
diskutil eject /dev/diskX
```

### 1.3 Kezdeti Indítás

1. Formázott SD-kártya behelyezése Pi-be
2. Csatlakoztatás:
   - HDMI → Monitor/TV
   - USB → Billentyűzet + Egér
   - Hálózati kábel (opcionális)
   - USB-C tápellátás UTOLSÓNAK (ez indítja meg a Pi-t)
3. Várakozz ~30 másodpercet (a boot-hoz)
4. Bejelentkezés:
   ```
   felhasználónév: pi
   jelszó: raspberry
   ```

---

## FÁZIS 2: KEZDETI KONFIGURÁCIÓ (20 perc)

### 2.1 Raspi-config Futtatása

```bash
sudo raspi-config
```

**Konfigurálj ezeket az opciókat:**

1. **System Options → Hostname**
   - Változtass: `tozsde-pi` (vagy a preferált név)
   - Újraindítás szükséges

2. **Interface Options → SSH**
   - SSH engedélyezése (így SSH-zdhetsz a laptopodról később)

3. **Advanced Options → Expand Filesystem**
   - Kiterjeszkedj a teljes SD-kártya méretére

4. **Localization Options → Timezone**
   - Állítsd be az időzónádat

5. **Kilépés** és válaszd **IGEN az újraindításhoz**

### 2.2 IP-cím Megkeresése

Újraindítás után bejelentkezz újra és futtasd:
```bash
hostname -I
# Kimenet: 192.168.x.x
```

**Jegyezd fel ezt!** Ezt fogod használni SSH-hoz a laptopodról.

### 2.3 Rendszer Frissítése

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

Ez 5-10 percet vesz igénybe. Kávé time! ☕

---

## FÁZIS 3: APP KLÓNOZÁSA & DEPLOY (45 perc)

### 3.1 SSH a Laptopodról (Nincs Több Billentyűzet/Egér!)

```bash
# A laptopod/asztali gép terminálából
ssh pi@192.168.x.x
# Vagy használd a hostname-t:
ssh pi@tozsde-pi.local

# Első alkalom: válaszolj "yes" az ujjlenyomat kérdésre
# Jelszó: raspberry (vagy az új jelszó ha megváltoztattad)
```

### 3.2 Repository Klónozása

```bash
# Klónozd az app repo-t (URL-t igazítsd az igényhez)
git clone https://github.com/your-repo/tozsde_webapp.git ~/tozsde_webapp
cd ~/tozsde_webapp

# Ellenőrizd a struktúrát
ls -la
# Látnod kell: app/, tests/, requirements.txt, deploy_rpi.sh, stb.
```

### 3.3 EGYKATTINTÁSOS Deploy Futtatása

```bash
# Ez az EGYETLEN parancs mindent megtesz:
bash deploy_rpi.sh

# Várható kimenet folyama:
# [✓] System dependencies installed
# [✓] Virtual environment created
# [✓] Python dependencies installed
# [✓] systemd service created and enabled
# [✓] Cron jobs configured
# [✓] Health check script created
# [✓] Log rotation configured
# [✓] Flask API service started successfully
# [✓] API responding on port 5000
# DEPLOYMENT COMPLETE! 🎉

# Teljes idő: 10-15 perc (az internet sebességétől függ)
```

**Mit telepít a script:**
- ✅ System csomagok (Python 3.11, pip, cron, curl)
- ✅ Python virtuális környezet
- ✅ requirements.txt függőségek
- ✅ systemd service (Flask API)
- ✅ 3 cron job (napi, heti, havi)
- ✅ Health check script (5 percenként)
- ✅ Log rotation (7 nap megőrzés)
- ✅ Összes service indítása
- ✅ Teljes ellenőrzés

---

## FÁZIS 4: ELLENŐRZÉS (5 perc)

### 4.1 API Health Test

```bash
# A Pi-ből vagy a laptopodról
curl http://tozsde-pi.local:5000/admin/health -H "X-Admin-Key: <key>"

# Várható válasz:
# {"status": "healthy", "timestamp": "2026-01-23T10:30:45", ...}
```

Megjegyzes: a health_check.sh script alapertelmezetten /api/health-et hasznal. Allitsd at /admin/health-re es add hozza az X-Admin-Key headert.

### 4.2 Ütemezett Feladatok Megtekintése

```bash
# Nézd meg az összes cron jobot
crontab -l

# Kimenet kell legyen:
# 0 6 * * * ...daily pipeline...
# 0 4 * * 1 ...weekly audit...
# 0 1 1 * * ...monthly optimize...
```

### 4.3 Service Státusz

```bash
# Flask API fut?
sudo systemctl status tozsde-api.service

# Kell látnod: Active (running)
```

### 4.4 Live Naplók Megtekintése

```bash
# Valós idejű Flask naplók
sudo journalctl -u tozsde-api.service -f

# Összes elérhető napló
ls -lh ~/tozsde_webapp/logs/
```

### 4.5 Manuális Futtatás Tesztelése

```bash
# Ma pipeline-jét próbáld ki (száraz futás)
source ~/tozsde_webapp/venv/bin/activate
cd ~/tozsde_webapp
python -m app.infrastructure.cron_tasks --daily --dry-run 2>&1 | head -20

# Ellenőrizd a naplót
cat ~/tozsde_webapp/logs/cron_daily.log | tail -10
```

---

## FÁZIS 5: POST-DEPLOYMENT (Folyamatos)

### 5.1 Napi Monitoring

```bash
# Minden reggel ellenőrizd az egészségét
curl http://tozsde-pi.local:5000/admin/health -H "X-Admin-Key: <key>"

# Figyeld a rendszer erőforrásokat (Pi 5 meleg lehet)
df -h /home
free -h
/opt/vc/bin/vcgencmd measure_temp  # <80°C kell legyen
```

### 5.2 Cron Végrehajtás Ellenőrzése

```bash
# 6:00 AM után ellenőrizd, hogy a napi pipeline futott
cat ~/tozsde_webapp/logs/cron_daily.log | tail -5

# Hétfő 4:00 AM után ellenőrizd a heti auditot
cat ~/tozsde_webapp/logs/cron_weekly.log | tail -5
```

### 5.3 Hibaelhárítás

| Tünet | Diagnózis | Megoldás |
|-------|-----------|----------|
| **API nem indul** | `sudo journalctl -u tozsde-api.service -n 50` | Ellenőrizd a hibát, restart: `sudo systemctl restart tozsde-api.service` |
| **Cron nem futott** | Ellenőrizd: `crontab -l` | Ellenőrizd az időzónát: `date`, az idő egyezik-e |
| **Lemez megtelt (>85%)** | `df -h /home` | Töröld a régi naplókat: `rm ~/tozsde_webapp/logs/*.gz` |
| **Pi túlmelegszik (>85°C)** | `/opt/vc/bin/vcgencmd measure_temp` | Adjunk ventilátort/hűtőbordákat, csökkentsd a backtest periódusokat |
| **Hálózat timeout** | `ping 8.8.8.8` | Restart: `sudo systemctl restart networking` |
| **Magas CPU** | `top` | Ellenőrizd, ha backtest fut, fontold meg rövidebb periódusokat |
| **Import hibák** | `python -m app` | Ellenőrizd a venv-et: `which python` kell mutasson: /home/pi/... |

### 5.4 Távololi Hozzáférés Bárhonnan

SSH tunnel a biztonságos távololi hozzáféréshez:

```bash
# Option 1: SSH alagút (biztonságos)
ssh -R 5000:localhost:5000 user@your-vps.com
# Majd a VPS-ről: curl http://localhost:5000/api/health

# Option 2: VPN (mint Tailscale, ingyenes, könnyű)
curl https://tailscale.com/install.sh | sh
sudo tailscale up
# Majd hozzáfér Tailscale IP-n bárhonnan
```

---

## TECHNIKAI RÉSZLETEK

### Fájl Helyek

| Útvonal | Cél |
|--------|-----|
| `/home/pi/tozsde_webapp/` | Alkalmazás gyökere |
| `/home/pi/tozsde_webapp/venv/` | Python virtuális környezet |
| `/home/pi/tozsde_webapp/logs/` | Alkalmazás naplók (cron_*.log, health_check.log) |
| `/home/pi/tozsde_webapp/app/data/` | SQLite adatbázis (market_data.db) |
| `/etc/systemd/system/tozsde-api.service` | Flask service definíció |

### Servicek & Timers

```bash
# Összes service megtekintése
systemctl list-units --type service

# Rendszer naplók
journalctl -n 50 --follow

# Service restart ha szükséges
sudo systemctl restart tozsde-api.service

# Auto-start letiltása (ha szükséges)
sudo systemctl disable tozsde-api.service
```

### Cron Job Formátum

```bash
# Szerkeszd a crontab-ot
crontab -e

# Formátum: perc óra nap hónap nap_hete parancs
# Példák:
0 6 * * * parancs  # Naponta 6:00-kor
0 4 * * 1 parancs  # Hétfőnként 4:00-kor
0 1 1 * * parancs  # Hó 1. napján 1:00-kor
*/5 * * * * parancs  # Minden 5 percben
```

### Erőforrás Korlátok (Fontos!)

**Raspberry Pi 4B (4GB):**
- Max párhuzamos backtest periódus: 2-3
- Ajánlott: Rövidebb történeti periódusok (1-2 év)
- Memória: Figyelj `free -h`, legyen >200MB szabad

**Raspberry Pi 5 (8GB):**
- Max párhuzamos backtest periódus: 4-5
- Elbír: 3+ év történetet
- Jobb hőkezelés: Tartsd ventilátort engedélyezve

**Lemez Terület:**
- Napi naplók: ~5-20MB nap
- SQLite adatbázis: Idővel növekszik
- Log rotation: Utolsó 7 napot tartja (lásd `/etc/logrotate.d/tozsde`)

---

## FONTOS: Egyszeri Setup Újraindítás Után

Ha valaha újraindítod a Pi-t (áramkimaradás, manuális restart):

```bash
# Engedélyezd a serviceket újra
sudo systemctl enable tozsde-api.service
sudo systemctl start tozsde-api.service

# Ellenőrizd, hogy cron még aktív
crontab -l  # Kell mutatnia az előző jobokat

# Ha nem, újra add hozzá:
crontab -e  # Olvasd hozzá a 3 sort a deploy_rpi.sh-ből
```

---

## KÖVETKEZŐ LÉPÉSEK

1. ✅ Hardver setup kész
2. ✅ OS flash-elve és konfigurálva
3. ✅ App telepítve EGYETLEN scripttel
4. ✅ Servicek futnak és ellenőrzöttek
5. ⏭️ **KÖVETKEZŐ:** Figyeld az első napi futást (holnap 6:00 AM)
6. ⏭️ **MAJD:** E-mail riasztások konfigurálása vagy dashboard

---

## KÉRDÉSEK?

- **API nem válaszol?** → Ellenőrizd: `curl http://tozsde-pi.local:5000/api/health`
- **Cron nem fut?** → Ellenőrizd: `crontab -l` és `/home/pi/tozsde_webapp/logs/cron_*.log`
- **Engedély problémák?** → Mindent a `pi` felhasználónak kell lennie. Ellenőrizd: `ls -l ~/tozsde_webapp/`
- **Memória/lemez problémák?** → Ellenőrizd: `df -h /home` és `free -h`

**Boldog kereskedést! 🚀**
