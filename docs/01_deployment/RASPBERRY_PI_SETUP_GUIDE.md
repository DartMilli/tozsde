# Raspberry Pi Setup Guide - Complete Walkthrough

**Target Hardware:** Raspberry Pi 4B or 5 (64-bit ARM)  
**Operating System:** Raspberry Pi OS Lite (64-bit, Debian-based)  
**Estimated Time:** 1.5-2 hours total (mostly waiting for downloads/reboots)  
**Required:** SD card (64GB+), USB-C power supply, HDMI cable, keyboard/mouse (initial setup only)

---

## PHASE 1: HARDWARE SETUP (15 minutes)

### 1.1 Unbox and Connect

**Equipment checklist:**
- [ ] Raspberry Pi 4B or 5 (with heatsinks recommended)
- [ ] Micro SD card (64GB+ recommended for logs & databases)
- [ ] USB-C power supply (27W for Pi 5, 15W minimum for Pi 4)
- [ ] Micro HDMI cable(s) - **2x for Pi 4, 1x for Pi 5**
- [ ] USB keyboard
- [ ] USB mouse
- [ ] Optional: Network cable (faster than WiFi during setup)
- [ ] Optional: Cooling fan (Pi 5 runs hot, especially with backtest)

### 1.2 Flash SD Card with OS

**On your laptop/desktop computer:**

**Option A: Using Raspberry Pi Imager (Easiest)**
1. Download: https://www.raspberrypi.com/software/
2. Insert SD card into card reader
3. Run Imager:
   - Choose Device: **Raspberry Pi 4** or **Raspberry Pi 5**
   - Choose OS: **Raspberry Pi OS (Other) → Raspberry Pi OS Lite (64-bit)**
   - Choose Storage: Your SD card
   - Click **NEXT**
   - Choose **EDIT SETTINGS** (optional, for SSH/WiFi setup)
   - Click **SAVE** → **YES** (will erase card)
   - Wait for write & verify (~5-10 minutes)

**Option B: Command Line (macOS/Linux)**
```bash
# Download latest Pi OS Lite 64-bit
curl -L https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2024-01-15/2024-01-15-raspios-bookworm-arm64-lite.img.zip -o ~/Downloads/rpi-os.zip
unzip ~/Downloads/rpi-os.zip

# Find SD card
diskutil list
# Look for "external, physical" with your card size

# Write image (replace diskX with your disk, e.g. disk4)
sudo dd if=2024-01-15-raspios-bookworm-arm64-lite.img of=/dev/rdiskX bs=4m
# Wait... (2-5 minutes)

# Eject card
diskutil eject /dev/diskX
```

### 1.3 Initial Boot

1. Insert flashed SD card into Pi
2. Connect:
   - HDMI → monitor
   - USB → keyboard + mouse
   - Network cable (optional but recommended)
   - USB-C power last (this starts the Pi)
3. Wait ~30 seconds for boot (colored square appears)
4. Login:
   ```
   username: pi
   password: raspberry
   ```

---

## PHASE 2: INITIAL CONFIGURATION (20 minutes)

### 2.1 Run raspi-config

```bash
sudo raspi-config
```

**Configure these options:**

1. **System Options → Hostname**
   - Change to: `tozsde-pi` (or your preference)
   - Reboot when asked

2. **Interface Options → SSH**
   - Enable SSH (so you can SSH from laptop later)

3. **Advanced Options → Expand Filesystem**
   - Expand to use full SD card space

4. **Localization Options → Timezone**
   - Set your timezone

5. **Exit** and choose **YES to reboot**

### 2.2 Find IP Address

After reboot, login again and run:
```bash
hostname -I
# Output: 192.168.x.x
```

**Write this down!** You'll use it to SSH from your laptop.

### 2.3 Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

This may take 5-10 minutes. Grab coffee! ☕

---

## PHASE 3: CLONE APP & RUN DEPLOY (45 minutes)

### 3.1 SSH From Your Laptop (no more keyboard/mouse needed!)

```bash
# From your laptop/desktop terminal
ssh pi@192.168.x.x
# Or use hostname:
ssh pi@tozsde-pi.local

# First time: answer "yes" to fingerprint question
# Password: raspberry (or your new password if you changed it)
```

### 3.2 Clone Repository

```bash
# Clone your app repo (adjust URL as needed)
git clone https://github.com/your-repo/tozsde_webapp.git ~/tozsde_webapp
cd ~/tozsde_webapp

# Verify structure
ls -la
# Should see: app/, tests/, requirements.txt, deploy_rpi.sh, etc.
```

### 3.3 Run ONE-CLICK Deploy

```bash
# This single command does EVERYTHING:
bash deploy_rpi.sh

# Expected output flow:
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

# Total time: 10-15 minutes (depends on internet speed)
```

**What the script does:**
- ✅ Installs system packages (Python 3.11, pip, curl, etc.)
- ✅ Creates Python virtual environment
- ✅ Installs Python dependencies from requirements.txt
- ✅ Creates systemd service for Flask API (auto-start on reboot)
- ✅ Schedules cron jobs:
  - Daily 6:00 AM: Market pipeline
  - Monday 4:00 AM: Weekly backtest audit
  - 1st of month 1:00 AM: Monthly GA optimization
- ✅ Sets up 5-minute health checks
- ✅ Configures log rotation
- ✅ Starts Flask API on port 5000
- ✅ Verifies everything works

---

## PHASE 4: VERIFICATION (5 minutes)

### 4.1 Test API Health

```bash
# From Pi or your laptop
curl http://tozsde-pi.local:5000/api/health

# Expected response:
# {"status": "healthy", "timestamp": "2026-01-23T10:30:45", ...}
```

### 4.2 View Scheduled Tasks

```bash
# See all your cron jobs
crontab -l

# Output should show:
# 0 6 * * * ...daily pipeline...
# 0 4 * * 1 ...weekly audit...
# 0 1 1 * * ...monthly optimize...
```

### 4.3 Check Service Status

```bash
# Flask API running?
sudo systemctl status tozsde-api.service

# Should show: Active (running)
```

### 4.4 View Live Logs

```bash
# Real-time Flask logs
sudo journalctl -u tozsde-api.service -f

# See all available logs
ls -lh ~/tozsde_webapp/logs/
```

### 4.5 Test Manual Execution

```bash
# Try running today's pipeline manually (dry run)
source ~/tozsde_webapp/venv/bin/activate
cd ~/tozsde_webapp
python -m app.daily_pipeline --dry-run 2>&1 | head -20

# Check if it logged
cat ~/tozsde_webapp/logs/cron_daily.log | tail -10
```

---

## PHASE 5: POST-DEPLOYMENT (Ongoing)

### 5.1 Daily Monitoring

```bash
# Check system health every morning
curl http://tozsde-pi.local:5000/api/health

# Monitor disk usage (Pi 5 runs hot)
df -h /home
free -h
/opt/vc/bin/vcgencmd measure_temp  # Should be <80°C
```

### 5.2 Check Cron Execution

```bash
# After 6:00 AM, verify daily pipeline ran
cat ~/tozsde_webapp/logs/cron_daily.log | tail -5

# After Monday 4:00 AM, check weekly audit
cat ~/tozsde_webapp/logs/cron_weekly.log | tail -5
```

### 5.3 Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| **API won't start** | `sudo journalctl -u tozsde-api.service -n 50` | Check error, restart: `sudo systemctl restart tozsde-api.service` |
| **Cron job didn't run** | Check crontab: `crontab -l` | Verify time zone: `date`, verify time matches |
| **Disk full (>85%)** | `df -h /home` | Delete old logs: `rm ~/tozsde_webapp/logs/*.gz` |
| **Pi overheating (>85°C)** | `/opt/vc/bin/vcgencmd measure_temp` | Add fan/heatsink, reduce backtest periods |
| **Network timeout** | `ping 8.8.8.8` | Restart: `sudo systemctl restart networking` |
| **High CPU usage** | `top` | Check if backtest running, consider shorter periods |
| **Import errors** | `python -m app` | Verify venv: `which python` should show /home/pi/... |

### 5.4 Remote Access from Anywhere

To access your Pi from outside your home network:

```bash
# Option 1: SSH tunnel (secure)
ssh -R 5000:localhost:5000 user@your-vps.com
# Then from VPS: curl http://localhost:5000/api/health

# Option 2: VPN (like Tailscale, free, easy)
curl https://tailscale.com/install.sh | sh
sudo tailscale up
# Then access via Tailscale IP from anywhere
```

---

## TECHNICAL DETAILS

### File Locations

| Path | Purpose |
|------|---------|
| `/home/pi/tozsde_webapp/` | Application root |
| `/home/pi/tozsde_webapp/venv/` | Python virtual environment |
| `/home/pi/tozsde_webapp/logs/` | Application logs (cron_*.log, health_check.log) |
| `/home/pi/tozsde_webapp/app/data/` | SQLite database (market_data.db) |
| `/etc/systemd/system/tozsde-api.service` | Flask service definition |

### Services & Timers

```bash
# View all services
systemctl list-units --type service

# View system logs
journalctl -n 50 --follow

# Restart service if needed
sudo systemctl restart tozsde-api.service

# Disable auto-start (if you need to)
sudo systemctl disable tozsde-api.service
```

### Cron Job Format

```bash
# Edit crontab
crontab -e

# Format: minute hour day month weekday command
# Examples:
0 6 * * * command  # Daily @ 6:00 AM
0 4 * * 1 command  # Monday @ 4:00 AM
0 1 1 * * command  # 1st of month @ 1:00 AM
*/5 * * * * command  # Every 5 minutes
```

### Resource Constraints (Important!)

**Raspberry Pi 4B (4GB):**
- Max parallel backtest periods: 2-3 (walk-forward)
- Recommended: Shorter historical periods (1-2 years)
- Memory: Monitor `free -h`, ensure >200MB free

**Raspberry Pi 5 (8GB):**
- Max parallel backtest periods: 4-5
- Can handle: 3+ years of history
- Better thermal management: Keep fan enabled

**Disk Space:**
- Daily logs: ~5-20MB per day
- SQLite database: Grows over time
- Log rotation keeps last 7 days (see `/etc/logrotate.d/tozsde`)

---

## IMPORTANT: One-Time Setup After Reboot

If you ever need to reboot the Pi (power loss, manual restart):

```bash
# Re-enable services after reboot
sudo systemctl enable tozsde-api.service
sudo systemctl start tozsde-api.service

# Verify cron still active
crontab -l  # Should show your jobs

# If not, re-add them:
crontab -e  # Add the 3 lines from deploy_rpi.sh
```

---

## NEXT STEPS

1. ✅ Hardware setup complete
2. ✅ OS flashed and configured
3. ✅ App deployed with ONE script
4. ✅ Services running and verified
5. ⏭️ **NEXT:** Monitor logs for first daily run (6:00 AM tomorrow)
6. ⏭️ **THEN:** Configure email alerts or dashboard

---

## Questions?

- **API not responding?** → Check: `curl http://tozsde-pi.local:5000/api/health`
- **Cron not running?** → Check: `crontab -l` and `/home/pi/tozsde_webapp/logs/cron_*.log`
- **Permission issues?** → All should be owned by `pi` user. Check: `ls -l ~/tozsde_webapp/`
- **Memory/disk issues?** → Check: `df -h /home` and `free -h`

**Happy trading! 🚀**
