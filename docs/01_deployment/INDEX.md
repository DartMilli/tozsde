# Deployment Documentation Index

## 🍓 Raspberry Pi Production Deployment

This folder contains everything needed to deploy ToZsDE on Raspberry Pi 4/5.

### 📄 Central Guide
- **[RASPBERRY_PI_SETUP_GUIDE.md](RASPBERRY_PI_SETUP_GUIDE.md)** - Complete walkthrough from unboxing to live trading
  - Hardware setup (15 min)
  - Initial configuration (20 min)  
  - One-click deployment (45 min)
  - Verification & testing (5 min)
  - Post-deployment monitoring
  - Troubleshooting guide

### 🚀 Quick Deploy
Located in project root: **`deploy_rpi.sh`**

The script automates:
- System dependencies (Python 3.11, pip, build tools)
- Python virtual environment
- Application requirements
- systemd service (Flask API)
- 3 cron jobs (daily, weekly, monthly)
- Health checks (5 min interval)
- Log rotation

**Usage:**
```bash
bash deploy_rpi.sh
```

**Runtime:** 10-15 minutes

---

## 📊 Deployment Details

| Component | Configuration |
|-----------|---|
| **Hardware** | Raspberry Pi 4B (8GB) or Pi 5 (8GB) |
| **OS** | Raspberry Pi OS Lite 64-bit |
| **Python** | 3.11+ |
| **API Server** | Flask on port 5000 |
| **Process Manager** | systemd service |
| **Task Scheduler** | cron (daily, weekly, monthly) |
| **Monitoring** | 5-minute health checks |
| **Storage** | SQLite (market_data.db) |

---

## 🎯 Typical Workflow

1. **Physical Setup** (15 min)
   - Flash Raspberry Pi OS to SD card
   - Boot Pi, connect peripherals
   - SSH enable via raspi-config

2. **Application Deployment** (45 min)
   - Clone repository
   - Run `bash deploy_rpi.sh`
   - System ready ✅

3. **Verification** (5 min)
   - Test API: `curl http://tozsde-pi.local:5000/api/health`
   - Check logs: `sudo journalctl -u tozsde-api.service -f`
   - Verify cron: `crontab -l`

4. **Production Monitoring** (Ongoing)
   - Daily API health checks
   - Log rotation (automatic)
   - Cron job execution verification

---

## 📝 File Structure

```
/home/pi/tozsde_webapp/
├── app/                    (application code)
├── tests/                  (test suite)
├── venv/                   (Python environment)
├── logs/                   (application logs)
│   ├── cron_daily.log
│   ├── cron_weekly.log
│   ├── cron_monthly.log
│   └── health_check.log
├── scripts/
│   └── health_check.sh     (5-min health probe)
├── config/
│   ├── systemd/
│   │   └── tozsde-api.service
│   └── settings.env
└── data/
    └── market_data.db      (SQLite database)
```

---

## ⚙️ Scheduled Tasks

### Daily Pipeline (6:00 AM)
```bash
0 6 * * * source /home/pi/tozsde_webapp/venv/bin/activate && \
           cd /home/pi/tozsde_webapp && \
           python -m app.daily_pipeline
```

### Weekly Audit (Monday 4:00 AM)
```bash
0 4 * * 1 source /home/pi/tozsde_webapp/venv/bin/activate && \
          cd /home/pi/tozsde_webapp && \
          python -m app.backtesting.audit_runner
```

### Monthly Optimization (1st of month, 1:00 AM)
```bash
0 1 1 * * source /home/pi/tozsde_webapp/venv/bin/activate && \
          cd /home/pi/tozsde_webapp && \
          python -m app.optimization.runner
```

### Health Check (Every 5 minutes)
```bash
*/5 * * * * /home/pi/tozsde_webapp/scripts/health_check.sh
```

---

## 🔧 Common Commands

### View Flask logs
```bash
sudo journalctl -u tozsde-api.service -f
```

### Restart service
```bash
sudo systemctl restart tozsde-api.service
```

### Check service status
```bash
sudo systemctl status tozsde-api.service
```

### View cron jobs
```bash
crontab -l
```

### Check system health
```bash
df -h /home          # Disk usage
free -h              # Memory
top                  # CPU usage
/opt/vc/bin/vcgencmd measure_temp  # Temperature
```

### Check cron execution logs
```bash
tail -f ~/tozsde_webapp/logs/cron_daily.log
tail -f ~/tozsde_webapp/logs/health_check.log
```

---

## 📚 Related Documentation

- **[../../README.md](../../README.md)** - Project overview
- **[../02_implementation/IMPLEMENTATION_PLAN.md](../02_implementation/IMPLEMENTATION_PLAN.md)** - SPRINT 1-5 specs
- **[../03_testing/FINAL_STATUS_REPORT.md](../03_testing/FINAL_STATUS_REPORT.md)** - Test results (139/139 passing)

---

**Status:** ✅ Production Ready
**Last Updated:** 2026-01-23
**Target:** Raspberry Pi 4/5 (64-bit ARM)
