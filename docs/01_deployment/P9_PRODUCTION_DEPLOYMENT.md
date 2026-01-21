# P9 PRODUCTION DEPLOYMENT GUIDE (No Docker)
**Target Environment:** Synology NAS / Linux server  
**Deployment Method:** Systemd services + Cron jobs + Process supervision  
**Date:** 2026-01-21

---

## OVERVIEW

This guide covers production deployment using **native Linux tools** instead of Docker:
- ✅ **systemd services** for Flask API and background processes
- ✅ **Cron jobs** for scheduled tasks (daily, optimization, audit)
- ✅ **Supervisor** for process monitoring and auto-restart
- ✅ **Virtual environment** isolation (no containers needed)
- ✅ **Systemd timers** as modern cron alternative

**Why not Docker?**
- NAS resources are limited (prefer lightweight)
- Direct access to system is faster
- Simpler troubleshooting and debugging
- No additional container runtime needed
- Native integration with NAS scheduling

---

## ARCHITECTURE

```
Synology NAS / Linux Server
├── /volume1/tozsde_webapp/              ← Main application
│   ├── venv/                            ← Virtual environment
│   ├── app/                             ← Source code
│   ├── logs/                            ← Log files
│   ├── data/                            ← SQLite database
│   └── scheduler/                       ← Cron & systemd scripts
│
├── /etc/systemd/system/                 ← Service definitions (root)
│   ├── tozsde-api.service               ← Flask API service
│   ├── tozsde-daily.service             ← Daily pipeline
│   ├── tozsde-daily.timer               ← Timer (6 AM daily)
│   ├── tozsde-optimize.timer            ← Timer (quarterly)
│   └── tozsde-reliability.timer         ← Timer (weekly)
│
├── /var/log/tozsde/                     ← Systemd journal logs (root)
│   └── (managed by journalctl)
│
└── /usr/local/bin/                      ← Wrapper scripts (root)
    ├── tozsde-start
    ├── tozsde-stop
    ├── tozsde-status
    └── tozsde-shell
```

---

## STEP 1: SYSTEM PREPARATION

### 1.1 Create Application Directory

```bash
# SSH into NAS
ssh admin@192.168.1.100

# Create app directory
sudo mkdir -p /volume1/tozsde_webapp
sudo mkdir -p /volume1/tozsde_webapp/logs
sudo mkdir -p /volume1/tozsde_webapp/data
sudo mkdir -p /volume1/tozsde_webapp/scheduler

# Set permissions
sudo chown -R admin:users /volume1/tozsde_webapp
chmod 755 /volume1/tozsde_webapp
```

### 1.2 Install Python & Dependencies

```bash
# Check Python installed
python3 --version    # Should be 3.8+

# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install python3-pip python3-venv
sudo apt-get install sqlite3
sudo apt-get install supervisor    # Optional, for monitoring
```

### 1.3 Create Virtual Environment

```bash
cd /volume1/tozsde_webapp

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Test installation
python -c "import app; print('OK')"
```

---

## STEP 2: DEPLOY APPLICATION

### 2.1 Clone/Copy Project Files

```bash
# Option A: Git clone
cd /volume1/tozsde_webapp
git clone <repo-url> .

# Option B: SCP from local machine
scp -r . admin@192.168.1.100:/volume1/tozsde_webapp/

# Verify
ls -la /volume1/tozsde_webapp/
```

### 2.2 Initialize Database

```bash
cd /volume1/tozsde_webapp

source venv/bin/activate

# Apply schema
python -m app.scripts.apply_schema

# Verify
sqlite3 app/data/market_data.db ".tables"
# Should show: ohlcv, trades, recommendations, ...
```

### 2.3 Set Environment Variables

```bash
# Create .env file
cat > /volume1/tozsde_webapp/.env << 'EOF'
# Flask
FLASK_ENV=production
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
FLASK_DEBUG=false

# Logging
LOGGING_LEVEL=INFO

# Email
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
NOTIFY_EMAIL=recipient@example.com

# Feature flags
ENABLE_RL=false
ENABLE_FLASK=true
ENABLE_RELIABILITY=true

# Admin
ADMIN_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(16))")
EOF

chmod 600 /volume1/tozsde_webapp/.env
```

---

## STEP 3: CREATE SYSTEMD SERVICES

### 3.1 Flask API Service

Create: `/etc/systemd/system/tozsde-api.service`

```ini
[Unit]
Description=Tozsde Trading AI - Flask API
After=network.target
Wants=tozsde-daily.timer

[Service]
Type=simple
User=admin
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/volume1/tozsde_webapp/.env

# Start Flask with gunicorn
ExecStart=/volume1/tozsde_webapp/venv/bin/gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --timeout 300 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  app.ui.app:app

# Auto-restart on failure
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=5

# Process management
StandardOutput=append:/volume1/tozsde_webapp/logs/flask.log
StandardError=append:/volume1/tozsde_webapp/logs/flask-error.log

[Install]
WantedBy=multi-user.target
```

Enable and test:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tozsde-api.service
sudo systemctl start tozsde-api.service
sudo systemctl status tozsde-api.service

# Check logs
journalctl -u tozsde-api.service -f
```

### 3.2 Daily Pipeline Service

Create: `/etc/systemd/system/tozsde-daily.service`

```ini
[Unit]
Description=Tozsde Trading AI - Daily Pipeline
After=network.target

[Service]
Type=oneshot
User=admin
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/volume1/tozsde_webapp/.env

ExecStart=/volume1/tozsde_webapp/venv/bin/python main.py --mode daily

StandardOutput=append:/volume1/tozsde_webapp/logs/daily-pipeline.log
StandardError=append:/volume1/tozsde_webapp/logs/daily-pipeline-error.log

# No auto-restart for one-time tasks
Restart=no
```

### 3.3 Optimization Service

Create: `/etc/systemd/system/tozsde-optimize.service`

```ini
[Unit]
Description=Tozsde Trading AI - Quarterly Optimization
After=network.target

[Service]
Type=oneshot
User=admin
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/volume1/tozsde_webapp/.env

ExecStart=/volume1/tozsde_webapp/venv/bin/python main.py --mode optimize

# Longer timeout for optimization
TimeoutStartSec=86400

StandardOutput=append:/volume1/tozsde_webapp/logs/optimize.log
StandardError=append:/volume1/tozsde_webapp/logs/optimize-error.log

Restart=no
```

### 3.4 Reliability Service

Create: `/etc/systemd/system/tozsde-reliability.service`

```ini
[Unit]
Description=Tozsde Trading AI - Weekly Reliability Check
After=network.target

[Service]
Type=oneshot
User=admin
WorkingDirectory=/volume1/tozsde_webapp
Environment="PATH=/volume1/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/volume1/tozsde_webapp/.env

ExecStart=/volume1/tozsde_webapp/venv/bin/python main.py --mode reliability

StandardOutput=append:/volume1/tozsde_webapp/logs/reliability.log
StandardError=append:/volume1/tozsde_webapp/logs/reliability-error.log

Restart=no
```

---

## STEP 4: CREATE SYSTEMD TIMERS (Modern Cron)

### 4.1 Daily Timer (6 AM)

Create: `/etc/systemd/system/tozsde-daily.timer`

```ini
[Unit]
Description=Daily Tozsde Pipeline Trigger
Requires=tozsde-daily.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 06:00:00

# Run if system was off at scheduled time
Persistent=true

# Randomize by 5 minutes to avoid load spikes
RandomizedDelaySec=5min

Unit=tozsde-daily.service

[Install]
WantedBy=timers.target
```

### 4.2 Quarterly Optimization Timer (1 AM on 1st of month, every 3 months)

Create: `/etc/systemd/system/tozsde-optimize.timer`

```ini
[Unit]
Description=Quarterly Tozsde Optimization Trigger
Requires=tozsde-optimize.service

[Timer]
# Every 3 months on the 1st at 1 AM
OnCalendar=*-01,04,07,10-01 01:00:00

Persistent=true
RandomizedDelaySec=10min

Unit=tozsde-optimize.service

[Install]
WantedBy=timers.target
```

### 4.3 Weekly Reliability Timer (4 AM on Monday)

Create: `/etc/systemd/system/tozsde-reliability.timer`

```ini
[Unit]
Description=Weekly Tozsde Reliability Check Trigger
Requires=tozsde-reliability.service

[Timer]
# Every Monday at 4 AM
OnCalendar=Mon *-*-* 04:00:00

Persistent=true
RandomizedDelaySec=5min

Unit=tozsde-reliability.service

[Install]
WantedBy=timers.target
```

Enable all timers:
```bash
sudo systemctl daemon-reload

sudo systemctl enable tozsde-daily.timer
sudo systemctl enable tozsde-optimize.timer
sudo systemctl enable tozsde-reliability.timer

sudo systemctl start tozsde-daily.timer
sudo systemctl start tozsde-optimize.timer
sudo systemctl start tozsde-reliability.timer

# Verify timers
sudo systemctl list-timers --all
```

---

## STEP 5: MONITORING & SUPERVISION (Optional: Supervisor)

### 5.1 Install Supervisor

```bash
sudo apt-get install supervisor
```

### 5.2 Configure Supervisor for Flask API

Create: `/etc/supervisor/conf.d/tozsde.conf`

```ini
[program:tozsde-api]
command=/volume1/tozsde_webapp/venv/bin/gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --timeout 300 \
  app.ui.app:app
directory=/volume1/tozsde_webapp
environment=PATH="/volume1/tozsde_webapp/venv/bin"
autostart=true
autorestart=true
startsecs=10
stopasgroup=true
killasgroup=true
stdout_logfile=/volume1/tozsde_webapp/logs/supervisor-api.log
stderr_logfile=/volume1/tozsde_webapp/logs/supervisor-api-error.log
```

Activate:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start tozsde-api
sudo supervisorctl status
```

---

## STEP 6: UTILITY SCRIPTS

### 6.1 Start Script

Create: `/usr/local/bin/tozsde-start`

```bash
#!/bin/bash
set -e

echo "Starting Tozsde Trading AI..."

# Start API service
sudo systemctl start tozsde-api.service

# Start timers
sudo systemctl start tozsde-daily.timer
sudo systemctl start tozsde-optimize.timer
sudo systemctl start tozsde-reliability.timer

echo "✅ Tozsde services started"
echo ""
echo "Check status:"
echo "  sudo systemctl status tozsde-api.service"
echo "  sudo systemctl list-timers --all"
echo ""
echo "View logs:"
echo "  journalctl -u tozsde-api.service -f"
```

### 6.2 Stop Script

Create: `/usr/local/bin/tozsde-stop`

```bash
#!/bin/bash
set -e

echo "Stopping Tozsde Trading AI..."

# Stop API service
sudo systemctl stop tozsde-api.service

# Stop timers (they'll auto-restart at next schedule)
# Usually leave running - only stop if maintenance needed

echo "✅ Tozsde services stopped"
```

### 6.3 Status Script

Create: `/usr/local/bin/tozsde-status`

```bash
#!/bin/bash

echo "Tozsde Trading AI - System Status"
echo "==================================="
echo ""

echo "API Service:"
sudo systemctl status tozsde-api.service --no-pager | head -10

echo ""
echo "Timers:"
sudo systemctl list-timers --all | grep tozsde

echo ""
echo "Recent API Logs:"
sudo journalctl -u tozsde-api.service -n 20 --no-pager

echo ""
echo "Database Status:"
sqlite3 /volume1/tozsde_webapp/app/data/market_data.db \
  "SELECT name FROM sqlite_master WHERE type='table';"
```

### 6.4 Shell Script

Create: `/usr/local/bin/tozsde-shell`

```bash
#!/bin/bash

cd /volume1/tozsde_webapp
source venv/bin/activate
python
```

Make executable:
```bash
sudo chmod +x /usr/local/bin/tozsde-start
sudo chmod +x /usr/local/bin/tozsde-stop
sudo chmod +x /usr/local/bin/tozsde-status
sudo chmod +x /usr/local/bin/tozsde-shell
```

---

## STEP 7: MONITORING & LOGGING

### 7.1 Check Service Logs

```bash
# Real-time Flask logs
journalctl -u tozsde-api.service -f

# Daily pipeline logs
journalctl -u tozsde-daily.service -f

# All tozsde logs
journalctl | grep tozsde
```

### 7.2 Check Timer Execution

```bash
# View all scheduled timers
sudo systemctl list-timers --all

# View last 50 timer runs
sudo systemctl list-timers --all --no-pager | tail -50

# Check specific timer
sudo systemctl list-timers tozsde-daily.timer
```

### 7.3 Log Rotation

Create: `/etc/logrotate.d/tozsde`

```
/volume1/tozsde_webapp/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 admin users
    sharedscripts
    postrotate
        systemctl reload tozsde-api.service > /dev/null 2>&1 || true
    endscript
}
```

---

## STEP 8: DEPLOYMENT VERIFICATION

### 8.1 Startup Checklist

```bash
# ✅ Services running
sudo systemctl status tozsde-api.service

# ✅ Timers scheduled
sudo systemctl list-timers --all

# ✅ Database accessible
sqlite3 /volume1/tozsde_webapp/app/data/market_data.db ".tables"

# ✅ API responding
curl http://localhost:5000/

# ✅ Logs being written
tail -f /volume1/tozsde_webapp/logs/flask.log

# ✅ Flask accessible from browser
# http://nas-ip:5000
```

### 8.2 Test Daily Pipeline Manually

```bash
# Run daily pipeline immediately
sudo systemctl start tozsde-daily.service

# Check logs
journalctl -u tozsde-daily.service -f
```

### 8.3 Test Timer Accuracy

```bash
# Verify next scheduled runs
sudo systemctl list-timers tozsde-daily.timer --no-pager

# Sample output should show:
# NEXT                 LEFT       LAST                 PASSED
# Tue 2026-01-22 06:00 17h left   Mon 2026-01-21 06:00 6h 30min ago
```

---

## STEP 9: OPERATIONAL TASKS

### 9.1 Update Application

```bash
cd /volume1/tozsde_webapp

# Stop services
sudo systemctl stop tozsde-api.service

# Pull latest code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Apply schema migrations
python -m app.scripts.apply_schema

# Restart
sudo systemctl start tozsde-api.service
```

### 9.2 Backup Database

```bash
# Manual backup
cp /volume1/tozsde_webapp/app/data/market_data.db \
   /volume1/tozsde_webapp/backups/market_data_$(date +%Y%m%d).db

# Automated daily backup via cron
cat > /etc/cron.d/tozsde-backup << 'EOF'
# Backup daily at 2 AM
0 2 * * * admin /bin/bash -c 'cp /volume1/tozsde_webapp/app/data/market_data.db /volume1/tozsde_webapp/backups/market_data_$(date +\%Y\%m\%d).db'
EOF
```

### 9.3 Monitor Health

```bash
# Create health check script
cat > /volume1/tozsde_webapp/scheduler/health_check.sh << 'EOF'
#!/bin/bash

# Check API responding
curl -s http://localhost:5000/ > /dev/null || echo "API DOWN"

# Check database
sqlite3 /volume1/tozsde_webapp/app/data/market_data.db "SELECT COUNT(*) FROM ohlcv" || echo "DB ERROR"

# Check disk space
DISK_USAGE=$(df /volume1 | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
  echo "DISK WARNING: $DISK_USAGE%"
fi

# Check systemd health
systemctl is-active tozsde-api.service > /dev/null || echo "SERVICE DOWN"
EOF

chmod +x /volume1/tozsde_webapp/scheduler/health_check.sh

# Run health check every hour
cat > /etc/cron.d/tozsde-healthcheck << 'EOF'
0 * * * * admin /volume1/tozsde_webapp/scheduler/health_check.sh >> /volume1/tozsde_webapp/logs/healthcheck.log 2>&1
EOF
```

---

## TROUBLESHOOTING

### Service Won't Start

```bash
# Check service status
sudo systemctl status tozsde-api.service

# Check logs
journalctl -u tozsde-api.service -n 50 -e

# Check permissions
ls -la /volume1/tozsde_webapp/

# Verify Python in venv
/volume1/tozsde_webapp/venv/bin/python --version

# Test import
/volume1/tozsde_webapp/venv/bin/python -c "import app"
```

### Timer Not Running

```bash
# Check timer status
sudo systemctl status tozsde-daily.timer

# Check timer next run
sudo systemctl list-timers tozsde-daily.timer

# Test timer manually
sudo systemctl start tozsde-daily.service

# Check logs
journalctl -u tozsde-daily.service
```

### Database Locked

```bash
# Check open connections
lsof | grep market_data.db

# Kill stuck process
kill -9 <PID>

# Vacuum database
sqlite3 /volume1/tozsde_webapp/app/data/market_data.db "VACUUM;"
```

---

## SUMMARY: DEPLOYMENT COMPONENTS

| Component | Type | Purpose |
|-----------|------|---------|
| `tozsde-api.service` | systemd | Flask API (always running) |
| `tozsde-daily.timer` | systemd | Daily 6 AM trigger |
| `tozsde-daily.service` | systemd | Daily pipeline execution |
| `tozsde-optimize.timer` | systemd | Quarterly optimization trigger |
| `tozsde-optimize.service` | systemd | Quarterly optimization execution |
| `tozsde-reliability.timer` | systemd | Weekly reliability check trigger |
| `tozsde-reliability.service` | systemd | Weekly reliability execution |
| `supervisor` (optional) | External | Process monitoring & restart |
| `logrotate` | System | Log file rotation |
| `health_check.sh` | Cron | System health monitoring |

---

## ADVANTAGES VS DOCKER

✅ **Lower resource usage** (no container overhead)  
✅ **Faster startup** (direct Python execution)  
✅ **Simpler debugging** (direct OS-level access)  
✅ **Native NAS integration** (works with DSM)  
✅ **No learning curve** (standard Linux tools)  
✅ **Better performance** (no container IPC overhead)  
✅ **Direct systemd integration** (native timers)  
✅ **Smaller attack surface** (fewer layers)

---

**Status:** ✅ Production-ready deployment without Docker

**Next:** Deploy to Synology NAS following these steps

