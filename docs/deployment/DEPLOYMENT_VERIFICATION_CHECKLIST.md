# Deployment Verification Checklist (Raspberry Pi)

**Status:** Hardware pending

## 1) Pre-Deployment (Hardware + OS)
- [ ] Raspberry Pi 4/5 available and powered
- [ ] Raspberry Pi OS Lite 64-bit flashed to SD card
- [ ] SSH enabled
- [ ] Network connectivity confirmed
- [ ] `git` installed and working

## 2) Deployment Script Execution
- [ ] Repo cloned to /home/pi/tozsde_webapp
- [ ] `bash deploy_rpi.sh` completed without errors
- [ ] `venv/` created with Python 3.11
- [ ] `requirements.txt` installed successfully

## 3) Services & Cron Jobs
- [ ] `tozsde-api.service` enabled and active
- [ ] Health check script installed and executable
- [ ] Cron jobs listed (`crontab -l`)
- [ ] Daily, weekly, monthly cron entries present

## 4) Post-Deployment Verification
- [ ] API responds: `curl http://localhost:5000/api/health`
- [ ] Systemd logs show clean startup
- [ ] Log rotation configured (`/etc/logrotate.d/tozsde`)
- [ ] `logs/` directory is writable

## 5) Health Check Validation
- [ ] Wait 5 minutes and confirm health check log updates
- [ ] Simulate API down; verify auto-restart
- [ ] Disk usage threshold logging works

## 6) First Daily Pipeline Dry-Run
- [ ] Manual run: `python -m app.infrastructure.cron_tasks --daily`
- [ ] Backup created in `backups/`
- [ ] Cron execution logged to `logs/cron_executions.jsonl`

## Notes
- If any step fails, record the exact error and timestamp in logs.
- Hardware-dependent steps remain pending until the Pi arrives.
