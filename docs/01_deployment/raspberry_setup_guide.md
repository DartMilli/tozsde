
# Raspberry Pi Setup Guide (Full Version)

## 1. Overview
This guide provides full installation and configuration instructions for deploying the Trading AI application on a Raspberry Pi from unboxing to production-ready scheduled automation.

## 2. Hardware Preparation
- Raspberry Pi (4/5 recommended)
- microSD card (16GB+)
- Power supply
- HDMI cable + monitor
- Keyboard
- Ethernet or Wi-Fi

## 3. Install Raspberry Pi OS
1. Download Raspberry Pi Imager.
2. Choose Raspberry Pi OS (64-bit).
3. Write to SD.
4. Enable SSH, set username, Wi-Fi, locale.

## 4. First Boot & SSH
Use `ssh pi@<IP>` and run:
```
sudo apt update && sudo apt upgrade -y
```

## 5. Project Setup
```
mkdir -p ~/trading_ai
cd ~/trading_ai
```
Clone repo or upload files.

## 6. Python Environment
```
sudo apt install python3 python3-pip python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Environment Variables
Add a `.env` file:
```
EMAIL_ADDRESS=example@mail.com
EMAIL_PASSWORD=1234
SECRET_KEY=supersecret
```

## 8. Test Application
```
python main.py runserver
```
Or for FastAPI:
```
uvicorn ui.app:app --host 0.0.0.0 --port 8000
```

## 9. System Service
Use the service config file (see raspberry_service_config.md).
Enable:
```
sudo systemctl daemon-reload
sudo systemctl enable trading_ai
sudo systemctl start trading_ai
```

## 10. Cron Jobs
Edit:
```
crontab -e
```
Add:
```
0 6 * * * /home/pi/trading_ai/venv/bin/python /home/pi/trading_ai/scheduler/cron_job.py --mode daily
```

## 11. Schema Initialization
```
python scripts/apply_schema.py
```

## 12. Monitoring
- `systemctl status trading_ai`
- `journalctl -u trading_ai -f`
- `df -h`
- `vcgencmd measure_temp`

## 13. Summary
System is ready for automated daily trading recommendations.
