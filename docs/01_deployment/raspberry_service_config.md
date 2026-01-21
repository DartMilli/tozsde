
# Raspberry Pi Service Configuration

## 1. Overview
This service ensures the Trading AI application runs continuously using systemd.

## 2. Create Service File
```
sudo nano /etc/systemd/system/trading_ai.service
```

## 3. Service File Contents
```
[Unit]
Description=Trading AI Service
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/trading_ai
ExecStart=/home/pi/trading_ai/venv/bin/python /home/pi/trading_ai/main.py runserver
Restart=on-failure
EnvironmentFile=/home/pi/trading_ai/.env

[Install]
WantedBy=multi-user.target
```

## 4. Enable Service
```
sudo systemctl daemon-reload
sudo systemctl enable trading_ai
sudo systemctl start trading_ai
```

## 5. Logs & Debugging
```
journalctl -u trading_ai -f
systemctl status trading_ai
```
