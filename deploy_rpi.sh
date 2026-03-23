#!/bin/bash
################################################################################
# ToZsDE Raspberry Pi 4/5 Automated Deployment Script
# ONE-CLICK SETUP: System deps -> Python env -> Flask service -> Cron jobs
# 
# Usage: bash deploy_rpi.sh
# 
# What it does:
#   v Installs system dependencies (Python 3.11, pip, curl, etc.)
#   v Creates Python virtual environment
#   v Installs application requirements
#   v Creates systemd service for Flask API (auto-restart)
#   v Schedules 3 cron jobs (daily, weekly, monthly)
#   v Sets up health checks (every 5 minutes)
#   v Configures log rotation
#   v Starts Flask service
#   v Verifies everything works
#
# Total runtime: 10-15 minutes
# Requires: Raspberry Pi OS Lite 64-bit, sudo access
################################################################################

set -e  # Exit immediately on any error

# Configuration
APP_DIR="/home/pi/tozsde_webapp"
VENV_DIR="$APP_DIR/venv"
LOGS_DIR="$APP_DIR/logs"
SCRIPTS_DIR="$APP_DIR/app/scripts"
USER="pi"
GROUP="pi"

# Optional RL training/cron configuration (opt-in)
TRAIN_RL_ON_DEPLOY="${TRAIN_RL_ON_DEPLOY:-true}"
TRAIN_RL_FORCE="${TRAIN_RL_FORCE:-false}"
TRAIN_RL_TICKER="${TRAIN_RL_TICKER:-VOO}"
TRAIN_RL_REWARD_STRATEGY="${TRAIN_RL_REWARD_STRATEGY:-portfolio_value}"
ENABLE_RL_CRON="${ENABLE_RL_CRON:-false}"
RL_CRON_MODE="${RL_CRON_MODE:-minimal}"
RL_CRON_TICKER="${RL_CRON_TICKER:-VOO}"
RL_CRON_REWARD_STRATEGY="${RL_CRON_REWARD_STRATEGY:-portfolio_value}"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Output functions
print_status() {
    echo -e "${GREEN}[v]${NC} $1"
}

print_error() {
    echo -e "${RED}[x]${NC} $1"
}

print_header() {
    echo -e "\n${YELLOW}${NC}"
    echo -e "${YELLOW}${NC} $1"
    echo -e "${YELLOW}${NC}\n"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================
print_header "PRE-FLIGHT CHECKS"

# Check if running on Pi (optional, just a warning)
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    print_info "Warning: Not detected as Raspberry Pi, but continuing..."
fi

# Check if app directory exists
if [ ! -d "$APP_DIR" ]; then
    print_error "Application directory not found: $APP_DIR"
    echo "Please clone the repository to $APP_DIR first:"
    echo "  git clone <repo-url> $APP_DIR"
    exit 1
fi

print_status "App directory found: $APP_DIR"

# Check if requirements.txt exists
if [ ! -f "$APP_DIR/requirements.txt" ]; then
    print_error "requirements.txt not found in $APP_DIR"
    exit 1
fi

print_status "requirements.txt found"

# Ensure admin key exists for admin endpoints
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    if command -v openssl >/dev/null 2>&1; then
        ADMIN_API_KEY=$(openssl rand -hex 16)
    else
        ADMIN_API_KEY=$(date +%s%N)
    fi
    cat > "$ENV_FILE" << EOF
ADMIN_API_KEY=$ADMIN_API_KEY
EOF
    chmod 600 "$ENV_FILE"
    print_status "Created .env with ADMIN_API_KEY"
else
    print_info ".env already exists; keeping existing ADMIN_API_KEY"
fi

# ==============================================================================
# STEP 1: SYSTEM DEPENDENCIES
# ==============================================================================
print_header "STEP 1: Installing System Dependencies"

print_info "Updating package list..."
sudo apt-get update -qq

print_info "Installing Python 3.11 and build tools..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libatlas-base-dev \
    libjasper-dev \
    libtiff-dev \
    libtiffxx-dev \
    libjasper1 \
    libjasper-dev \
    logrotate \
    cron \
    > /dev/null 2>&1

print_status "System dependencies installed"

# ==============================================================================
# STEP 2: PREPARE DIRECTORIES
# ==============================================================================
print_header "STEP 2: Preparing Directories"

mkdir -p "$LOGS_DIR"
mkdir -p "$APP_DIR/config/systemd"
mkdir -p "$SCRIPTS_DIR"

# Set permissions
sudo chown -R $USER:$GROUP "$APP_DIR" 2>/dev/null || true
chmod 755 "$APP_DIR"
chmod 755 "$LOGS_DIR"
chmod 755 "$SCRIPTS_DIR"

print_status "Directories prepared: $APP_DIR"
print_status "Logs directory: $LOGS_DIR"
print_status "Scripts directory: $SCRIPTS_DIR"

# ==============================================================================
# STEP 3: PYTHON VIRTUAL ENVIRONMENT
# ==============================================================================
print_header "STEP 3: Setting Up Python Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists, using existing..."
else
    print_info "Creating virtual environment with Python 3.11..."
    python3.11 -m venv "$VENV_DIR"
    print_status "Virtual environment created"
fi

# Activate venv for pip operations
source "$VENV_DIR/bin/activate"

print_info "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q

print_info "Installing Python dependencies from requirements.txt..."
pip install -r "$APP_DIR/requirements.txt" -q

# Test import
print_info "Testing application import..."
if python -c "import app" 2>/dev/null; then
    print_status "Application imports successfully"
else
    print_error "Application import failed - check requirements.txt"
    exit 1
fi

# ==============================================================================
# STEP 3B: OPTIONAL RL TRAINING (ON DEPLOY)
# ==============================================================================
if [ "$TRAIN_RL_ON_DEPLOY" = "true" ]; then
    print_header "STEP 3B: Optional RL Training"
    print_info "RL training check enabled (ticker=${TRAIN_RL_TICKER}, reward=${TRAIN_RL_REWARD_STRATEGY})"

    TRAIN_REQUIRED="true"
    if [ "$TRAIN_RL_FORCE" != "true" ]; then
        set +e
        python "$APP_DIR/scripts/training_fingerprint.py" --check > /tmp/training_check.log 2>&1
        CHECK_EXIT=$?
        set -e
        if [ $CHECK_EXIT -eq 0 ]; then
            TRAIN_REQUIRED="false"
            print_info "Training fingerprint OK; skipping RL training"
        else
            print_info "Training required (fingerprint exit=$CHECK_EXIT)"
        fi
    else
        print_info "TRAIN_RL_FORCE=true; training will run regardless of fingerprint"
    fi

    if [ "$TRAIN_REQUIRED" = "true" ]; then
        print_info "Running walk-forward (GA) before RL training..."
        ENABLE_RL=true python "$APP_DIR/main.py" walk-forward "$TRAIN_RL_TICKER"

        print_info "Training DQN..."
        ENABLE_RL=true python "$APP_DIR/scripts/train_rl.py" \
            --ticker "$TRAIN_RL_TICKER" \
            --model-type DQN \
            --reward-strategy "$TRAIN_RL_REWARD_STRATEGY"

        print_info "Training PPO..."
        ENABLE_RL=true python "$APP_DIR/scripts/train_rl.py" \
            --ticker "$TRAIN_RL_TICKER" \
            --model-type PPO \
            --reward-strategy "$TRAIN_RL_REWARD_STRATEGY"

        python "$APP_DIR/scripts/training_fingerprint.py" --write
        print_status "Optional RL training completed"
    fi
fi

# ==============================================================================
# STEP 4: SYSTEMD SERVICE (Flask API)
# ==============================================================================
print_header "STEP 4: Creating systemd Service (Flask API)"

cat > "$APP_DIR/config/systemd/tozsde-api.service" << 'EOF'
[Unit]
Description=ToZsDE Trading API - Raspberry Pi
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/tozsde_webapp
Environment="PATH=/home/pi/tozsde_webapp/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/pi/tozsde_webapp/venv/bin/python -m app.ui.app
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
StandardOutputMaxMem=16M

[Install]
WantedBy=multi-user.target
EOF

# Install service
sudo cp "$APP_DIR/config/systemd/tozsde-api.service" /etc/systemd/system/
sudo systemctl daemon-reload

# Enable service to auto-start on boot
sudo systemctl enable tozsde-api.service > /dev/null 2>&1

print_status "systemd service created: tozsde-api.service"
print_status "Service will auto-start on reboot"

# ==============================================================================
# STEP 5: CRON JOBS (Scheduled Tasks)
# ==============================================================================
print_header "STEP 5: Setting Up Cron Jobs"

# Create cron entries (idempotent - removes old, adds new)
# Remove any existing tozsde cron entries
(crontab -l 2>/dev/null | grep -v "tozsde_webapp" | grep -v "^$" | grep -v "^#.*tozsde") | crontab - 2>/dev/null || true

# Helper function to add cron entry
add_cron_entry() {
    local schedule="$1"
    local command="$2"
    (crontab -l 2>/dev/null; echo "$schedule $command") | crontab -
}

# Daily pipeline - 6:00 AM
DAILY_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.infrastructure.cron_tasks --daily >> /home/pi/tozsde_webapp/logs/cron_daily.log 2>&1"
add_cron_entry "0 6 * * *" "$DAILY_CMD"

# Weekly audit - Monday 4:00 AM
WEEKLY_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.infrastructure.cron_tasks --weekly >> /home/pi/tozsde_webapp/logs/cron_weekly.log 2>&1"
add_cron_entry "0 4 * * 1" "$WEEKLY_CMD"

# Monthly optimization - 1st of month 1:00 AM
MONTHLY_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && python -m app.infrastructure.cron_tasks --monthly >> /home/pi/tozsde_webapp/logs/cron_monthly.log 2>&1"
add_cron_entry "0 1 1 * *" "$MONTHLY_CMD"

# Optional RL training cron (opt-in)
if [ "$ENABLE_RL_CRON" = "true" ]; then
    if [ "$RL_CRON_MODE" = "full" ]; then
        RL_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && ENABLE_RL=true python main.py monthly >> /home/pi/tozsde_webapp/logs/cron_monthly_rl.log 2>&1"
    else
        RL_CMD="source /home/pi/tozsde_webapp/venv/bin/activate && cd /home/pi/tozsde_webapp && ENABLE_RL=true python main.py walk-forward $RL_CRON_TICKER && ENABLE_RL=true python scripts/train_rl.py --ticker $RL_CRON_TICKER --model-type DQN --reward-strategy $RL_CRON_REWARD_STRATEGY && ENABLE_RL=true python scripts/train_rl.py --ticker $RL_CRON_TICKER --model-type PPO --reward-strategy $RL_CRON_REWARD_STRATEGY >> /home/pi/tozsde_webapp/logs/cron_monthly_rl.log 2>&1"
    fi
    add_cron_entry "0 2 1 * *" "$RL_CMD"
    print_status "Optional RL training cron configured: 1st of month, 2:00 AM"
fi

# Health check - every 5 minutes
HEALTH_CHECK="$SCRIPTS_DIR/health_check.sh"
add_cron_entry "*/5 * * * *" "$HEALTH_CHECK >> /dev/null 2>&1"

print_status "Cron jobs configured:"
echo "  - Daily pipeline:       6:00 AM (every day)"
echo "  - Weekly audit:         4:00 AM (every Monday)"
echo "  - Monthly optimization: 1:00 AM (1st of month)"
echo "  - Health check:         Every 5 minutes"

# ==============================================================================
# STEP 6: HEALTH CHECK SCRIPT
# ==============================================================================
print_header "STEP 6: Creating Health Check Script"

cat > "$SCRIPTS_DIR/health_check.sh" << 'EOF'
#!/bin/bash
# Health check for Flask API
# Runs every 5 minutes via cron

HEALTH_URL="http://localhost:5000/admin/health"
TIMEOUT=5
LOG_FILE="/home/pi/tozsde_webapp/logs/health_check.log"
MAX_LOG_SIZE=10485760  # 10MB
ENV_FILE="/home/pi/tozsde_webapp/.env"
ADMIN_API_KEY=""
HEADER=()

if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

if [ -n "$ADMIN_API_KEY" ]; then
    HEADER=(-H "X-Admin-Key: $ADMIN_API_KEY")
fi

# Create log file if it doesn't exist
touch "$LOG_FILE"

# Rotate log if too large
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
    mv "$LOG_FILE" "$LOG_FILE.$(date +%s)"
    gzip "$LOG_FILE".* 2>/dev/null || true
fi

# Check if API is responding
if timeout $TIMEOUT curl -f -s "${HEADER[@]}" "$HEALTH_URL" > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] v OK" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] x FAILED - attempting restart" >> "$LOG_FILE"
    
    # Try to restart service
    systemctl restart tozsde-api.service
    sleep 3
    
    # Verify recovery
    if timeout $TIMEOUT curl -f -s "${HEADER[@]}" "$HEALTH_URL" > /dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] v Service recovered" >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] x Service still down - see logs" >> "$LOG_FILE"
    fi
fi

# Check disk space
DISK_USAGE=$(df -h /home | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Disk usage: ${DISK_USAGE}% (cleanup logs)" >> "$LOG_FILE"
fi
EOF

chmod +x "$SCRIPTS_DIR/health_check.sh"
print_status "Health check script created: $SCRIPTS_DIR/health_check.sh"

# ==============================================================================
# STEP 7: LOG ROTATION
# ==============================================================================
print_header "STEP 7: Configuring Log Rotation"

cat > /tmp/tozsde_logrotate << 'EOF'
/home/pi/tozsde_webapp/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 pi pi
    postrotate
        systemctl reload tozsde-api.service > /dev/null 2>&1 || true
    endscript
}
EOF

sudo mv /tmp/tozsde_logrotate /etc/logrotate.d/tozsde
sudo chmod 644 /etc/logrotate.d/tozsde

print_status "Log rotation configured:"
echo "  - Frequency: Daily"
echo "  - Retention: 7 days"
echo "  - Compression: Enabled"
echo "  - Location: /etc/logrotate.d/tozsde"

# ==============================================================================
# STEP 8: START SERVICES
# ==============================================================================
print_header "STEP 8: Starting Services"

print_info "Starting Flask API service..."
sudo systemctl start tozsde-api.service
sleep 2

if sudo systemctl is-active --quiet tozsde-api.service; then
    print_status "Flask API service started successfully"
else
    print_error "Failed to start Flask API service"
    echo -e "\n${YELLOW}Recent logs:${NC}"
    sudo journalctl -u tozsde-api.service -n 20 --no-pager
    exit 1
fi

# ==============================================================================
# STEP 9: VERIFICATION & TESTING
# ==============================================================================
print_header "STEP 9: Verification & Testing"

print_info "Waiting for API to be ready..."
ADMIN_API_KEY=""
HEALTH_HEADER=()
if [ -f "$APP_DIR/.env" ]; then
    set -a
    . "$APP_DIR/.env"
    set +a
fi
if [ -n "$ADMIN_API_KEY" ]; then
    HEALTH_HEADER=(-H "X-Admin-Key: $ADMIN_API_KEY")
fi
API_READY=0
for i in {1..20}; do
    if curl -f -s "${HEALTH_HEADER[@]}" http://localhost:5000/admin/health > /dev/null 2>&1; then
        API_READY=1
        break
    fi
    if [ $((i % 5)) -eq 0 ]; then
        print_info "Still waiting (attempt $i/20)..."
    fi
    sleep 1
done

if [ $API_READY -eq 1 ]; then
    print_status "v API responding on port 5000"
    
    # Test health endpoint
    HEALTH=$(curl -s "${HEALTH_HEADER[@]}" http://localhost:5000/admin/health)
    if echo "$HEALTH" | grep -q "healthy"; then
        print_status "v Health endpoint working"
    fi
else
    print_error "API not responding after 20 seconds"
    echo "Check logs: sudo journalctl -u tozsde-api.service -n 30"
    exit 1
fi

# Show cron jobs
print_info "Scheduled cron jobs:"
echo "---"
crontab -l | grep -v "^#" | grep -v "^$"
echo "---"

# Show service status
print_info "Service status:"
sudo systemctl status tozsde-api.service --no-pager | head -10

# ==============================================================================
# FINAL STATUS
# ==============================================================================
print_header "DEPLOYMENT COMPLETE! "

echo -e "${GREEN}v System dependencies installed${NC}"
echo -e "${GREEN}v Python 3.11 environment created${NC}"
echo -e "${GREEN}v Application requirements installed${NC}"
echo -e "${GREEN}v Flask API service running${NC}"
echo -e "${GREEN}v Cron jobs scheduled${NC}"
echo -e "${GREEN}v Health checks active${NC}"
echo -e "${GREEN}v Log rotation configured${NC}"

echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo "  1. Test API:"
echo "     curl http://raspberrypi.local:5000/admin/health -H \"X-Admin-Key: <key>\""
echo ""
echo "  2. View logs:"
echo "     sudo journalctl -u tozsde-api.service -f"
echo ""
echo "  3. Check cron jobs:"
echo "     crontab -l"
echo ""
echo "  4. Monitor system:"
echo "     df -h /home     # Disk space"
echo "     free -h         # Memory"
echo "     top             # CPU usage"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  - First daily run: Tomorrow at 6:00 AM"
echo "  - Check logs: /home/pi/tozsde_webapp/logs/"
echo "  - SSH access: ssh pi@raspberrypi.local"
echo "  - To restart service: sudo systemctl restart tozsde-api.service"
echo ""
echo -e "${GREEN}Happy trading! ${NC}"
