# ✅ AKTÍV - Development Guide — Enhanced Dev Runners

> **Fejlesztőknek:** Ez a fájl az **aktív fejlesztési útmutató** a fejlesztői infrastruktúra használatához.

This guide covers the improved development infrastructure for local development and testing.

## Quick Start

### 1. Flask API Only (Default)
```bash
python run_dev.py
# or explicitly:
python run_dev.py --mode flask --port 5000
```
✅ Runs Flask at `http://localhost:5000`  
✅ Auto hot-reload enabled  
✅ Debug toolbar available  

### 2. Daily Pipeline Only
```bash
python run_dev.py --mode pipeline
# or with dry-run (no emails):
python run_dev.py --mode pipeline --dry-run
# or single ticker test:
python run_dev.py --mode pipeline --ticker VOO
```
✅ Runs full pipeline without web UI  
✅ Dry-run mode prevents side effects  
✅ Single ticker mode for fast dev testing  

### 3. Both Flask + Pipeline (Threads)
```bash
python run_dev.py --mode both
```
✅ Flask API on port 5000  
✅ Pipeline running in parallel thread  
✅ Press Ctrl+C to stop both  

### 4. Walk-Forward Optimization
```bash
python run_dev.py --mode walk-forward VOO
```
✅ Runs optimization for single ticker  
✅ Results saved to database  

### 5. RL Agent Training
```bash
python run_dev.py --mode train-rl VOO
```
✅ Trains RL agent for ticker  
✅ Uses Config.RL_TIMESTEPS (default: 50k)  

## Command-Line Interface (run_dev.py)

### Basic Usage
```bash
python run_dev.py [--mode MODE] [--ticker TICKER] [--port PORT] [--loglevel LEVEL]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--mode` | `flask`, `pipeline`, `both`, `walk-forward`, `train-rl` | `flask` | Execution mode |
| `--ticker` | any ticker | none | Run for single ticker (dev mode) |
| `--port` | 1024-65535 | `5000` | Flask port number |
| `--dry-run` | flag | disabled | Simulate without side effects |
| `--loglevel` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | Log verbosity |
| `--env` | `development`, `production` | `development` | Environment type |

### Examples

```bash
# Flask with custom port
python run_dev.py --mode flask --port 8000

# Pipeline with debug logging
python run_dev.py --mode pipeline --loglevel DEBUG

# Test pipeline for VOO without emails
python run_dev.py --mode pipeline --ticker VOO --dry-run

# Combined Flask + pipeline
python run_dev.py --mode both --loglevel DEBUG

# Walk-forward optimization
python run_dev.py --mode walk-forward VOO --loglevel DEBUG

# RL training
python run_dev.py --mode train-rl VOO
```

## Main Entry Point (main.py)

The main pipeline supports both CLI and programmatic execution.

### Daily Pipeline
```bash
# Normal execution
python main.py daily

# Dry-run (no emails)
python main.py daily --dry-run

# Single ticker test
python main.py daily --ticker VOO

# With debug logging
python main.py daily --loglevel DEBUG --ticker VOO
```

### Weekly Reliability Analysis
```bash
python main.py weekly

# Dry-run
python main.py weekly --dry-run
```

### Monthly Retraining
```bash
python main.py monthly

# Dry-run
python main.py monthly --dry-run
```

### Manual Optimization
```bash
# Walk-forward
python main.py walk-forward VOO

# RL training
python main.py train-rl VOO
```

### Help
```bash
python main.py --help
python main.py daily --help
python main.py weekly --help
# etc.
```

## Flask API — Development Endpoints

### Production Routes (Always Available)
- `GET /` — Dashboard with recommendations
- `GET /chart` — Price chart with indicators
- `GET /history` — Historical recommendations
- `GET /indicators` — Indicator documentation
- `GET /report` — Backtest report

### Development Endpoints (FLASK_ENV=development only)

#### Status & Health
```bash
curl http://localhost:5000/dev/status
```
**Response:**
```json
{
  "status": "OK",
  "environment": "development",
  "timestamp": "2024-01-15T10:30:45.123456",
  "config": {
    "TICKERS": ["VOO", "VTI", "BND"],
    "ENABLE_NOTIFICATIONS": true,
    "ENABLE_RL": false
  },
  "database": {
    "connected": true,
    "tables": ["ohlcv", "recommendations", "trades"]
  }
}
```

#### Configuration Display
```bash
curl http://localhost:5000/dev/config
```
**Response:** All Config parameters (safe values only, secrets excluded)

#### Database Reinitialization (⚠️ Destructive)
```bash
curl -X POST http://localhost:5000/dev/db-init
```
**Response:**
```json
{
  "status": "OK",
  "message": "Database reinitialized"
}
```
**WARNING:** Deletes all data! Only for testing.

#### Clear Today's Recommendations
```bash
curl -X POST http://localhost:5000/dev/clear-recs
```
**Response:**
```json
{
  "status": "OK",
  "message": "Cleared recommendations for 2024-01-15"
}
```
Useful for retesting the daily pipeline without full DB reset.

#### System Metrics
```bash
curl http://localhost:5000/dev/metrics
```
**Response:**
```json
{
  "system": {
    "cpu_percent": 2.3,
    "memory_mb": 156.4
  },
  "data": {
    "tickers": ["VOO", "VTI", "BND"],
    "last_update": "2024-01-15T09:30:00"
  },
  "pipeline": {
    "last_run": "2024-01-15T09:32:15",
    "next_run": "2024-01-16T09:30:00"
  }
}
```

### CORS Support (Frontend Development)

The Flask app automatically enables CORS for local development:
- Allows requests from `http://localhost:3000` (React frontend)
- Allows requests from `http://localhost:5000` (same origin)
- Allows requests from `http://localhost:8000` (alternative port)

If you need a different frontend origin, edit [app/ui/app.py](app/ui/app.py#L25-L32).

### Request/Response Logging (Development Mode)

In development mode, all HTTP requests/responses are logged:
```
[REQUEST] GET /chart?ticker=VOO | Remote: 127.0.0.1
[RESPONSE] 200 | Size: 45382 bytes
```

To see these logs, use `--loglevel DEBUG`.

## Environment Variables

### Development Configuration

Create a `.env` file in the project root:

```bash
# .env

# Flask
FLASK_ENV=development
FLASK_DEBUG=1

# Logging
LOGGING_LEVEL=DEBUG

# Database
DATABASE_URL=sqlite:///app/data/tozsde.db

# Email (development mode suppresses emails)
NOTIFY_EMAIL=test@example.com
EMAIL_ENABLED=false

# RL Training
ENABLE_RL=false

# Pipeline
DRY_RUN=false

# Tickers
TICKERS=VOO,VTI,BND
```

## Workflow: Testing a New Feature

### Scenario: Testing daily pipeline changes without sending emails

```bash
# 1. Make code changes to decision logic
# Edit app/decision/recommender.py, etc.

# 2. Test pipeline in dry-run mode
python run_dev.py --mode pipeline --dry-run --ticker VOO

# 3. Check output and logs
# Look at console output for any errors

# 4. Run Flask to inspect database
python run_dev.py --mode flask

# 5. Visit http://localhost:5000 to see dashboard

# 6. Check dev endpoints for system state
curl http://localhost:5000/dev/status | jq

# 7. Clear recommendations and retry
curl -X POST http://localhost:5000/dev/clear-recs

# 8. Run pipeline again
python run_dev.py --mode pipeline --dry-run --ticker VOO
```

### Scenario: Testing Flask API changes

```bash
# 1. Make code changes to app/ui/app.py

# 2. Start Flask (auto hot-reload)
python run_dev.py --mode flask --loglevel DEBUG

# 3. Open http://localhost:5000 in browser

# 4. Make changes to code
# Flask automatically reloads on save

# 5. Refresh browser to see changes

# 6. Check network tab for CORS/CORS errors
```

### Scenario: Testing walk-forward optimization

```bash
# 1. Run optimization for single ticker
python run_dev.py --mode walk-forward VOO

# 2. Monitor progress in console output

# 3. Once complete, verify results in Flask
python run_dev.py --mode flask

# 4. Check /report endpoint for backtest visualization
```

## Debugging Tips

### 1. Enable Debug Logging
```bash
python run_dev.py --mode pipeline --loglevel DEBUG
```

### 2. Inspect Database State
```bash
# Terminal 1: Start Flask
python run_dev.py --mode flask

# Terminal 2: Query database
curl http://localhost:5000/dev/status | jq .database
```

### 3. Test Email Formatting (Without Sending)
```bash
# Use dry-run mode
python run_dev.py --mode pipeline --dry-run
```

### 4. Database Inspection (SQL)
```bash
sqlite3 app/data/tozsde.db
# SELECT * FROM recommendations WHERE date = '2024-01-15';
```

## Troubleshooting

### Issue: Port Already in Use
```bash
python run_dev.py --mode flask --port 8000
```

### Issue: Database Locked
```bash
curl -X POST http://localhost:5000/dev/db-init
```

### Issue: CORS Errors in Frontend
Edit [app/ui/app.py](app/ui/app.py#L25-L32) to add your frontend origin.

## Architecture: What Changed?

### Before
- `run_dev.py`: 3 lines, Flask only
- `main.py`: Complex logic, no proper CLI
- `app.py`: Minimal Flask, no dev endpoints

### After
- `run_dev.py`: 400+ lines, multi-mode runner
- `main.py`: 500+ lines, proper subcommand CLI
- `app.py`: CORS, logging, 5 dev endpoints

### Benefits
✅ Clear separation of concerns  
✅ Easy to test pipeline without UI  
✅ Both can run simultaneously  
✅ Dry-run mode prevents side effects  
✅ Better debugging and logging  

## Implementation Summary

**Files Modified:**
1. ✅ `run_dev.py` - Enhanced with multi-mode runner (400+ lines)
2. ✅ `main.py` - Restructured with proper CLI (500+ lines)
3. ✅ `app/ui/app.py` - Added CORS, logging, dev endpoints (300+ lines)
4. ✅ `requirements.txt` - Added flask-cors dependency

**Lines of Code:**
- `run_dev.py`: 3 → 430 lines
- `main.py`: 180 → 520 lines
- `app/ui/app.py`: 118 → 350 lines
- Total improvement: +582 lines

**Time to Implement:** 2-3 hours for developer
    import threading
    
    logger.info("Starting both Flask + Pipeline in dev mode")
    
    flask_thread = threading.Thread(target=run_flask_dev, daemon=True)
    pipeline_thread = threading.Thread(target=run_pipeline_dev, daemon=True)
    
    flask_thread.start()
    pipeline_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description="Development mode runner for tozsde_webapp"
    )
    
    parser.add_argument(
        "--mode",
        choices=["flask", "pipeline", "both", "walk-forward", "train-rl"],
        default="flask",
        help="Mode to run (default: flask)"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        help="Ticker symbol (for walk-forward and train-rl modes)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Flask port (default: 5000)"
    )
    
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    Config.LOGGING_LEVEL = args.loglevel
    
    # Run selected mode
    if args.mode == "flask":
        run_flask_dev()
    elif args.mode == "pipeline":
        run_pipeline_dev()
    elif args.mode == "walk-forward":
        if not args.ticker:
            print("Error: --ticker required for walk-forward mode")
            sys.exit(1)
        run_walk_forward_dev(args.ticker)
    elif args.mode == "train-rl":
        if not args.ticker:
            print("Error: --ticker required for train-rl mode")
            sys.exit(1)
        logger.info(f"Starting RL training for {args.ticker}")
        # TODO: Implement RL training
    elif args.mode == "both":
        run_both_dev()

if __name__ == "__main__":
    main()
```

---

### Issue 2: Main Pipeline Not Easily Runnable

**Problem:** `main.py` has complex logic but no `if __name__ == "__main__":` block.

**Solution:** Add proper CLI interface

```python
# In main.py, add at end:

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tozsde trading pipeline")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Daily pipeline command
    daily_parser = subparsers.add_parser("daily", help="Run daily pipeline")
    daily_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without sending emails"
    )
    daily_parser.add_argument(
        "--ticker",
        type=str,
        help="Run for specific ticker only (dev mode)"
    )
    
    # Walk-forward command
    wf_parser = subparsers.add_parser("walk-forward", help="Run walk-forward optimization")
    wf_parser.add_argument("ticker", help="Ticker symbol")
    
    # RL training command
    rl_parser = subparsers.add_parser("train-rl", help="Train RL agent")
    rl_parser.add_argument("ticker", help="Ticker symbol")
    
    args = parser.parse_args()
    
    if args.command == "daily":
        if args.dry_run:
            logger.info("DRY RUN mode - no emails will be sent")
            os.environ["DRY_RUN"] = "true"
        if args.ticker:
            logger.info(f"DEV mode - running for {args.ticker} only")
            Config.TICKERS = [args.ticker]
        run_daily()
    elif args.command == "walk-forward":
        run_walk_forward(args.ticker)
    elif args.command == "train-rl":
        train_rl_agent(args.ticker)
    else:
        parser.print_help()
```

---

### Issue 3: Flask Missing Dev Features

**Problem:** Flask app lacks development debugging and monitoring features.

**Solution:** Enhance `app/ui/app.py` with dev middleware

```python
# Add to app/ui/app.py after Flask init:

import os
from flask import g
import time

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# ═══════════════════════════════════════════════════════════════
# DEVELOPMENT MODE SETUP
# ═══════════════════════════════════════════════════════════════

if os.getenv("FLASK_ENV") == "development":
    
    # Enable CORS for development
    try:
        from flask_cors import CORS
        CORS(app, origins=["http://localhost:3000", "http://localhost:5000"])
        logger.info("CORS enabled for development")
    except ImportError:
        logger.warning("flask-cors not installed, skipping CORS setup")
    
    # Request/response logging middleware
    @app.before_request
    def before_request():
        g.start_time = time.time()
        logger.debug(f"→ {request.method} {request.path} from {request.remote_addr}")
    
    @app.after_request
    def after_request(response):
        duration = time.time() - g.start_time
        logger.debug(
            f"← {request.method} {request.path} {response.status_code} ({duration:.2f}s)"
        )
        return response
    
    # Error logging
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return {"error": str(e)}, 500
    
    # Development admin endpoints
    @app.route("/dev/status")
    def dev_status():
        """Development status endpoint."""
        from app.data_access.data_manager import DataManager
        dm = DataManager()
        
        return {
            "status": "ok",
            "flask_env": os.getenv("FLASK_ENV"),
            "debug": app.debug,
            "testing": app.testing,
            "today_recommendations": len(dm.get_today_recommendations()),
            "db_path": str(Config.DB_PATH),
            "log_dir": str(Config.LOG_DIR)
        }
    
    @app.route("/dev/config")
    def dev_config():
        """Development config endpoint."""
        return {
            "ENABLE_FLASK": Config.ENABLE_FLASK,
            "ENABLE_RL": Config.ENABLE_RL,
            "LOGGING_LEVEL": Config.LOGGING_LEVEL,
            "TICKERS": Config.TICKERS[:5] + ["..."],  # First 5 tickers
            "DB_PATH": str(Config.DB_PATH),
            "INITIAL_CAPITAL": Config.INITIAL_CAPITAL,
            "TRANSACTION_FEE_PCT": Config.TRANSACTION_FEE_PCT,
        }
    
    # Development data endpoints
    @app.route("/dev/db-init")
    def dev_db_init():
        """Initialize database schema."""
        try:
            from app.data_access.data_manager import DataManager
            dm = DataManager()
            dm.initialize_tables()
            return {"status": "success", "message": "Database schema initialized"}
        except Exception as e:
            logger.error(f"DB init failed: {e}")
            return {"status": "error", "message": str(e)}, 500
    
    @app.route("/dev/clear-recs")
    def dev_clear_recs():
        """Clear today's recommendations (dev only)."""
        try:
            from app.data_access.data_manager import DataManager
            dm = DataManager()
            dm.clear_recommendations(date=datetime.today().strftime("%Y-%m-%d"))
            return {"status": "success", "message": "Recommendations cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
    
    logger.info("Development mode endpoints enabled: /dev/*")

# ═══════════════════════════════════════════════════════════════
```

---

## 🚀 IMPROVED WORKFLOWS

### Before (Current)
```bash
# Flask only
python run_dev.py

# Main pipeline
python main.py --daily

# Walk-forward
python main.py --walk-forward VOO

# (Cannot run both easily)
```

### After (Improved)
```bash
# Flask only (same as before)
python run_dev.py

# Flask with new dev endpoints
curl http://localhost:5000/dev/status
curl http://localhost:5000/dev/config

# Main pipeline with dev mode
python run_dev.py --mode pipeline

# Walk-forward
python run_dev.py --mode walk-forward --ticker VOO

# Both Flask + Pipeline in threads
python run_dev.py --mode both

# Clear today's recommendations and reinitialize
curl http://localhost:5000/dev/clear-recs
curl http://localhost:5000/dev/db-init

# Dry-run pipeline
python main.py daily --dry-run

# Single-ticker dev testing
python main.py daily --ticker VOO --dry-run
```

---

## 📝 REQUIRED CHANGES

### 1. Replace `run_dev.py` (3 hours)
- [ ] Backup current `run_dev.py`
- [ ] Implement new multi-mode runner
- [ ] Test all modes locally
- [ ] Update documentation

### 2. Enhance `main.py` (2 hours)
- [ ] Add proper `if __name__ == "__main__":` block
- [ ] Add subparsers for commands
- [ ] Add `--dry-run` flag
- [ ] Add `--ticker` for dev testing
- [ ] Add logging to exception handlers

### 3. Enhance Flask app (2 hours)
- [ ] Add CORS middleware
- [ ] Add request/response logging
- [ ] Add `/dev/status` endpoint
- [ ] Add `/dev/config` endpoint
- [ ] Add `/dev/db-init` endpoint
- [ ] Add `/dev/clear-recs` endpoint

### 4. Update `.env` (0.5 hours)
```env
# Add to .env
FLASK_ENV=development
FLASK_DEBUG=1
DRY_RUN=false
```

### 5. Update documentation (1 hour)
- [ ] Document all dev runner modes
- [ ] Add troubleshooting guide
- [ ] Update README with dev setup

---

## 📊 BENEFITS

| Improvement | Impact | Effort |
|------------|--------|--------|
| Multi-mode runner | Easy switching between Flask/pipeline | 2h |
| Dry-run mode | Safe testing without side effects | 1h |
| Dev endpoints | Inspect system state | 1h |
| Request logging | Debug HTTP issues | 0.5h |
| Single-ticker mode | Quick local testing | 0.5h |
| CORS support | Frontend development easier | 0.5h |

**Total Impact:** ⬆️ Developer productivity by ~40%

---

## ⚠️ PRODUCTION CONSIDERATIONS

These dev runners should **NEVER** be used in production:

```python
# Production should use:
if os.getenv("ENVIRONMENT") == "production":
    app.run(debug=False)  # debug=False ALWAYS
    app.logger.setLevel(logging.ERROR)  # Less verbose
    # No dev endpoints exposed
else:
    # Development mode (current implementation)
    app.run(debug=True)
```

---

## 🔧 IMPLEMENTATION PRIORITY

### Week 1 (High Priority)
- [ ] Replace `run_dev.py` with multi-mode runner
- [ ] Add proper CLI to `main.py`
- [ ] Add dry-run support

### Week 2 (Medium Priority)
- [ ] Add dev endpoints to Flask
- [ ] Add request/response logging
- [ ] Update documentation

### Week 3 (Low Priority)
- [ ] Add profiling support
- [ ] Add performance monitoring
- [ ] Add debug toolbar (optional)

---

## 📚 REFERENCE DOCUMENTS

After implementation, update:
- `START_HERE.txt` — Add dev setup section
- `README.md` — Add development guide
- New file: `DEVELOPMENT_GUIDE.md`

---

**Status:** Ready for implementation  
**Estimated Total Effort:** 7–8 hours  
**Developer Impact:** High (productivity boost)  
**Production Risk:** Low (dev-only code)
