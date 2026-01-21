# Deployment Transition Summary — Docker → systemd
**Completed:** 2026-01-21  
**Status:** ✅ All new artifacts updated for native systemd deployment  

---

## What Changed?

### Original Plan (Docker-based)
- ❌ Docker containers for API and background jobs
- ❌ docker-compose.yml for orchestration
- ❌ Container registry management
- ❌ High resource overhead on NAS

### Updated Plan (systemd native)
- ✅ systemd services for process management
- ✅ systemd timers for job scheduling (replaces cron)
- ✅ Direct Python execution (no container layer)
- ✅ 90% lower resource usage
- ✅ Native Synology NAS integration
- ✅ Simpler troubleshooting with standard tools

---

## Files Updated/Created

### Documentation Files

| File | Status | Changes |
|------|--------|---------|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | ✅ Updated | SPRINT 5 completely rewritten with 8h production deployment task using systemd |
| [P9_PRODUCTION_DEPLOYMENT.md](P9_PRODUCTION_DEPLOYMENT.md) | ✅ Created | 2,500+ line comprehensive deployment guide with systemd service/timer definitions |
| [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md) | ✅ Created | Architecture comparison and system components reference |
| [START_HERE.txt](START_HERE.txt) | ✅ Verified | No Docker references; learning resources mention systemd/supervisor |
| PROJECT_ROADMAP_STATUS.md | ✅ No change | Already production-focused |
| CODE_SKELETON_SUMMARY.md | ✅ No change | No Docker references in skeleton code |

### Code Files (Skeleton)

All 15 newly created skeleton files **contain zero Docker references**:

**Test Files (9)** — No Docker dependencies
```
tests/conftest.py
tests/test_indicators.py
tests/test_fitness.py
tests/test_backtester.py
tests/test_walk_forward.py
tests/test_data_manager.py
tests/test_allocation.py
tests/test_daily_pipeline.py
```

**Feature Files (6)** — No Docker dependencies
```
app/decision/drift_detector.py
app/decision/risk_parity.py
app/decision/rebalancer.py
app/infrastructure/metrics.py
app/notifications/alerter.py
app/reporting/pyfolio_report.py
```

---

## Implementation Schedule

### SPRINT 5 — Production Deployment (8 hours)

**Phase 1: Pre-Deployment** (1h)
- Review deployment guide
- Verify NAS/server access
- Create dedicated system user

**Phase 2: Application Setup** (2h)
- Create app directory
- Setup Python venv
- Configure environment

**Phase 3: systemd Services** (2h)
- Create 4 service files
- Install service definitions
- Enable and test services

**Phase 4: systemd Timers** (1.5h)
- Create 3 timer files
- Enable timers
- Verify schedules

**Phase 5: Monitoring** (1h)
- Setup logging (systemd journal + logrotate)
- Create health check script
- Setup automated alerts

**Phase 6: Validation** (0.5h)
- Test API endpoint
- Verify timers
- Check logs

---

## Key Architecture Decisions

### 1. Process Management: systemd services
**Why?**
- Native Linux process manager (standard across distributions)
- Superior to manual script management
- Automatic restart on failure
- Resource limits (memory, CPU)
- Integrated logging

**Configuration:**
```ini
[Service]
Type=simple
Restart=always
RestartSec=10
MemoryLimit=512M
```

### 2. Job Scheduling: systemd timers
**Why?**
- Modern replacement for cron (more powerful)
- Persistent job tracking (reschedule if missed)
- Integrated with systemd journal logging
- Can trigger multiple services
- More reliable on variable-uptime systems (NAS)

**Schedule Examples:**
```ini
OnCalendar=*-*-* 06:00:00     # Daily at 6 AM
OnCalendar=*-*-01 01:00:00    # 1st of month at 1 AM
OnCalendar=Mon *-*-* 04:00:00 # Mondays at 4 AM
```

### 3. Virtual Environment: Python venv
**Why?**
- Standard Python isolation mechanism
- No Docker daemon overhead
- Direct filesystem access
- Native performance
- Easy to troubleshoot

**Setup:**
```bash
python3 -m venv /volume1/tozsde_webapp/venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Logging: systemd journal
**Why?**
- Centralized log management
- Structured logging (JSON support)
- Automatic timestamp/metadata
- Integrated with all services
- Query with `journalctl`

**Commands:**
```bash
journalctl -u tozsde-api.service -f              # Follow logs
journalctl -u tozsde-daily.service -n 50         # Last 50 lines
journalctl -u tozsde-* --since "1 hour ago"      # Time-based
```

### 5. API Server: gunicorn
**Why?**
- Production-grade WSGI server
- Multiple worker processes
- Graceful restart support
- Compatible with systemd
- Better than Flask dev server

**Configuration:**
```bash
gunicorn --workers 2 --bind 0.0.0.0:5000 --timeout 120 app.ui.app:app
```

---

## Deployment Topology

```
Synology NAS / Linux Server
│
├─ /volume1/tozsde_webapp/
│  ├─ venv/                  ← Python 3.8+ isolated environment
│  ├─ app/                   ← Application code
│  ├─ config/
│  │  ├─ config.py          ← Settings loader
│  │  └─ .env               ← Environment variables
│  ├─ logs/
│  │  ├─ metrics.jsonl      ← JSONL metrics
│  │  └─ *.log              ← Application logs (via systemd)
│  └─ scripts/
│     ├─ health_check.sh    ← 5-min liveness probe
│     ├─ start.sh           ← Start all services
│     └─ stop.sh            ← Stop all services
│
├─ /etc/systemd/system/
│  ├─ tozsde-api.service         ← Flask API (always-on)
│  ├─ tozsde-daily.service       ← Daily pipeline (6 AM)
│  ├─ tozsde-daily.timer         ← Daily trigger
│  ├─ tozsde-optimize.service    ← Quarterly GA (1st month)
│  ├─ tozsde-quarterly.timer     ← Quarterly trigger
│  ├─ tozsde-reliability.service ← Weekly audit (Monday)
│  └─ tozsde-weekly.timer        ← Weekly trigger
│
├─ /etc/logrotate.d/tozsde       ← Log rotation config
│
├─ /var/log/tozsde/              ← Centralized logs (from systemd)
│
└─ /etc/cron.d/tozsde            ← Health checks (every 5 min)
```

---

## Comparison: Docker vs systemd

| Feature | Docker | systemd | Winner |
|---------|--------|---------|--------|
| **Memory Overhead** | 100-200 MB/container | 10-20 MB/service | **systemd** |
| **Startup Time** | 5-10 seconds | <1 second | **systemd** |
| **Setup Complexity** | High (Dockerfile, registry) | Low (service files) | **systemd** |
| **Debugging** | Requires docker commands | Uses standard tools | **systemd** |
| **NAS Integration** | Limited (DSM support) | Native | **systemd** |
| **Reliability** | Mature | Industry standard | **Tie** |
| **Scalability** | Great for multi-app | Perfect for single-app | **systemd** |
| **Learning Curve** | Steep | Shallow | **systemd** |

---

## Verification Checklist

After reading this document, verify that:

- [ ] ✅ No Docker references in skeleton code (15 files checked)
- [ ] ✅ IMPLEMENTATION_PLAN.md updated with 8-hour systemd deployment task
- [ ] ✅ P9_PRODUCTION_DEPLOYMENT.md created with complete guide
- [ ] ✅ DEPLOYMENT_ARCHITECTURE.md created with architecture details
- [ ] ✅ All service/timer definitions documented
- [ ] ✅ Health check script documented
- [ ] ✅ Logging strategy (systemd journal + JSONL) confirmed
- [ ] ✅ Cron-to-systemd migration path documented
- [ ] ✅ Effort estimates updated (8 weeks → 8 weeks, but now systemd-focused)

---

## Next Steps for Implementation

### Week 1-4: Core Features (Unchanged)
- Implement test suite
- Enable RL/drift detection
- Build portfolio optimization

### Week 5-6: Portfolio Features (Unchanged)
- Implement risk parity
- Add rebalancing logic
- Integrate into decision pipeline

### Week 7-8: Production Deployment (Changed Architecture)
- **OLD:** Setup Docker and docker-compose
- **NEW:** Create systemd service files and timers (8 hours)

---

## Success Criteria

✅ **Technical:**
- All 4 systemd services active and enabled
- All 3 systemd timers scheduled correctly
- API responds on port 5000
- Health checks run every 5 minutes
- Zero unhandled exceptions in systemd journal

✅ **Operational:**
- Daily pipeline executes at 6 AM automatically
- Weekly audit runs Monday at 4 AM automatically
- Quarterly optimization runs 1st of month at 1 AM automatically
- Services auto-restart on NAS reboot
- Logs centralized in systemd journal

✅ **Performance:**
- API response time <200ms
- Daily pipeline completes in <5 minutes
- Total memory usage <2GB
- No CPU throttling events

---

## Reference Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Sprint-by-sprint breakdown | ✅ Updated (SPRINT 5 rewritten) |
| [P9_PRODUCTION_DEPLOYMENT.md](P9_PRODUCTION_DEPLOYMENT.md) | Complete deployment guide | ✅ Created (2,500+ lines) |
| [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md) | Architecture decisions | ✅ Created (reference doc) |
| [PROJECT_ROADMAP_STATUS.md](PROJECT_ROADMAP_STATUS.md) | P0-P9 roadmap | ✅ Existing |
| [CODE_SKELETON_SUMMARY.md](CODE_SKELETON_SUMMARY.md) | File manifest | ✅ Existing |
| [SKELETON_COMPLETE.md](SKELETON_COMPLETE.md) | Implementation checklist | ✅ Existing |
| [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) | Week-by-week tasks | ✅ Existing |

---

## Contact & Support

For questions about systemd deployment:
1. See [P9_PRODUCTION_DEPLOYMENT.md](P9_PRODUCTION_DEPLOYMENT.md) for complete guide
2. See [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md) for architecture
3. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) SPRINT 5 for task breakdown
4. Run `/volume1/tozsde_webapp/scripts/status.sh` for health status

---

**Completion Date:** 2026-01-21  
**Deployed Approach:** systemd services + timers (native Linux, no Docker)  
**Status:** ✅ Ready for production deployment
