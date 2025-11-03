# Health Monitoring System Guide

## Overview

The Health Monitoring System provides **comprehensive visibility** into your trading engine with **heavy logging at every step**, ensuring you always know what's working, what's enabled, and what's not.

---

## Key Features

### 🔍 **Complete Visibility**
- Every component logs its status at initialization
- Every health check logs its findings
- Every alert logs its trigger and context
- Every remediation logs its action and result
- System status logged at startup and on-demand

### 📊 **What It Monitors**

1. **Statistical Anomalies**
   - Win rate drops (>2 std dev)
   - Unusual P&L patterns
   - Trade volume drops (>50%)
   - All with extensive logging

2. **Pattern Health**
   - Individual pattern performance tracking
   - Degradation detection
   - Overfitting detection
   - Every pattern status logged

3. **Error Monitoring**
   - Error spike detection (>3x baseline)
   - Recurring error patterns
   - Impact analysis
   - All errors categorized and logged

4. **System Health**
   - Database connectivity
   - Service status (all components)
   - Resource usage (CPU, memory, disk)
   - Active features detection
   - Recent activity tracking

5. **Auto-Remediation**
   - Safe corrective actions
   - Pause failing patterns
   - Log detailed context
   - All actions logged and reversible

---

## Logging Philosophy

### **Everything Is Logged**

```
✅ Component initialization
✅ Service health checks
✅ Database connectivity
✅ Feature detection (what's enabled)
✅ Health check cycles (every step)
✅ Alert generation
✅ Remediation actions
✅ Resource usage
✅ Recent activity
✅ Errors and exceptions
✅ Startup status
✅ Shutdown status
```

### **Log Levels**

- `INFO`: Normal operations, status checks, component initialization
- `WARNING`: Degraded performance, non-critical issues
- `ERROR`: Failures, critical alerts
- `EXCEPTION`: Unhandled errors with full stack traces

---

## Usage

### **1. Standalone Monitoring**

```python
from src.cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator
from src.cloud.training.config.settings import EngineSettings

settings = EngineSettings.load()
monitor = HealthMonitorOrchestrator(
    settings=settings,
    dsn="postgresql://user:pass@localhost/huracan"
)

# Run single health check
alerts = monitor.run_health_check()

# Or run continuously (every 5 minutes)
monitor.run_continuous(interval_seconds=300)
```

### **2. Integrated with Training Pipeline**

```python
from src.cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline
from src.cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator

# Start monitoring in background
monitor = HealthMonitorOrchestrator(settings, dsn)
# Run in separate thread
import threading
monitor_thread = threading.Thread(target=monitor.run_continuous, args=(300,))
monitor_thread.daemon = True
monitor_thread.start()

# Run training
pipeline = RLTrainingPipeline(settings, dsn)
pipeline.train_on_universe(symbols, exchange)

# Monitoring runs in parallel, alerts you to issues
```

### **3. Check System Status On-Demand**

```python
from src.cloud.training.monitoring.system_status import SystemStatusReporter

reporter = SystemStatusReporter(dsn)

# Full health report
report = reporter.generate_full_report()

print(f"Overall Status: {report.overall_status}")
print(f"Services: {len(report.services)}")
print(f"Active Features: {report.active_features}")
print(f"Trades (24h): {report.recent_activity['trades_24h']}")
```

---

## What Gets Logged

### **Startup Logs**

```
INFO: ===== INITIALIZING HEALTH MONITOR =====
INFO: component_initialized component=AnomalyDetector status=OK
INFO: component_initialized component=PatternHealthMonitor status=OK
INFO: component_initialized component=ErrorMonitor status=OK
INFO: component_initialized component=AlertManager telegram_enabled=True status=OK
INFO: component_initialized component=AutoRemediator status=OK
INFO: component_initialized component=SystemStatusReporter status=OK
INFO: ===== HEALTH MONITOR INITIALIZED =====

INFO: ===== SYSTEM STARTUP STATUS CHECK =====
INFO: checking_database_status operation=DB_HEALTH_CHECK
INFO: database_connection_ok status=CONNECTED
INFO: table_exists table=trade_memory status=OK
INFO: table_exists table=pattern_library status=OK
INFO: database_data_counts trades=15234 patterns=42 status=COUNTED
INFO: service_status_detail service=Database enabled=True running=True healthy=True
INFO: feature_active feature=HISTORICAL_TRAINING_DATA
INFO: feature_active feature=PATTERN_LIBRARY
INFO: active_features_detected total=2 features=['HISTORICAL_TRAINING_DATA', 'PATTERN_LIBRARY']
INFO: startup_status_summary overall_status=HEALTHY services_total=5 services_healthy=5
INFO: ===== STARTUP CHECK COMPLETE =====
```

### **Health Check Cycle Logs**

```
INFO: ===== STARTING HEALTH CHECK =====
INFO: health_check_step step=1 operation=SYSTEM_STATUS_CHECK
INFO: system_status_checked overall_status=HEALTHY services_healthy=5 services_total=5
INFO: health_check_step step=2 operation=ANOMALY_DETECTION
INFO: anomaly_detection_completed alerts_generated=0 critical=0 warning=0
INFO: health_check_step step=3 operation=PATTERN_HEALTH_CHECK
INFO: pattern_health_checked patterns_checked=12 alerts_generated=1 failing_patterns=0
INFO: pattern_status_detail pattern_id=1 pattern_name=ETH_MEAN_REVERSION win_rate=0.62 status=HEALTHY
INFO: health_check_step step=4 operation=OVERFITTING_DETECTION
INFO: health_check_step step=5 operation=ERROR_MONITORING
INFO: health_check_step step=6 operation=ALERT_PROCESSING total_alerts=1
INFO: health_check_step step=7 operation=ALERT_DELIVERY
INFO: ===== HEALTH CHECK COMPLETE ===== duration_seconds=2.3 total_alerts=1
```

### **Alert Generation Logs**

```
WARNING: win_rate_anomaly z_score=-2.4 baseline_trades=156 current_trades=45
INFO: alert_generated alert_id=win_rate_anomaly_1234567890 severity=WARNING title="Win Rate Anomaly Detected"
INFO: alert_queued severity=WARNING title="Win Rate Anomaly Detected"
```

### **Remediation Logs**

```
INFO: evaluating_remediation alert_id=pattern_123_failing severity=CRITICAL
INFO: attempting_pattern_pause pattern_id=123 dry_run=False reason=critical_failure
INFO: pattern_paused pattern_id=123 pattern_name=BTC_BREAKOUT previous_win_rate=0.38
INFO: remediation_action_completed action_id=pause_pattern_123_1234567890 success=True reversible=True
```

### **Resource Monitoring Logs**

```
INFO: checking_resource_usage operation=RESOURCE_MONITOR
INFO: resource_usage_checked cpu=23.4% memory=45.2% disk=12.1%
WARNING: high_memory_usage memory_percent=87.3
```

---

## Configuration

Edit `config/monitoring.yaml` to adjust thresholds:

```yaml
monitoring:
  enabled: true
  check_interval_seconds: 300  # How often to run checks

  anomaly_detection:
    win_rate_stddev_threshold: 2.0  # Sensitivity
    min_trades_for_analysis: 30     # Required sample size

  pattern_health:
    min_win_rate_threshold: 0.45    # Auto-pause below this
    degradation_threshold_pct: 0.15  # Alert on 15% decline

  alerts:
    critical_immediate: true         # Send critical alerts now
    warning_digest_interval_minutes: 60  # Batch warnings hourly
    daily_report_time_utc: "08:00"
```

---

## Telegram Alerts

### **Critical Alert Example**

```
🚨 CRITICAL: Win Rate Anomaly Detected
========================================
Win rate dropped to 43.2% (baseline: 58.1%, -14.9%)
Z-score: -2.8 (2.8 std deviations below normal)
Recent trades: 45 (last 24h)

🔧 Suggested Actions:
1. Review recent losing trades for common patterns
2. Check if market regime changed (volatility spike, spread widening)
3. Verify data quality and execution issues
4. Consider pausing trading until issue identified

Time: 2025-01-15 14:32:00 UTC
```

### **Warning Digest Example**

```
⚠️ WARNING Alert Digest (3 alerts)
==================================================

📌 Pattern Health Issue (2)
  • Pattern 'BTC_BREAKOUT' degrading: 48% win rate (was 62%)
  • Pattern 'ETH_MOMENTUM' failing: 42% win rate (threshold: 45%)
  ... and 0 more

📌 Error Spike (1)
  • Error rate spiked 3.2x above baseline: EXCHANGE_TIMEOUT
```

### **Daily Report Example**

```
📊 Daily Health Report
==================================================
Date: 2025-01-15
Total alerts: 7

🚨 CRITICAL (1)
  • Win Rate Anomaly Detected

⚠️ WARNING (4)
  • Pattern Health Issue: BTC_BREAKOUT
  • Pattern Health Issue: ETH_MOMENTUM
  • Error Spike Detected: EXCHANGE_TIMEOUT
  • Trade Volume Drop Detected
  ... and 0 more

ℹ️ INFO (2)
  • Recurring Error Pattern: DATA_QUALITY_ERROR
  • Pattern Health Issue: SOL_REVERSAL
```

---

## How to Know What's Working

### **Check Logs**

All logs go to `structlog` in JSON format:

```bash
# Filter for specific components
tail -f logs/engine.log | grep "component_initialized"

# Filter for health checks
tail -f logs/engine.log | grep "health_check"

# Filter for alerts
tail -f logs/engine.log | grep "alert_"

# Filter for remediation
tail -f logs/engine.log | grep "remediation"

# Filter for startup
tail -f logs/engine.log | grep "STARTUP"
```

### **Query System Status**

```python
monitor.get_current_status()
# Returns:
# {
#   "running": True,
#   "total_checks": 42,
#   "last_check": "2025-01-15T14:30:00Z",
#   "pending_alerts": {"CRITICAL": 0, "WARNING": 2, "INFO": 1},
#   "remediation_actions_24h": 3
# }
```

### **Check Telegram**

All alerts sent to your configured Telegram chat.

---

## Safety Features

### **Auto-Remediation Rules**

1. ✅ **NEVER modifies code or config files**
2. ✅ **All actions are reversible**
3. ✅ **Everything is logged**
4. ✅ **Only runtime state changes**
5. ✅ **User can always override**

### **What Auto-Remediation CAN Do**

- Pause failing patterns (set reliability_score = 0)
- Log detailed context for investigation
- Record all actions in database

### **What Auto-Remediation CANNOT Do**

- Modify Python code
- Change configuration files
- Delete data
- Execute arbitrary commands

### **Reversing Actions**

```python
# Get recent actions
actions = monitor.auto_remediator.get_action_history(hours=24)

# Reverse specific action
for action in actions:
    if action.action_type == "PAUSE_PATTERN":
        monitor.auto_remediator.reverse_action(action)
        # Logs: "action_reversed action_id=... success=True"
```

---

## Troubleshooting

### **No Logs Appearing**

Check structlog configuration in your app:

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(structlog.INFO),
)
```

### **Telegram Alerts Not Sending**

Check configuration:

```python
settings = EngineSettings.load()
print(f"Telegram enabled: {settings.notifications.telegram_enabled}")
print(f"Webhook URL set: {bool(settings.notifications.telegram_webhook_url)}")
print(f"Chat ID set: {bool(settings.notifications.telegram_chat_id)}")
```

Check logs for errors:

```bash
tail -f logs/engine.log | grep "telegram"
```

### **Too Many Alerts**

Adjust thresholds in `config/monitoring.yaml`:

```yaml
anomaly_detection:
  win_rate_stddev_threshold: 3.0  # Less sensitive (was 2.0)
  min_trades_for_analysis: 50     # Require more data (was 30)
```

---

## Files Created

```
src/cloud/training/monitoring/
├── __init__.py
├── types.py                  # Type definitions
├── anomaly_detector.py       # Statistical anomaly detection
├── pattern_health.py         # Pattern performance monitoring
├── error_monitor.py          # Log analysis and error detection
├── alert_manager.py          # Alert routing and delivery
├── auto_remediation.py       # Safe corrective actions
├── system_status.py          # System health reporting
└── health_monitor.py         # Main orchestrator

config/
└── monitoring.yaml           # Configuration
```

---

## Summary

The Health Monitoring System provides:

- ✅ **Complete visibility** - Know exactly what's happening
- ✅ **Heavy logging** - Every step logged for debugging
- ✅ **Service checks** - Verify components are running
- ✅ **Feature detection** - Know what's enabled vs disabled
- ✅ **Proactive alerts** - Catch issues before they cost money
- ✅ **Safe remediation** - Auto-fix critical issues
- ✅ **Telegram integration** - Get notified immediately
- ✅ **Comprehensive reporting** - Daily health summaries

**You'll always know what's working and what's not!**
