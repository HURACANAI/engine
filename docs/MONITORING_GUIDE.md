# Engine Monitoring Guide

Complete guide to monitoring your Engine in real-time.

## Quick Commands

### One-Time Status Check
```bash
python scripts/monitor_engine.py status
```

### Watch Mode (Live Updates)
```bash
python scripts/monitor_engine.py --watch
# Updates every 5 seconds (default)
# Press Ctrl+C to stop
```

### Check Shadow Trades Count
```bash
python scripts/monitor_engine.py shadow-trades
```

### Check Engine Status
```bash
python scripts/monitor_engine.py engines
```

### Check Training Progress
```bash
python scripts/monitor_engine.py training
```

### JSON Output (for scripts)
```bash
python scripts/monitor_engine.py status --json
```

## Dashboard Access

### Live Terminal Dashboard
```bash
python -m observability.ui.live_dashboard
```

Shows:
- Real-time shadow trade feed
- Live learning metrics
- Gate pass/fail rates
- Model training progress
- System health indicators
- Auto-refresh every 1 second

### Enhanced Dashboard
```bash
python -m observability.ui.enhanced_dashboard
```

Shows:
- Live shadow trade feed
- Real-time learning metrics (AUC improving?)
- Model confidence heatmap
- Gate pass/fail rates (updated every minute)
- Performance vs targets
- Active learning indicators
- Circuit breaker status
- Concept drift warnings
- Confidence-based position scaling

### Trade Viewer
```bash
python -m observability.ui.trade_viewer
```

Interactive menu to:
- View recent trades
- Filter by mode (scalp/runner)
- Filter by regime
- Filter by symbol
- View best/worst trades
- See detailed trade breakdowns

### Gate Inspector
```bash
python -m observability.ui.gate_inspector
```

Understand gate decisions:
- View pass rates
- Block accuracy
- Threshold suggestions
- Blocked signal analysis

### Model Tracker
```bash
python -m observability.ui.model_tracker_ui
```

Track model evolution:
- Compare versions
- Hamilton readiness
- AUC/ECE improvements
- Training history

## Telegram Monitoring

The engine automatically sends Telegram notifications for:
- ✅ Training progress (started, downloading, completed, failed)
- ✅ Batch progress updates
- ✅ Shadow trades (if enabled)
- ✅ System health checks
- ✅ Validation failures
- ✅ Errors and warnings

**Setup:**
1. Get your Telegram bot token from @BotFather
2. Get your chat ID using:
   ```bash
   python scripts/get_telegram_chat_id.py
   ```
3. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
   ```

## Database Queries

### Check Shadow Trades Count
```sql
-- Total shadow trades
SELECT COUNT(*) FROM shadow_trades;

-- Today's shadow trades
SELECT COUNT(*) FROM shadow_trades 
WHERE ts >= CURRENT_DATE;

-- By mode
SELECT mode, COUNT(*) as count, 
       AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate
FROM shadow_trades
GROUP BY mode;

-- Recent trades (last 24h)
SELECT COUNT(*) FROM shadow_trades 
WHERE ts >= NOW() - INTERVAL '24 hours';
```

### Check Training Progress
```sql
-- Recent training sessions
SELECT symbol, created_at, kind, notes
FROM model_registry
ORDER BY created_at DESC
LIMIT 20;

-- Training metrics
SELECT symbol, sharpe, hit_rate, pnl_bps, trades_oos
FROM model_metrics
ORDER BY created_at DESC
LIMIT 20;
```

### Check Engine Status
```sql
-- Recent activity
SELECT * FROM event_log
WHERE event_type LIKE '%engine%'
ORDER BY timestamp DESC
LIMIT 50;
```

## Log Files

### Training Logs
```bash
# View latest training log
tail -f logs/training.log

# Filter for specific symbol
tail -f logs/training.log | grep "BTC/USDT"

# Filter for errors
tail -f logs/training.log | grep -i error

# Filter for progress
tail -f logs/training.log | grep -E "(downloading|completed|batch)"
```

### System Logs
```bash
# View system logs
tail -f logs/system.log

# Filter for health checks
tail -f logs/system.log | grep "health"
```

## Monitoring During Training

### Watch Training Progress
```bash
# Terminal 1: Run training
python -m src.cloud.training.pipelines.daily_retrain

# Terminal 2: Watch progress
python scripts/monitor_engine.py --watch

# Terminal 3: Watch logs
tail -f logs/training.log
```

### Check Download Progress
The training logs show download progress:
```
downloading_historical_data symbol=BTC/USDT start_at=... end_at=... window_days=150
data_download_complete symbol=BTC/USDT rows=216000
```

### Check Batch Progress
Telegram notifications show:
- Batch number and total batches
- Tasks completed in current batch
- Symbols being processed

## Shadow Trades Monitoring

### Count Shadow Trades
```bash
# Quick count
python scripts/monitor_engine.py shadow-trades

# Detailed stats
python -m observability.ui.trade_viewer
# Then select "View recent trades"
```

### Query Shadow Trades
```sql
-- All shadow trades
SELECT * FROM shadow_trades ORDER BY ts DESC LIMIT 100;

-- By mode
SELECT mode, COUNT(*) as count, 
       AVG(shadow_pnl_bps) as avg_pnl_bps,
       AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate
FROM shadow_trades
GROUP BY mode;

-- Best trades
SELECT symbol, mode, shadow_pnl_bps, duration_sec
FROM shadow_trades
WHERE shadow_pnl_bps > 0
ORDER BY shadow_pnl_bps DESC
LIMIT 20;

-- Worst trades
SELECT symbol, mode, shadow_pnl_bps, duration_sec
FROM shadow_trades
WHERE shadow_pnl_bps < 0
ORDER BY shadow_pnl_bps ASC
LIMIT 20;
```

## Engine Status

### All 23 Alpha Engines
The engine status shows:
- Trend Engine
- Range Engine
- Breakout Engine
- Tape Engine
- Leader Engine
- Sweep Engine
- Scalper/Latency-Arb Engine
- Volatility Engine
- Correlation/Cluster Engine
- Funding/Carry Engine
- Arbitrage Engine
- Adaptive/Meta-Learning Engine
- Evolutionary/Auto-Discovery Engine
- Risk Engine
- Flow-Prediction Engine
- Cross-Venue Latency Engine
- Market-Maker/Inventory Engine
- Anomaly-Detection Engine
- Regime-Classifier Engine
- Momentum Reversal Engine
- Divergence Engine
- Support/Resistance Bounce Engine
- Meta-Label Engine

### Check Engine Performance
```sql
-- Engine performance metrics
SELECT engine_name, 
       COUNT(*) as signals,
       AVG(confidence) as avg_confidence,
       AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate
FROM engine_signals
GROUP BY engine_name
ORDER BY signals DESC;
```

## Troubleshooting

### Engine Appears "Cold" (0% CPU/GPU)
1. Check if training is actually running:
   ```bash
   python scripts/monitor_engine.py training
   ```

2. Check logs:
   ```bash
   tail -f logs/training.log
   ```

3. Check Telegram notifications (if configured)

4. Check database for recent activity:
   ```sql
   SELECT * FROM event_log ORDER BY timestamp DESC LIMIT 20;
   ```

### No Shadow Trades
1. Check if shadow trading is enabled
2. Check gate pass rates (may be too strict)
3. Check logs for errors
4. Verify database connection

### Training Not Progressing
1. Check for rate limit errors in logs
2. Check database connection
3. Check disk space
4. Check Telegram for error notifications

## Advanced Monitoring

### Custom Queries
Create your own monitoring queries:
```python
from scripts.monitor_engine import EngineMonitor

monitor = EngineMonitor()
shadow = monitor.get_shadow_trades_count()
print(f"Total shadow trades: {shadow['total']}")
```

### Integration with External Tools
The `--json` flag allows integration with external monitoring tools:
```bash
python scripts/monitor_engine.py status --json | jq '.shadow_trades.total'
```

### Automated Monitoring Script
```bash
#!/bin/bash
# monitor.sh
while true; do
    python scripts/monitor_engine.py status
    sleep 60
done
```

## Summary

**Quick Status:**
```bash
python scripts/monitor_engine.py status
```

**Live Dashboard:**
```bash
python -m observability.ui.live_dashboard
```

**Watch Mode:**
```bash
python scripts/monitor_engine.py --watch
```

**Telegram:**
Configure once, receive automatic notifications for all events.

**Database:**
Query directly for detailed analysis.

