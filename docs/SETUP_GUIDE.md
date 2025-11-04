# Huracan Engine - Complete Setup Guide

This guide will walk you through setting up the complete RL-powered Huracan Engine with health monitoring.

---

## Prerequisites

1. **Python 3.11+** installed
2. **PostgreSQL 14+** with superuser access (for pgvector extension)
3. **Ray** cluster (optional, can run locally)
4. **Telegram Bot** (optional, for alerts)

---

## Step 1: Install Dependencies

All required dependencies have been added to `pyproject.toml`. Install them:

```bash
cd "/Users/haq/Engine (HF1)/engine"

# Activate virtual environment
source .venv/bin/activate

# Dependencies are already installed (torch, psycopg2-binary, psutil)
# If you need to reinstall:
pip install torch==2.1.0 psycopg2-binary==2.9.9 psutil==5.9.6
```

---

## Step 2: Setup PostgreSQL Database

### 2.1 Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE huracan;

# Exit
\q
```

### 2.2 Set DATABASE_URL

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export DATABASE_URL='postgresql://your_user:your_password@localhost:5432/huracan'

# Or for this session only:
export DATABASE_URL='postgresql://your_user:your_password@localhost:5432/huracan'
```

### 2.3 Run Database Setup Script

This will:
- Install pgvector extension
- Create all RL training tables
- Verify everything is working

```bash
cd "/Users/haq/Engine (HF1)/engine"
./scripts/setup_database.sh
```

**Expected output:**
```
========================================
  Huracan Engine Database Setup
========================================

✅ DATABASE_URL found
1️⃣  Checking database connection...
   ✅ Database connection successful

2️⃣  Installing pgvector extension...
   ✅ pgvector extension installed

3️⃣  Creating RL training schema...
   ✅ Schema created successfully

4️⃣  Verifying tables...
   ✅ trade_memory
   ✅ post_exit_tracking
   ✅ win_analysis
   ✅ loss_analysis
   ✅ pattern_library
   ✅ model_performance

5️⃣  Checking pgvector functionality...
   ✅ pgvector is working

========================================
✅ Database setup complete!
========================================
```

---

## Step 3: Configure Settings

### 3.1 Update config/base.yaml

Add PostgreSQL configuration:

```yaml
postgres:
  dsn: "${DATABASE_URL}"  # Will read from environment variable
```

The RL settings are already configured:

```yaml
training:
  window_days: 150
  walk_forward:
    train_days: 20
    test_days: 5
    min_trades: 300
  rl_agent:
    enabled: true              # ✅ RL training enabled
    learning_rate: 0.0003
    gamma: 0.99
    clip_epsilon: 0.2
    entropy_coef: 0.01
    n_epochs: 10
    batch_size: 64
    state_dim: 80
    device: "cpu"
  shadow_trading:
    enabled: true              # ✅ Shadow trading enabled
    position_size_gbp: 1000
    max_hold_minutes: 120
    stop_loss_bps: 15
    take_profit_bps: 20
    min_confidence_threshold: 0.52
  memory:
    vector_similarity_threshold: 0.7
    min_pattern_occurrences: 10
    pattern_update_frequency: "daily"
    max_similar_patterns: 20
  monitoring:
    enabled: true              # ✅ Health monitoring enabled
    check_interval_seconds: 300
    win_rate_stddev_threshold: 2.0
    profit_stddev_threshold: 2.0
    volume_stddev_threshold: 2.5
    auto_remediation_enabled: true
    pause_failing_patterns: true
    pattern_failure_threshold: 0.45
```

### 3.2 Configure Telegram Alerts (Optional but Recommended)

1. **Create a Telegram Bot:**
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` and follow instructions
   - Copy the bot token

2. **Get your Chat ID:**
   - Send a message to your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Look for `"chat":{"id":<YOUR_CHAT_ID>}`

3. **Update config/base.yaml:**

```yaml
notifications:
  telegram_enabled: true
  telegram_webhook_url: "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage"
  telegram_chat_id: "<YOUR_CHAT_ID>"
```

---

## Step 4: Verify Installation

### 4.1 Check Python Imports

```bash
cd "/Users/haq/Engine (HF1)/engine"
source .venv/bin/activate

python -c "
import torch
import psycopg2
import psutil
from src.cloud.training.memory.store import MemoryStore
from src.cloud.training.agents.rl_agent import RLTradingAgent
from src.cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator
print('✅ All imports successful!')
"
```

### 4.2 Test Database Connection

```bash
python -c "
import psycopg2
import os
dsn = os.environ['DATABASE_URL']
conn = psycopg2.connect(dsn)
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM trade_memory')
print(f'✅ Database connected! trade_memory rows: {cur.fetchone()[0]}')
conn.close()
"
```

---

## Step 5: Run Your First Training

### Option A: Test Single Symbol (Recommended First)

Create a test script `test_single_symbol.py`:

```python
#!/usr/bin/env python3
"""Test RL training on a single symbol."""

import os
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.services.exchange import ExchangeClient
from src.cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline

# Load settings
settings = EngineSettings.load()
dsn = os.environ['DATABASE_URL']

# Setup exchange (read-only, no credentials needed for historical data)
exchange = ExchangeClient("binance", sandbox=False)

# Initialize RL pipeline
rl_pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)

# Train on BTC/USDT for last 90 days
print("🚀 Starting RL training on BTC/USDT...")
metrics = rl_pipeline.train_on_symbol(
    symbol="BTC/USDT",
    exchange=exchange,
    lookback_days=90,
)

print("\n✅ Training complete!")
print(f"   Total trades: {metrics['total_trades']}")
print(f"   Winning trades: {metrics['winning_trades']}")
print(f"   Win rate: {metrics['win_rate']:.2%}")
print(f"   Avg profit: £{metrics['avg_profit_gbp']:.2f}")
print(f"   Patterns learned: {metrics['patterns_learned']}")
```

Run it:

```bash
cd "/Users/haq/Engine (HF1)/engine"
source .venv/bin/activate
python test_single_symbol.py
```

### Option B: Run Full Daily Retrain

This will:
1. Train LightGBM models on 20 coins
2. Run RL shadow trading on each coin
3. Store patterns in memory database
4. Run health checks before/after training
5. Send Telegram alerts

```bash
cd "/Users/haq/Engine (HF1)/engine"
source .venv/bin/activate

# Set environment variables
export DATABASE_URL='postgresql://user:pass@localhost:5432/huracan'
export HURACAN_ENV='local'

# Run the daily retrain pipeline
python -m src.cloud.training.pipelines.daily_retrain
```

**Expected log output:**

```json
{"event": "run_start", "timestamp": "2025-01-15T14:00:00.000Z"}
{"event": "===== INITIALIZING HEALTH MONITORING =====", "check_interval": 300}
{"event": "===== STARTUP STATUS CHECK =====", "operation": "STARTUP_HEALTH_CHECK"}
{"event": "startup_status_summary", "overall_status": "HEALTHY", "services_healthy": 5, "services_total": 5}
{"event": "===== PRE-TRAINING HEALTH CHECK ====="}
{"event": "pre_training_health_check_complete", "alerts": 0, "critical": 0, "warning": 0}
{"event": "universe_selected", "count": 20, "symbols": ["BTC/USDT", "ETH/USDT", ...]}
{"event": "===== STARTING RL TRAINING =====", "symbol": "BTC/USDT", "lookback_days": 150}
{"event": "shadow_trading_start", "symbol": "BTC/USDT", "rows": 50000}
...
{"event": "===== RL TRAINING COMPLETE =====", "symbol": "BTC/USDT", "total_trades": 1234, "win_rate": 0.58}
{"event": "rl_training_completed_for_symbol", "symbol": "BTC/USDT"}
...
{"event": "run_complete"}
{"event": "===== POST-TRAINING HEALTH CHECK ====="}
{"event": "post_training_health_check_complete", "alerts": 0}
```

---

## Step 6: Monitor the System

### 6.1 Run Standalone Health Monitor

In a separate terminal, run continuous health monitoring:

```bash
cd "/Users/haq/Engine (HF1)/engine"
source .venv/bin/activate
export DATABASE_URL='postgresql://user:pass@localhost:5432/huracan'

python scripts/run_health_monitor.py
```

This will:
- Check health every 5 minutes
- Detect anomalies (win rate drops, error spikes, etc.)
- Send Telegram alerts
- Auto-pause failing patterns

### 6.2 Watch Logs

```bash
# Watch all logs
tail -f logs/engine.log | jq .

# Filter specific events
tail -f logs/engine.log | jq 'select(.event | contains("rl_training"))'
tail -f logs/engine.log | jq 'select(.event | contains("health_check"))'
tail -f logs/engine.log | jq 'select(.event | contains("alert"))'
```

### 6.3 Check Telegram

You'll receive alerts like:

```
🚨 CRITICAL: Win Rate Anomaly
========================================
Win rate dropped to 43% (baseline: 58%)
Z-score: -2.8 std deviations below normal

🔧 Suggested Actions:
1. Review recent losing trades
2. Check market regime change
3. Verify data quality
```

---

## Step 7: Query the Memory Database

### 7.1 See What the System Learned

```sql
-- Top performing patterns
SELECT
    pattern_name,
    win_rate,
    avg_profit_gbp,
    total_trades,
    confidence_score
FROM pattern_library
WHERE win_rate > 0.55
ORDER BY confidence_score DESC
LIMIT 10;

-- Recent winning trades
SELECT
    symbol,
    entry_price,
    exit_price,
    profit_gbp,
    hold_duration_minutes,
    created_at
FROM trade_memory
WHERE is_winner = true
ORDER BY created_at DESC
LIMIT 20;

-- Losing patterns to avoid
SELECT
    failure_reason,
    COUNT(*) as count,
    AVG(loss_gbp) as avg_loss
FROM loss_analysis
GROUP BY failure_reason
ORDER BY count DESC;

-- Post-exit analysis (learning optimal holds)
SELECT
    symbol,
    exit_type,
    missed_profit_gbp,
    optimal_exit_minutes
FROM post_exit_tracking
WHERE missed_profit_gbp > 1.0
ORDER BY missed_profit_gbp DESC
LIMIT 10;
```

---

## Troubleshooting

### Issue: "pgvector extension not found"

```bash
# On macOS with Homebrew:
brew install pgvector

# On Ubuntu/Debian:
sudo apt-get install postgresql-14-pgvector

# Then reconnect and run:
CREATE EXTENSION vector;
```

### Issue: "Database connection refused"

```bash
# Check if PostgreSQL is running:
pg_isready

# Start PostgreSQL:
# macOS:
brew services start postgresql@14

# Linux:
sudo systemctl start postgresql
```

### Issue: "RL training fails with CUDA error"

The default config uses CPU. If you have a GPU and want to use it:

```yaml
# config/base.yaml
training:
  rl_agent:
    device: "cuda"  # Change from "cpu"
```

### Issue: "Import errors"

```bash
# Make sure you're in the virtual environment:
source .venv/bin/activate

# Reinstall dependencies:
pip install torch==2.1.0 psycopg2-binary==2.9.9 psutil==5.9.6
```

### Issue: "No Telegram alerts"

1. Test the bot manually:
```bash
curl -X POST "https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage" \
  -d "chat_id=<YOUR_CHAT_ID>" \
  -d "text=Test message"
```

2. Check logs for notification errors:
```bash
tail -f logs/engine.log | jq 'select(.event | contains("telegram"))'
```

---

## What Happens Next?

Once running, your system will:

1. **Every night at 02:00 UTC:**
   - Download latest candle data for 20 coins
   - Train LightGBM models with walk-forward validation
   - Run RL shadow trading on 150+ days of history per coin
   - Analyze every win and loss
   - Update pattern library with new learnings
   - Run health checks and send alerts

2. **Every 5 minutes (if health monitor running):**
   - Check statistical anomalies
   - Monitor pattern performance degradation
   - Detect error spikes
   - Auto-pause failing patterns
   - Send Telegram alerts for issues

3. **Continuously learning:**
   - Every trade outcome stored in memory
   - Patterns clustered by similarity
   - Win/loss root cause analysis
   - Post-exit tracking for optimal holds
   - RL agent improves decision-making

---

## Expected Results (After 1 Month)

Based on £1-2 per trade target:

- **Pattern Library:** 40-60 patterns identified
- **Win Rate:** 55-60% (after sufficient data)
- **Trades per Day:** 50-100 across 20 coins
- **Avg Profit:** £1.20-£1.80 per winning trade
- **Daily P&L:** £60-£180 (at target performance)

---

## Next Steps

1. ✅ Run the database setup script
2. ✅ Configure Telegram alerts
3. ✅ Test on single symbol (BTC/USDT)
4. ✅ Run full daily retrain
5. ✅ Start health monitor in background
6. ✅ Query database to see patterns
7. ✅ Analyze what's working

---

## Support & Documentation

- **RL Training Guide:** [docs/RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md)
- **Health Monitoring Guide:** [docs/HEALTH_MONITORING_GUIDE.md](HEALTH_MONITORING_GUIDE.md)
- **Complete System Overview:** [docs/COMPLETE_SYSTEM_OVERVIEW.md](COMPLETE_SYSTEM_OVERVIEW.md)
- **Logs:** `tail -f logs/engine.log`
- **Database:** Connect with `psql $DATABASE_URL`

---

🎉 **Your powerhouse RL trading engine is ready to learn and trade!** 🎉
