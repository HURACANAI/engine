# Training Dashboards Overview

## Available Dashboards

You now have **multiple dashboard options** to monitor SOL/USDT training in real-time:

### 1. Ultra-Detailed Dashboard ⭐ **RECOMMENDED**
**File**: `scripts/ultra_detailed_dashboard.py`

The ultimate training monitoring tool - shows EVERYTHING happening during training.

**Features**:
- ✅ Comprehensive performance metrics (8 key indicators)
- ✅ Advanced analysis (profit factor, expectancy, risk:reward)
- ✅ Complete trade details (last 15 trades + deep dive)
- ✅ Market regime breakdown with visualizations
- ✅ Exit reason analysis
- ✅ 24-hour activity patterns
- ✅ Decision reasoning for each trade
- ✅ Real-time updates every 1.5 seconds

**Best for**: Understanding what's really happening, debugging, and deep analysis

**Start**:
```bash
python scripts/ultra_detailed_dashboard.py
```

**Documentation**: [ULTRA_DETAILED_DASHBOARD.md](ULTRA_DETAILED_DASHBOARD.md)

---

### 2. Standard Dashboard
**File**: `scripts/training_dashboard.py`

Simple, fast overview of training progress.

**Features**:
- ✅ Basic performance metrics
- ✅ Recent trades (last 10)
- ✅ Win/loss statistics
- ✅ Real-time updates every 2 seconds

**Best for**: Quick checks, minimal screen space, familiar users

**Start**:
```bash
python scripts/training_dashboard.py
```

---

### 3. Other Dashboard Variants
**Files**:
- `scripts/simple_web_dashboard.py` - Web-based version
- `scripts/beautiful_dashboard.py` - Alternative styling
- `scripts/grok_ai_dashboard.py` - AI-enhanced insights
- `scripts/advanced_dashboard.py` - Earlier detailed version

**Note**: These are experimental/legacy versions. Use the ultra-detailed dashboard for the best experience.

---

## Quick Comparison

| Feature | Standard | Ultra-Detailed |
|---------|----------|----------------|
| Performance metrics | Basic | Advanced |
| Trade history | 10 trades | 15 trades + deep dive |
| Regime analysis | ❌ | ✅ |
| Exit reasons | ❌ | ✅ |
| Hourly patterns | ❌ | ✅ |
| Decision reasoning | ❌ | ✅ |
| Professional metrics | ❌ | ✅ (profit factor, expectancy) |
| Update frequency | 2s | 1.5s |

**Full comparison**: [DASHBOARD_COMPARISON.md](DASHBOARD_COMPARISON.md)

---

## Getting Started (60 Seconds)

### Step 1: Start Training
```bash
python scripts/train_sol_full.py
```

### Step 2: Open Dashboard (New Terminal)
```bash
# For complete visibility
python scripts/ultra_detailed_dashboard.py

# OR for quick overview
python scripts/training_dashboard.py
```

### Step 3: Watch The Magic
- Trades appear as model learns
- Metrics update in real-time
- See exactly what's happening

**Quick start guide**: [DASHBOARD_QUICK_START.md](DASHBOARD_QUICK_START.md)

---

## What You'll See

### Ultra-Detailed Dashboard Layout

```
┌────────────────────────────────────────────────────┐
│  🚀 Header: Time, Uptime, Status                   │
└────────────────────────────────────────────────────┘

┌─────────────────────────┬──────────────────────────┐
│  📊 Overview            │  📊 Advanced Metrics     │
│  Trades, Win%, P&L      │  Profit Factor, R:R      │
└─────────────────────────┴──────────────────────────┘

┌─────────────────────────┬──────────────────────────┐
│  📈 Recent Trades (15)  │  🔍 Latest Trade Detail  │
│  Full trade history     │  Complete breakdown      │
└─────────────────────────┴──────────────────────────┘

┌──────────┬───────────────┬──────────────────────────┐
│  🌍      │  🚪 Exit      │  📅 Hourly Activity     │
│  Regime  │  Reasons      │  24h Pattern            │
└──────────┴───────────────┴──────────────────────────┘
```

---

## Key Metrics Explained

### What They Mean

| Metric | What It Shows | Target |
|--------|---------------|--------|
| **Win Rate** | % of profitable trades | >50% |
| **Profit Factor** | Total wins ÷ total losses | >1.5 |
| **Expectancy** | Average profit per trade | >£0.50 |
| **Risk:Reward** | Avg win ÷ avg loss | >1.5 |
| **Confidence** | Model certainty (0-100%) | >50% |
| **Hold Duration** | Minutes per trade | 5-120 |

### Interpreting Results

**🟢 Excellent Training**:
- Win rate: >55%
- Profit factor: >1.8
- Positive expectancy
- Most exits: take-profit

**🟡 Learning Phase**:
- Win rate: 45-55%
- Profit factor: 1.2-1.5
- Small positive expectancy
- Mixed exit reasons

**🔴 Needs Attention**:
- Win rate: <40%
- Profit factor: <1.0
- Negative expectancy
- Most exits: stop-loss

---

## Common Questions

### Q: Which dashboard should I use?
**A**: Start with **ultra-detailed** to understand what's happening. Switch to standard once comfortable.

### Q: Can I run multiple dashboards?
**A**: Yes! They're all read-only and query the same database.

### Q: Will dashboards slow down training?
**A**: No. Dashboards only read data and have negligible impact (<1ms overhead).

### Q: Why don't I see any trades?
**A**: Training takes 5-10 minutes to:
1. Load historical data
2. Build features
3. Start executing shadow trades

Be patient!

### Q: What if data looks wrong?
**A**:
1. Check PostgreSQL is running: `psql -U haq -d huracan -c "SELECT 1"`
2. Verify training is active: `ps aux | grep train_sol`
3. Check trades exist: `psql -U haq -d huracan -c "SELECT COUNT(*) FROM trade_memory"`

---

## Advanced Usage

### Run Both Dashboards Simultaneously

**Terminal 1**: Training
```bash
python scripts/train_sol_full.py
```

**Terminal 2**: Ultra-detailed dashboard
```bash
python scripts/ultra_detailed_dashboard.py
```

**Terminal 3**: Standard dashboard (for comparison)
```bash
python scripts/training_dashboard.py
```

### Custom Refresh Rates

Edit the dashboard file:
```python
# In ultra_detailed_dashboard.py, line 724:
await asyncio.sleep(1.5)  # Change to 0.5 for faster updates

# In training_dashboard.py, line 310:
await asyncio.sleep(2.0)  # Change to 1.0 for faster updates
```

### Export Dashboard Data

```bash
# Export all trades to CSV
psql -U haq -d huracan -c "COPY (
  SELECT * FROM trade_memory
  WHERE symbol='SOL/USDT'
  ORDER BY trade_id DESC
) TO '/tmp/trades.csv' CSV HEADER"

# Export performance summary
psql -U haq -d huracan -c "
  SELECT
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE is_winner) as wins,
    AVG(net_profit_gbp) as avg_profit,
    SUM(net_profit_gbp) as total_profit
  FROM trade_memory
  WHERE symbol='SOL/USDT'
" > /tmp/summary.txt
```

---

## Troubleshooting

### Dashboard won't start
```
❌ Failed to connect to database
```

**Solution**:
```bash
# Start PostgreSQL
brew services start postgresql@14

# Verify connection
psql -U haq -d huracan -c "SELECT 1"
```

### No trades showing
```
Total Trades: 0
```

**Solution**: This is normal! Wait 5-10 minutes for:
1. Data loading
2. Feature building
3. First trades to execute

**Check training progress**:
```bash
# See if training is running
ps aux | grep train_sol

# Check logs
tail -f logs/training.log
```

### Stale data (not updating)
**Solution**:
1. Check timestamp in header - should update every 1-2 seconds
2. Restart dashboard: Press Ctrl+C, then run again
3. Verify training is still running

### Wrong symbol showing
**Solution**: Dashboards filter to `SOL/USDT` by default. Check training script is using the same symbol.

---

## Performance Monitoring Best Practices

### During Training

1. ✅ **First 30 minutes**: Watch for first trades, verify system working
2. ✅ **After 100 trades**: Check win rate trend, profit factor
3. ✅ **After 500 trades**: Analyze regime distribution, exit patterns
4. ✅ **End of training**: Review advanced metrics, export data

### Red Flags to Watch For

⚠️ **All stop-losses**: Model not reaching profit targets
⚠️ **Single regime only**: Not adapting to different conditions
⚠️ **Declining win rate**: Model may be overfitting
⚠️ **Low confidence**: Need more training data or iterations

---

## Documentation Index

### Quick Start
- **[Quick Start Guide](DASHBOARD_QUICK_START.md)** - Start here! (5 min read)

### Detailed Information
- **[Ultra-Detailed Dashboard Docs](ULTRA_DETAILED_DASHBOARD.md)** - Complete feature guide (15 min read)
- **[Dashboard Comparison](DASHBOARD_COMPARISON.md)** - Standard vs Ultra-detailed (10 min read)

### Technical Details
- **[Training Pipeline Docs](../src/cloud/training/pipelines/enhanced_rl_pipeline.py)** - What creates the data
- **[Shadow Trader Docs](../src/cloud/training/backtesting/shadow_trader.py)** - How trades are executed
- **[Memory Store Docs](../src/cloud/training/memory/store.py)** - Where data is stored

---

## Summary

You now have **professional-grade monitoring** for your SOL/USDT training:

✅ **Real-time visibility** - See every trade as it happens
✅ **Deep insights** - Understand WHY decisions are made
✅ **Advanced metrics** - Professional performance analysis
✅ **Pattern recognition** - What works and when
✅ **Complete transparency** - Nothing hidden

**Start monitoring**:
```bash
# Terminal 1
python scripts/train_sol_full.py

# Terminal 2
python scripts/ultra_detailed_dashboard.py
```

**See everything happening in your training - in real-time!** 🚀

---

## Contributing

Want to add features to the dashboard? Dashboards are in `scripts/` directory:
- `ultra_detailed_dashboard.py` - Main detailed dashboard
- `training_dashboard.py` - Standard dashboard

Feel free to:
- Add new panels
- Create custom visualizations
- Modify layouts
- Build new dashboard variants

All dashboards are read-only, so experiments are safe!

---

## Support

Issues with dashboards?
1. Check [Quick Start Guide](DASHBOARD_QUICK_START.md) troubleshooting section
2. Review [full documentation](ULTRA_DETAILED_DASHBOARD.md)
3. Check PostgreSQL connection and training status
4. Verify data exists in database

---

**Happy monitoring!** 📊🚀
