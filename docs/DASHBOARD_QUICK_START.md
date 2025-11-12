# Ultra-Detailed Dashboard - Quick Start Guide

## 30-Second Start

```bash
# Terminal 1: Start training
python scripts/train_sol_full.py

# Terminal 2: Watch it happen
python scripts/ultra_detailed_dashboard.py
```

That's it! You'll now see EVERYTHING happening during training.

## What You'll See

### Top Section: Overview + Advanced Metrics
```
Total Trades | Win Rate | P&L | Avg Win/Loss | Hold Time | Confidence
        ↓
Profit Factor | Expectancy | Risk:Reward | Best/Worst Trades
```
**What to look for**: Win rate >50%, Profit factor >1.5, Positive expectancy

### Middle Section: Recent Trades + Latest Trade Details
```
Last 15 trades with full details        Deep dive into most recent trade
Entry→Exit prices, P&L, Hold time  →    Full context: Why? When? How well?
Regime, Exit reason, Result             Entry conditions, Exit analysis
```
**What to look for**: Diverse exit reasons, increasing confidence over time

### Bottom Section: Regime + Exit Reasons + Hourly Activity
```
Market conditions          How trades close         When trades happen
Trend | Range | Panic  →   TP | SL | Signal    →   Hour-by-hour breakdown
```
**What to look for**: Balance across regimes, mostly take-profits, active trading

## Reading the Dashboard

### Colors Mean Things

| Color | Meaning | Examples |
|-------|---------|----------|
| 🟢 Green | Positive, good | Profit, wins, trend regime |
| 🔴 Red | Negative, warning | Loss, stop-loss, panic regime |
| 🟡 Yellow | Neutral, caution | Range regime, timeout exits |
| 🔵 Blue | Info, analysis | Model signals, general data |
| ⚪ White/Dim | Neutral data | Unknown, timestamps |

### Symbols Guide

| Symbol | Meaning |
|--------|---------|
| ✅ | Win / Success |
| ❌ | Loss / Failure |
| 📊 | Statistics / Metrics |
| 📈 | Upward trend / Positive |
| 📉 | Downward trend / Negative |
| 🎯 | Target / Goal |
| ⏱️ | Time / Duration |
| 💰 | Money / P&L |
| 🧠 | AI / Decision logic |
| 🌍 | Market / Environment |
| 🚪 | Exit / Close |
| 📅 | Calendar / Time-based |

## Key Metrics Explained (One Line Each)

- **Win Rate**: % of winning trades (target: >50%)
- **Profit Factor**: Total wins / total losses (target: >1.5)
- **Expectancy**: Average profit per trade (must be positive)
- **Risk:Reward**: Win size / loss size (target: >1.5)
- **Confidence**: How sure the model is (0-100%)
- **Hold Duration**: How long trades are open (minutes)
- **Regime**: Market condition (trend/range/panic)
- **BPS**: Basis points (1 bps = 0.01%)

## Common Patterns

### 🟢 Healthy Training
```
Win Rate: 55%
Profit Factor: 1.8
Expectancy: £1.20/trade
Exit Reasons: 60% take-profit, 35% stop-loss, 5% signal
Regime: Balanced across trend/range
```
**Action**: Keep training! It's learning well.

### 🟡 Needs Adjustment
```
Win Rate: 48%
Profit Factor: 1.2
Expectancy: £0.30/trade
Exit Reasons: 40% take-profit, 55% stop-loss, 5% timeout
Regime: Heavy on panic regime
```
**Action**: Model struggling but improving. Monitor for trends.

### 🔴 Problematic
```
Win Rate: 35%
Profit Factor: 0.8
Expectancy: -£0.50/trade
Exit Reasons: 20% take-profit, 75% stop-loss, 5% timeout
Regime: Only trading in one regime
```
**Action**: Stop and investigate. Check data quality and config.

## Troubleshooting (2 Minutes)

### Problem: "Failed to connect to database"
**Fix**:
```bash
# Check PostgreSQL is running
psql -U haq -d huracan -c "SELECT 1"
```

### Problem: "No trades yet"
**Wait**: Training takes ~5-10 minutes to start executing trades
**Check**: Is training script running?
```bash
ps aux | grep train_sol
```

### Problem: Dashboard not updating
**Check**: Look at timestamp in header - should update every 1.5 seconds
**Fix**: Restart dashboard (Ctrl+C, then run again)

### Problem: Data looks wrong
**Check**: Are you looking at the right symbol? (Should be SOL/USDT)
**Fix**:
```bash
# Verify trades in database
psql -U haq -d huracan -c "SELECT COUNT(*) FROM trade_memory WHERE symbol='SOL/USDT'"
```

## Quick Analysis Checklist

When checking the dashboard, look at:

1. ✅ **Is training active?**
   - Timestamp updating?
   - Recent trades showing?

2. ✅ **Is model learning?**
   - Win rate improving over time?
   - Confidence increasing?

3. ✅ **Are trades profitable?**
   - Total P&L positive?
   - Profit factor >1.0?

4. ✅ **Is risk managed?**
   - Stop-losses working?
   - No excessive losses?

5. ✅ **Is model adaptive?**
   - Trading in multiple regimes?
   - Diverse exit reasons?

## Advanced Tips

### Watch for Patterns
- **Early training**: Low confidence, mixed results, learning phase
- **Mid training**: Confidence rising, win rate improving, patterns emerging
- **Late training**: High confidence, consistent wins, refined strategy

### Time-Based Analysis
- Check hourly activity: Are certain hours more profitable?
- May indicate time-based patterns in SOL/USDT

### Regime Analysis
- Which regime has highest win rate?
- Should model avoid certain conditions?

### Exit Analysis
- Too many stop-losses? Position sizing too aggressive
- Too many timeouts? Hold duration too long
- Mostly take-profits? Model is learning well!

## Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Win Rate | >40% | >50% | >60% |
| Profit Factor | >1.0 | >1.5 | >2.0 |
| Expectancy | >£0 | >£0.50 | >£1.00 |
| Risk:Reward | >1.0 | >1.5 | >2.0 |
| Avg Confidence | >30% | >50% | >70% |

## Keyboard Shortcuts

- **Ctrl+C**: Stop dashboard (safely)
- **Ctrl+L**: Clear terminal before restarting
- **↑ Arrow**: Scroll through command history

## Next Steps

1. ✅ Start training and dashboard
2. ✅ Watch for 10-15 minutes to see patterns
3. ✅ Check win rate and profit factor
4. ✅ Analyze regime and exit distributions
5. ✅ Read [full documentation](ULTRA_DETAILED_DASHBOARD.md) for deeper insights

## One-Line Summary of Each Panel

| Panel | One-Line Summary |
|-------|------------------|
| **Header** | Current time and uptime |
| **Overview** | High-level: how many trades, how profitable |
| **Advanced Metrics** | Professional analysis: profit factor, expectancy, R:R |
| **Recent Trades** | Last 15 trades with key details |
| **Latest Trade** | Complete breakdown of most recent trade |
| **Regime Analysis** | What market conditions trades happen in |
| **Exit Reasons** | How trades are closing |
| **Hourly Activity** | When trades are happening |

## Getting Help

- 📖 Full docs: [ULTRA_DETAILED_DASHBOARD.md](ULTRA_DETAILED_DASHBOARD.md)
- 🔄 Comparison: [DASHBOARD_COMPARISON.md](DASHBOARD_COMPARISON.md)
- 💬 Questions: Check the code comments in `scripts/ultra_detailed_dashboard.py`

## Remember

The dashboard is **read-only** - it doesn't affect training. You can:
- ✅ Start/stop it anytime
- ✅ Run multiple instances
- ✅ Modify and experiment
- ✅ Run alongside standard dashboard

**Just watch and learn!**

---

**That's everything you need to start using the ultra-detailed dashboard effectively.**

Happy monitoring! 🚀
