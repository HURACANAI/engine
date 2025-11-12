# Ultra-Detailed Training Dashboard

## Overview

The ultra-detailed dashboard provides **comprehensive real-time visibility** into every aspect of the SOL/USDT training process. It shows you EVERYTHING that's happening as the model learns from historical data.

## Running the Dashboard

```bash
# Start the dashboard
python scripts/ultra_detailed_dashboard.py

# While training is running in another terminal
python scripts/train_sol_full.py
```

## Dashboard Sections

### 🎯 Header
- **Current Time**: Live UTC timestamp
- **Uptime**: How long the dashboard has been running
- **Connection Status**: Database connection health

### 📊 Overview Panel
Shows high-level performance metrics:
- **Total Trades**: Complete count of executed trades
- **Win Rate**: Percentage of winning trades (target: 50%+)
- **Total P&L**: Cumulative profit/loss in GBP
- **Average Win**: Mean profit per winning trade (£ and bps)
- **Average Loss**: Mean loss per losing trade (£ and bps)
- **Average Hold Time**: Mean duration of trades in minutes
  - Also shows range (shortest to longest trade)
- **Average Confidence**: Mean model confidence across all trades

### 📊 Advanced Metrics Panel
Sophisticated performance analysis:
- **Profit Factor**: Ratio of gross profit to gross loss
  - Good: >1.5 (winning more than losing)
  - Target: 1.5+
- **Expectancy**: Average expected profit per trade
  - Formula: (Win% × AvgWin) - (Loss% × AvgLoss)
  - Positive expectancy = profitable strategy
- **Risk:Reward Ratio**: Average win size vs average loss size
  - Target: 1:1.5+ (wins should be larger than losses)
- **Best Trade**: Largest winning trade
  - Shows multiple of average win
- **Worst Trade**: Largest losing trade
  - Shows multiple of average loss

### 📈 Recent Trades Panel (Last 15)
Real-time feed of the most recent trades showing:
- **ID**: Trade identifier for tracking
- **Time**: Entry time (HH:MM:SS format)
- **Entry→Exit**: Price movement
  - Example: $180.45→$182.10
- **P&L**: Profit/Loss in GBP
  - Green for positive, red for negative
- **Hold**: Duration in minutes
- **Conf**: Model confidence score (%)
- **Regime**: Market condition at entry
  - `trend` (green) - Directional movement
  - `range` (yellow) - Sideways/choppy
  - `panic` (red) - High volatility
- **Exit**: Why the trade was closed
  - `TAKE_PROFIT` - Hit profit target
  - `STOP_LOSS` - Hit loss limit
  - `TIMEOUT` - Max hold time reached
  - `MODEL_SIGNAL` - Model decided to exit
- **Result**: WIN ✅ or LOSS ❌

### 🔍 Latest Trade Details
Ultra-detailed breakdown of the most recent trade:

**📥 Entry Information:**
- Full timestamp with date
- Exact entry price
- Trade direction (LONG/SHORT)
- Model confidence percentage
- Market regime classification
- Market volatility (in bps)
- Bid-ask spread (in bps)

**📤 Exit Information:**
- Exit price
- Exit reason (detailed)
- Hold duration in minutes

**💰 Performance:**
- Net profit/loss in GBP
- Gross profit in basis points
- Final result (WIN/LOSS)

**🧠 Decision Reasoning:**
- Human-readable explanation of why the model entered this trade
- Confidence factors considered
- Pattern matching details

### 🌍 Market Regime Analysis
Breakdown of trades by market condition:
- **Regime Types:**
  - `TREND`: Strong directional movement (green)
  - `RANGE`: Sideways/choppy market (yellow)
  - `PANIC`: High volatility/uncertainty (red)
  - `UNKNOWN`: Unclassified (gray)
- **Trade Count**: Number of trades in each regime
- **Percentage**: Distribution across regimes
- **Visual Bar**: Graphical representation

### 🚪 Exit Reasons
Analysis of how trades are being closed:
- **TAKE_PROFIT**: Successfully hit profit target
- **STOP_LOSS**: Hit loss limit (risk management)
- **TIMEOUT**: Maximum hold duration reached
- **MODEL_SIGNAL**: Model determined optimal exit
- Shows count and percentage for each

### 📅 24h Trading Activity
Hourly breakdown of trading patterns:
- **Hour**: Trading hour (00:00 to 23:00)
- **Trades**: Number of trades in that hour
- **P&L**: Profit/loss for that hour
- **Activity Bar**: Visual representation of trade volume
- Helps identify:
  - Peak trading hours
  - Most profitable times
  - Patterns in trade execution

## Understanding the Data

### What Each Trade Shows

Every trade represents a complete learning cycle:

1. **Entry Decision**
   - Model analyzes 177 features (141 market + 36 engineered)
   - Queries memory store for similar historical patterns
   - Calculates confidence score
   - Checks market regime
   - Decides whether to enter and position size

2. **Position Management**
   - Monitors price movement every candle (1-hour intervals)
   - Tracks unrealized P&L
   - Evaluates exit conditions:
     - Take profit: +20 bps
     - Stop loss: -15 bps
     - Timeout: 120 minutes
     - Model signal: Optimal exit detected

3. **Exit & Learning**
   - Records actual outcome
   - Compares to optimal exit (what SHOULD have happened)
   - Stores pattern in memory for future reference
   - Updates model with reward/penalty
   - Analyzes win/loss for pattern recognition

### Key Metrics Explained

**Basis Points (bps)**
- 1 bps = 0.01%
- 100 bps = 1%
- Used for precise profit/loss measurement
- Example: +20 bps on $1000 = £2 profit

**Confidence Score**
- Ranges from 0% to 100%
- Based on:
  - Sample count (how many similar patterns exist)
  - Pattern reliability (historical win rate)
  - Regime match (does current condition match pattern)
  - Meta-features (orderbook, signals, etc.)
- Minimum threshold: 20% (configurable)

**Market Regime**
- Detected using sophisticated multi-factor analysis
- Considers:
  - Price trends (5m, 1h, 24h)
  - Volatility (short-term vs long-term)
  - Volume patterns
  - Momentum indicators
- Critical for context-aware trading

**Hold Duration**
- How long the trade was open
- Configuration:
  - Min: 1 minute (1 candle)
  - Max: 120 minutes (2 hours)
  - Avg: ~60 minutes
- Affects:
  - Opportunity cost
  - Risk exposure
  - Learning efficiency

## Real-Time Updates

The dashboard updates every **1.5 seconds** with:
- ✅ New trades as they execute
- ✅ Updated performance metrics
- ✅ Recalculated statistics
- ✅ Live regime analysis
- ✅ Current system status

## Performance Targets

### Healthy Training Indicators

✅ **Win Rate: 50%+**
- Indicates model is learning profitable patterns
- Below 40% may indicate overfitting or poor data quality

✅ **Profit Factor: 1.5+**
- Winning significantly more than losing
- Below 1.0 = losing overall

✅ **Positive Expectancy**
- Each trade has positive expected value
- Confirms long-term profitability

✅ **Confidence Scores Increasing**
- Model becoming more certain over time
- More historical patterns to learn from

✅ **Diverse Exit Reasons**
- Not all stop-losses (bad)
- Mix of take-profit and model signals (good)

✅ **Regime Distribution**
- Should see trades in multiple regimes
- Model learning to handle different conditions

### Warning Signs

⚠️ **Win Rate < 40%**
- Model may need retraining
- Check feature quality

⚠️ **All Stop-Loss Exits**
- Not reaching profit targets
- May need to adjust position sizing or timeframes

⚠️ **Single Regime Only**
- Model not adapting to different conditions
- May need more diverse training data

⚠️ **Low Confidence (<30% average)**
- Insufficient historical patterns
- Need more training iterations

⚠️ **Extreme Trades**
- Losses >3x average loss
- May indicate edge cases or data quality issues

## Data Sources

All data comes from the `trade_memory` PostgreSQL table, which stores:
- Complete trade history
- Entry/exit prices and times
- Market conditions at trade time
- Model confidence and reasoning
- Performance metrics
- Pattern embeddings for similarity search

## Integration with Training

The dashboard is designed to run **alongside** the training process:

**Terminal 1**: Run training
```bash
python scripts/train_sol_full.py
```

**Terminal 2**: Monitor with dashboard
```bash
python scripts/ultra_detailed_dashboard.py
```

As training executes:
1. Shadow trader simulates trades on historical data
2. Trades are stored in PostgreSQL
3. Dashboard queries database in real-time
4. You see every decision the model makes
5. Model learns from outcomes
6. Process repeats with improved strategy

## Troubleshooting

### Dashboard won't start
```
❌ Failed to connect to database
```
**Solution**: Ensure PostgreSQL is running:
```bash
psql -U haq -d huracan -c "SELECT 1"
```

### No trades showing
```
No trades yet
```
**Solution**: This is normal if:
- Training hasn't started yet
- Model confidence is below threshold
- Historical data is being loaded

Wait for training to begin executing trades.

### Stale data (not updating)
**Solution**: Check that training is actively running:
```bash
ps aux | grep train_sol
```

## Advanced Usage

### Custom Refresh Rate

Edit line 724 in `ultra_detailed_dashboard.py`:
```python
await asyncio.sleep(1.5)  # Change to 0.5 for faster updates
```

### Filter by Date

Modify SQL queries to focus on specific time periods:
```sql
WHERE entry_timestamp >= NOW() - INTERVAL '24 hours'
```

### Export Data

The dashboard reads from PostgreSQL, so you can export anytime:
```bash
psql -U haq -d huracan -c "COPY (SELECT * FROM trade_memory WHERE symbol='SOL/USDT') TO '/tmp/trades.csv' CSV HEADER"
```

## Summary

The ultra-detailed dashboard provides **complete transparency** into:
- ✅ What trades the model is taking
- ✅ Why it's making those decisions
- ✅ How well it's performing
- ✅ What patterns it's learning
- ✅ Where it needs improvement

Use it to:
- 🎯 Verify training is progressing
- 🎯 Identify areas for improvement
- 🎯 Understand model behavior
- 🎯 Monitor performance metrics
- 🎯 Debug issues in real-time

**Everything you need to know about what's happening during training, in one beautiful dashboard.**
