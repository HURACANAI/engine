# Web Dashboard - Live Training Monitor

## 🚀 Quick Start

```bash
# Start the web dashboard server
python scripts/web_dashboard_server.py

# Open in your browser
open http://localhost:5055/
```

That's it! The dashboard is now live and updating in real-time.

## Overview

The web dashboard provides a **beautiful, real-time interface** for monitoring SOL/USDT training directly in your browser. No terminal required!

### ✨ Features

- 🎯 **Win Rate Tracking** - Live win/loss statistics
- 💰 **P&L Monitoring** - Real-time profit/loss with expectancy and profit factor
- 📊 **Performance Metrics** - Average win/loss, risk:reward ratio, confidence scores
- 🌍 **Regime Analysis** - Visual breakdown of market conditions
- 🚪 **Exit Reasons** - How trades are closing
- 📅 **24h Activity** - Hourly trading patterns
- 📈 **Recent Trades Table** - Last 15 trades with full details
- ⚡ **Real-Time Updates** - Auto-refreshes every 1.5 seconds

## How It Works

The web dashboard consists of two parts:

1. **Flask Backend** ([scripts/web_dashboard_server.py](../scripts/web_dashboard_server.py))
   - Queries PostgreSQL database
   - Serves data via REST API and Server-Sent Events (SSE)
   - Runs on port 5055

2. **HTML/CSS/JS Frontend** ([templates/dashboard.html](../templates/dashboard.html))
   - Beautiful, responsive design
   - Real-time updates using EventSource API
   - Interactive charts and visualizations

## Starting the Dashboard

### Method 1: Direct Start (Recommended)

```bash
python scripts/web_dashboard_server.py
```

Output:
```
============================================================
🚀 SOL/USDT Training Dashboard
============================================================
📊 Dashboard URL: http://localhost:5055/
🔌 API Endpoint:  http://localhost:5055/api/data
💓 Health Check:  http://localhost:5055/api/health
============================================================
Press Ctrl+C to stop
============================================================
```

### Method 2: Background Process

```bash
# Start in background
nohup python scripts/web_dashboard_server.py > /tmp/dashboard.log 2>&1 &

# Check it's running
curl http://localhost:5055/api/health

# View logs
tail -f /tmp/dashboard.log

# Stop it
lsof -ti:5055 | xargs kill
```

### Method 3: With Training

Terminal 1:
```bash
python scripts/train_sol_full.py
```

Terminal 2:
```bash
python scripts/web_dashboard_server.py
```

Browser:
```
http://localhost:5055/
```

## Dashboard Layout

### Top Section
```
┌─────────────────────────────────────────────────────┐
│  🚀 SOL/USDT Training Dashboard                     │
│  [Live Indicator] Updated: HH:MM:SS                 │
└─────────────────────────────────────────────────────┘
```

### Metrics Grid
```
┌──────────────┬──────────────┬──────────────┐
│  🎯 Win Rate │  💰 Total P&L│  📊 Perform  │
│              │              │              │
│  56.0%       │  +£45.32     │  Avg Win     │
│  Wins: 28    │  Expect:     │  Avg Loss    │
│  Loss: 22    │  Profit:     │  Risk:Reward │
└──────────────┴──────────────┴──────────────┘

┌──────────────┬──────────────┬──────────────┐
│  🌍 Regimes  │  🚪 Exits    │  📅 24h Act  │
│              │              │              │
│  TREND  55%  │  TP: 60%     │  [Bar Chart] │
│  RANGE  35%  │  SL: 35%     │              │
│  PANIC  10%  │  SIG: 5%     │              │
└──────────────┴──────────────┴──────────────┘
```

### Recent Trades Table
```
┌─────────────────────────────────────────────────────┐
│  📈 Recent Trades                                   │
├─────────────────────────────────────────────────────┤
│ ID  Time  Entry   Exit    P&L   BPS  Hold  Result  │
│ #50 14:32 $180.45 $182.10 £2.10 +20  45m   ✅ WIN │
│ #49 13:15 $179.80 $179.20 -£0.85 -15 30m   ❌ LOSS│
└─────────────────────────────────────────────────────┘
```

## API Endpoints

### GET /
**Description**: Main dashboard page (HTML interface)

**Returns**: Beautiful web dashboard

### GET /api/data
**Description**: Get all current dashboard data as JSON

**Response**:
```json
{
  "overview": {
    "total_trades": 50,
    "winning_trades": 28,
    "losing_trades": 22,
    "win_rate": 56.0,
    "total_profit_gbp": 45.32,
    "avg_profit_gbp": 3.21,
    "avg_loss_gbp": -1.85,
    "profit_factor": 1.85,
    "expectancy": 0.91,
    "risk_reward": 1.73,
    "avg_confidence": 62.5
  },
  "regimes": [
    {"regime": "trend", "count": 28},
    {"regime": "range", "count": 18},
    {"regime": "panic", "count": 4}
  ],
  "exit_reasons": [
    {"reason": "TAKE_PROFIT", "count": 30},
    {"reason": "STOP_LOSS", "count": 18},
    {"reason": "MODEL_SIGNAL", "count": 2}
  ],
  "recent_trades": [...],
  "hourly": {...},
  "pnl_series": [...],
  "timestamp": "2025-11-12T14:30:00.000Z"
}
```

### GET /api/stream
**Description**: Server-Sent Events stream for real-time updates

**Response**: Continuous stream of data updates every 1.5 seconds

**Usage**:
```javascript
const eventSource = new EventSource('/api/stream');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Update dashboard...
};
```

### GET /api/health
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "ok",
  "database": "connected"
}
```

## Features in Detail

### 🎯 Win Rate Card
- **Big Number**: Current win rate percentage
  - Green if ≥50%
  - Red if <50%
- **Winning Trades**: Count of successful trades
- **Losing Trades**: Count of unsuccessful trades

### 💰 Total P&L Card
- **Big Number**: Cumulative profit/loss in GBP
  - Green for positive
  - Red for negative
- **Expectancy**: Expected profit per trade
- **Profit Factor**: Ratio of total wins to total losses

### 📊 Performance Card
- **Avg Win**: Average profit per winning trade (£ and bps)
- **Avg Loss**: Average loss per losing trade (£ and bps)
- **Risk:Reward**: Ratio of average win to average loss
- **Avg Confidence**: Mean model confidence score

### 🌍 Market Regimes Card
Visual breakdown of trades by market condition:
- **TREND**: Directional movement (green badge)
- **RANGE**: Sideways/choppy (yellow badge)
- **PANIC**: High volatility (red badge)

Shows count and percentage for each regime.

### 🚪 Exit Reasons Card
Analysis of how trades are closing:
- **TAKE_PROFIT**: Hit profit target
- **STOP_LOSS**: Hit loss limit
- **MODEL_SIGNAL**: Model optimal exit
- **TIMEOUT**: Max hold duration

### 📅 24h Activity Chart
Interactive bar chart showing:
- Trades per hour
- P&L per hour
- Visual patterns in trading activity

Helps identify:
- Peak trading hours
- Most profitable times
- Activity patterns

### 📈 Recent Trades Table
Last 15 trades with complete details:
- Trade ID
- Entry/exit time
- Entry/exit price
- P&L (colored)
- Basis points
- Hold duration
- Confidence score
- Market regime (badged)
- Exit reason
- Result (WIN/LOSS badge)

## Customization

### Refresh Rate

Edit [scripts/web_dashboard_server.py](../scripts/web_dashboard_server.py:413):
```python
yield f"data: {json.dumps(data)}\n\n"
time.sleep(1.5)  # Change to 0.5 for faster updates
```

### Port

Edit [scripts/web_dashboard_server.py](../scripts/web_dashboard_server.py:434):
```python
app.run(host='0.0.0.0', port=5055, debug=False, threaded=True)
#                             ^^^^
#                             Change port here
```

### Styling

Edit [templates/dashboard.html](../templates/dashboard.html) to customize:
- Colors (search for color codes like `#667eea`, `#4caf50`)
- Layout (modify grid columns, card sizes)
- Fonts (change `font-family`)
- Animations (adjust transition times)

## Troubleshooting

### Dashboard won't start

**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 5055
lsof -ti:5055 | xargs kill -9

# Try starting again
python scripts/web_dashboard_server.py
```

### Dashboard shows no data

**Check 1**: Is PostgreSQL running?
```bash
psql -U haq -d huracan -c "SELECT COUNT(*) FROM trade_memory"
```

**Check 2**: Is training generating trades?
```bash
psql -U haq -d huracan -c "SELECT COUNT(*) FROM trade_memory WHERE symbol='SOL/USDT'"
```

**Check 3**: Check API endpoint directly
```bash
curl http://localhost:5055/api/data | jq .
```

### Dashboard not updating

**Check 1**: Look at browser console (F12)
- Should see EventSource connections
- Should see data updates

**Check 2**: Check server logs
```bash
tail -f /tmp/dashboard.log
```

**Check 3**: Refresh the page (Ctrl+R or Cmd+R)

### Slow performance

**Issue**: Dashboard feels sluggish

**Solution 1**: Increase refresh interval
```python
# In web_dashboard_server.py
time.sleep(3.0)  # Update every 3 seconds instead of 1.5
```

**Solution 2**: Limit recent trades
```python
# In web_dashboard_server.py, line 177
LIMIT 10  # Instead of 20
```

**Solution 3**: Use caching (already implemented)
- Dashboard caches data for 1 second
- Reduces database load

## Architecture

### Data Flow

```
PostgreSQL Database
        ↓
  [Flask Backend]
  - Query database
  - Calculate metrics
  - Cache results (1s)
        ↓
   [REST API]
  /api/data - JSON
  /api/stream - SSE
        ↓
 [Browser Client]
  - Fetch initial data
  - Connect to EventSource
  - Update UI in real-time
```

### Technology Stack

**Backend**:
- Flask 3.x - Web framework
- psycopg2 - PostgreSQL driver
- Server-Sent Events (SSE) - Real-time streaming

**Frontend**:
- Vanilla JavaScript - No framework overhead
- CSS Grid - Responsive layout
- EventSource API - Real-time updates
- Fetch API - Initial data load

### Performance

- **Update Frequency**: 1.5 seconds
- **Database Queries**: ~6 per update
- **Query Time**: <50ms total
- **Cache Duration**: 1 second
- **Memory Usage**: ~50MB
- **CPU Usage**: <5%

## Comparison: Terminal vs Web

| Aspect | Terminal Dashboard | Web Dashboard |
|--------|-------------------|---------------|
| **Interface** | Text-based (Rich) | Browser-based (HTML/CSS) |
| **Access** | Terminal only | Any browser |
| **Sharing** | Screenshot | Share URL |
| **Portability** | Local only | Network accessible |
| **Styling** | Limited colors | Full CSS |
| **Interactivity** | Minimal | High (hover, click) |
| **Mobile Friendly** | No | Yes (responsive) |
| **Setup** | Run script | Run server + browser |

## Use Cases

### During Training
```bash
# Terminal 1: Training
python scripts/train_sol_full.py

# Terminal 2: Web dashboard
python scripts/web_dashboard_server.py

# Browser: Monitor
http://localhost:5055/
```

### Remote Monitoring
```bash
# Start dashboard server
python scripts/web_dashboard_server.py

# Access from another machine
http://your-machine-ip:5055/
```

### Presenting Results
- Open dashboard in browser
- Full screen (F11)
- Show live training to stakeholders
- Professional interface

### Development
- Monitor training while coding
- Quick glance at performance
- No terminal switching needed

## Advanced Tips

### Run as System Service (macOS)

Create `~/Library/LaunchAgents/com.huracan.dashboard.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.huracan.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/engine/scripts/web_dashboard_server.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.huracan.dashboard.plist
```

### Multiple Instances

Run dashboards on different ports for different symbols:
```python
# Edit port in web_dashboard_server.py
app.run(port=5056)  # BTC dashboard
app.run(port=5057)  # ETH dashboard
```

### Reverse Proxy (Production)

Use nginx to serve dashboard:
```nginx
server {
    listen 80;
    server_name dashboard.huracan.com;

    location / {
        proxy_pass http://localhost:5055;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Security Notes

⚠️ **Important**:
- Dashboard is currently **not authenticated**
- Only run on trusted networks
- Do not expose to public internet without authentication
- Consider adding basic auth if needed

## Summary

The web dashboard provides:
- ✅ Beautiful, professional interface
- ✅ Real-time updates (1.5s)
- ✅ Comprehensive metrics
- ✅ Easy sharing and presentation
- ✅ Mobile-friendly responsive design
- ✅ No terminal required

**Perfect for**:
- Monitoring during training
- Presenting to stakeholders
- Remote access
- Professional reporting

---

## Quick Reference

**Start**: `python scripts/web_dashboard_server.py`
**URL**: http://localhost:5055/
**API**: http://localhost:5055/api/data
**Health**: http://localhost:5055/api/health
**Stop**: `Ctrl+C` or `lsof -ti:5055 | xargs kill`

**Enjoy your beautiful, real-time training dashboard!** 🚀
