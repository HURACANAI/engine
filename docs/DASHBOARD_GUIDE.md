# 📊 Dashboard Guide

Monitor training in real-time with three dashboard options.

---

## 🚀 Quick Start

### Option 1: Web Dashboard (Recommended)
```bash
# Terminal 1: Start training
python scripts/train_sol_full.py

# Terminal 2: Start web dashboard
python scripts/web_dashboard_server.py

# Open browser
open http://localhost:5055/
```

### Option 2: Ultra-Detailed Terminal
```bash
# Terminal 1: Start training
python scripts/train_sol_full.py

# Terminal 2: Start dashboard
python scripts/ultra_detailed_dashboard.py
```

### Option 3: Standard Terminal
```bash
# Terminal 1: Start training
python scripts/train_sol_full.py

# Terminal 2: Start dashboard
python scripts/training_dashboard.py
```

---

## 📊 Dashboard Comparison

| Feature | Web Dashboard | Ultra Terminal | Standard Terminal |
|---------|--------------|----------------|-------------------|
| **Best For** | Presentations, sharing | Deep analysis | Quick checks |
| **Win Rate** | ✅ | ✅ | ✅ |
| **P&L Tracking** | ✅ | ✅ | ✅ |
| **Recent Trades** | 15 trades | 15 trades | 10 trades |
| **Regime Analysis** | ✅ Visual | ✅ Visual | ❌ |
| **Exit Reasons** | ✅ Detailed | ✅ Detailed | ❌ |
| **24h Activity** | ✅ Chart | ✅ Chart | ❌ |
| **Mobile Friendly** | ✅ | ❌ | ❌ |
| **Shareable** | URL | Screenshot | Screenshot |
| **Update Speed** | 1.5s | 1.5s | 2s |

---

## 🌐 Web Dashboard

**URL**: http://localhost:5055/

### Features
- ✅ Real-time updates (1.5s refresh)
- ✅ Beautiful browser interface
- ✅ Mobile-friendly
- ✅ Network accessible
- ✅ Professional for presentations

### What You'll See
- Win rate and P&L metrics
- Recent trades table
- Market regime analysis
- Exit reasons breakdown
- 24-hour activity chart
- Performance indicators

### API Endpoints
- Main: http://localhost:5055/
- Data: http://localhost:5055/api/data
- Health: http://localhost:5055/api/health

---

## ⚡ Ultra-Detailed Terminal

### Features
- ✅ Maximum information density
- ✅ Decision reasoning
- ✅ Pattern analysis
- ✅ Complete trade context

### Panels
1. **Overview** - High-level metrics
2. **Advanced Metrics** - Profit factor, expectancy, R:R
3. **Recent Trades** - Last 15 trades
4. **Latest Trade** - Complete breakdown
5. **Regime Analysis** - Market conditions
6. **Exit Reasons** - How trades close
7. **Hourly Activity** - When trades happen

---

## 📊 Standard Terminal

### Features
- ✅ Quick overview
- ✅ Minimal screen space
- ✅ Fast and simple

### Shows
- Win rate
- Total P&L
- Recent trades (10)
- Basic metrics

---

## 🎯 When to Use Each

### Use Web Dashboard When:
- Presenting to stakeholders
- Monitoring from another machine
- Need mobile access
- Sharing with team

### Use Ultra-Detailed When:
- Deep analysis needed
- Debugging model behavior
- Understanding decisions
- Learning the system

### Use Standard When:
- Quick performance check
- Minimal screen space
- Familiar with system
- Just need basics

---

## 🔧 Troubleshooting

### Dashboard Not Updating
```bash
# Check if training is running
ps aux | grep train_sol

# Restart dashboard
# Press Ctrl+C, then run again
```

### No Data Showing
- Wait 5-10 minutes for trades to execute
- Check database has trades:
  ```bash
  psql -U haq -d huracan -c "SELECT COUNT(*) FROM trade_memory"
  ```

### Web Dashboard Not Loading
```bash
# Kill and restart
lsof -ti:5055 | xargs kill -9
python scripts/web_dashboard_server.py
```

---

## 📈 Key Metrics Explained

- **Win Rate**: % of winning trades (target: >50%)
- **Profit Factor**: Total wins / total losses (target: >1.5)
- **Expectancy**: Average profit per trade (must be positive)
- **Risk:Reward**: Win size / loss size (target: >1.5)
- **Confidence**: Model certainty (0-100%)
- **Hold Duration**: Time trades are open (minutes)
- **Regime**: Market condition (trend/range/panic)

---

## 📚 Related Files

- Scripts: `scripts/web_dashboard_server.py`, `scripts/ultra_detailed_dashboard.py`, `scripts/training_dashboard.py`
- Template: `templates/dashboard.html`

---

**Choose the dashboard that fits your needs!** 🚀

