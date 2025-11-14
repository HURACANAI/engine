# 📊 Enhanced Comprehensive Dashboard

## Overview

The Huracan Engine now includes a **super comprehensive, advanced, and intuitive** real-time dashboard that shows **EVERYTHING** the engine is doing with **updates every second**.

## Access

**Dashboard URL:** http://localhost:5055/

The dashboard automatically starts when you run the enhanced dashboard server.

## Features

### ✨ Super Comprehensive
- **Real-time metrics** from all data sources
- **Comprehensive analytics** including P&L, win rates, regime distribution, confidence heatmap
- **Performance by symbol** with detailed breakdowns
- **Gate performance** metrics
- **Learning metrics** (AUC, ECE, model improvements)
- **System health** monitoring (CPU, memory, disk, network)

### 🎯 Super Advanced
- **Real-time charts** using Chart.js
  - P&L over time (cumulative)
  - Win rate trends
  - Market regime distribution
  - Confidence heatmap by regime
  - Performance by symbol
- **Interactive visualizations** with smooth animations
- **Deep insights** into engine performance
- **Comprehensive trade analysis**

### 💎 Super Intuitive
- **Beautiful, modern UI** with gradient backgrounds
- **Color-coded metrics** (green for positive, red for negative)
- **Clear visual indicators** for system health
- **Responsive design** that works on all screen sizes
- **Smooth animations** and transitions

### 🔄 Real-Time Updates
- **Updates every second** via Server-Sent Events (SSE)
- **No page refresh needed** - data streams in real-time
- **Smooth chart animations** as data updates
- **Live status indicators** showing connection status

## What It Shows

### 1. **Training Progress** (Real-time)
- Current training stage
- Progress percentage with visual bar
- Detailed status messages
- Stage-specific details

### 2. **Key Metrics**
- **Win Rate**: Overall win rate with winning/losing trades breakdown
- **Total P&L**: Cumulative profit/loss with expectancy, profit factor, risk:reward
- **Performance**: Avg win/loss, max win/loss, avg confidence, avg hold time
- **System Health**: CPU usage, memory usage, disk usage, database status

### 3. **Real-Time Charts**
- **P&L Over Time**: Cumulative P&L chart for last 7 days
- **Win Rate Trend**: Win rate and confidence trends over time
- **Market Regimes**: Distribution of trades by market regime
- **Confidence Heatmap**: Average confidence by regime
- **Performance by Symbol**: P&L breakdown by trading symbol

### 4. **Gate Performance**
- Gate blocking statistics
- Block accuracy metrics
- Total blocks per gate

### 5. **Learning Metrics**
- **AUC**: Area Under Curve (model performance)
- **ECE**: Expected Calibration Error
- **Delta AUC**: Improvement in model performance
- **Samples Processed**: Total training samples

### 6. **Performance by Symbol**
- Detailed table showing:
  - Total trades per symbol
  - Win rate per symbol
  - Total P&L per symbol
  - Average P&L per symbol
  - Average win/loss per symbol
  - Average confidence per symbol
  - Max win/loss per symbol

### 7. **Recent Trades** (Last 100)
- Trade ID, symbol, time
- Entry/exit prices
- P&L and basis points
- Hold duration
- Model confidence
- Market regime
- Win/loss status

## Usage

### Start Enhanced Dashboard Server

```bash
python scripts/enhanced_dashboard_server.py
```

Or use the comprehensive dashboard server (which now uses the enhanced dashboard):

```bash
python scripts/comprehensive_dashboard_server.py
```

### Access Dashboard

Open your browser and navigate to:
```
http://localhost:5055/
```

The dashboard will automatically:
- Connect to the data stream
- Start updating every second
- Display all metrics and charts
- Show real-time system health

## Technical Details

### Data Sources
- **PostgreSQL**: Trade data from `trade_memory` table
- **SQLite Journal**: Shadow trades from `observability/data/sqlite/journal.db`
- **SQLite Learning**: Training sessions from `observability/data/sqlite/learning.db`
- **Training Progress**: JSON file from `training_progress.json`
- **System Health**: Real-time metrics from `psutil`

### Update Frequency
- **Server-side**: Data fetched every 0.5 seconds (cached)
- **Client-side**: Updates streamed every 1 second via SSE
- **Charts**: Updated smoothly with Chart.js animations

### API Endpoints
- **GET /**: Main dashboard page
- **GET /api/data**: Get all dashboard data as JSON
- **GET /api/stream**: Server-Sent Events stream for real-time updates
- **GET /api/health**: Health check endpoint

## Requirements

- **Python 3.8+**
- **Flask**: Web server framework
- **psycopg2**: PostgreSQL database connector
- **psutil**: System health metrics
- **Chart.js**: Client-side charting library (loaded from CDN)

## Configuration

### Database Configuration
Edit the `DB_CONFIG` in the server script:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'huracan',
    'user': 'haq',
}
```

### Port Configuration
Default port is `5055`. To change it, edit the server script:
```python
app.run(host='0.0.0.0', port=5055, debug=False, threaded=True)
```

## Troubleshooting

### Dashboard not updating
- Check that the server is running
- Check browser console for errors
- Verify database connection
- Check that data exists in the database

### Charts not showing
- Check that Chart.js is loaded (check browser console)
- Verify that data is being fetched (check Network tab)
- Check that data format is correct

### Database connection errors
- Verify database is running
- Check database credentials
- Verify database exists and has data
- Check that `trade_memory` table exists

### Missing data
- Check that trades exist in the database
- Verify that observability modules are available
- Check that SQLite databases exist (if using observability features)

## Notes

- The dashboard reads training progress from `training_progress.json`
- Database connection is required for trade data
- System health metrics require `psutil` package
- Observability modules are optional but enhance the dashboard
- Charts update smoothly with Chart.js animations
- All data updates in real-time without page refresh

## Future Enhancements

- Add filters for time range selection
- Add symbol filtering
- Add regime filtering
- Add export functionality
- Add more advanced analytics
- Add comparison views
- Add historical data analysis
