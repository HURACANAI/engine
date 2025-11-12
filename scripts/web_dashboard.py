#!/usr/bin/env python3
"""
Beautiful Web Dashboard for Training Monitoring

A sleek, modern web interface to monitor SOL/USDT training in real-time.
Open in your browser and watch the AI learn!

Usage:
    python3 scripts/web_dashboard.py

Then open: http://localhost:5050
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import psycopg2
import os
import re
from datetime import datetime
from collections import deque
from threading import Lock

app = Flask(__name__)
CORS(app)

# Global state
class DashboardState:
    def __init__(self):
        self.lock = Lock()
        self.log_file = "/tmp/sol_final_run.log"
        self.log_fp = None
        self.log_position = 0
        self.activity_log = deque(maxlen=50)

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.current_idx = 0
        self.total_candles = 2160
        self.latest_price = 0.0
        self.confidence_threshold = 0.2

        # DB config
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'huracan',
            'user': 'haq',
        }

state = DashboardState()


def get_db_stats():
    """Fetch stats from database"""
    try:
        conn = psycopg2.connect(**state.db_config)
        cur = conn.cursor()

        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_winner = true) as wins,
                COUNT(*) FILTER (WHERE is_winner = false) as losses,
                COALESCE(SUM(net_profit_gbp), 0) as profit
            FROM trade_memory
            WHERE symbol = 'SOL/USDT'
        """)

        row = cur.fetchone()
        if row:
            with state.lock:
                state.total_trades = row[0]
                state.winning_trades = row[1]
                state.losing_trades = row[2]
                state.total_profit = float(row[3])
                state.win_rate = (state.winning_trades / state.total_trades * 100) if state.total_trades > 0 else 0.0

        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")


def read_logs():
    """Read new log lines"""
    try:
        with state.lock:
            if state.log_fp is None:
                if not os.path.exists(state.log_file):
                    return
                state.log_fp = open(state.log_file, 'r')
                state.log_fp.seek(0, 2)
                state.log_position = state.log_fp.tell()

            state.log_fp.seek(state.log_position)
            new_lines = state.log_fp.readlines()
            state.log_position = state.log_fp.tell()

            for line in new_lines:
                line = line.rstrip()
                if not line:
                    continue

                # Extract useful info
                if 'shadow_entry' in line and 'idx=' in line:
                    match = re.search(r'idx=\[35m(\d+)\[0m', line)
                    if match:
                        state.current_idx = int(match.group(1))

                    match = re.search(r'price=\[35m([\d.]+)\[0m', line)
                    if match:
                        state.latest_price = float(match.group(1))

                # Create simple messages
                simple_msg = None
                timestamp = datetime.now().strftime('%H:%M:%S')

                if 'shadow_entry' in line:
                    simple_msg = f"💰 Considering trade at ${state.latest_price:.2f}"
                elif 'trade_stored' in line:
                    match = re.search(r'trade_id=\[35m(\d+)\[0m', line)
                    if match:
                        simple_msg = f"✅ Trade #{match.group(1)} completed"
                elif 'analyzing_loss' in line:
                    simple_msg = "📚 Learning from this trade..."
                elif 'is_winner.*True' in line:
                    simple_msg = "🎉 Profitable trade!"
                elif 'is_winner.*False' in line:
                    simple_msg = "📉 Unprofitable trade, learning why"
                elif 'shadow_trading_start' in line:
                    simple_msg = "🤖 Starting practice trading"
                elif 'historical_data_loaded' in line:
                    simple_msg = "📊 Price data loaded successfully"

                if simple_msg:
                    state.activity_log.append({
                        'time': timestamp,
                        'message': simple_msg
                    })

    except Exception as e:
        print(f"Log read error: {e}")


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """API endpoint for dashboard stats"""
    get_db_stats()
    read_logs()

    with state.lock:
        progress_percent = (state.current_idx / state.total_candles * 100) if state.total_candles > 0 else 0

        return jsonify({
            'progress': {
                'percent': round(progress_percent, 1),
                'current': state.current_idx,
                'total': state.total_candles,
                'current_price': state.latest_price,
            },
            'performance': {
                'total_trades': state.total_trades,
                'winning_trades': state.winning_trades,
                'losing_trades': state.losing_trades,
                'win_rate': round(state.win_rate, 1),
                'total_profit': round(state.total_profit, 2),
            },
            'activity': list(state.activity_log)[-20:]  # Last 20 activities
        })


# Create templates directory
templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
os.makedirs(templates_dir, exist_ok=True)

# Create HTML template
html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOL Training Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 10px;
        }

        .header .subtitle {
            color: #666;
            font-size: 1.1em;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .card h2 {
            font-size: 1.3em;
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .stat {
            margin: 15px 0;
        }

        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }

        .stat-value.positive {
            color: #10b981;
        }

        .stat-value.negative {
            color: #ef4444;
        }

        .progress-bar {
            background: #e5e7eb;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }

        .activity-feed {
            grid-column: 1 / -1;
        }

        .activity-item {
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            background: #f9fafb;
            border-radius: 5px;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .activity-time {
            color: #666;
            font-size: 0.85em;
            min-width: 80px;
        }

        .activity-message {
            flex: 1;
        }

        .badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .badge.success {
            background: #d1fae5;
            color: #065f46;
        }

        .badge.warning {
            background: #fef3c7;
            color: #92400e;
        }

        .badge.info {
            background: #dbeafe;
            color: #1e40af;
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .emoji {
            font-size: 1.5em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SOL Trading Bot Training</h1>
            <p class="subtitle">Watching AI Learn to Trade in Real-Time</p>
            <p style="color: #999; margin-top: 10px;" id="last-update">Loading...</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2><span class="emoji">📊</span> Training Progress</h2>
                <div class="stat">
                    <div class="stat-label">Overall Progress</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-bar" style="width: 0%">
                            <span id="progress-text">0%</span>
                        </div>
                    </div>
                </div>
                <div class="stat">
                    <div class="stat-label">Hours Analyzed</div>
                    <div class="stat-value" id="candles-processed">0 / 2,160</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value" style="color: #667eea;" id="current-price">$0.00</div>
                </div>
            </div>

            <div class="card">
                <h2><span class="emoji">💰</span> Performance</h2>
                <div class="stat">
                    <div class="stat-label">Practice Trades</div>
                    <div class="stat-value" id="total-trades">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-value" id="win-rate">0%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Total P&L (Practice)</div>
                    <div class="stat-value" id="total-profit">£0.00</div>
                </div>
            </div>

            <div class="card">
                <h2><span class="emoji">📈</span> Trade Breakdown</h2>
                <div class="stat">
                    <div class="stat-label">Successful Trades</div>
                    <div class="stat-value positive" id="winning-trades">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Learning Trades</div>
                    <div class="stat-value negative" id="losing-trades">0</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Status</div>
                    <span class="badge info pulse">🔴 Live Training</span>
                </div>
            </div>

            <div class="card activity-feed">
                <h2><span class="emoji">🔴</span> Live Activity Feed</h2>
                <div id="activity-container">
                    <p style="color: #999; text-align: center; padding: 20px;">Waiting for activity...</p>
                </div>
            </div>
        </div>

        <div style="text-align: center; color: white; margin-top: 20px; opacity: 0.8;">
            <p>💡 This is a practice environment - no real money is used</p>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    // Update progress
                    document.getElementById('progress-bar').style.width = data.progress.percent + '%';
                    document.getElementById('progress-text').textContent = data.progress.percent + '%';
                    document.getElementById('candles-processed').textContent =
                        data.progress.current.toLocaleString() + ' / ' + data.progress.total.toLocaleString();
                    document.getElementById('current-price').textContent = '$' + data.progress.current_price.toFixed(2);

                    // Update performance
                    document.getElementById('total-trades').textContent = data.performance.total_trades.toLocaleString();
                    document.getElementById('win-rate').textContent = data.performance.win_rate + '%';
                    document.getElementById('winning-trades').textContent = data.performance.winning_trades.toLocaleString();
                    document.getElementById('losing-trades').textContent = data.performance.losing_trades.toLocaleString();

                    // Update profit with color
                    const profitEl = document.getElementById('total-profit');
                    profitEl.textContent = '£' + data.performance.total_profit.toFixed(2);
                    profitEl.className = 'stat-value ' + (data.performance.total_profit >= 0 ? 'positive' : 'negative');

                    // Update activity feed
                    const activityContainer = document.getElementById('activity-container');
                    if (data.activity && data.activity.length > 0) {
                        activityContainer.innerHTML = data.activity.reverse().map(item => `
                            <div class="activity-item">
                                <span class="activity-time">${item.time}</span>
                                <span class="activity-message">${item.message}</span>
                            </div>
                        `).join('');
                    }

                    // Update timestamp
                    document.getElementById('last-update').textContent =
                        'Last updated: ' + new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error:', error));
        }

        // Update every 2 seconds
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>'''

# Write template file
with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
    f.write(html_template)


if __name__ == '__main__':
    print('\n' + '='*80)
    print('  🚀 SOL Training Dashboard Server')
    print('='*80)
    print('\n  Starting web server...')
    print('\n  📱 Open your browser and go to:')
    print('\n     👉  http://localhost:5050')
    print('\n  Press Ctrl+C to stop the server')
    print('\n' + '='*80 + '\n')

    app.run(host='0.0.0.0', port=5050, debug=False)
