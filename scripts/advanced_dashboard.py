#!/usr/bin/env python3
"""
Advanced In-Depth Web Dashboard
Comprehensive training monitoring with detailed metrics
"""

from flask import Flask, jsonify
from flask_cors import CORS
import psycopg2
import os
import re
from datetime import datetime, timedelta
from collections import deque
from threading import Lock

app = Flask(__name__)
CORS(app)

class State:
    def __init__(self):
        self.lock = Lock()
        self.log_file = "/tmp/sol_training_live.log"
        self.log_fp = None
        self.activity = deque(maxlen=100)
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'win_rate': 0.0,
            'current_idx': 0,
            'total_candles': 2161,
            'current_price': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_volume': 0.0,
            'trades_per_hour': 0.0,
            'last_trade_time': None,
            'start_time': datetime.now(),
            'elapsed_time': '0h 0m',
            'estimated_completion': 'Calculating...',
            'speed': 0.0,  # candles per second
        }
        self.trade_history = deque(maxlen=50)  # Last 50 trades
        self.recent_wins = deque(maxlen=10)
        self.recent_losses = deque(maxlen=10)

state = State()

def update_from_db():
    try:
        conn = psycopg2.connect(host='localhost', port=5432, database='huracan', user='haq')
        cur = conn.cursor()

        # Main stats
        cur.execute("""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE is_winner = true),
                   COALESCE(SUM(net_profit_gbp), 0),
                   COALESCE(AVG(CASE WHEN is_winner THEN net_profit_gbp END), 0),
                   COALESCE(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0),
                   COALESCE(MAX(CASE WHEN is_winner THEN net_profit_gbp END), 0),
                   COALESCE(MIN(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0),
                   COALESCE(SUM(position_size_gbp), 0)
            FROM trade_memory WHERE symbol = 'SOL/USDT'
        """)
        row = cur.fetchone()
        if row:
            with state.lock:
                state.stats['total_trades'] = row[0]
                state.stats['wins'] = row[1]
                state.stats['losses'] = row[0] - row[1]
                state.stats['profit'] = float(row[2])
                state.stats['win_rate'] = (row[1] / row[0] * 100) if row[0] > 0 else 0
                state.stats['avg_win'] = float(row[3])
                state.stats['avg_loss'] = float(row[4])
                state.stats['largest_win'] = float(row[5])
                state.stats['largest_loss'] = float(row[6])
                state.stats['total_volume'] = float(row[7])

        # Recent trades
        cur.execute("""
            SELECT trade_id, entry_price, exit_price, net_profit_gbp,
                   is_winner, exit_time, entry_reason
            FROM trade_memory
            WHERE symbol = 'SOL/USDT'
            ORDER BY trade_id DESC
            LIMIT 50
        """)
        trades = []
        for t in cur.fetchall():
            trades.append({
                'id': t[0],
                'entry': float(t[1]),
                'exit': float(t[2]),
                'profit': float(t[3]),
                'winner': t[4],
                'time': t[5].strftime('%H:%M:%S') if t[5] else 'N/A',
                'reason': t[6] or 'N/A'
            })
        with state.lock:
            state.trade_history = deque(trades, maxlen=50)

        # Calculate consecutive wins/losses
        cur.execute("""
            SELECT is_winner FROM trade_memory
            WHERE symbol = 'SOL/USDT'
            ORDER BY trade_id DESC
            LIMIT 100
        """)
        results = [r[0] for r in cur.fetchall()]
        if results:
            with state.lock:
                # Current streak
                current_streak = 0
                for r in results:
                    if r == results[0]:
                        current_streak += 1
                    else:
                        break

                if results[0]:
                    state.stats['consecutive_wins'] = current_streak
                    state.stats['consecutive_losses'] = 0
                else:
                    state.stats['consecutive_losses'] = current_streak
                    state.stats['consecutive_wins'] = 0

                # Max streaks
                max_win_streak = 0
                max_loss_streak = 0
                current_win = 0
                current_loss = 0

                for r in results:
                    if r:
                        current_win += 1
                        current_loss = 0
                        max_win_streak = max(max_win_streak, current_win)
                    else:
                        current_loss += 1
                        current_win = 0
                        max_loss_streak = max(max_loss_streak, current_loss)

                state.stats['max_consecutive_wins'] = max_win_streak
                state.stats['max_consecutive_losses'] = max_loss_streak

        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

def read_logs():
    try:
        with state.lock:
            if state.log_fp is None:
                if os.path.exists(state.log_file):
                    state.log_fp = open(state.log_file, 'r')
                    state.log_fp.seek(0, 2)
                else:
                    return

            new_lines = state.log_fp.readlines()
            for line in new_lines:
                if 'idx=' in line:
                    m = re.search(r'idx=\[35m(\d+)', line)
                    if m:
                        state.stats['current_idx'] = int(m.group(1))
                        # Calculate speed and ETA
                        elapsed = (datetime.now() - state.stats['start_time']).total_seconds()
                        if elapsed > 0:
                            state.stats['speed'] = state.stats['current_idx'] / elapsed
                            remaining = state.stats['total_candles'] - state.stats['current_idx']
                            if state.stats['speed'] > 0:
                                eta_seconds = remaining / state.stats['speed']
                                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                                state.stats['estimated_completion'] = eta_time.strftime('%H:%M:%S')

                        # Calculate elapsed time
                        hours = int(elapsed // 3600)
                        minutes = int((elapsed % 3600) // 60)
                        state.stats['elapsed_time'] = f"{hours}h {minutes}m"

                if 'price=' in line:
                    m = re.search(r'price=\[35m([\d.]+)', line)
                    if m: state.stats['current_price'] = float(m.group(1))

                msg = None
                if 'shadow_entry' in line:
                    msg = f"🔵 Entry at ${state.stats['current_price']:.2f}"
                elif 'trade_stored' in line:
                    msg = "💾 Trade saved to database"
                elif 'winner.*True' in line or 'is_winner.*True' in line:
                    msg = "🎉 Winner!"
                elif 'winner.*False' in line or 'is_winner.*False' in line:
                    msg = "❌ Loss"
                elif 'analyzing_loss' in line:
                    msg = "🔍 Analyzing what went wrong..."
                elif 'preventable.*True' in line:
                    msg = "⚠️  Loss was preventable"
                elif 'reward=' in line:
                    m = re.search(r'reward=\[35m([-\d.]+)', line)
                    if m: msg = f"📊 Reward: {float(m.group(1)):.4f}"

                if msg:
                    state.activity.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'msg': msg
                    })
    except Exception as e:
        print(f"Log Error: {e}")

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Advanced SOL Training Dashboard</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}
.container{max-width:1600px;margin:0 auto}
.header{background:white;padding:30px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.1);
margin-bottom:30px;text-align:center}
.header h1{font-size:2.5em;color:#667eea;margin-bottom:10px}
.header .subtitle{color:#666;font-size:1.1em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px;margin-bottom:20px}
.grid-wide{grid-column:1/-1}
.card{background:white;padding:25px;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,0.1)}
.card h2{font-size:1.3em;color:#667eea;margin-bottom:15px;border-bottom:2px solid #f0f0f0;padding-bottom:10px}
.stat{margin:15px 0}.stat-label{color:#666;font-size:0.9em;margin-bottom:5px}
.stat-value{font-size:2em;font-weight:bold;color:#333}
.stat-value.positive{color:#10b981}.stat-value.negative{color:#ef4444}
.stat-value.medium{font-size:1.5em}.stat-value.small{font-size:1.2em}
.progress-bar{background:#e5e7eb;border-radius:10px;height:30px;overflow:hidden;margin:10px 0}
.progress-fill{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);height:100%;
transition:width 0.5s ease;display:flex;align-items:center;justify-content:center;
color:white;font-weight:bold}
.activity-feed{max-height:400px;overflow-y:auto}
.activity-item{padding:12px;border-left:4px solid #667eea;margin:8px 0;
background:#f9fafb;border-radius:5px;display:flex;gap:15px;font-size:0.9em}
.activity-time{color:#666;font-size:0.85em;min-width:80px;font-weight:600}
.badge{display:inline-block;padding:5px 15px;border-radius:20px;font-size:0.85em;font-weight:600}
.badge.live{background:#fee2e2;color:#991b1b;animation:pulse 2s infinite}
.badge.success{background:#d1fae5;color:#065f46}
.badge.warning{background:#fef3c7;color:#92400e}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.trade-table{width:100%;border-collapse:collapse;font-size:0.85em}
.trade-table th{background:#f9fafb;padding:10px;text-align:left;color:#666;font-weight:600}
.trade-table td{padding:10px;border-bottom:1px solid #f0f0f0}
.trade-table tr:hover{background:#f9fafb}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.metric-box{background:#f9fafb;padding:15px;border-radius:10px}
.metric-box .label{color:#666;font-size:0.85em;margin-bottom:5px}
.metric-box .value{font-size:1.5em;font-weight:bold;color:#333}
</style></head><body><div class="container">

<div class="header">
<h1>⚡ Advanced SOL Training Dashboard</h1>
<p class="subtitle">Comprehensive Real-Time Monitoring & Analytics</p>
<p style="color:#999;margin-top:10px" id="update">Loading...</p>
</div>

<div class="grid">
<!-- Progress Card -->
<div class="card">
<h2>📊 Training Progress</h2>
<div class="stat">
<div class="stat-label">Completion</div>
<div class="progress-bar"><div class="progress-fill" id="bar" style="width:0%">
<span id="pct">0%</span></div></div>
</div>
<div class="stat">
<div class="stat-label">Hours Analyzed</div>
<div class="stat-value medium" id="hours">0 / 2,161</div>
</div>
<div class="stat">
<div class="stat-label">Current SOL Price</div>
<div class="stat-value" style="color:#667eea" id="price">$0.00</div>
</div>
<div class="stat">
<div class="stat-label">Training Speed</div>
<div class="stat-value small" id="speed">0 candles/sec</div>
</div>
<div class="stat">
<div class="stat-label">Elapsed Time</div>
<div class="stat-value small" id="elapsed">0h 0m</div>
</div>
<div class="stat">
<div class="stat-label">Est. Completion</div>
<div class="stat-value small" id="eta">Calculating...</div>
</div>
</div>

<!-- Performance Overview -->
<div class="card">
<h2>💰 Performance Overview</h2>
<div class="stat">
<div class="stat-label">Total Trades Executed</div>
<div class="stat-value" id="trades">0</div>
</div>
<div class="stat">
<div class="stat-label">Win Rate</div>
<div class="stat-value" id="rate">0%</div>
</div>
<div class="stat">
<div class="stat-label">Total P&L (Practice)</div>
<div class="stat-value" id="profit">£0.00</div>
</div>
<div class="stat">
<div class="stat-label">Total Volume Traded</div>
<div class="stat-value small" id="volume">£0</div>
</div>
</div>

<!-- Win/Loss Breakdown -->
<div class="card">
<h2>📈 Win/Loss Breakdown</h2>
<div class="metric-grid">
<div class="metric-box">
<div class="label">Winning Trades</div>
<div class="value positive" id="wins">0</div>
</div>
<div class="metric-box">
<div class="label">Losing Trades</div>
<div class="value negative" id="losses">0</div>
</div>
<div class="metric-box">
<div class="label">Avg Win</div>
<div class="value positive" id="avg-win">£0.00</div>
</div>
<div class="metric-box">
<div class="label">Avg Loss</div>
<div class="value negative" id="avg-loss">£0.00</div>
</div>
<div class="metric-box">
<div class="label">Largest Win</div>
<div class="value positive" id="max-win">£0.00</div>
</div>
<div class="metric-box">
<div class="label">Largest Loss</div>
<div class="value negative" id="max-loss">£0.00</div>
</div>
</div>
</div>

<!-- Streak Analysis -->
<div class="card">
<h2>🔥 Streak Analysis</h2>
<div class="metric-grid">
<div class="metric-box">
<div class="label">Current Win Streak</div>
<div class="value positive" id="cur-win-streak">0</div>
</div>
<div class="metric-box">
<div class="label">Current Loss Streak</div>
<div class="value negative" id="cur-loss-streak">0</div>
</div>
<div class="metric-box">
<div class="label">Best Win Streak</div>
<div class="value positive" id="max-win-streak">0</div>
</div>
<div class="metric-box">
<div class="label">Worst Loss Streak</div>
<div class="value negative" id="max-loss-streak">0</div>
</div>
</div>
<div class="stat" style="margin-top:15px">
<span class="badge live">🔴 Training Live</span>
<span class="badge success" id="status-badge">Active</span>
</div>
</div>

<!-- Live Activity Feed -->
<div class="card grid-wide">
<h2>🔴 Live Activity Feed</h2>
<div class="activity-feed" id="activity">
<p style="color:#999;text-align:center;padding:20px">Waiting for activity...</p>
</div>
</div>

<!-- Recent Trades Table -->
<div class="card grid-wide">
<h2>📋 Recent Trade History</h2>
<div style="overflow-x:auto">
<table class="trade-table" id="trades-table">
<thead><tr>
<th>Trade ID</th>
<th>Entry Price</th>
<th>Exit Price</th>
<th>Profit/Loss</th>
<th>Result</th>
<th>Time</th>
</tr></thead>
<tbody id="trades-body">
<tr><td colspan="6" style="text-align:center;color:#999">No trades yet...</td></tr>
</tbody>
</table>
</div>
</div>

</div></div>

<script>
function update(){
fetch('/api/stats').then(r=>r.json()).then(d=>{
// Progress
document.getElementById('bar').style.width=d.pct+'%';
document.getElementById('pct').textContent=d.pct+'%';
document.getElementById('hours').textContent=d.idx.toLocaleString()+' / '+d.total.toLocaleString();
document.getElementById('price').textContent='$'+d.price.toFixed(2);
document.getElementById('speed').textContent=d.speed.toFixed(2)+' candles/sec';
document.getElementById('elapsed').textContent=d.elapsed;
document.getElementById('eta').textContent=d.eta;

// Performance
document.getElementById('trades').textContent=d.trades.toLocaleString();
document.getElementById('rate').textContent=d.rate+'%';
document.getElementById('volume').textContent='£'+d.volume.toLocaleString();

const p=document.getElementById('profit');
p.textContent='£'+d.profit.toFixed(2);
p.className='stat-value '+(d.profit>=0?'positive':'negative');

// Breakdown
document.getElementById('wins').textContent=d.wins.toLocaleString();
document.getElementById('losses').textContent=d.losses.toLocaleString();
document.getElementById('avg-win').textContent='£'+d.avg_win.toFixed(2);
document.getElementById('avg-loss').textContent='£'+d.avg_loss.toFixed(2);
document.getElementById('max-win').textContent='£'+d.max_win.toFixed(2);
document.getElementById('max-loss').textContent='£'+d.max_loss.toFixed(2);

// Streaks
document.getElementById('cur-win-streak').textContent=d.cur_win_streak;
document.getElementById('cur-loss-streak').textContent=d.cur_loss_streak;
document.getElementById('max-win-streak').textContent=d.max_win_streak;
document.getElementById('max-loss-streak').textContent=d.max_loss_streak;

// Activity
const a=document.getElementById('activity');
if(d.activity&&d.activity.length>0){
a.innerHTML=d.activity.reverse().map(i=>`<div class="activity-item">
<span class="activity-time">${i.time}</span><span>${i.msg}</span></div>`).join('');
}

// Trades table
const tb=document.getElementById('trades-body');
if(d.trade_history&&d.trade_history.length>0){
tb.innerHTML=d.trade_history.map(t=>`<tr>
<td>#${t.id}</td>
<td>$${t.entry.toFixed(2)}</td>
<td>$${t.exit.toFixed(2)}</td>
<td style="color:${t.profit>=0?'#10b981':'#ef4444'}">£${t.profit.toFixed(2)}</td>
<td>${t.winner?'✅ Win':'❌ Loss'}</td>
<td>${t.time}</td>
</tr>`).join('');
}

document.getElementById('update').textContent='Updated: '+new Date().toLocaleTimeString();
}).catch(e=>console.error(e));}
update();setInterval(update,2000);
</script>
</body></html>'''

@app.route('/api/stats')
def stats():
    update_from_db()
    read_logs()
    with state.lock:
        pct = (state.stats['current_idx'] / state.stats['total_candles'] * 100) if state.stats['total_candles'] > 0 else 0
        return jsonify({
            'pct': round(pct, 1),
            'idx': state.stats['current_idx'],
            'total': state.stats['total_candles'],
            'price': state.stats['current_price'],
            'trades': state.stats['total_trades'],
            'wins': state.stats['wins'],
            'losses': state.stats['losses'],
            'rate': round(state.stats['win_rate'], 1),
            'profit': state.stats['profit'],
            'volume': state.stats['total_volume'],
            'avg_win': state.stats['avg_win'],
            'avg_loss': state.stats['avg_loss'],
            'max_win': state.stats['largest_win'],
            'max_loss': state.stats['largest_loss'],
            'cur_win_streak': state.stats['consecutive_wins'],
            'cur_loss_streak': state.stats['consecutive_losses'],
            'max_win_streak': state.stats['max_consecutive_wins'],
            'max_loss_streak': state.stats['max_consecutive_losses'],
            'speed': state.stats['speed'],
            'elapsed': state.stats['elapsed_time'],
            'eta': state.stats['estimated_completion'],
            'activity': list(state.activity)[-30:],
            'trade_history': list(state.trade_history)[:20]
        })

if __name__ == '__main__':
    print('\n'+'='*80)
    print('  ⚡ Advanced SOL Training Dashboard')
    print('='*80)
    print('\n  📱 Open: http://localhost:5051\n')
    print('='*80+'\n')
    app.run(host='0.0.0.0', port=5051, debug=False)
