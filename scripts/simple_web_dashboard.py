#!/usr/bin/env python3
"""
Simple Self-Contained Web Dashboard
No templates needed - everything in one file!
"""

from flask import Flask, jsonify
from flask_cors import CORS
import psycopg2
import os
import re
from datetime import datetime
from collections import deque
from threading import Lock

app = Flask(__name__)
CORS(app)

class State:
    def __init__(self):
        self.lock = Lock()
        self.log_file = "/tmp/sol_training_live.log"
        self.log_fp = None
        self.activity = deque(maxlen=50)
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'win_rate': 0.0,
            'current_idx': 0,
            'total_candles': 2160,
            'current_price': 0.0
        }

state = State()

def update_from_db():
    try:
        conn = psycopg2.connect(host='localhost', port=5432, database='huracan', user='haq')
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE is_winner = true),
                   COALESCE(SUM(net_profit_gbp), 0)
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
        conn.close()
    except: pass

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
                    if m: state.stats['current_idx'] = int(m.group(1))
                if 'price=' in line:
                    m = re.search(r'price=\[35m([\d.]+)', line)
                    if m: state.stats['current_price'] = float(m.group(1))

                msg = None
                if 'shadow_entry' in line:
                    msg = f"💰 Trade at ${state.stats['current_price']:.2f}"
                elif 'trade_stored' in line:
                    msg = "✅ Trade completed"
                elif 'winner.*True' in line:
                    msg = "🎉 Profitable!"
                elif 'winner.*False' in line:
                    msg = "📚 Learning..."

                if msg:
                    state.activity.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'msg': msg
                    })
    except: pass

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SOL Training Dashboard</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}
.container{max-width:1200px;margin:0 auto}
.header{background:white;padding:30px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.1);
margin-bottom:30px;text-align:center}
.header h1{font-size:2.5em;color:#667eea;margin-bottom:10px}
.header .subtitle{color:#666;font-size:1.1em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:20px}
.card{background:white;padding:25px;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,0.1)}
.card h2{font-size:1.3em;color:#667eea;margin-bottom:15px}
.stat{margin:15px 0}.stat-label{color:#666;font-size:0.9em;margin-bottom:5px}
.stat-value{font-size:2em;font-weight:bold;color:#333}
.stat-value.positive{color:#10b981}.stat-value.negative{color:#ef4444}
.progress-bar{background:#e5e7eb;border-radius:10px;height:30px;overflow:hidden;margin:10px 0}
.progress-fill{background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);height:100%;
transition:width 0.5s ease;display:flex;align-items:center;justify-content:center;
color:white;font-weight:bold}
.activity-feed{grid-column:1/-1}
.activity-item{padding:15px;border-left:4px solid #667eea;margin:10px 0;
background:#f9fafb;border-radius:5px;display:flex;gap:15px}
.activity-time{color:#666;font-size:0.85em;min-width:80px}
.badge{display:inline-block;padding:5px 15px;border-radius:20px;font-size:0.85em;font-weight:600}
.badge.live{background:#fee2e2;color:#991b1b;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
</style></head><body><div class="container">
<div class="header"><h1>🚀 SOL Training Bot</h1>
<p class="subtitle">Watching AI Learn in Real-Time</p>
<p style="color:#999;margin-top:10px" id="update">Loading...</p></div>
<div class="grid">
<div class="card"><h2>📊 Progress</h2><div class="stat"><div class="stat-label">Training Progress</div>
<div class="progress-bar"><div class="progress-fill" id="bar" style="width:0%">
<span id="pct">0%</span></div></div></div>
<div class="stat"><div class="stat-label">Hours Analyzed</div>
<div class="stat-value" id="hours">0 / 2,160</div></div>
<div class="stat"><div class="stat-label">Current Price</div>
<div class="stat-value" style="color:#667eea" id="price">$0.00</div></div></div>
<div class="card"><h2>💰 Performance</h2><div class="stat"><div class="stat-label">Practice Trades</div>
<div class="stat-value" id="trades">0</div></div>
<div class="stat"><div class="stat-label">Success Rate</div>
<div class="stat-value" id="rate">0%</div></div>
<div class="stat"><div class="stat-label">Total P&L</div>
<div class="stat-value" id="profit">£0.00</div></div></div>
<div class="card"><h2>📈 Breakdown</h2><div class="stat"><div class="stat-label">Wins</div>
<div class="stat-value positive" id="wins">0</div></div>
<div class="stat"><div class="stat-label">Losses</div>
<div class="stat-value negative" id="losses">0</div></div>
<div class="stat"><div class="stat-label">Status</div>
<span class="badge live">🔴 Live</span></div></div>
<div class="card activity-feed"><h2>🔴 Live Activity</h2>
<div id="activity"><p style="color:#999;text-align:center;padding:20px">Waiting...</p></div></div>
</div></div><script>
function update(){fetch('/api/stats').then(r=>r.json()).then(d=>{
document.getElementById('bar').style.width=d.pct+'%';
document.getElementById('pct').textContent=d.pct+'%';
document.getElementById('hours').textContent=d.idx.toLocaleString()+' / '+d.total.toLocaleString();
document.getElementById('price').textContent='$'+d.price.toFixed(2);
document.getElementById('trades').textContent=d.trades.toLocaleString();
document.getElementById('rate').textContent=d.rate+'%';
document.getElementById('wins').textContent=d.wins.toLocaleString();
document.getElementById('losses').textContent=d.losses.toLocaleString();
const p=document.getElementById('profit');
p.textContent='£'+d.profit.toFixed(2);
p.className='stat-value '+(d.profit>=0?'positive':'negative');
const a=document.getElementById('activity');
if(d.activity&&d.activity.length>0){
a.innerHTML=d.activity.reverse().map(i=>`<div class="activity-item">
<span class="activity-time">${i.time}</span><span>${i.msg}</span></div>`).join('');
}document.getElementById('update').textContent='Updated: '+new Date().toLocaleTimeString();
}).catch(e=>console.error(e));}
update();setInterval(update,2000);
</script></body></html>'''

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
            'activity': list(state.activity)[-20:]
        })

if __name__ == '__main__':
    print('\n'+'='*80)
    print('  🚀 SOL Training Dashboard')
    print('='*80)
    print('\n  📱 Open: http://localhost:5050\n')
    print('='*80+'\n')
    app.run(host='0.0.0.0', port=5050, debug=False)
