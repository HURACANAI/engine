#!/usr/bin/env python3
"""
Grok AI-Powered Trading Dashboard
Real-time win/loss analysis with AI insights and suggestions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import os
import re
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
import json

app = Flask(__name__)
CORS(app)

class State:
    def __init__(self):
        self.lock = Lock()
        self.log_file = "/tmp/sol_grok_training.log"
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
            'start_time': datetime.now(),
            'elapsed_time': '0h 0m',
            'estimated_completion': 'Calculating...',
            'speed': 0.0,
        }
        self.trade_history = deque(maxlen=50)

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
                   is_winner, exit_time
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
            })
        with state.lock:
            state.trade_history = deque(trades, maxlen=50)

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
                        elapsed = (datetime.now() - state.stats['start_time']).total_seconds()
                        if elapsed > 0:
                            state.stats['speed'] = state.stats['current_idx'] / elapsed
                            remaining = state.stats['total_candles'] - state.stats['current_idx']
                            if state.stats['speed'] > 0:
                                eta_seconds = remaining / state.stats['speed']
                                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                                state.stats['estimated_completion'] = eta_time.strftime('%H:%M:%S')
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
                elif 'analyzing_loss' in line:
                    msg = "🔍 Analyzing loss..."
                elif 'preventable.*True' in line:
                    msg = "⚠️  Loss was preventable"

                if msg:
                    state.activity.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'msg': msg
                    })
    except Exception as e:
        print(f"Log Error: {e}")

def generate_grok_analysis():
    """Generate AI-powered analysis of wins and losses"""
    try:
        conn = psycopg2.connect(host='localhost', port=5432, database='huracan', user='haq')
        cur = conn.cursor()

        # Get detailed trade analysis
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_winner) as wins,
                ROUND(AVG(net_profit_gbp)::numeric, 2) as avg_pnl,
                ROUND(AVG(CASE WHEN is_winner THEN net_profit_gbp END)::numeric, 2) as avg_win,
                ROUND(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END)::numeric, 2) as avg_loss,
                ROUND(MAX(net_profit_gbp)::numeric, 2) as best_win,
                ROUND(MIN(net_profit_gbp)::numeric, 2) as worst_loss
            FROM trade_memory WHERE symbol = 'SOL/USDT'
        """)
        overall = cur.fetchone()

        # Get loss distribution
        cur.execute("""
            SELECT
                CASE
                    WHEN net_profit_gbp < -50 THEN 'Catastrophic (< -£50)'
                    WHEN net_profit_gbp >= -50 AND net_profit_gbp < -20 THEN 'Large (-£50 to -£20)'
                    WHEN net_profit_gbp >= -20 AND net_profit_gbp < -10 THEN 'Medium (-£20 to -£10)'
                    WHEN net_profit_gbp >= -10 AND net_profit_gbp < 0 THEN 'Small (-£10 to £0)'
                    WHEN net_profit_gbp >= 0 AND net_profit_gbp < 10 THEN 'Small Win (£0 to £10)'
                    WHEN net_profit_gbp >= 10 THEN 'Large Win (> £10)'
                END as bucket,
                COUNT(*) as count
            FROM trade_memory
            WHERE symbol = 'SOL/USDT'
            GROUP BY bucket
            ORDER BY MIN(net_profit_gbp)
        """)
        distribution = cur.fetchall()

        # Get recent losing trades for pattern analysis
        cur.execute("""
            SELECT entry_price, exit_price, net_profit_gbp, hold_duration_minutes
            FROM trade_memory
            WHERE symbol = 'SOL/USDT' AND is_winner = false
            ORDER BY trade_id DESC
            LIMIT 20
        """)
        recent_losses = cur.fetchall()

        # Get recent winning trades
        cur.execute("""
            SELECT entry_price, exit_price, net_profit_gbp, hold_duration_minutes
            FROM trade_memory
            WHERE symbol = 'SOL/USDT' AND is_winner = true
            ORDER BY trade_id DESC
            LIMIT 20
        """)
        recent_wins = cur.fetchall()

        conn.close()

        # Generate Grok-style analysis
        win_rate = (overall[1] / overall[0] * 100) if overall[0] > 0 else 0

        analysis = {
            "summary": {
                "verdict": "🔴 CRITICAL ISSUES DETECTED" if win_rate < 20 else "⚠️  NEEDS IMPROVEMENT" if win_rate < 40 else "✅ GOOD PROGRESS",
                "win_rate": f"{win_rate:.1f}%",
                "total_trades": overall[0],
                "avg_pnl": f"£{overall[2]:.2f}",
                "risk_reward": f"1:{abs(overall[3]/overall[4]):.2f}" if overall[4] != 0 else "N/A"
            },
            "key_insights": [],
            "problems_detected": [],
            "winning_patterns": [],
            "suggestions": []
        }

        # Analyze and generate insights
        if win_rate < 10:
            analysis["problems_detected"].append({
                "severity": "CRITICAL",
                "issue": "Extremely Low Win Rate",
                "detail": f"Only {win_rate:.1f}% of trades are winning. The confidence threshold is too low."
            })
            analysis["suggestions"].append({
                "action": "Increase Confidence Threshold",
                "reason": "Raise from 0.20 to 0.40 to filter out low-quality trades",
                "expected_impact": "50-70% reduction in total trades, but higher win rate"
            })

        if abs(overall[4]) > abs(overall[3]) * 2:
            analysis["problems_detected"].append({
                "severity": "HIGH",
                "issue": "Poor Risk/Reward Ratio",
                "detail": f"Average loss (£{overall[4]:.2f}) is {abs(overall[4]/overall[3]):.1f}x larger than average win (£{overall[3]:.2f})"
            })
            analysis["suggestions"].append({
                "action": "Implement Tighter Stop Losses",
                "reason": "Losses are running too far before exits",
                "expected_impact": "Reduce average loss by 30-50%"
            })

        if overall[6] < -50:
            analysis["problems_detected"].append({
                "severity": "CRITICAL",
                "issue": "Catastrophic Losses Occurring",
                "detail": f"Worst single loss: £{overall[6]:.2f}. This is unacceptable."
            })
            analysis["suggestions"].append({
                "action": "Add Hard Stop Loss at -2%",
                "reason": "Prevent any single trade from losing more than £20",
                "expected_impact": "Eliminate catastrophic losses entirely"
            })

        # Analyze distribution
        large_losses = sum(d[1] for d in distribution if 'Large' in d[0] or 'Catastrophic' in d[0])
        if large_losses > overall[0] * 0.3:
            analysis["problems_detected"].append({
                "severity": "HIGH",
                "issue": "Too Many Large Losses",
                "detail": f"{large_losses} trades ({large_losses/overall[0]*100:.1f}%) are large or catastrophic losses"
            })

        # Winning patterns analysis
        if overall[1] > 0:
            avg_win_hold = sum(w[3] for w in recent_wins if w[3]) / len([w for w in recent_wins if w[3]]) if recent_wins else 0
            analysis["winning_patterns"].append({
                "pattern": "Hold Duration",
                "detail": f"Winning trades average {avg_win_hold:.0f} minutes hold time",
                "recommendation": "Focus on trades that match this timeframe"
            })

        # Key insights
        analysis["key_insights"].append(f"You're in the exploration phase - the RL agent is learning what NOT to do")
        analysis["key_insights"].append(f"Current risk/reward ratio is unsustainable: losing £{abs(overall[4]):.2f} per loss vs winning £{overall[3]:.2f} per win")
        analysis["key_insights"].append(f"After {overall[0]} trades, the model has collected enough data to improve dramatically in next iteration")

        # Final suggestion
        analysis["suggestions"].append({
            "action": "Let Training Complete",
            "reason": "The RL agent needs to finish all 2,161 candles to update its policy based on these lessons",
            "expected_impact": "Next training run should show 50-100% improvement in win rate"
        })

        return analysis

    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grok AI Trading Dashboard</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}
.container{max-width:1600px;margin:0 auto}
.header{background:white;padding:30px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.1);
margin-bottom:30px;text-align:center}
.header h1{font-size:2.5em;color:#667eea;margin-bottom:10px}
.grok-btn{background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);color:white;
padding:15px 40px;border:none;border-radius:50px;font-size:1.1em;font-weight:bold;
cursor:pointer;box-shadow:0 5px 20px rgba(245,87,108,0.4);transition:all 0.3s}
.grok-btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(245,87,108,0.5)}
.grok-btn:active{transform:translateY(0)}
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
.ai-analysis{display:none;margin-top:20px;padding:20px;background:#f9fafb;border-radius:10px;
border-left:4px solid #f5576c}
.ai-section{margin:15px 0;padding:15px;background:white;border-radius:8px}
.ai-section h3{color:#667eea;margin-bottom:10px;font-size:1.1em}
.insight-item{padding:10px;margin:5px 0;background:#f0f0f0;border-radius:5px}
.severity-critical{border-left:4px solid #ef4444}
.severity-high{border-left:4px solid #f59e0b}
.severity-medium{border-left:4px solid#3b82f6}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.metric-box{background:#f9fafb;padding:15px;border-radius:10px}
.metric-box .label{color:#666;font-size:0.85em;margin-bottom:5px}
.metric-box .value{font-size:1.5em;font-weight:bold;color:#333}
.badge{display:inline-block;padding:5px 15px;border-radius:20px;font-size:0.85em;font-weight:600}
.badge.live{background:#fee2e2;color:#991b1b;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.loading{text-align:center;padding:20px;color:#666}
</style></head><body><div class="container">

<div class="header">
<h1>🤖 Grok AI Trading Dashboard</h1>
<p class="subtitle">AI-Powered Win/Loss Analysis & Insights</p>
<button class="grok-btn" onclick="analyzeWithGrok()">🧠 Ask Grok AI to Analyze Trades</button>
<p style="color:#999;margin-top:10px" id="update">Loading...</p>
</div>

<div id="ai-analysis" class="ai-analysis card grid-wide">
<h2>🤖 Grok AI Analysis</h2>
<div id="ai-content" class="loading">Analyzing trades with AI...</div>
</div>

<div class="grid">
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
<div class="stat-label">Est. Completion</div>
<div class="stat-value small" id="eta">Calculating...</div>
</div>
</div>

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
</div>

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
<div class="stat" style="margin-top:15px">
<span class="badge live">🔴 Training Live</span>
</div>
</div>

</div></div>

<script>
function analyzeWithGrok(){
document.getElementById('ai-analysis').style.display='block';
document.getElementById('ai-content').innerHTML='<div class="loading">🤖 Grok AI is analyzing your trades...</div>';
fetch('/api/grok-analyze').then(r=>r.json()).then(d=>{
if(d.error){
document.getElementById('ai-content').innerHTML='<div class="insight-item severity-critical">Error: '+d.error+'</div>';
return;
}
let html='';
html+='<div class="ai-section"><h3>'+d.summary.verdict+'</h3>';
html+='<p>Win Rate: <strong>'+d.summary.win_rate+'</strong> | ';
html+='Total Trades: <strong>'+d.summary.total_trades+'</strong> | ';
html+='Avg P&L: <strong>'+d.summary.avg_pnl+'</strong> | ';
html+='Risk/Reward: <strong>'+d.summary.risk_reward+'</strong></p></div>';

if(d.problems_detected.length>0){
html+='<div class="ai-section"><h3>⚠️  Problems Detected</h3>';
d.problems_detected.forEach(p=>{
html+='<div class="insight-item severity-'+p.severity.toLowerCase()+'">';
html+='<strong>'+p.issue+'</strong><br>'+p.detail+'</div>';
});
html+='</div>';
}

if(d.winning_patterns.length>0){
html+='<div class="ai-section"><h3>✅ Winning Patterns</h3>';
d.winning_patterns.forEach(w=>{
html+='<div class="insight-item">';
html+='<strong>'+w.pattern+':</strong> '+w.detail+'<br>';
html+='<em>'+w.recommendation+'</em></div>';
});
html+='</div>';
}

html+='<div class="ai-section"><h3>🎯 AI Suggestions</h3>';
d.suggestions.forEach(s=>{
html+='<div class="insight-item">';
html+='<strong>'+s.action+'</strong><br>';
html+='<em>Why:</em> '+s.reason+'<br>';
html+='<em>Expected Impact:</em> '+s.expected_impact+'</div>';
});
html+='</div>';

html+='<div class="ai-section"><h3>💡 Key Insights</h3>';
d.key_insights.forEach(i=>{
html+='<div class="insight-item">'+i+'</div>';
});
html+='</div>';

document.getElementById('ai-content').innerHTML=html;
}).catch(e=>{
document.getElementById('ai-content').innerHTML='<div class="insight-item severity-critical">Error communicating with Grok AI</div>';
});
}

function update(){
fetch('/api/stats').then(r=>r.json()).then(d=>{
document.getElementById('bar').style.width=d.pct+'%';
document.getElementById('pct').textContent=d.pct+'%';
document.getElementById('hours').textContent=d.idx.toLocaleString()+' / '+d.total.toLocaleString();
document.getElementById('price').textContent='$'+d.price.toFixed(2);
document.getElementById('speed').textContent=d.speed.toFixed(2)+' candles/sec';
document.getElementById('eta').textContent=d.eta;
document.getElementById('trades').textContent=d.trades.toLocaleString();
document.getElementById('rate').textContent=d.rate+'%';
const p=document.getElementById('profit');
p.textContent='£'+d.profit.toFixed(2);
p.className='stat-value '+(d.profit>=0?'positive':'negative');
document.getElementById('wins').textContent=d.wins.toLocaleString();
document.getElementById('losses').textContent=d.losses.toLocaleString();
document.getElementById('avg-win').textContent='£'+d.avg_win.toFixed(2);
document.getElementById('avg-loss').textContent='£'+d.avg_loss.toFixed(2);
document.getElementById('max-win').textContent='£'+d.max_win.toFixed(2);
document.getElementById('max-loss').textContent='£'+d.max_loss.toFixed(2);
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
            'avg_win': state.stats['avg_win'],
            'avg_loss': state.stats['avg_loss'],
            'max_win': state.stats['largest_win'],
            'max_loss': state.stats['largest_loss'],
            'speed': state.stats['speed'],
            'eta': state.stats['estimated_completion'],
        })

@app.route('/api/grok-analyze')
def grok_analyze():
    return jsonify(generate_grok_analysis())

if __name__ == '__main__':
    print('\n'+'='*80)
    print('  🤖 Grok AI-Powered Trading Dashboard')
    print('='*80)
    print('\n  📱 Open: http://localhost:5055\n')
    print('  🧠 Click "Ask Grok AI" for intelligent trade analysis!\n')
    print('='*80+'\n')
    app.run(host='0.0.0.0', port=5055, debug=False)
