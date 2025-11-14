#!/usr/bin/env python3
"""
Real-Time Web Dashboard for SOL/USDT Training

Beautiful web interface showing comprehensive training metrics in real-time.
Access at: http://localhost:5055/

Features:
- Real-time updates using Server-Sent Events (SSE)
- Beautiful responsive design
- Interactive charts and visualizations
- Complete training transparency

Usage:
    python scripts/web_dashboard_server.py

Requirements:
    pip install flask psycopg2-binary
"""

from flask import Flask, render_template, jsonify, Response
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import json
import time
import threading
import os
from pathlib import Path

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_DIR = PROJECT_ROOT / 'templates'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'huracan',
    'user': 'haq',
}

class DashboardData:
    """Manages dashboard data fetching and caching"""

    def __init__(self):
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 1  # Cache for 1 second

    def get_db_connection(self):
        """Get PostgreSQL connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch all dashboard data"""
        # Check cache
        now = time.time()
        if self.cache_time and (now - self.cache_time) < self.cache_duration:
            return self.cache

        conn = self.get_db_connection()
        if not conn:
            return self._empty_data()

        try:
            data = {}

            with conn.cursor() as cur:
                # Overall stats
                cur.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as winning_trades,
                        COUNT(*) FILTER (WHERE is_winner = false) as losing_trades,
                        COALESCE(SUM(net_profit_gbp), 0) as total_profit,
                        COALESCE(AVG(CASE WHEN is_winner THEN net_profit_gbp END), 0) as avg_profit,
                        COALESCE(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as avg_loss,
                        COALESCE(MAX(CASE WHEN is_winner THEN net_profit_gbp END), 0) as largest_win,
                        COALESCE(MIN(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as largest_loss,
                        COALESCE(AVG(CASE WHEN is_winner THEN gross_profit_bps END), 0) as avg_win_bps,
                        COALESCE(AVG(CASE WHEN NOT is_winner THEN gross_profit_bps END), 0) as avg_loss_bps,
                        COALESCE(AVG(hold_duration_minutes), 0) as avg_hold,
                        COALESCE(AVG(model_confidence), 0) as avg_confidence
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                """)

                row = cur.fetchone()
                if row:
                    total_trades = row[0]
                    winning_trades = row[1]
                    losing_trades = row[2]
                    total_profit = float(row[3])
                    avg_profit = float(row[4])
                    avg_loss = float(row[5])
                    largest_win = float(row[6])
                    largest_loss = float(row[7])
                    avg_win_bps = float(row[8])
                    avg_loss_bps = float(row[9])
                    avg_hold = float(row[10])
                    avg_confidence = float(row[11])

                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

                    # Calculate advanced metrics
                    if total_trades > 0 and abs(avg_loss) > 0.01:
                        profit_factor = abs(avg_profit * winning_trades / max(0.01, abs(avg_loss * losing_trades)))
                        expectancy = ((win_rate/100) * avg_profit - ((100-win_rate)/100) * abs(avg_loss))
                        risk_reward = abs(avg_profit / max(0.01, abs(avg_loss)))
                    else:
                        profit_factor = 0
                        expectancy = 0
                        risk_reward = 0

                    data['overview'] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'win_rate': win_rate,
                        'total_profit_gbp': total_profit,
                        'avg_profit_gbp': avg_profit,
                        'avg_loss_gbp': avg_loss,
                        'largest_win_gbp': largest_win,
                        'largest_loss_gbp': largest_loss,
                        'avg_win_bps': avg_win_bps,
                        'avg_loss_bps': avg_loss_bps,
                        'avg_hold_duration': avg_hold,
                        'avg_confidence': avg_confidence * 100,
                        'profit_factor': profit_factor,
                        'expectancy': expectancy,
                        'risk_reward': risk_reward,
                    }
                else:
                    data['overview'] = self._empty_overview()

                # Regime breakdown
                cur.execute("""
                    SELECT
                        COALESCE(market_regime, 'unknown') as regime,
                        COUNT(*) as count
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    GROUP BY market_regime
                    ORDER BY count DESC
                """)
                data['regimes'] = [{'regime': row[0], 'count': row[1]} for row in cur.fetchall()]

                # Exit reasons
                cur.execute("""
                    SELECT
                        exit_reason,
                        COUNT(*) as count
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT' AND exit_reason IS NOT NULL
                    GROUP BY exit_reason
                    ORDER BY count DESC
                """)
                data['exit_reasons'] = [{'reason': row[0], 'count': row[1]} for row in cur.fetchall()]

                # Recent trades (last 20)
                cur.execute("""
                    SELECT
                        trade_id,
                        entry_timestamp,
                        entry_price,
                        exit_price,
                        direction,
                        net_profit_gbp,
                        gross_profit_bps,
                        is_winner,
                        exit_reason,
                        model_confidence,
                        hold_duration_minutes,
                        market_regime,
                        volatility_bps,
                        spread_at_entry_bps
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    ORDER BY trade_id DESC
                    LIMIT 20
                """)

                trades = []
                for row in cur.fetchall():
                    trades.append({
                        'trade_id': row[0],
                        'entry_timestamp': row[1].isoformat() if row[1] else None,
                        'entry_price': float(row[2]) if row[2] else 0,
                        'exit_price': float(row[3]) if row[3] else 0,
                        'direction': row[4],
                        'net_profit_gbp': float(row[5]) if row[5] else 0,
                        'gross_profit_bps': float(row[6]) if row[6] else 0,
                        'is_winner': row[7],
                        'exit_reason': row[8],
                        'model_confidence': float(row[9]) * 100 if row[9] else 0,
                        'hold_duration_minutes': row[10] if row[10] else 0,
                        'market_regime': row[11] or 'unknown',
                        'volatility_bps': float(row[12]) if row[12] else 0,
                        'spread_bps': float(row[13]) if row[13] else 0,
                    })
                data['recent_trades'] = trades

                # Hourly breakdown (last 24 hours)
                cur.execute("""
                    SELECT
                        EXTRACT(HOUR FROM entry_timestamp) as hour,
                        COUNT(*) as trade_count,
                        COALESCE(SUM(net_profit_gbp), 0) as profit
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                        AND entry_timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY EXTRACT(HOUR FROM entry_timestamp)
                    ORDER BY hour
                """)
                hourly = {}
                for row in cur.fetchall():
                    hourly[int(row[0])] = {
                        'trade_count': row[1],
                        'profit': float(row[2])
                    }
                data['hourly'] = hourly

                # P&L over time (last 100 trades)
                cur.execute("""
                    SELECT
                        trade_id,
                        net_profit_gbp,
                        entry_timestamp
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    ORDER BY trade_id DESC
                    LIMIT 100
                """)
                pnl_series = []
                cumulative = 0
                for row in reversed(list(cur.fetchall())):
                    cumulative += float(row[1]) if row[1] else 0
                    pnl_series.append({
                        'trade_id': row[0],
                        'cumulative_pnl': cumulative,
                        'timestamp': row[2].isoformat() if row[2] else None
                    })
                data['pnl_series'] = pnl_series

            # Cache the data
            self.cache = data
            self.cache_time = now

            return data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._empty_data()
        finally:
            conn.close()

    def _empty_data(self) -> Dict[str, Any]:
        """Return empty data structure"""
        return {
            'overview': self._empty_overview(),
            'regimes': [],
            'exit_reasons': [],
            'recent_trades': [],
            'hourly': {},
            'pnl_series': []
        }

    def _empty_overview(self) -> Dict[str, float]:
        """Return empty overview"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit_gbp': 0.0,
            'avg_profit_gbp': 0.0,
            'avg_loss_gbp': 0.0,
            'largest_win_gbp': 0.0,
            'largest_loss_gbp': 0.0,
            'avg_win_bps': 0.0,
            'avg_loss_bps': 0.0,
            'avg_hold_duration': 0.0,
            'avg_confidence': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'risk_reward': 0.0,
        }

# Global dashboard data instance
dashboard_data = DashboardData()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """Get all dashboard data as JSON"""
    data = dashboard_data.fetch_data()
    data['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    # Add training progress if available
    progress_file = PROJECT_ROOT / 'training_progress.json'
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                training_progress = json.load(f)
                data['training_progress'] = training_progress
        except Exception as e:
            print(f"Error reading training progress: {e}")
            data['training_progress'] = None
    else:
        data['training_progress'] = None
    
    return jsonify(data)

@app.route('/api/stream')
def stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        while True:
            try:
                data = dashboard_data.fetch_data()
                data['timestamp'] = datetime.now(timezone.utc).isoformat()
                
                # Add training progress if available
                progress_file = PROJECT_ROOT / 'training_progress.json'
                if progress_file.exists():
                    try:
                        with open(progress_file, 'r') as f:
                            training_progress = json.load(f)
                            data['training_progress'] = training_progress
                    except Exception:
                        data['training_progress'] = None
                else:
                    data['training_progress'] = None
                
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1.5)  # Update every 1.5 seconds
            except GeneratorExit:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(5)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    conn = dashboard_data.get_db_connection()
    if conn:
        conn.close()
        return jsonify({'status': 'ok', 'database': 'connected'})
    else:
        return jsonify({'status': 'error', 'database': 'disconnected'}), 503

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SOL/USDT Training Dashboard")
    print("=" * 60)
    print(f"📊 Dashboard URL: http://localhost:5055/")
    print(f"🔌 API Endpoint:  http://localhost:5055/api/data")
    print(f"💓 Health Check:  http://localhost:5055/api/health")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5055, debug=False, threaded=True)
