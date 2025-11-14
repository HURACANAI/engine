#!/usr/bin/env python3
"""
Comprehensive Real-Time Dashboard for Huracan Engine

Shows EVERYTHING the engine is doing:
- Training progress (all stages)
- Data loading/downloading
- Model training metrics
- Shadow trades
- Gate decisions
- Learning updates
- System health
- Error logs
- Real-time activity feed
- Dropbox sync status
- Performance metrics

Access at: http://localhost:5055/
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
import subprocess
import sys
from decimal import Decimal

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_DIR = PROJECT_ROOT / 'templates'
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'huracan',
    'user': 'haq',
}

# Try to import observability modules
OBSERVABILITY_AVAILABLE = False
try:
    from observability.analytics.trade_journal import TradeJournal
    from observability.analytics.learning_tracker import LearningTracker
    from observability.analytics.metrics_computer import MetricsComputer
    from observability.analytics.model_evolution import ModelEvolutionTracker
    from observability.core.event_logger import EventLogger
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    print("⚠️  Observability modules not available - some features will be limited")


class ComprehensiveDashboardData:
    """Manages comprehensive dashboard data fetching"""

    def __init__(self):
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 0.1  # Cache for 0.1 seconds for real-time feel (updates every second)
        
        # Initialize observability modules if available
        # Use global keyword to access the global variable
        global OBSERVABILITY_AVAILABLE
        self.observability_available = OBSERVABILITY_AVAILABLE
        
        if OBSERVABILITY_AVAILABLE:
            try:
                self.trade_journal = TradeJournal()
                self.learning_tracker = LearningTracker()
                self.metrics_computer = MetricsComputer()
                self.model_tracker = ModelEvolutionTracker()
            except Exception as e:
                print(f"⚠️  Could not initialize observability: {e}")
                self.observability_available = False

    def get_db_connection(self):
        """Get PostgreSQL connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            return None

    def read_training_progress(self) -> Optional[Dict[str, Any]]:
        """Read training progress from JSON file"""
        progress_file = PROJECT_ROOT / 'training_progress.json'
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def read_recent_logs(self, lines: int = 50) -> List[str]:
        """Read recent log entries"""
        log_files = [
            PROJECT_ROOT / 'logs' / 'learning' / 'engine.log',
            PROJECT_ROOT / 'logs' / 'engine.log',
        ]
        
        logs = []
        for log_file in log_files:
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        file_logs = f.readlines()
                        logs.extend(file_logs[-lines:])
                except Exception:
                    pass
        
        return logs[-lines:] if logs else []

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_available_gb': 0,
                'disk_percent': 0,
                'disk_free_gb': 0,
            }

    def _json_serializer(self, obj):
        """Custom JSON serializer for Decimal and other types"""
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch all comprehensive dashboard data"""
        now = time.time()
        if self.cache_time and (now - self.cache_time) < self.cache_duration:
            return self.cache

        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'training_progress': self.read_training_progress(),
            'system_health': self.get_system_health(),
            'recent_logs': self.read_recent_logs(30),
        }

        # Database data
        conn = self.get_db_connection()
        conn_for_queries = conn  # Keep reference for later queries
        if conn:
            try:
                with conn.cursor() as cur:
                    # Overall stats (all symbols)
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_trades,
                            COUNT(*) FILTER (WHERE is_winner = true) as winning_trades,
                            COUNT(*) FILTER (WHERE is_winner = false) as losing_trades,
                            COALESCE(SUM(net_profit_gbp), 0) as total_profit,
                            COALESCE(AVG(CASE WHEN is_winner = true THEN net_profit_gbp END), 0) as avg_profit,
                            COALESCE(AVG(CASE WHEN is_winner = false THEN net_profit_gbp END), 0) as avg_loss,
                            COALESCE(AVG(model_confidence), 0) as avg_confidence,
                            COALESCE(AVG(hold_duration_minutes), 0) as avg_hold,
                            COALESCE(MAX(CASE WHEN is_winner = true THEN net_profit_gbp END), 0) as max_win,
                            COALESCE(MIN(CASE WHEN is_winner = false THEN net_profit_gbp END), 0) as max_loss
                        FROM trade_memory
                    """)
                    
                    row = cur.fetchone()
                    if row:
                        total_trades = row[0]
                        winning_trades = row[1]
                        losing_trades = row[2]
                        total_profit = float(row[3])
                        avg_profit = float(row[4])
                        avg_loss = float(row[5])
                        avg_confidence = float(row[6])
                        avg_hold = float(row[7]) if len(row) > 7 else 0
                        max_win = float(row[8]) if len(row) > 8 else 0
                        max_loss = float(row[9]) if len(row) > 9 else 0
                        
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
                            'avg_confidence': avg_confidence * 100,
                            'avg_hold_duration': avg_hold,
                            'max_win_gbp': max_win,
                            'max_loss_gbp': max_loss,
                            'profit_factor': profit_factor,
                            'expectancy': expectancy,
                            'risk_reward': risk_reward,
                        }
                    else:
                        data['overview'] = self._empty_overview()

                    # Recent trades (last 50)
                    cur.execute("""
                        SELECT
                            trade_id, symbol, entry_timestamp, entry_price, exit_price,
                            direction, net_profit_gbp, gross_profit_bps, is_winner,
                            exit_reason, model_confidence, hold_duration_minutes,
                            market_regime, volatility_bps, spread_at_entry_bps
                        FROM trade_memory
                        ORDER BY trade_id DESC
                        LIMIT 50
                    """)
                    
                    trades = []
                    for row in cur.fetchall():
                        try:
                            trades.append({
                                'trade_id': row[0],
                                'symbol': row[1],
                                'entry_timestamp': row[2].isoformat() if row[2] else None,
                                'entry_price': float(row[3]) if row[3] else 0,
                                'exit_price': float(row[4]) if row[4] else 0,
                                'direction': row[5] if len(row) > 5 else 'LONG',
                                'net_profit_gbp': float(row[6]) if len(row) > 6 and row[6] else 0,
                                'gross_profit_bps': float(row[7]) if len(row) > 7 and row[7] else 0,
                                'is_winner': row[8] if len(row) > 8 else False,
                                'exit_reason': row[9] if len(row) > 9 else None,
                                'model_confidence': float(row[10]) * 100 if len(row) > 10 and row[10] else 0,
                                'hold_duration_minutes': row[11] if len(row) > 11 and row[11] else 0,
                                'market_regime': row[12] if len(row) > 12 and row[12] else 'unknown',
                                'volatility_bps': float(row[13]) if len(row) > 13 and row[13] else 0,
                                'spread_bps': float(row[14]) if len(row) > 14 and row[14] else 0,
                            })
                        except Exception as e:
                            print(f"Error parsing trade row: {e}")
                            continue
                    data['recent_trades'] = trades

                    # Symbol breakdown
                    cur.execute("""
                        SELECT
                            symbol,
                            COUNT(*) as trade_count,
                            COUNT(*) FILTER (WHERE is_winner = true) as wins,
                            COALESCE(SUM(net_profit_gbp), 0) as profit
                        FROM trade_memory
                        GROUP BY symbol
                        ORDER BY trade_count DESC
                    """)
                    data['symbol_breakdown'] = [
                        {
                            'symbol': row[0],
                            'trade_count': row[1],
                            'wins': row[2],
                            'profit': float(row[3]),
                        }
                        for row in cur.fetchall()
                    ]

            except Exception as e:
                print(f"❌ Database query error: {e}")
                import traceback
                traceback.print_exc()
                # Keep connection open for additional queries - don't close here
        else:
            data['overview'] = self._empty_overview()
            data['recent_trades'] = []
            data['symbol_breakdown'] = []
            conn_for_queries = None

        # Add missing fields that enhanced dashboard expects
        data['pnl_series'] = []
        data['win_rate_trend'] = []
        data['regime_distribution'] = {'regimes': [], 'by_symbol': {}}
        data['confidence_heatmap'] = {}
        data['performance_by_symbol'] = []
        data['gate_performance'] = {'gates': [], 'total_blocked': 0}
        data['learning_metrics'] = {}
        
        # Try to fetch P&L series if we have data
        if conn_for_queries and data.get('recent_trades'):
            try:
                with conn_for_queries.cursor() as cur:
                    # Get last 7 days of trades for P&L chart
                    cur.execute("""
                        SELECT
                            trade_id,
                            entry_timestamp,
                            net_profit_gbp,
                            symbol,
                            is_winner,
                            model_confidence,
                            market_regime
                        FROM trade_memory
                        WHERE entry_timestamp >= NOW() - INTERVAL '7 days'
                          AND net_profit_gbp IS NOT NULL
                        ORDER BY entry_timestamp ASC
                        LIMIT 1000
                    """)
                    
                    cumulative = 0
                    for row in cur.fetchall():
                        try:
                            pnl = float(row[2]) if row[2] is not None else 0
                            cumulative += pnl
                            data['pnl_series'].append({
                                'trade_id': row[0],
                                'timestamp': row[1].isoformat() if row[1] else None,
                                'pnl': pnl,
                                'cumulative_pnl': cumulative,
                                'symbol': row[3],
                                'is_winner': row[4] if row[4] is not None else False,
                                'confidence': float(row[5]) * 100 if row[5] is not None else 0,
                                'regime': row[6] or 'unknown',
                            })
                        except:
                            continue
            except Exception as e:
                print(f"Error fetching P&L series: {e}")
        
        # Fetch regime distribution
        if conn_for_queries:
            try:
                with conn_for_queries.cursor() as cur:
                    cur.execute("""
                        SELECT
                            COALESCE(market_regime, 'unknown') as regime,
                            COUNT(*) as count,
                            COUNT(*) FILTER (WHERE is_winner = true) as wins,
                            AVG(net_profit_gbp) as avg_pnl,
                            AVG(model_confidence) as avg_confidence
                        FROM trade_memory
                        GROUP BY market_regime
                        ORDER BY count DESC
                    """)
                    
                    regimes = []
                    for row in cur.fetchall():
                        total = row[1] or 0
                        wins = row[2] or 0
                        win_rate = (wins / total * 100) if total > 0 else 0
                        regimes.append({
                            'regime': row[0],
                            'count': total,
                            'wins': wins,
                            'win_rate': win_rate,
                            'avg_pnl': float(row[3]) if row[3] else 0,
                            'avg_confidence': float(row[4]) * 100 if row[4] else 0,
                        })
                    data['regime_distribution'] = {'regimes': regimes, 'by_symbol': {}}
            except Exception as e:
                print(f"Error fetching regime distribution: {e}")
        
        # Fetch performance by symbol
        if conn_for_queries:
            try:
                with conn_for_queries.cursor() as cur:
                    cur.execute("""
                        SELECT
                            symbol,
                            COUNT(*) as total_trades,
                            COUNT(*) FILTER (WHERE is_winner = true) as wins,
                            SUM(net_profit_gbp) as total_pnl,
                            AVG(net_profit_gbp) as avg_pnl,
                            AVG(CASE WHEN is_winner = true THEN net_profit_gbp END) as avg_win,
                            AVG(CASE WHEN is_winner = false THEN net_profit_gbp END) as avg_loss,
                            AVG(model_confidence) as avg_confidence,
                            AVG(hold_duration_minutes) as avg_hold,
                            MAX(CASE WHEN is_winner = true THEN net_profit_gbp END) as max_win,
                            MIN(CASE WHEN is_winner = false THEN net_profit_gbp END) as max_loss
                        FROM trade_memory
                        GROUP BY symbol
                        ORDER BY total_trades DESC
                    """)
                    
                    symbols = []
                    for row in cur.fetchall():
                        total = row[1] or 0
                        wins = row[2] or 0
                        win_rate = (wins / total * 100) if total > 0 else 0
                        symbols.append({
                            'symbol': row[0],
                            'total_trades': total,
                            'wins': wins,
                            'win_rate': win_rate,
                            'total_pnl': float(row[3]) if row[3] else 0,
                            'avg_pnl': float(row[3]) / total if total > 0 else 0,
                            'avg_win': float(row[5]) if row[5] else 0,
                            'avg_loss': float(row[6]) if row[6] else 0,
                            'avg_confidence': float(row[7]) * 100 if row[7] else 0,
                            'avg_hold': float(row[8]) if row[8] else 0,
                            'max_win': float(row[9]) if row[9] else 0,
                            'max_loss': float(row[10]) if row[10] else 0,
                        })
                    data['performance_by_symbol'] = symbols
            except Exception as e:
                print(f"Error fetching performance by symbol: {e}")
        
        # Close connection after all queries
        if conn_for_queries:
            try:
                conn_for_queries.close()
            except:
                pass

        # Observability data
        if OBSERVABILITY_AVAILABLE:
            try:
                # Learning progress
                data['learning'] = {
                    'sessions_today': 0,
                    'last_training': None,
                }
                
                # Model status
                data['models'] = {
                    'active_models': 0,
                    'last_update': None,
                }
            except Exception:
                pass

        self.cache = data
        self.cache_time = now
        return data

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
            'avg_confidence': 0.0,
            'avg_hold_duration': 0.0,
            'max_win_gbp': 0.0,
            'max_loss_gbp': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'risk_reward': 0.0,
        }


# Global dashboard data instance
dashboard_data = ComprehensiveDashboardData()


@app.route('/')
def index():
    """Main dashboard page"""
    # Always use comprehensive dashboard (the one with all features)
    try:
        return render_template('comprehensive_dashboard.html')
    except Exception as e:
        print(f"⚠️  Error rendering dashboard: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Dashboard Error</h1><p>{str(e)}</p>", 500


@app.route('/api/data')
def get_data():
    """Get all dashboard data as JSON"""
    data = dashboard_data.fetch_data()
    # Convert Decimal to float for JSON serialization
    def convert_decimals(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        return obj
    data = convert_decimals(data)
    return jsonify(data)


@app.route('/api/stream')
def stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        while True:
            try:
                data = dashboard_data.fetch_data()
                # Convert Decimal to float for JSON serialization
                def convert_decimals(obj):
                    if isinstance(obj, Decimal):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_decimals(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_decimals(item) for item in obj]
                    return obj
                data = convert_decimals(data)
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1.0)  # Update every 1 second (exactly)
            except GeneratorExit:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Retry faster on error

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/health')
def health():
    """Health check endpoint"""
    conn = dashboard_data.get_db_connection()
    db_status = 'connected' if conn else 'disconnected'
    if conn:
        conn.close()
    return jsonify({
        'status': 'ok',
        'database': db_status,
        'observability': 'available' if OBSERVABILITY_AVAILABLE else 'unavailable',
    })


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Comprehensive Huracan Engine Dashboard")
    print("=" * 80)
    print(f"📊 Dashboard URL: http://localhost:5055/")
    print(f"🔌 API Endpoint:  http://localhost:5055/api/data")
    print(f"💓 Health Check:  http://localhost:5055/api/health")
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5055, debug=False, threaded=True)

