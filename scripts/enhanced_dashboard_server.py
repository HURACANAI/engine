#!/usr/bin/env python3
"""
Enhanced Comprehensive Real-Time Dashboard for Huracan Engine

Super advanced dashboard with:
- Real-time charts and visualizations
- Comprehensive metrics from all sources
- Updates every second with smooth animations
- Beautiful, intuitive UI
- Deep insights into engine performance

Access at: http://localhost:5055/
"""

from flask import Flask, render_template, jsonify, Response
from werkzeug.wrappers.response import Response as WerkzeugResponse
import psycopg2  # type: ignore
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Type
import json
import time
import sys
from pathlib import Path
import sqlite3
from decimal import Decimal

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEMPLATE_DIR = PROJECT_ROOT / 'templates'
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Enable CORS for all routes
@app.after_request
def after_request(response: WerkzeugResponse) -> WerkzeugResponse:  # type: ignore
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')  # type: ignore
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')  # type: ignore
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')  # type: ignore
    return response

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'huracan',
    'user': 'haq',
}

# Try to import observability modules
OBSERVABILITY_AVAILABLE = False
TradeJournal: Optional[Type[Any]] = None
LearningTracker: Optional[Type[Any]] = None
MetricsComputer: Optional[Type[Any]] = None
ModelEvolutionTracker: Optional[Type[Any]] = None

try:
    from observability.analytics.trade_journal import TradeJournal  # type: ignore
    from observability.analytics.learning_tracker import LearningTracker  # type: ignore
    from observability.analytics.metrics_computer import MetricsComputer  # type: ignore
    from observability.analytics.model_evolution import ModelEvolutionTracker  # type: ignore
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    print("⚠️  Observability modules not available - some features will be limited")


def convert_decimals(obj: Any) -> Any:
    """Recursively convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_decimals(item) for item in obj)
    return obj


class EnhancedDashboardData:
    """Enhanced dashboard data manager with comprehensive data fetching"""

    def __init__(self):
        self.cache = {}
        self.cache_time = None
        self.cache_duration = 0.5  # Cache for 0.5 seconds (updates every 1 second)
        
        # Initialize observability modules if available
        if OBSERVABILITY_AVAILABLE and TradeJournal is not None and LearningTracker is not None and MetricsComputer is not None and ModelEvolutionTracker is not None:
            try:
                self.trade_journal = TradeJournal()  # type: ignore
                self.learning_tracker = LearningTracker()  # type: ignore
                self.metrics_computer = MetricsComputer()  # type: ignore
                self.model_tracker = ModelEvolutionTracker()  # type: ignore
            except Exception as e:
                print(f"⚠️  Could not initialize observability: {e}")
                self.observability_available = False
        else:
            self.observability_available = False

    def get_db_connection(self) -> Optional[Any]:
        """Get PostgreSQL connection"""
        try:
            # Ensure port is an int for type checking
            db_config = dict(DB_CONFIG)
            db_config['port'] = int(db_config['port'])
            conn: Any = psycopg2.connect(**db_config)  # type: ignore
            return conn
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_sqlite_connection(self, db_path: str) -> Optional[Any]:
        """Get SQLite connection"""
        try:
            db_file = PROJECT_ROOT / db_path
            if db_file.exists():
                conn: Any = sqlite3.connect(str(db_file))
                conn.row_factory = sqlite3.Row  # type: ignore
                return conn
            return None
        except Exception:
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

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_sent_mb': network.bytes_sent / (1024**2),
                'network_recv_mb': network.bytes_recv / (1024**2),
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_available_gb': 0,
                'memory_total_gb': 0,
                'disk_percent': 0,
                'disk_free_gb': 0,
                'disk_total_gb': 0,
                'network_sent_mb': 0,
                'network_recv_mb': 0,
            }

    def fetch_pnl_series(self, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch P&L series for charting"""
        conn: Any = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cur: Any = conn.cursor()  # type: ignore
            try:
                # Get trades from last N days
                cur.execute(f"""
                    SELECT
                        trade_id,
                        entry_timestamp,
                        net_profit_gbp,
                        symbol,
                        is_winner,
                        model_confidence,
                        market_regime
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '{days} days'
                      AND net_profit_gbp IS NOT NULL
                    ORDER BY entry_timestamp ASC
                """)
                
                trades: List[Dict[str, Any]] = []
                cumulative = 0.0
                rows: Any = cur.fetchall()  # type: ignore
                for row in rows:
                    try:
                        pnl_val: Any = row[2] if len(row) > 2 else None  # type: ignore
                        pnl = float(pnl_val) if pnl_val is not None else 0.0  # type: ignore
                        cumulative += pnl
                        timestamp_val: Any = row[1] if len(row) > 1 else None  # type: ignore
                        timestamp_str: Optional[str] = timestamp_val.isoformat() if timestamp_val else None  # type: ignore
                        trades.append({
                            'trade_id': row[0],  # type: ignore
                            'timestamp': timestamp_str,
                            'pnl': pnl,
                            'cumulative_pnl': cumulative,
                            'symbol': row[3] if len(row) > 3 else '',  # type: ignore
                            'is_winner': bool(row[4]) if len(row) > 4 and row[4] is not None else False,  # type: ignore
                            'confidence': float(row[5]) * 100 if len(row) > 5 and row[5] is not None else 0.0,  # type: ignore
                            'regime': str(row[6]) if len(row) > 6 and row[6] else 'unknown',  # type: ignore
                        })
                    except Exception as e:
                        print(f"Error parsing P&L row: {e}")
                        continue
                return trades
            finally:
                cur.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching P&L series: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if conn:
                conn.close()  # type: ignore

    def fetch_win_rate_trend(self, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch win rate trend over time"""
        conn: Any = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cur: Any = conn.cursor()  # type: ignore
            try:
                # Group by hour
                cur.execute(f"""
                    SELECT
                        DATE_TRUNC('hour', entry_timestamp) as hour,
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as wins,
                        AVG(model_confidence) as avg_confidence,
                        AVG(net_profit_gbp) as avg_pnl
                    FROM trade_memory
                    WHERE entry_timestamp >= NOW() - INTERVAL '{days} days'
                      AND net_profit_gbp IS NOT NULL
                    GROUP BY DATE_TRUNC('hour', entry_timestamp)
                    ORDER BY hour ASC
                """)
                
                trends: List[Dict[str, Any]] = []
                rows: Any = cur.fetchall()  # type: ignore
                for row in rows:
                    try:
                        total: int = int(row[1]) if len(row) > 1 and row[1] is not None else 0  # type: ignore
                        wins: int = int(row[2]) if len(row) > 2 and row[2] is not None else 0  # type: ignore
                        win_rate = (wins / total * 100) if total > 0 else 0.0
                        hour_val: Any = row[0] if len(row) > 0 else None  # type: ignore
                        timestamp_str: Optional[str] = hour_val.isoformat() if hour_val else None  # type: ignore
                        trends.append({
                            'timestamp': timestamp_str,
                            'total_trades': total,
                            'wins': wins,
                            'win_rate': win_rate,
                            'avg_confidence': float(row[3]) * 100 if len(row) > 3 and row[3] is not None else 0.0,  # type: ignore
                            'avg_pnl': float(row[4]) if len(row) > 4 and row[4] is not None else 0.0,  # type: ignore
                        })
                    except Exception as e:
                        print(f"Error parsing win rate trend row: {e}")
                        continue
                return trends
            finally:
                cur.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching win rate trend: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if conn:
                conn.close()  # type: ignore

    def fetch_regime_distribution(self) -> Dict[str, Any]:
        """Fetch market regime distribution"""
        conn: Any = self.get_db_connection()
        if not conn:
            return {'regimes': [], 'by_symbol': {}}
        
        try:
            cur: Any = conn.cursor()  # type: ignore
            try:
                # Overall regime distribution
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
                
                regimes: List[Dict[str, Any]] = []
                rows: Any = cur.fetchall()  # type: ignore
                for row in rows:
                    total: int = int(row[1]) if len(row) > 1 and row[1] is not None else 0  # type: ignore
                    wins: int = int(row[2]) if len(row) > 2 and row[2] is not None else 0  # type: ignore
                    win_rate = (wins / total * 100) if total > 0 else 0.0
                    regimes.append({
                        'regime': str(row[0]) if len(row) > 0 else 'unknown',  # type: ignore
                        'count': total,
                        'wins': wins,
                        'win_rate': win_rate,
                        'avg_pnl': float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,  # type: ignore
                        'avg_confidence': float(row[4]) * 100 if len(row) > 4 and row[4] is not None else 0.0,  # type: ignore
                    })
                
                # By symbol
                cur.execute("""
                    SELECT
                        symbol,
                        COALESCE(market_regime, 'unknown') as regime,
                        COUNT(*) as count
                    FROM trade_memory
                    GROUP BY symbol, market_regime
                    ORDER BY symbol, count DESC
                """)
                
                by_symbol: Dict[str, List[Dict[str, Any]]] = {}
                rows2: Any = cur.fetchall()  # type: ignore
                for row in rows2:
                    symbol: str = str(row[0]) if len(row) > 0 else ''  # type: ignore
                    if symbol not in by_symbol:
                        by_symbol[symbol] = []
                    by_symbol[symbol].append({
                        'regime': str(row[1]) if len(row) > 1 else 'unknown',  # type: ignore
                        'count': int(row[2]) if len(row) > 2 and row[2] is not None else 0,  # type: ignore
                    })
                
                return {'regimes': regimes, 'by_symbol': by_symbol}
            finally:
                cur.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching regime distribution: {e}")
            import traceback
            traceback.print_exc()
            return {'regimes': [], 'by_symbol': {}}
        finally:
            if conn:
                conn.close()  # type: ignore

    def fetch_confidence_heatmap(self) -> Dict[str, Any]:
        """Fetch confidence heatmap by regime"""
        conn: Any = self.get_db_connection()
        if not conn:
            print("⚠️  Confidence Heatmap: Database connection not available")
            return {}
        
        try:
            cur: Any = conn.cursor()  # type: ignore
            try:
                # Query with proper grouping - group by the COALESCE expression
                cur.execute("""
                    SELECT
                        COALESCE(market_regime, 'unknown') as regime,
                        AVG(CAST(model_confidence AS FLOAT)) as avg_confidence,
                        COALESCE(STDDEV(CAST(model_confidence AS FLOAT)), 0.0) as std_confidence,
                        COUNT(*) as count
                    FROM trade_memory
                    WHERE model_confidence IS NOT NULL
                        AND CAST(model_confidence AS FLOAT) >= 0
                        AND CAST(model_confidence AS FLOAT) <= 1
                    GROUP BY COALESCE(market_regime, 'unknown')
                    HAVING COUNT(*) > 0
                    ORDER BY avg_confidence DESC
                """)
                
                heatmap: Dict[str, Dict[str, Any]] = {}
                rows: Any = cur.fetchall()  # type: ignore
                
                print(f"📊 Confidence Heatmap: Found {len(rows)} regimes")
                
                for row in rows:
                    regime: str = str(row[0]) if len(row) > 0 and row[0] is not None else 'unknown'  # type: ignore
                    avg_conf: float = float(row[1]) if len(row) > 1 and row[1] is not None else 0.0  # type: ignore
                    std_conf: float = float(row[2]) if len(row) > 2 and row[2] is not None else 0.0  # type: ignore
                    count: int = int(row[3]) if len(row) > 3 and row[3] is not None else 0  # type: ignore
                    
                    # Convert to percentage if confidence is stored as decimal (0-1)
                    # If avg_conf > 1, assume it's already a percentage
                    if avg_conf > 1.0:
                        # Already in percentage form
                        avg_confidence_pct = avg_conf
                        std_confidence_pct = std_conf
                    else:
                        # Convert from decimal (0-1) to percentage (0-100)
                        avg_confidence_pct = avg_conf * 100.0
                        std_confidence_pct = std_conf * 100.0
                    
                    heatmap[regime] = {
                        'avg_confidence': round(avg_confidence_pct, 2),
                        'std_confidence': round(std_confidence_pct, 2),
                        'count': count,
                    }
                    
                    print(f"📊 Confidence Heatmap: {regime} - {avg_confidence_pct:.2f}% (n={count})")
                
                if not heatmap:
                    print("⚠️  Confidence Heatmap: No data found (no trades with confidence values)")
                
                return heatmap
            finally:
                cur.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching confidence heatmap: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            if conn:
                conn.close()  # type: ignore

    def fetch_performance_by_symbol(self) -> List[Dict[str, Any]]:
        """Fetch performance metrics by symbol"""
        conn: Any = self.get_db_connection()
        if not conn:
            return []
        
        try:
            cur: Any = conn.cursor()  # type: ignore
            try:
                cur.execute("""
                    SELECT
                        symbol,
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as wins,
                        COALESCE(SUM(net_profit_gbp), 0) as total_pnl,
                        COALESCE(AVG(net_profit_gbp), 0) as avg_pnl,
                        COALESCE(AVG(CASE WHEN is_winner = true THEN net_profit_gbp END), 0) as avg_win,
                        COALESCE(AVG(CASE WHEN is_winner = false THEN net_profit_gbp END), 0) as avg_loss,
                        COALESCE(AVG(model_confidence), 0) as avg_confidence,
                        COALESCE(AVG(hold_duration_minutes), 0) as avg_hold,
                        COALESCE(MAX(CASE WHEN is_winner = true THEN net_profit_gbp END), 0) as max_win,
                        COALESCE(MIN(CASE WHEN is_winner = false THEN net_profit_gbp END), 0) as max_loss
                    FROM trade_memory
                    WHERE symbol IS NOT NULL
                    GROUP BY symbol
                    HAVING COUNT(*) > 0
                    ORDER BY total_trades DESC
                """)
                
                symbols: List[Dict[str, Any]] = []
                rows: Any = cur.fetchall()  # type: ignore
                print(f"📊 Fetched {len(rows)} symbols for performance_by_symbol")
                for row in rows:
                    try:
                        total: int = int(row[1]) if len(row) > 1 and row[1] is not None else 0  # type: ignore
                        wins: int = int(row[2]) if len(row) > 2 and row[2] is not None else 0  # type: ignore
                        win_rate = (wins / total * 100) if total > 0 else 0.0
                        symbol_name: str = str(row[0]) if len(row) > 0 and row[0] is not None else ''  # type: ignore
                        if not symbol_name:
                            continue  # Skip empty symbols
                        symbols.append({
                            'symbol': symbol_name,
                            'total_trades': total,
                            'wins': wins,
                            'win_rate': win_rate,
                            'total_pnl': float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,  # type: ignore
                            'avg_pnl': float(row[4]) if len(row) > 4 and row[4] is not None else 0.0,  # type: ignore
                            'avg_win': float(row[5]) if len(row) > 5 and row[5] is not None else 0.0,  # type: ignore
                            'avg_loss': float(row[6]) if len(row) > 6 and row[6] is not None else 0.0,  # type: ignore
                            'avg_confidence': float(row[7]) * 100 if len(row) > 7 and row[7] is not None else 0.0,  # type: ignore
                            'avg_hold': float(row[8]) if len(row) > 8 and row[8] is not None else 0.0,  # type: ignore
                            'max_win': float(row[9]) if len(row) > 9 and row[9] is not None else 0.0,  # type: ignore
                            'max_loss': float(row[10]) if len(row) > 10 and row[10] is not None else 0.0,  # type: ignore
                        })
                    except Exception as e:
                        print(f"⚠️  Error parsing symbol row: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                print(f"✅ Returning {len(symbols)} symbols after processing")
                return symbols
            finally:
                cur.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching performance by symbol: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if conn:
                conn.close()  # type: ignore

    def fetch_gate_performance(self) -> Dict[str, Any]:
        """Fetch gate performance metrics"""
        # Try to get from SQLite journal if available
        journal_conn: Any = self.get_sqlite_connection('observability/data/sqlite/journal.db')
        if not journal_conn:
            print("⚠️  Gate Performance: SQLite journal database not found at 'observability/data/sqlite/journal.db'")
            return {'gates': [], 'total_blocked': 0, 'error': 'Database not found'}
        
        try:
            # Get gate performance from last 7 days (more data)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            cursor: Any = journal_conn.cursor()  # type: ignore
            try:
                # Check if table exists first
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='shadow_trades'
                """)
                if not cursor.fetchone():
                    print("⚠️  Gate Performance: shadow_trades table does not exist")
                    return {'gates': [], 'total_blocked': 0, 'error': 'Table not found'}
                
                # Query with better NULL handling
                cursor.execute("""
                    SELECT
                        blocked_by,
                        COUNT(*) as total_blocked,
                        SUM(CASE WHEN was_good_block = 1 THEN 1 ELSE 0 END) as good_blocks,
                        CASE 
                            WHEN COUNT(*) > 0 THEN 
                                AVG(CASE WHEN was_good_block = 1 THEN 1.0 ELSE 0.0 END) * 100.0
                            ELSE 0.0
                        END as block_accuracy
                    FROM shadow_trades
                    WHERE ts >= ? AND blocked_by IS NOT NULL AND blocked_by != ''
                    GROUP BY blocked_by
                    ORDER BY total_blocked DESC
                """, (cutoff,))
                
                gates: List[Dict[str, Any]] = []
                total_blocked = 0
                rows: Any = cursor.fetchall()  # type: ignore
                
                print(f"📊 Gate Performance: Found {len(rows)} gates")
                
                for row in rows:
                    blocked_count: int = int(row[1]) if len(row) > 1 and row[1] is not None else 0  # type: ignore
                    if blocked_count == 0:
                        continue
                    
                    total_blocked += blocked_count
                    gate_name: str = str(row[0]) if len(row) > 0 and row[0] is not None else 'unknown'  # type: ignore
                    good_blocks: int = int(row[2]) if len(row) > 2 and row[2] is not None else 0  # type: ignore
                    block_accuracy: float = float(row[3]) if len(row) > 3 and row[3] is not None else 0.0  # type: ignore
                    
                    gates.append({
                        'gate': gate_name,
                        'total_blocked': blocked_count,
                        'good_blocks': good_blocks,
                        'block_accuracy': block_accuracy,
                    })
                
                print(f"📊 Gate Performance: Returning {len(gates)} gates, {total_blocked} total blocked")
                return {'gates': gates, 'total_blocked': total_blocked}
            finally:
                cursor.close()  # type: ignore
        except Exception as e:
            print(f"❌ Error fetching gate performance: {e}")
            import traceback
            traceback.print_exc()
            return {'gates': [], 'total_blocked': 0, 'error': str(e)}
        finally:
            if journal_conn:
                journal_conn.close()  # type: ignore

    def fetch_learning_metrics(self) -> Dict[str, Any]:
        """Fetch learning metrics"""
        learning_conn: Any = self.get_sqlite_connection('observability/data/sqlite/learning.db')
        if not learning_conn:
            return {}
        
        try:
            cursor: Any = learning_conn.cursor()  # type: ignore
            try:
                # Get latest training session
                cursor.execute("""
                    SELECT
                        auc, ece, brier, wr, delta_auc, samples_processed, ts
                    FROM training_sessions
                    ORDER BY ts DESC
                    LIMIT 1
                """)
                
                row: Any = cursor.fetchone()  # type: ignore
                if row:
                    return {
                        'auc': float(row[0]) if len(row) > 0 and row[0] is not None else 0.0,  # type: ignore
                        'ece': float(row[1]) if len(row) > 1 and row[1] is not None else 0.0,  # type: ignore
                        'brier': float(row[2]) if len(row) > 2 and row[2] is not None else 0.0,  # type: ignore
                        'wr': float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,  # type: ignore
                        'delta_auc': float(row[4]) if len(row) > 4 and row[4] is not None else 0.0,  # type: ignore
                        'samples_processed': int(row[5]) if len(row) > 5 and row[5] is not None else 0,  # type: ignore
                        'last_training': str(row[6]) if len(row) > 6 and row[6] is not None else None,  # type: ignore
                    }
                return {}
            finally:
                cursor.close()  # type: ignore
        except Exception as e:
            print(f"⚠️  Error fetching learning metrics: {e}")
            return {}
        finally:
            if learning_conn:
                learning_conn.close()  # type: ignore

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch all comprehensive dashboard data"""
        now = time.time()
        if self.cache_time and (now - self.cache_time) < self.cache_duration:
            return self.cache

        data: Dict[str, Any] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'training_progress': self.read_training_progress(),
            'system_health': self.get_system_health(),
        }

        # Database data
        conn: Any = self.get_db_connection()
        if conn:
            try:
                cur: Any = conn.cursor()  # type: ignore
                try:
                    # Overall stats
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
                    
                    row: Any = cur.fetchone()  # type: ignore
                    if row:
                        total_trades: int = int(row[0]) if len(row) > 0 and row[0] is not None else 0  # type: ignore
                        winning_trades: int = int(row[1]) if len(row) > 1 and row[1] is not None else 0  # type: ignore
                        losing_trades: int = int(row[2]) if len(row) > 2 and row[2] is not None else 0  # type: ignore
                        total_profit: float = float(row[3]) if len(row) > 3 and row[3] is not None else 0.0  # type: ignore
                        avg_profit: float = float(row[4]) if len(row) > 4 and row[4] is not None else 0.0  # type: ignore
                        avg_loss: float = float(row[5]) if len(row) > 5 and row[5] is not None else 0.0  # type: ignore
                        avg_confidence: float = float(row[6]) if len(row) > 6 and row[6] is not None else 0.0  # type: ignore
                        avg_hold: float = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0  # type: ignore
                        max_win: float = float(row[8]) if len(row) > 8 and row[8] is not None else 0.0  # type: ignore
                        max_loss: float = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0  # type: ignore
                        
                        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                        
                        # Calculate advanced metrics
                        if total_trades > 0 and abs(avg_loss) > 0.01:
                            profit_factor = abs(avg_profit * winning_trades / max(0.01, abs(avg_loss * losing_trades)))
                            expectancy = ((win_rate/100) * avg_profit - ((100-win_rate)/100) * abs(avg_loss))
                            risk_reward = abs(avg_profit / max(0.01, abs(avg_loss)))
                        else:
                            profit_factor = 0.0
                            expectancy = 0.0
                            risk_reward = 0.0
                        
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

                    # Recent trades (last 100)
                    cur.execute("""
                        SELECT
                            trade_id, symbol, entry_timestamp, entry_price, exit_price,
                            direction, net_profit_gbp, gross_profit_bps, is_winner,
                            exit_reason, model_confidence, hold_duration_minutes,
                            market_regime, volatility_bps, spread_at_entry_bps
                        FROM trade_memory
                        ORDER BY trade_id DESC
                        LIMIT 100
                    """)
                    
                    trades: List[Dict[str, Any]] = []
                    rows: Any = cur.fetchall()  # type: ignore
                    print(f"📋 Fetched {len(rows)} recent trades from database")
                    for row in rows:
                        try:
                            entry_ts: Any = row[2] if len(row) > 2 else None  # type: ignore
                            entry_ts_str: Optional[str] = entry_ts.isoformat() if entry_ts else None  # type: ignore
                            trades.append({  # type: ignore
                                'trade_id': int(row[0]) if len(row) > 0 and row[0] is not None else 0,  # type: ignore
                                'symbol': str(row[1]) if len(row) > 1 and row[1] is not None else '',  # type: ignore
                                'entry_timestamp': entry_ts_str,
                                'entry_price': float(row[3]) if len(row) > 3 and row[3] is not None else 0.0,  # type: ignore
                                'exit_price': float(row[4]) if len(row) > 4 and row[4] is not None else 0.0,  # type: ignore
                                'direction': str(row[5]) if len(row) > 5 and row[5] is not None else 'LONG',  # type: ignore
                                'net_profit_gbp': float(row[6]) if len(row) > 6 and row[6] is not None else 0.0,  # type: ignore
                                'gross_profit_bps': float(row[7]) if len(row) > 7 and row[7] is not None else 0.0,  # type: ignore
                                'is_winner': bool(row[8]) if len(row) > 8 and row[8] is not None else False,  # type: ignore
                                'exit_reason': str(row[9]) if len(row) > 9 and row[9] is not None else None,  # type: ignore
                                'model_confidence': float(row[10]) * 100 if len(row) > 10 and row[10] is not None else 0.0,  # type: ignore
                                'hold_duration_minutes': int(row[11]) if len(row) > 11 and row[11] is not None else 0,  # type: ignore
                                'market_regime': str(row[12]) if len(row) > 12 and row[12] is not None else 'unknown',  # type: ignore
                                'volatility_bps': float(row[13]) if len(row) > 13 and row[13] is not None else 0.0,  # type: ignore
                                'spread_bps': float(row[14]) if len(row) > 14 and row[14] is not None else 0.0,  # type: ignore
                            })
                        except Exception as e:
                            print(f"⚠️  Error parsing trade row: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    print(f"✅ Processed {len(trades)} recent trades")
                    data['recent_trades'] = trades
                finally:
                    cur.close()  # type: ignore

            except Exception as e:
                print(f"❌ Database query error: {e}")
                import traceback
                traceback.print_exc()
                data['overview'] = self._empty_overview()
                data['recent_trades'] = []
            finally:
                if conn:
                    conn.close()  # type: ignore
        else:
            print("❌ No database connection available")
            data['overview'] = self._empty_overview()
            data['recent_trades'] = []

        # Fetch additional metrics
        try:
            data['pnl_series'] = self.fetch_pnl_series(days=7)
            data['win_rate_trend'] = self.fetch_win_rate_trend(days=7)
            data['regime_distribution'] = self.fetch_regime_distribution()
            data['confidence_heatmap'] = self.fetch_confidence_heatmap()
            data['performance_by_symbol'] = self.fetch_performance_by_symbol()
            data['gate_performance'] = self.fetch_gate_performance()
            data['learning_metrics'] = self.fetch_learning_metrics()
            
            # Debug logging
            print(f"📊 Data summary:")
            print(f"   - performance_by_symbol: {len(data.get('performance_by_symbol', []))} symbols")
            print(f"   - recent_trades: {len(data.get('recent_trades', []))} trades")
            print(f"   - pnl_series: {len(data.get('pnl_series', []))} points")
            print(f"   - win_rate_trend: {len(data.get('win_rate_trend', []))} points")
            print(f"   - regime_distribution: {len(data.get('regime_distribution', {}).get('regimes', []))} regimes")
        except Exception as e:
            print(f"⚠️  Error fetching additional metrics: {e}")
            import traceback
            traceback.print_exc()
            # Ensure these fields exist even if empty
            if 'performance_by_symbol' not in data:
                data['performance_by_symbol'] = []
            if 'recent_trades' not in data:
                data['recent_trades'] = []

        # Cache the data
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
dashboard_data = EnhancedDashboardData()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('enhanced_dashboard.html')


@app.route('/api/data')
def get_data():
    """Get all dashboard data as JSON"""
    try:
        data = dashboard_data.fetch_data()
        # Convert all Decimals to floats for JSON serialization
        data = convert_decimals(data)
        
        # Ensure performance_by_symbol and recent_trades are always arrays
        if 'performance_by_symbol' not in data:
            data['performance_by_symbol'] = []
        if not isinstance(data.get('performance_by_symbol'), list):
            data['performance_by_symbol'] = []
            
        if 'recent_trades' not in data:
            data['recent_trades'] = []
        if not isinstance(data.get('recent_trades'), list):
            data['recent_trades'] = []
        
        # Debug logging
        print(f"📤 /api/data response:")
        print(f"   - performance_by_symbol: {len(data.get('performance_by_symbol', []))} items (type: {type(data.get('performance_by_symbol'))})")
        print(f"   - recent_trades: {len(data.get('recent_trades', []))} items (type: {type(data.get('recent_trades'))})")
        
        return jsonify(data)
    except Exception as e:
        print(f"❌ Error in /api/data endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e), 
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance_by_symbol': [],
            'recent_trades': []
        }), 500


@app.route('/api/stream')
def stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        while True:
            try:
                data = dashboard_data.fetch_data()
                # Convert all Decimals to floats for JSON serialization
                data = convert_decimals(data)
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1.0)  # Update every 1 second (exactly)
            except GeneratorExit:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/health')
def health():
    """Health check endpoint"""
    conn: Any = dashboard_data.get_db_connection()
    db_status = 'connected' if conn else 'disconnected'
    if conn:
        conn.close()  # type: ignore
    return jsonify({
        'status': 'ok',
        'database': db_status,
        'observability': 'available' if OBSERVABILITY_AVAILABLE else 'unavailable',
    })


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Enhanced Comprehensive Huracan Engine Dashboard")
    print("=" * 80)
    print(f"📊 Dashboard URL: http://localhost:5055/")
    print(f"🔌 API Endpoint:  http://localhost:5055/api/data")
    print(f"💓 Stream:        http://localhost:5055/api/stream")
    print(f"❤️  Health Check:  http://localhost:5055/api/health")
    print("=" * 80)
    print("✨ Features: Real-time updates every second, comprehensive metrics,")
    print("            beautiful charts, deep insights into engine performance")
    print("=" * 80)
    print("Press Ctrl+C to stop")
    print("=" * 80)

    # Check if default port is available, use alternative if not
    import socket
    default_port = 5055
    port = default_port
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', default_port))
        s.close()
    except OSError:
        # Port is in use, find an available port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        print(f"⚠️  Port {default_port} is in use, using port {port} instead")
    
    print(f"🚀 Starting dashboard server on http://localhost:{port}/")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

