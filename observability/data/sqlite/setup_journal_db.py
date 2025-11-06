"""
Setup script for journal.db - Trade journal database

This script creates the SQLite database for storing trade data.
Run once to initialize, safe to run multiple times (idempotent).

Tables:
- trades: Core trade information
- trade_features: Features at trade entry time
- trade_outcomes: Detailed P&L breakdown
- shadow_trades: Trades that passed signals but were blocked by gates

Usage:
    python -m observability.data.sqlite.setup_journal_db
"""

import sqlite3
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


def setup_journal_db(db_path: str = "observability/data/sqlite/journal.db"):
    """
    Create journal.db with all required tables.

    Args:
        db_path: Path to database file
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # ============================================================================
    # TRADES TABLE
    # ============================================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades(
            trade_id TEXT PRIMARY KEY,
            ts_open TEXT NOT NULL,
            ts_close TEXT,

            -- Symbol & Mode
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,  -- 'scalp' or 'runner'
            regime TEXT NOT NULL,  -- 'TREND', 'CHOP', 'VOLATILE'

            -- Entry
            side TEXT NOT NULL,  -- 'long' or 'short'
            entry_price REAL NOT NULL,
            size_gbp REAL NOT NULL,
            size_asset REAL NOT NULL,

            -- Exit (NULL if still open)
            exit_price REAL,
            exit_reason TEXT,  -- 'TP', 'SL', 'timeout', 'manual'

            -- P&L (NULL if still open)
            pnl_gbp REAL,
            pnl_pct REAL,
            return_bps REAL,

            -- Fees
            fee_entry_gbp REAL,
            fee_exit_gbp REAL,
            slippage_bps REAL,

            -- Model info
            model_id TEXT,
            code_git_sha TEXT,

            -- Status
            status TEXT NOT NULL,  -- 'open', 'closed', 'error'

            -- Metadata
            event_id TEXT,  -- Link to Event in DuckDB/Parquet
            tags TEXT  -- JSON array
        )
    """)

    # Indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trades_ts_open ON trades(ts_open)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trades_symbol ON trades(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trades_mode ON trades(mode)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trades_regime ON trades(regime)")

    # ============================================================================
    # TRADE_FEATURES TABLE
    # ============================================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_features(
            trade_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,

            -- Signal features
            confidence REAL,
            signal_strength REAL,
            predicted_return REAL,
            predicted_sharpe REAL,

            -- Market context (at entry time)
            volatility_1h REAL,
            spread_bps REAL,
            liquidity_score REAL,
            recent_trend_30m REAL,
            volume_vs_avg REAL,
            order_book_imbalance REAL,

            -- Gate outputs
            meta_label_prob REAL,
            cost_adjusted_sharpe REAL,
            execution_quality REAL,

            -- All features (JSON)
            features_json TEXT,

            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS ix_trade_features_ts ON trade_features(ts)")

    # ============================================================================
    # TRADE_OUTCOMES TABLE
    # ============================================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_outcomes(
            trade_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,

            -- Realized outcome
            actual_return_bps REAL NOT NULL,
            actual_duration_sec REAL NOT NULL,
            actual_sharpe REAL,

            -- Predicted vs Actual
            predicted_return_bps REAL,
            prediction_error_bps REAL,

            -- Drawdown
            max_drawdown_bps REAL,
            max_runup_bps REAL,

            -- Target hits
            hit_tp BOOLEAN,
            hit_sl BOOLEAN,
            time_to_tp_sec REAL,
            time_to_sl_sec REAL,

            -- Quality metrics
            execution_slippage_bps REAL,
            total_cost_bps REAL,
            net_return_bps REAL,

            -- Meta
            outcome_json TEXT,  -- Full outcome data

            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS ix_trade_outcomes_ts ON trade_outcomes(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_trade_outcomes_return ON trade_outcomes(actual_return_bps)")

    # ============================================================================
    # SHADOW_TRADES TABLE
    # ============================================================================
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades(
            shadow_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,

            -- Trade details
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,

            -- Why blocked
            blocked_by TEXT NOT NULL,  -- Gate name that blocked
            block_reason TEXT NOT NULL,

            -- Gate inputs
            gate_inputs TEXT,  -- JSON

            -- Counterfactual: What would have happened?
            cf_pnl_bps REAL,  -- If we had taken the trade
            cf_duration_sec REAL,
            cf_hit_tp BOOLEAN,
            cf_hit_sl BOOLEAN,

            -- Was this a good block?
            was_good_block BOOLEAN,  -- TRUE if trade would have lost money

            -- Model info
            model_id TEXT,

            -- Metadata
            event_id TEXT,
            tags TEXT
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_trades_ts ON shadow_trades(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_trades_blocked_by ON shadow_trades(blocked_by)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_trades_mode ON shadow_trades(mode)")

    # ============================================================================
    # VIEWS FOR COMMON QUERIES
    # ============================================================================

    # Win rate by mode
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_win_rate_by_mode AS
        SELECT
            mode,
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl_gbp > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_gbp <= 0 THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(CASE WHEN pnl_gbp > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
            ROUND(SUM(pnl_gbp), 2) as total_pnl_gbp,
            ROUND(AVG(return_bps), 1) as avg_return_bps
        FROM trades
        WHERE status = 'closed'
        GROUP BY mode
    """)

    # Win rate by regime
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_win_rate_by_regime AS
        SELECT
            regime,
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl_gbp > 0 THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN pnl_gbp > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
            ROUND(SUM(pnl_gbp), 2) as total_pnl_gbp
        FROM trades
        WHERE status = 'closed'
        GROUP BY regime
    """)

    # Recent trades
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_recent_trades AS
        SELECT
            t.trade_id,
            t.ts_open,
            t.symbol,
            t.mode,
            t.side,
            t.entry_price,
            t.exit_price,
            t.pnl_gbp,
            t.return_bps,
            t.exit_reason,
            tf.confidence,
            tf.meta_label_prob
        FROM trades t
        LEFT JOIN trade_features tf ON t.trade_id = tf.trade_id
        WHERE t.status = 'closed'
        ORDER BY t.ts_close DESC
        LIMIT 100
    """)

    # Shadow trades summary
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_shadow_trades_summary AS
        SELECT
            blocked_by,
            COUNT(*) as total_blocked,
            SUM(CASE WHEN was_good_block = 1 THEN 1 ELSE 0 END) as good_blocks,
            SUM(CASE WHEN was_good_block = 0 THEN 1 ELSE 0 END) as bad_blocks,
            ROUND(AVG(CASE WHEN was_good_block = 1 THEN 1.0 ELSE 0.0 END), 3) as block_accuracy,
            ROUND(SUM(cf_pnl_bps), 1) as missed_pnl_bps
        FROM shadow_trades
        WHERE cf_pnl_bps IS NOT NULL
        GROUP BY blocked_by
    """)

    conn.commit()
    conn.close()

    logger.info("journal_db_setup_complete", db_path=str(db_path))
    print(f"✓ journal.db created at {db_path}")
    print("  Tables: trades, trade_features, trade_outcomes, shadow_trades")
    print("  Views: v_win_rate_by_mode, v_win_rate_by_regime, v_recent_trades, v_shadow_trades_summary")


def get_schema_info(db_path: str = "observability/data/sqlite/journal.db"):
    """Print database schema information"""
    conn = sqlite3.connect(db_path)

    print("\n" + "="*80)
    print("JOURNAL.DB SCHEMA")
    print("="*80)

    # Get all tables
    tables = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """).fetchall()

    for (table_name,) in tables:
        print(f"\n📊 {table_name.upper()}")
        print("-" * 80)

        # Get columns
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        for col in columns:
            col_id, name, type_, notnull, default, pk = col
            pk_mark = " 🔑" if pk else ""
            null_mark = " NOT NULL" if notnull else ""
            print(f"  {name:30s} {type_:15s}{null_mark}{pk_mark}")

    # Get views
    views = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='view'
        ORDER BY name
    """).fetchall()

    if views:
        print("\n" + "="*80)
        print("VIEWS")
        print("="*80)
        for (view_name,) in views:
            print(f"  • {view_name}")

    conn.close()


if __name__ == '__main__':
    print("Setting up journal.db...")
    print("="*80)

    setup_journal_db()
    get_schema_info()

    print("\n" + "="*80)
    print("✓ journal.db setup complete")
    print("="*80)
