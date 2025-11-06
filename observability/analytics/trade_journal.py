"""
Trade Journal

Queryable database of all trades with rich analytics.

Provides:
- Full trade history with filters
- Win/loss analytics
- Performance by symbol, mode, regime
- Time-series analysis
- Trade replay (what happened)

This answers: "Show me all runner trades on ETH that lost money in TREND regime"

Usage:
    journal = TradeJournal()

    # Record trade
    journal.record_trade(
        trade_id="trade_001",
        symbol="ETH-USD",
        mode="scalp",
        ...
    )

    # Query trades
    trades = journal.query_trades(
        mode="runner",
        regime="TREND",
        min_return_bps=-10,
        limit=100
    )

    # Get analytics
    stats = journal.get_stats(mode="scalp", days=7)
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog
import pandas as pd

logger = structlog.get_logger(__name__)


class TradeJournal:
    """
    Queryable trade journal with rich analytics.

    Uses journal.db created by setup_journal_db.py
    """

    def __init__(self, db_path: str = "observability/data/sqlite/journal.db"):
        """
        Initialize trade journal.

        Args:
            db_path: Path to journal database
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"journal.db not found at {self.db_path}. "
                "Run: python -m observability.data.sqlite.setup_journal_db"
            )

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        logger.info("trade_journal_initialized", db_path=str(self.db_path))

    def record_trade(
        self,
        trade_id: str,
        ts_open: str,
        symbol: str,
        mode: str,
        regime: str,
        side: str,
        entry_price: float,
        size_gbp: float,
        size_asset: float,
        fee_entry_gbp: float,
        model_id: Optional[str] = None,
        code_git_sha: Optional[str] = None,
        event_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Record trade entry.

        Args:
            trade_id: Unique trade ID
            ts_open: Timestamp of entry
            symbol: Trading symbol
            mode: 'scalp' or 'runner'
            regime: Market regime
            side: 'long' or 'short'
            entry_price: Entry price
            size_gbp: Position size in GBP
            size_asset: Position size in asset
            fee_entry_gbp: Entry fee
            model_id: Model ID that generated signal
            code_git_sha: Git SHA of code
            event_id: Link to Event
            tags: Tags

        Returns:
            trade_id
        """
        import json

        self.conn.execute("""
            INSERT INTO trades(
                trade_id, ts_open, symbol, mode, regime, side,
                entry_price, size_gbp, size_asset, fee_entry_gbp,
                model_id, code_git_sha, status, event_id, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, ts_open, symbol, mode, regime, side,
            entry_price, size_gbp, size_asset, fee_entry_gbp,
            model_id, code_git_sha, 'open', event_id,
            json.dumps(tags) if tags else None
        ))

        self.conn.commit()

        logger.info(
            "trade_recorded",
            trade_id=trade_id,
            symbol=symbol,
            mode=mode,
            side=side,
            size_gbp=size_gbp
        )

        return trade_id

    def close_trade(
        self,
        trade_id: str,
        ts_close: str,
        exit_price: float,
        exit_reason: str,
        fee_exit_gbp: float,
        slippage_bps: float
    ):
        """
        Close a trade and compute P&L.

        Args:
            trade_id: Trade ID
            ts_close: Close timestamp
            exit_price: Exit price
            exit_reason: 'TP', 'SL', 'timeout', 'manual'
            fee_exit_gbp: Exit fee
            slippage_bps: Slippage in bps
        """
        # Get trade entry
        trade = self.conn.execute("""
            SELECT * FROM trades WHERE trade_id = ?
        """, (trade_id,)).fetchone()

        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        if trade['status'] == 'closed':
            logger.warning("trade_already_closed", trade_id=trade_id)
            return

        # Compute P&L
        entry_price = trade['entry_price']
        size_asset = trade['size_asset']
        side = trade['side']

        if side == 'long':
            pnl_asset = size_asset * (exit_price - entry_price) / entry_price
        else:  # short
            pnl_asset = size_asset * (entry_price - exit_price) / entry_price

        pnl_gbp = pnl_asset * entry_price  # Approximate
        total_fees = trade['fee_entry_gbp'] + fee_exit_gbp
        pnl_gbp -= total_fees

        pnl_pct = pnl_gbp / trade['size_gbp']
        return_bps = pnl_pct * 10000

        # Update trade
        self.conn.execute("""
            UPDATE trades SET
                ts_close = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl_gbp = ?,
                pnl_pct = ?,
                return_bps = ?,
                fee_exit_gbp = ?,
                slippage_bps = ?,
                status = 'closed'
            WHERE trade_id = ?
        """, (
            ts_close, exit_price, exit_reason,
            pnl_gbp, pnl_pct, return_bps,
            fee_exit_gbp, slippage_bps,
            trade_id
        ))

        self.conn.commit()

        logger.info(
            "trade_closed",
            trade_id=trade_id,
            pnl_gbp=pnl_gbp,
            return_bps=return_bps,
            reason=exit_reason
        )

    def record_trade_features(
        self,
        trade_id: str,
        ts: str,
        features: Dict[str, float],
        market_context: Dict[str, float],
        gate_outputs: Dict[str, float]
    ):
        """Record features at trade entry time"""
        import json

        self.conn.execute("""
            INSERT INTO trade_features(
                trade_id, ts,
                confidence, signal_strength, predicted_return, predicted_sharpe,
                volatility_1h, spread_bps, liquidity_score, recent_trend_30m,
                volume_vs_avg, order_book_imbalance,
                meta_label_prob, cost_adjusted_sharpe, execution_quality,
                features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, ts,
            features.get('confidence'), features.get('signal_strength'),
            features.get('predicted_return'), features.get('predicted_sharpe'),
            market_context.get('volatility_1h'), market_context.get('spread_bps'),
            market_context.get('liquidity_score'), market_context.get('recent_trend_30m'),
            market_context.get('volume_vs_avg'), market_context.get('order_book_imbalance'),
            gate_outputs.get('meta_label_prob'), gate_outputs.get('cost_adjusted_sharpe'),
            gate_outputs.get('execution_quality'),
            json.dumps({**features, **market_context, **gate_outputs})
        ))

        self.conn.commit()

    def record_trade_outcome(
        self,
        trade_id: str,
        ts: str,
        actual_return_bps: float,
        actual_duration_sec: float,
        actual_sharpe: Optional[float] = None,
        predicted_return_bps: Optional[float] = None,
        max_drawdown_bps: Optional[float] = None,
        max_runup_bps: Optional[float] = None,
        hit_tp: Optional[bool] = None,
        hit_sl: Optional[bool] = None
    ):
        """Record detailed trade outcome"""
        prediction_error = None
        if predicted_return_bps is not None:
            prediction_error = actual_return_bps - predicted_return_bps

        self.conn.execute("""
            INSERT INTO trade_outcomes(
                trade_id, ts, actual_return_bps, actual_duration_sec, actual_sharpe,
                predicted_return_bps, prediction_error_bps,
                max_drawdown_bps, max_runup_bps, hit_tp, hit_sl,
                execution_slippage_bps, total_cost_bps, net_return_bps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, ts, actual_return_bps, actual_duration_sec, actual_sharpe,
            predicted_return_bps, prediction_error,
            max_drawdown_bps, max_runup_bps, hit_tp, hit_sl,
            None, None, actual_return_bps  # TODO: compute from trade
        ))

        self.conn.commit()

    def query_trades(
        self,
        mode: Optional[str] = None,
        regime: Optional[str] = None,
        symbol: Optional[str] = None,
        min_return_bps: Optional[float] = None,
        max_return_bps: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: str = 'closed',
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query trades with filters.

        Args:
            mode: Filter by mode ('scalp', 'runner')
            regime: Filter by regime
            symbol: Filter by symbol
            min_return_bps: Minimum return
            max_return_bps: Maximum return
            start_date: Start date
            end_date: End date
            status: 'open', 'closed', or 'all'
            limit: Max results

        Returns:
            List of trades
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        if regime:
            query += " AND regime = ?"
            params.append(regime)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if status != 'all':
            query += " AND status = ?"
            params.append(status)

        if min_return_bps is not None:
            query += " AND return_bps >= ?"
            params.append(min_return_bps)

        if max_return_bps is not None:
            query += " AND return_bps <= ?"
            params.append(max_return_bps)

        if start_date:
            query += " AND ts_open >= ?"
            params.append(start_date)

        if end_date:
            query += " AND ts_open <= ?"
            params.append(end_date)

        query += " ORDER BY ts_open DESC LIMIT ?"
        params.append(limit)

        trades = self.conn.execute(query, params).fetchall()

        return [dict(t) for t in trades]

    def get_stats(
        self,
        mode: Optional[str] = None,
        regime: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance statistics.

        Args:
            mode: Filter by mode
            regime: Filter by regime
            days: Number of days to look back

        Returns:
            Dict with win_rate, avg_return, total_pnl, etc.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl_gbp > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl_gbp <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN pnl_gbp > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(pnl_gbp) as total_pnl_gbp,
                AVG(return_bps) as avg_return_bps,
                MAX(return_bps) as max_return_bps,
                MIN(return_bps) as min_return_bps,
                AVG(ABS(return_bps)) as avg_abs_return_bps
            FROM trades
            WHERE status = 'closed' AND ts_close >= ?
        """
        params = [cutoff]

        if mode:
            query += " AND mode = ?"
            params.append(mode)

        if regime:
            query += " AND regime = ?"
            params.append(regime)

        stats = self.conn.execute(query, params).fetchone()

        return dict(stats) if stats else {}

    def get_performance_by_mode(self, days: int = 30) -> pd.DataFrame:
        """Get performance breakdown by mode"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        query = """
            SELECT * FROM v_win_rate_by_mode
        """
        # Note: View doesn't support date filtering, would need to recreate query

        df = pd.read_sql_query(
            "SELECT * FROM v_win_rate_by_mode",
            self.conn
        )

        return df

    def get_performance_by_regime(self, days: int = 30) -> pd.DataFrame:
        """Get performance breakdown by regime"""
        df = pd.read_sql_query(
            "SELECT * FROM v_win_rate_by_regime",
            self.conn
        )

        return df

    def get_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        """Get recent trades"""
        df = pd.read_sql_query(
            f"SELECT * FROM v_recent_trades LIMIT {limit}",
            self.conn
        )

        return df


if __name__ == '__main__':
    # Example usage
    print("Trade Journal Example")
    print("=" * 80)

    journal = TradeJournal(db_path="observability/data/sqlite/journal.db")

    # Record a trade
    trade_id = "example_trade_001"

    journal.record_trade(
        trade_id=trade_id,
        ts_open=datetime.utcnow().isoformat(),
        symbol="ETH-USD",
        mode="scalp",
        regime="TREND",
        side="long",
        entry_price=2045.50,
        size_gbp=100.0,
        size_asset=0.0489,
        fee_entry_gbp=0.10,
        model_id="sha256:abc123",
        tags=["test", "example"]
    )
    print(f"\n✓ Trade recorded: {trade_id}")

    # Record features
    journal.record_trade_features(
        trade_id=trade_id,
        ts=datetime.utcnow().isoformat(),
        features={"confidence": 0.78, "signal_strength": 0.82},
        market_context={"volatility_1h": 0.34, "spread_bps": 4.2},
        gate_outputs={"meta_label_prob": 0.78, "cost_adjusted_sharpe": 2.1}
    )
    print("✓ Features recorded")

    # Close trade
    import time
    time.sleep(0.1)
    journal.close_trade(
        trade_id=trade_id,
        ts_close=datetime.utcnow().isoformat(),
        exit_price=2052.30,
        exit_reason="TP",
        fee_exit_gbp=0.10,
        slippage_bps=1.2
    )
    print("✓ Trade closed")

    # Query
    trades = journal.query_trades(mode="scalp", limit=10)
    print(f"\n✓ Found {len(trades)} scalp trades")

    # Stats
    stats = journal.get_stats(mode="scalp", days=30)
    print(f"\n✓ Stats:")
    print(f"  Total trades: {stats.get('total_trades', 0)}")
    print(f"  Win rate: {stats.get('win_rate', 0):.1%}")
    print(f"  Total P&L: £{stats.get('total_pnl_gbp', 0):.2f}")

    print("\n✓ Trade journal ready!")
