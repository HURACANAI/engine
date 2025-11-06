"""
Market Context Logger

Logs market conditions at the time of each signal/shadow trade.

This helps answer:
- "What market conditions lead to winning shadow trades?"
- "Should we avoid trading in high volatility?"
- "What's the optimal spread for shadow trading?"

Usage:
    context_logger = MarketContextLogger()

    # Log market snapshot
    context_logger.log_context(
        ts=datetime.utcnow().isoformat(),
        symbol="ETH-USD",
        price=2045.50,
        volatility_1h=0.34,
        spread_bps=4.2,
        volume_ratio=1.2,
        regime="TREND"
    )

    # Analyze: What conditions work best?
    analysis = context_logger.analyze_optimal_conditions(days=7)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
import structlog

logger = structlog.get_logger(__name__)


class MarketContextLogger:
    """
    Log market context for analysis.

    Helps identify optimal trading conditions for shadow trades.
    """

    def __init__(self, db_path: str = "observability/data/sqlite/market_context.db"):
        """
        Initialize market context logger.

        Args:
            db_path: Path to market context database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self._create_tables()

        logger.info("market_context_logger_initialized", db_path=str(self.db_path))

    def _create_tables(self):
        """Create market context tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots(
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Price
                price REAL NOT NULL,

                -- Volatility
                volatility_1h REAL,
                volatility_24h REAL,

                -- Liquidity
                spread_bps REAL,
                bid_size REAL,
                ask_size REAL,
                liquidity_score REAL,

                -- Volume
                volume_1h REAL,
                volume_24h REAL,
                volume_ratio REAL,  -- current / 24h avg

                -- Trend
                trend_30m REAL,
                trend_1h REAL,
                trend_4h REAL,

                -- Order book
                order_book_imbalance REAL,  -- (bid_volume - ask_volume) / total

                -- Regime
                regime TEXT,

                -- Link to event
                signal_id TEXT,
                shadow_trade_id TEXT
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_market_snapshots_ts
            ON market_snapshots(ts)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_market_snapshots_symbol
            ON market_snapshots(symbol)
        """)

        self.conn.commit()

    def log_context(
        self,
        ts: str,
        symbol: str,
        price: float,
        volatility_1h: Optional[float] = None,
        spread_bps: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        regime: Optional[str] = None,
        signal_id: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Log market context snapshot.

        Args:
            ts: Timestamp
            symbol: Trading symbol
            price: Current price
            volatility_1h: 1-hour volatility
            spread_bps: Bid-ask spread in bps
            volume_ratio: Volume vs 24h average
            regime: Market regime
            signal_id: Link to signal event
            **kwargs: Additional context fields

        Returns:
            snapshot_id
        """
        cursor = self.conn.execute("""
            INSERT INTO market_snapshots(
                ts, symbol, price,
                volatility_1h, spread_bps, volume_ratio, regime,
                signal_id,
                volatility_24h, bid_size, ask_size, liquidity_score,
                volume_1h, volume_24h, trend_30m, trend_1h, trend_4h,
                order_book_imbalance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, symbol, price,
            volatility_1h, spread_bps, volume_ratio, regime,
            signal_id,
            kwargs.get('volatility_24h'), kwargs.get('bid_size'), kwargs.get('ask_size'),
            kwargs.get('liquidity_score'), kwargs.get('volume_1h'), kwargs.get('volume_24h'),
            kwargs.get('trend_30m'), kwargs.get('trend_1h'), kwargs.get('trend_4h'),
            kwargs.get('order_book_imbalance')
        ))

        snapshot_id = cursor.lastrowid
        self.conn.commit()

        logger.debug(
            "market_context_logged",
            snapshot_id=snapshot_id,
            symbol=symbol,
            volatility=volatility_1h,
            spread=spread_bps
        )

        return snapshot_id

    def analyze_optimal_conditions(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze optimal market conditions for shadow trading.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with optimal ranges for each metric
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Get statistics
        stats = self.conn.execute("""
            SELECT
                AVG(volatility_1h) as avg_volatility,
                MIN(volatility_1h) as min_volatility,
                MAX(volatility_1h) as max_volatility,
                AVG(spread_bps) as avg_spread,
                MIN(spread_bps) as min_spread,
                MAX(spread_bps) as max_spread,
                AVG(volume_ratio) as avg_volume_ratio,
                COUNT(*) as total_snapshots
            FROM market_snapshots
            WHERE ts >= ?
        """, (cutoff,)).fetchone()

        # By regime
        by_regime = self.conn.execute("""
            SELECT
                regime,
                COUNT(*) as count,
                AVG(volatility_1h) as avg_volatility,
                AVG(spread_bps) as avg_spread
            FROM market_snapshots
            WHERE ts >= ? AND regime IS NOT NULL
            GROUP BY regime
        """, (cutoff,)).fetchall()

        return {
            "period_days": days,
            "total_snapshots": stats['total_snapshots'],
            "volatility": {
                "avg": stats['avg_volatility'],
                "min": stats['min_volatility'],
                "max": stats['max_volatility']
            },
            "spread": {
                "avg": stats['avg_spread'],
                "min": stats['min_spread'],
                "max": stats['max_spread']
            },
            "volume_ratio": {
                "avg": stats['avg_volume_ratio']
            },
            "by_regime": [dict(r) for r in by_regime]
        }


if __name__ == '__main__':
    # Example usage
    print("Market Context Logger Example")
    print("=" * 80)

    logger_instance = MarketContextLogger(db_path="observability/data/test_market_context.db")

    # Log market snapshots
    print("\n📊 Logging market snapshots...")

    for i in range(10):
        snapshot_id = logger_instance.log_context(
            ts=datetime.utcnow().isoformat(),
            symbol="ETH-USD",
            price=2045.50 + i,
            volatility_1h=0.30 + i * 0.01,
            spread_bps=4.0 + i * 0.2,
            volume_ratio=1.0 + i * 0.1,
            regime="TREND" if i % 2 == 0 else "RANGE",
            signal_id=f"sig_{i:03d}"
        )

    print(f"✓ Logged 10 market snapshots")

    # Analyze
    print("\n📈 Analyzing optimal conditions...")
    analysis = logger_instance.analyze_optimal_conditions(days=7)

    print(f"  Total snapshots: {analysis['total_snapshots']}")
    print(f"  Avg volatility: {analysis['volatility']['avg']:.3f}")
    print(f"  Avg spread: {analysis['spread']['avg']:.1f} bps")
    print(f"  Avg volume ratio: {analysis['volume_ratio']['avg']:.2f}x")

    print(f"\n  By regime:")
    for regime_data in analysis['by_regime']:
        print(f"    {regime_data['regime']}: {regime_data['count']} snapshots, "
              f"volatility {regime_data['avg_volatility']:.3f}")

    print("\n✓ Market context logger ready!")
