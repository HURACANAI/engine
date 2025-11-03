"""Memory store for trade patterns using vector similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradeMemory:
    """Represents a single trade stored in memory."""
    trade_id: Optional[int]
    symbol: str
    entry_timestamp: datetime
    entry_price: float
    entry_features: Dict[str, Any]
    entry_embedding: np.ndarray
    position_size_gbp: float
    direction: str
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    hold_duration_minutes: Optional[int] = None
    gross_profit_bps: Optional[float] = None
    net_profit_gbp: Optional[float] = None
    fees_gbp: Optional[float] = None
    slippage_bps: Optional[float] = None
    market_regime: Optional[str] = None
    volatility_bps: Optional[float] = None
    spread_at_entry_bps: Optional[float] = None
    is_winner: Optional[bool] = None
    win_quality: Optional[str] = None
    model_version: Optional[str] = None
    model_confidence: Optional[float] = None


@dataclass
class SimilarPattern:
    """Result from similarity search."""
    trade_id: int
    symbol: str
    entry_timestamp: datetime
    similarity_score: float  # 0-1, higher is more similar
    is_winner: bool
    net_profit_gbp: float
    hold_duration_minutes: int
    market_regime: str


@dataclass
class PatternStats:
    """Aggregated statistics for similar patterns."""
    total_occurrences: int
    wins: int
    losses: int
    win_rate: float
    avg_profit_gbp: float
    avg_hold_minutes: int
    sharpe_ratio: float
    reliability_score: float


class MemoryStore:
    """Manages trade memory with vector similarity search capabilities."""

    def __init__(self, dsn: str, embedding_dim: int = 128) -> None:
        self._dsn = dsn
        self._embedding_dim = embedding_dim
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            logger.info("memory_store_connected")

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("memory_store_closed")

    def store_trade(self, trade: TradeMemory) -> int:
        """
        Store a trade in memory.

        Returns:
            trade_id: The inserted trade ID
        """
        self.connect()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trade_memory (
                    symbol, entry_timestamp, entry_price, entry_features, entry_embedding,
                    position_size_gbp, direction, exit_timestamp, exit_price, exit_reason,
                    hold_duration_minutes, gross_profit_bps, net_profit_gbp, fees_gbp,
                    slippage_bps, market_regime, volatility_bps, spread_at_entry_bps,
                    is_winner, win_quality, model_version, model_confidence
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING trade_id
                """,
                (
                    trade.symbol,
                    trade.entry_timestamp,
                    trade.entry_price,
                    Json(trade.entry_features),
                    trade.entry_embedding.tolist(),
                    trade.position_size_gbp,
                    trade.direction,
                    trade.exit_timestamp,
                    trade.exit_price,
                    trade.exit_reason,
                    trade.hold_duration_minutes,
                    trade.gross_profit_bps,
                    trade.net_profit_gbp,
                    trade.fees_gbp,
                    trade.slippage_bps,
                    trade.market_regime,
                    trade.volatility_bps,
                    trade.spread_at_entry_bps,
                    trade.is_winner,
                    trade.win_quality,
                    trade.model_version,
                    trade.model_confidence,
                ),
            )
            trade_id = cur.fetchone()[0]
            self._conn.commit()
            logger.info("trade_stored", trade_id=trade_id, symbol=trade.symbol)
            return trade_id

    def find_similar_patterns(
        self,
        embedding: np.ndarray,
        symbol: Optional[str] = None,
        market_regime: Optional[str] = None,
        top_k: int = 20,
        min_similarity: float = 0.7,
    ) -> List[SimilarPattern]:
        """
        Find similar historical trades using vector similarity.

        Args:
            embedding: Feature embedding to search for
            symbol: Optional filter by symbol
            market_regime: Optional filter by market regime
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of similar patterns ordered by similarity
        """
        self.connect()

        where_clauses = ["1 - (entry_embedding <=> %s::vector) >= %s"]
        params: List[Any] = [embedding.tolist(), min_similarity]

        if symbol:
            where_clauses.append("symbol = %s")
            params.append(symbol)

        if market_regime:
            where_clauses.append("market_regime = %s")
            params.append(market_regime)

        where_clause = " AND ".join(where_clauses)
        params.append(top_k)

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    trade_id,
                    symbol,
                    entry_timestamp,
                    1 - (entry_embedding <=> %s::vector) as similarity_score,
                    is_winner,
                    net_profit_gbp,
                    hold_duration_minutes,
                    market_regime
                FROM trade_memory
                WHERE {where_clause}
                ORDER BY entry_embedding <=> %s::vector
                LIMIT %s
                """,
                [embedding.tolist()] + params,
            )

            results = []
            for row in cur.fetchall():
                results.append(
                    SimilarPattern(
                        trade_id=row["trade_id"],
                        symbol=row["symbol"],
                        entry_timestamp=row["entry_timestamp"],
                        similarity_score=row["similarity_score"],
                        is_winner=row["is_winner"],
                        net_profit_gbp=row["net_profit_gbp"],
                        hold_duration_minutes=row["hold_duration_minutes"],
                        market_regime=row["market_regime"],
                    )
                )

            logger.info("similar_patterns_found", count=len(results), top_k=top_k)
            return results

    def get_pattern_stats(self, similar_patterns: List[SimilarPattern]) -> PatternStats:
        """Calculate aggregate statistics from similar patterns."""
        if not similar_patterns:
            return PatternStats(
                total_occurrences=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                avg_profit_gbp=0.0,
                avg_hold_minutes=0,
                sharpe_ratio=0.0,
                reliability_score=0.0,
            )

        wins = sum(1 for p in similar_patterns if p.is_winner)
        losses = len(similar_patterns) - wins
        win_rate = wins / len(similar_patterns)

        profits = [p.net_profit_gbp for p in similar_patterns]
        avg_profit = np.mean(profits)
        avg_hold = int(np.mean([p.hold_duration_minutes for p in similar_patterns]))

        # Sharpe calculation (simplified)
        if len(profits) > 1:
            sharpe = np.mean(profits) / (np.std(profits) + 1e-6)
        else:
            sharpe = 0.0

        # Reliability: combination of win rate and sample size
        sample_size_factor = min(1.0, len(similar_patterns) / 50.0)
        reliability = win_rate * sample_size_factor

        return PatternStats(
            total_occurrences=len(similar_patterns),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_profit_gbp=avg_profit,
            avg_hold_minutes=avg_hold,
            sharpe_ratio=sharpe,
            reliability_score=reliability,
        )

    def update_trade_exit(
        self,
        trade_id: int,
        exit_timestamp: datetime,
        exit_price: float,
        exit_reason: str,
        hold_duration_minutes: int,
        gross_profit_bps: float,
        net_profit_gbp: float,
        fees_gbp: float,
        is_winner: bool,
        win_quality: str,
    ) -> None:
        """Update a trade with exit information."""
        self.connect()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE trade_memory
                SET
                    exit_timestamp = %s,
                    exit_price = %s,
                    exit_reason = %s,
                    hold_duration_minutes = %s,
                    gross_profit_bps = %s,
                    net_profit_gbp = %s,
                    fees_gbp = %s,
                    is_winner = %s,
                    win_quality = %s
                WHERE trade_id = %s
                """,
                (
                    exit_timestamp,
                    exit_price,
                    exit_reason,
                    hold_duration_minutes,
                    gross_profit_bps,
                    net_profit_gbp,
                    fees_gbp,
                    is_winner,
                    win_quality,
                    trade_id,
                ),
            )
            self._conn.commit()
            logger.info("trade_exit_updated", trade_id=trade_id, is_winner=is_winner)

    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get recent performance metrics for a symbol."""
        self.connect()
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as wins,
                    AVG(net_profit_gbp) as avg_profit,
                    SUM(net_profit_gbp) as total_profit,
                    AVG(hold_duration_minutes) as avg_hold_minutes
                FROM trade_memory
                WHERE symbol = %s
                  AND entry_timestamp >= NOW() - INTERVAL '%s days'
                  AND is_winner IS NOT NULL
                """,
                (symbol, days),
            )
            row = cur.fetchone()

            total = row["total_trades"] or 0
            wins = row["wins"] or 0

            return {
                "symbol": symbol,
                "total_trades": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "avg_profit_gbp": float(row["avg_profit"] or 0.0),
                "total_profit_gbp": float(row["total_profit"] or 0.0),
                "avg_hold_minutes": int(row["avg_hold_minutes"] or 0),
            }
