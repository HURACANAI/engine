"""Memory store for trade patterns using vector similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
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

    def _ensure_connection(self) -> psycopg2.extensions.connection:
        """Return an active database connection or raise if unavailable."""
        self.connect()
        if self._conn is None or self._conn.closed:
            raise RuntimeError("Memory store database connection is not available")
        return self._conn

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
        conn = self._ensure_connection()
        with conn.cursor() as cur:
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
            returned = cur.fetchone()
            if returned is None:
                raise RuntimeError("Failed to retrieve inserted trade_id")
            trade_id = int(returned[0])
            conn.commit()
            logger.info("trade_stored", trade_id=trade_id, symbol=trade.symbol)
            return trade_id

    def find_similar_patterns(
        self,
        embedding: np.ndarray,
        symbol: Optional[str] = None,
        market_regime: Optional[str] = None,
        top_k: int = 20,
        min_similarity: float = 0.7,
        regime_weight: float = 0.3,
        use_regime_boost: bool = True,
    ) -> List[SimilarPattern]:
        """
        Find similar historical trades using context-aware vector similarity.

        Args:
            embedding: Feature embedding to search for
            symbol: Optional filter by symbol
            market_regime: Optional filter by market regime (for boost/filter)
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold (0-1)
            regime_weight: Weight for regime matching bonus (0-1, default 0.3)
            use_regime_boost: If True, boost scores for matching regimes instead of filtering

        Returns:
            List of similar patterns ordered by context-aware similarity

        Note:
            When use_regime_boost=True and market_regime is provided:
            - Patterns from same regime get similarity boosted by regime_weight
            - Patterns from different regimes still returned but with lower scores
            This prevents over-filtering while favoring contextually similar trades
        """
        conn = self._ensure_connection()

        where_clauses = ["1 - (entry_embedding <=> %s::vector) >= %s"]
        params: List[Any] = [embedding.tolist(), min_similarity]

        if symbol:
            where_clauses.append("symbol = %s")
            params.append(symbol)

        # Handle regime filtering vs boosting
        if market_regime and not use_regime_boost:
            # Traditional filtering: only return matching regimes
            where_clauses.append("market_regime = %s")
            params.append(market_regime)

        where_clause = " AND ".join(where_clauses)

        # Build regime-aware similarity calculation
        if use_regime_boost and market_regime:
            # Boost similarity for matching regimes
            # base_similarity * (1 + regime_weight) if match, else base_similarity
            similarity_calc = f"""
                CASE
                    WHEN market_regime = %s THEN
                        (1 - (entry_embedding <=> %s::vector)) * (1.0 + {regime_weight})
                    ELSE
                        (1 - (entry_embedding <=> %s::vector))
                END as context_similarity
            """
            query_params = [embedding.tolist(), min_similarity]
            if symbol:
                query_params.append(symbol)
            query_params.extend([market_regime, embedding.tolist(), embedding.tolist()])
            query_params.append(top_k * 2)  # Fetch more, then re-sort and limit

            order_clause = "context_similarity DESC"
        else:
            # Standard vector similarity
            similarity_calc = "1 - (entry_embedding <=> %s::vector) as context_similarity"
            query_params = [embedding.tolist()] + params
            query_params.append(top_k)
            order_clause = "entry_embedding <=> %s::vector"

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if use_regime_boost and market_regime:
                cur.execute(
                    f"""
                    SELECT
                        trade_id,
                        symbol,
                        entry_timestamp,
                        {similarity_calc},
                        is_winner,
                        net_profit_gbp,
                        hold_duration_minutes,
                        market_regime
                    FROM trade_memory
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT %s
                    """,
                    query_params,
                )
            else:
                cur.execute(
                    f"""
                    SELECT
                        trade_id,
                        symbol,
                        entry_timestamp,
                        {similarity_calc},
                        is_winner,
                        net_profit_gbp,
                        hold_duration_minutes,
                        market_regime
                    FROM trade_memory
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT %s
                    """,
                    query_params,
                )

            results = []
            rows = cur.fetchall()

            for row in rows:
                # Use context_similarity (regime-boosted) instead of similarity_score
                similarity = row.get("context_similarity", row.get("similarity_score", 0.0))

                results.append(
                    SimilarPattern(
                        trade_id=row["trade_id"],
                        symbol=row["symbol"],
                        entry_timestamp=row["entry_timestamp"],
                        similarity_score=similarity,  # Now context-aware!
                        is_winner=row["is_winner"],
                        net_profit_gbp=row["net_profit_gbp"],
                        hold_duration_minutes=row["hold_duration_minutes"],
                        market_regime=row["market_regime"],
                    )
                )

            # If we fetched extra for regime boosting, trim to top_k
            if use_regime_boost and market_regime and len(results) > top_k:
                results = results[:top_k]

            regime_matches = sum(1 for r in results if r.market_regime == market_regime) if market_regime else 0
            logger.info(
                "context_aware_patterns_found",
                count=len(results),
                top_k=top_k,
                regime_matches=regime_matches,
                regime_boost_enabled=use_regime_boost and market_regime is not None,
            )
            return results

    def sample_replay_experiences(self, limit: int = 256) -> List[Dict[str, Any]]:
        """Randomly sample historical trades for replay seeding."""

        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    symbol,
                    entry_embedding,
                    net_profit_gbp,
                    position_size_gbp,
                    hold_duration_minutes,
                    market_regime,
                    spread_at_entry_bps,
                    volatility_bps
                FROM trade_memory
                WHERE net_profit_gbp IS NOT NULL
                ORDER BY RANDOM()
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

        samples: List[Dict[str, Any]] = []
        for row in rows:
            embedding = np.array(row.get("entry_embedding", []), dtype=np.float32)
            samples.append({
                "symbol": row.get("symbol"),
                "entry_embedding": embedding,
                "net_profit_gbp": row.get("net_profit_gbp", 0.0),
                "position_size_gbp": row.get("position_size_gbp", 1_000.0),
                "hold_duration_minutes": row.get("hold_duration_minutes", 0),
                "market_regime": row.get("market_regime"),
                "spread_bps": row.get("spread_at_entry_bps", 5.0),
                "volatility_bps": row.get("volatility_bps", 0.0),
            })

        logger.info("replay_samples_loaded", count=len(samples))
        return samples

    def record_model_performance(self, model_version: str, evaluation_date: date, metrics: Dict[str, Any]) -> None:
        """Persist aggregate model metrics for cross-module consumption."""

        conn = self._ensure_connection()
        payload = {
            "model_version": model_version,
            "evaluation_date": evaluation_date,
            "trades_total": metrics.get("total_trades"),
            "trades_won": metrics.get("wins"),
            "trades_lost": metrics.get("losses"),
            "win_rate": metrics.get("win_rate"),
            "total_profit_gbp": metrics.get("total_profit_gbp"),
            "avg_profit_per_trade_gbp": metrics.get("avg_profit_per_trade_gbp"),
            "largest_win_gbp": metrics.get("largest_win_gbp"),
            "largest_loss_gbp": metrics.get("largest_loss_gbp"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "sortino_ratio": metrics.get("sortino_ratio"),
            "max_drawdown_gbp": metrics.get("max_drawdown_gbp"),
            "patterns_learned": metrics.get("patterns_learned"),
            "insights_generated": metrics.get("insights_generated"),
            "strategy_updates": metrics.get("strategy_updates"),
        }

        columns = ",".join(payload.keys())
        placeholders = ",".join(["%s"] * len(payload))
        updates = ",".join([f"{col} = EXCLUDED.{col}" for col in list(payload.keys())[2:]])

        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO model_performance ({columns})
                VALUES ({placeholders})
                ON CONFLICT(model_version, evaluation_date)
                DO UPDATE SET {updates}
                """,
                list(payload.values()),
            )
            conn.commit()
            logger.info("model_performance_recorded", model_version=model_version)

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

        profits = [float(p.net_profit_gbp) for p in similar_patterns]
        avg_profit = float(np.mean(profits)) if profits else 0.0
        avg_hold = int(np.mean([p.hold_duration_minutes for p in similar_patterns])) if similar_patterns else 0

        # Sharpe calculation (simplified)
        if len(profits) > 1:
            sharpe = float(np.mean(profits) / (np.std(profits) + 1e-6))
        else:
            sharpe = 0.0

        # Reliability: combination of win rate and sample size
        sample_size_factor = min(1.0, len(similar_patterns) / 50.0)
        reliability = float(win_rate * sample_size_factor)

        return PatternStats(
            total_occurrences=len(similar_patterns),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_profit_gbp=float(avg_profit),
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
        conn = self._ensure_connection()
        with conn.cursor() as cur:
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
            conn.commit()
            logger.info("trade_exit_updated", trade_id=trade_id, is_winner=is_winner)

    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """Get recent performance metrics for a symbol."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
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
            row = cur.fetchone() or {}

            total = int(row.get("total_trades") or 0)
            wins = int(row.get("wins") or 0)

            return {
                "symbol": symbol,
                "total_trades": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "avg_profit_gbp": float(row.get("avg_profit") or 0.0),
                "total_profit_gbp": float(row.get("total_profit") or 0.0),
                "avg_hold_minutes": int(row.get("avg_hold_minutes") or 0),
            }

    @staticmethod
    def get_regime_similarity_threshold(market_regime: Optional[str]) -> float:
        """
        Get dynamic similarity threshold based on market regime.

        In unstable regimes (panic), require higher similarity for confidence.
        In stable regimes (trend), can be less strict.

        Args:
            market_regime: Current market regime

        Returns:
            Minimum similarity threshold (0-1)

        Regime-specific thresholds:
        - TREND: 0.65 (can be less strict - patterns are reliable)
        - RANGE: 0.70 (moderate - mean reversion requires precision)
        - PANIC: 0.80 (very strict - only take high-conviction trades)
        - UNKNOWN: 0.75 (conservative default)
        """
        if not market_regime:
            return 0.70  # Default moderate threshold

        regime = market_regime.lower()

        regime_thresholds = {
            "trend": 0.65,    # Less strict - trending markets are more predictable
            "trending": 0.65,
            "range": 0.70,    # Moderate - range trading requires decent similarity
            "ranging": 0.70,
            "panic": 0.80,    # Very strict - high volatility = need high confidence
            "high_vol": 0.80,
            "unknown": 0.75,  # Conservative when regime unclear
            "low_vol": 0.68,
            "medium_vol": 0.70,
        }

        return regime_thresholds.get(regime, 0.70)  # Default to moderate
