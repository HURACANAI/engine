"""Pattern matching service for finding similar historical setups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import structlog

from ..memory.store import MemoryStore, SimilarPattern

logger = structlog.get_logger(__name__)


@dataclass
class PatternSignature:
    """Signature representing a market pattern."""
    pattern_id: Optional[int]
    pattern_name: str
    feature_signature: Dict[str, Any]
    market_regime: str
    embedding: np.ndarray

    # Performance metrics
    total_occurrences: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_profit_bps: float = 0.0
    avg_hold_minutes: int = 0

    # Statistical metrics
    sharpe_ratio: float = 0.0
    max_drawdown_bps: float = 0.0
    profit_factor: float = 0.0

    # Confidence
    reliability_score: float = 0.0
    sample_size_adequate: bool = False

    # Learned optimal parameters
    optimal_position_size_multiplier: float = 1.0
    optimal_exit_threshold_bps: float = 15.0


class PatternMatcher:
    """
    Identifies, clusters, and manages recurring market patterns.

    This service learns which patterns work and which don't, storing them
    for quick lookup during live trading.
    """

    def __init__(self, dsn: str, memory_store: MemoryStore):
        self._dsn = dsn
        self._memory = memory_store
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def find_matching_pattern(
        self,
        features: Dict[str, Any],
        embedding: np.ndarray,
        market_regime: str,
        similarity_threshold: float = 0.75,
    ) -> Optional[PatternSignature]:
        """
        Find if current setup matches any known pattern.

        Args:
            features: Current market features
            embedding: Feature embedding vector
            market_regime: Current market regime
            similarity_threshold: Minimum similarity to match

        Returns:
            PatternSignature if match found, None otherwise
        """
        self.connect()

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    pattern_id,
                    pattern_name,
                    feature_signature,
                    market_regime,
                    pattern_embedding,
                    total_occurrences,
                    wins,
                    losses,
                    win_rate,
                    avg_profit_bps,
                    avg_hold_minutes,
                    sharpe_ratio,
                    max_drawdown_bps,
                    profit_factor,
                    reliability_score,
                    sample_size_adequate,
                    optimal_position_size_multiplier,
                    optimal_exit_threshold_bps,
                    1 - (pattern_embedding <=> %s::vector) as similarity
                FROM pattern_library
                WHERE market_regime = %s
                  AND 1 - (pattern_embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT 1
                """,
                (embedding.tolist(), market_regime, embedding.tolist(), similarity_threshold),
            )

            row = cur.fetchone()

            if not row:
                return None

            return PatternSignature(
                pattern_id=row["pattern_id"],
                pattern_name=row["pattern_name"],
                feature_signature=row["feature_signature"],
                market_regime=row["market_regime"],
                embedding=np.array(row["pattern_embedding"], dtype=np.float32),
                total_occurrences=row["total_occurrences"],
                wins=row["wins"],
                losses=row["losses"],
                win_rate=float(row["win_rate"]),
                avg_profit_bps=float(row["avg_profit_bps"]),
                avg_hold_minutes=row["avg_hold_minutes"],
                sharpe_ratio=float(row["sharpe_ratio"]),
                max_drawdown_bps=float(row["max_drawdown_bps"]),
                profit_factor=float(row["profit_factor"]),
                reliability_score=float(row["reliability_score"]),
                sample_size_adequate=row["sample_size_adequate"],
                optimal_position_size_multiplier=float(row["optimal_position_size_multiplier"]),
                optimal_exit_threshold_bps=float(row["optimal_exit_threshold_bps"]),
            )

    def create_pattern(
        self,
        pattern_name: str,
        features: Dict[str, Any],
        embedding: np.ndarray,
        market_regime: str,
    ) -> int:
        """Create a new pattern in the library."""
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pattern_library (
                    pattern_name, feature_signature, pattern_embedding, market_regime,
                    total_occurrences, wins, losses, win_rate,
                    optimal_position_size_multiplier, optimal_exit_threshold_bps
                )
                VALUES (%s, %s, %s, %s, 0, 0, 0, 0.0, 1.0, 15.0)
                RETURNING pattern_id
                """,
                (
                    pattern_name,
                    Json(features),
                    embedding.tolist(),
                    market_regime,
                ),
            )

            pattern_id = cur.fetchone()[0]
            self._conn.commit()

            logger.info("pattern_created", pattern_id=pattern_id, name=pattern_name)
            return pattern_id

    def update_pattern_performance(
        self,
        pattern_id: int,
        trade_won: bool,
        profit_bps: float,
        hold_minutes: int,
    ) -> None:
        """Update pattern statistics with new trade result."""
        self.connect()

        with self._conn.cursor() as cur:
            # Increment counters
            cur.execute(
                """
                UPDATE pattern_library
                SET
                    total_occurrences = total_occurrences + 1,
                    wins = wins + CASE WHEN %s THEN 1 ELSE 0 END,
                    losses = losses + CASE WHEN %s THEN 0 ELSE 1 END,
                    last_updated = NOW()
                WHERE pattern_id = %s
                """,
                (trade_won, trade_won, pattern_id),
            )

            # Recalculate metrics
            cur.execute(
                """
                UPDATE pattern_library
                SET
                    win_rate = CAST(wins AS FLOAT) / NULLIF(total_occurrences, 0),
                    sample_size_adequate = total_occurrences >= 30
                WHERE pattern_id = %s
                """,
                (pattern_id,),
            )

            self._conn.commit()

            logger.info(
                "pattern_updated",
                pattern_id=pattern_id,
                won=trade_won,
                profit_bps=profit_bps,
            )

    def get_top_patterns(self, min_win_rate: float = 0.55, min_sample_size: int = 20) -> List[PatternSignature]:
        """Get top performing patterns."""
        self.connect()

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    pattern_id, pattern_name, feature_signature, market_regime,
                    pattern_embedding, total_occurrences, wins, losses, win_rate,
                    avg_profit_bps, avg_hold_minutes, sharpe_ratio, max_drawdown_bps,
                    profit_factor, reliability_score, sample_size_adequate,
                    optimal_position_size_multiplier, optimal_exit_threshold_bps
                FROM pattern_library
                WHERE win_rate >= %s
                  AND total_occurrences >= %s
                ORDER BY reliability_score DESC, win_rate DESC
                LIMIT 50
                """,
                (min_win_rate, min_sample_size),
            )

            patterns = []
            for row in cur.fetchall():
                patterns.append(
                    PatternSignature(
                        pattern_id=row["pattern_id"],
                        pattern_name=row["pattern_name"],
                        feature_signature=row["feature_signature"],
                        market_regime=row["market_regime"],
                        embedding=np.array(row["pattern_embedding"], dtype=np.float32),
                        total_occurrences=row["total_occurrences"],
                        wins=row["wins"],
                        losses=row["losses"],
                        win_rate=float(row["win_rate"]),
                        avg_profit_bps=float(row["avg_profit_bps"]),
                        avg_hold_minutes=row["avg_hold_minutes"],
                        sharpe_ratio=float(row["sharpe_ratio"]),
                        max_drawdown_bps=float(row["max_drawdown_bps"]),
                        profit_factor=float(row["profit_factor"]),
                        reliability_score=float(row["reliability_score"]),
                        sample_size_adequate=row["sample_size_adequate"],
                        optimal_position_size_multiplier=float(row["optimal_position_size_multiplier"]),
                        optimal_exit_threshold_bps=float(row["optimal_exit_threshold_bps"]),
                    )
                )

            logger.info("top_patterns_retrieved", count=len(patterns))
            return patterns

    def blacklist_pattern(self, pattern_id: int, reason: str) -> None:
        """Blacklist a pattern (set reliability to 0)."""
        self.connect()

        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pattern_library
                SET reliability_score = 0.0
                WHERE pattern_id = %s
                """,
                (pattern_id,),
            )
            self._conn.commit()

            logger.warning("pattern_blacklisted", pattern_id=pattern_id, reason=reason)

    def learn_optimal_parameters(self, pattern_id: int) -> None:
        """
        Learn optimal position size and exit threshold for a pattern
        by analyzing historical trades.
        """
        self.connect()

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all trades for this pattern (simplified - would need pattern-trade mapping)
            # For now, just use average of winning trades
            cur.execute(
                """
                SELECT
                    AVG(CASE WHEN is_winner THEN position_size_gbp / 1000.0 ELSE NULL END) as avg_win_size,
                    AVG(CASE WHEN is_winner THEN gross_profit_bps ELSE NULL END) as avg_win_profit
                FROM trade_memory
                WHERE is_winner = TRUE
                  AND trade_id IN (
                      SELECT trade_id
                      FROM trade_memory
                      ORDER BY entry_embedding <=> (
                          SELECT pattern_embedding
                          FROM pattern_library
                          WHERE pattern_id = %s
                      )
                      LIMIT 50
                  )
                """,
                (pattern_id,),
            )

            row = cur.fetchone()

            if row and row["avg_win_size"]:
                optimal_size = float(row["avg_win_size"])
                optimal_exit = float(row["avg_win_profit"]) * 0.9  # 90% of average win

                cur.execute(
                    """
                    UPDATE pattern_library
                    SET
                        optimal_position_size_multiplier = %s,
                        optimal_exit_threshold_bps = %s
                    WHERE pattern_id = %s
                    """,
                    (optimal_size, optimal_exit, pattern_id),
                )

                self._conn.commit()

                logger.info(
                    "optimal_parameters_learned",
                    pattern_id=pattern_id,
                    size=optimal_size,
                    exit=optimal_exit,
                )
