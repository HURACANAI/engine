"""Tracks price action after trade exit to learn optimal holding periods."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import polars as pl
import psycopg2
from psycopg2.extras import RealDictCursor
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PostExitInsights:
    """Insights from tracking price after exit."""
    trade_id: int
    price_1h_later: float
    price_4h_later: float
    price_24h_later: float
    max_price_reached: float
    max_price_time_minutes: int
    min_price_reached: float
    min_price_time_minutes: int
    missed_profit_bps: float
    optimal_exit_time_minutes: int
    should_have_held_longer: bool
    should_have_exited_earlier: bool
    insight_summary: str


class PostExitTracker:
    """
    Monitors price movement after trade exits to understand if we exited too early/late.

    This is CRITICAL for learning optimal holding periods - one of your key requirements.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)

    def _ensure_connection(self) -> psycopg2.extensions.connection:
        """Return an open connection, raising if unavailable."""
        self.connect()
        if self._conn is None or self._conn.closed:
            raise RuntimeError("Database connection is not available")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def track_post_exit(
        self,
        trade_id: int,
        exit_timestamp: datetime,
        exit_price: float,
        direction: str,
        future_prices: pl.DataFrame,  # DataFrame with columns: ts, close, high, low
        position_size_gbp: float,
    ) -> PostExitInsights:
        """
        Track price movement after exit to determine if we should have held longer.

        Args:
            trade_id: Trade identifier
            exit_timestamp: When we exited
            exit_price: Price at exit
            direction: 'LONG' or 'SHORT'
            future_prices: Price data AFTER exit (1-24 hours)
            position_size_gbp: Position size for profit calculations

        Returns:
            PostExitInsights with learning data
        """
        logger.info("tracking_post_exit", trade_id=trade_id, exit_price=exit_price)

        if future_prices.height == 0:
            logger.warning("no_future_data", trade_id=trade_id)
            return self._create_empty_insights(trade_id)

        # Sort by time
        future_prices = future_prices.sort("ts")

        # Find prices at specific time intervals
        price_1h = self._get_price_at_offset(future_prices, exit_timestamp, minutes=60)
        price_4h = self._get_price_at_offset(future_prices, exit_timestamp, minutes=240)
        price_24h = self._get_price_at_offset(future_prices, exit_timestamp, minutes=1440)

        # Find max/min prices reached
        max_info = self._find_extreme_price(future_prices, exit_timestamp, "high", direction)
        min_info = self._find_extreme_price(future_prices, exit_timestamp, "low", direction)

        # Calculate missed opportunity
        if direction == "LONG":
            # For longs, we care about how much higher it went
            optimal_price = max_info["price"]
            missed_profit_bps = ((optimal_price - exit_price) / exit_price) * 10000
        else:
            # For shorts, we care about how much lower it went
            optimal_price = min_info["price"]
            missed_profit_bps = ((exit_price - optimal_price) / exit_price) * 10000

        missed_profit_bps = max(0, missed_profit_bps)  # Can't be negative
        optimal_exit_minutes = max_info["minutes_after"] if direction == "LONG" else min_info["minutes_after"]

        # Determine if we should have held longer
        should_have_held_longer = self._should_have_held_longer(
            missed_profit_bps=missed_profit_bps,
            optimal_exit_minutes=optimal_exit_minutes,
        )

        # Determine if we should have exited earlier
        should_have_exited_earlier = self._should_have_exited_earlier(
            current_price=price_1h or exit_price,
            exit_price=exit_price,
            direction=direction,
        )

        # Generate insights
        insight_summary = self._generate_insight_summary(
            exit_price=exit_price,
            optimal_price=optimal_price,
            missed_profit_bps=missed_profit_bps,
            optimal_exit_minutes=optimal_exit_minutes,
            should_have_held_longer=should_have_held_longer,
            should_have_exited_earlier=should_have_exited_earlier,
        )

        insights = PostExitInsights(
            trade_id=trade_id,
            price_1h_later=price_1h or exit_price,
            price_4h_later=price_4h or exit_price,
            price_24h_later=price_24h or exit_price,
            max_price_reached=max_info["price"],
            max_price_time_minutes=max_info["minutes_after"],
            min_price_reached=min_info["price"],
            min_price_time_minutes=min_info["minutes_after"],
            missed_profit_bps=missed_profit_bps,
            optimal_exit_time_minutes=optimal_exit_minutes,
            should_have_held_longer=should_have_held_longer,
            should_have_exited_earlier=should_have_exited_earlier,
            insight_summary=insight_summary,
        )

        # Store in database
        self._store_tracking(insights)

        logger.info(
            "post_exit_tracked",
            trade_id=trade_id,
            missed_profit_bps=missed_profit_bps,
            should_have_held_longer=should_have_held_longer,
        )

        return insights

    def _get_price_at_offset(
        self,
        prices: pl.DataFrame,
        exit_timestamp: datetime,
        minutes: int,
    ) -> Optional[float]:
        """Get price at specific time offset after exit."""
        target_time = exit_timestamp + timedelta(minutes=minutes)

        # Find closest timestamp
        filtered = prices.filter(pl.col("ts") >= target_time)

        if filtered.height > 0:
            return float(filtered.row(0, named=True)["close"])

        return None

    def _find_extreme_price(
        self,
        prices: pl.DataFrame,
        exit_timestamp: datetime,
        column: str,  # 'high' or 'low'
        direction: str,
    ) -> dict:
        """Find maximum or minimum price reached after exit."""
        if prices.height == 0:
            return {"price": 0.0, "minutes_after": 0}

        # Find extreme
        if column == "high":
            extreme_row = prices.sort(column, descending=True).row(0, named=True)
        else:
            extreme_row = prices.sort(column, descending=False).row(0, named=True)

        extreme_price = float(extreme_row[column])
        extreme_time = extreme_row["ts"]

        minutes_after = int((extreme_time - exit_timestamp).total_seconds() / 60)

        return {
            "price": extreme_price,
            "minutes_after": max(0, minutes_after),
        }

    def _should_have_held_longer(
        self,
        missed_profit_bps: float,
        optimal_exit_minutes: int,
    ) -> bool:
        """Determine if we exited too early."""
        # If we missed >5 bps and optimal exit was within reasonable time
        if missed_profit_bps > 5.0 and optimal_exit_minutes < 240:  # Within 4 hours
            return True

        return False

    def _should_have_exited_earlier(
        self,
        current_price: float,
        exit_price: float,
        direction: str,
    ) -> bool:
        """Determine if we held too long."""
        if direction == "LONG":
            # If price dropped significantly after our exit
            drop_bps = ((exit_price - current_price) / exit_price) * 10000
            return drop_bps > 10.0  # Dropped more than 10 bps
        else:
            # For shorts
            rise_bps = ((current_price - exit_price) / exit_price) * 10000
            return rise_bps > 10.0

    def _generate_insight_summary(
        self,
        exit_price: float,
        optimal_price: float,
        missed_profit_bps: float,
        optimal_exit_minutes: int,
        should_have_held_longer: bool,
        should_have_exited_earlier: bool,
    ) -> str:
        """Generate human-readable insight."""
        if should_have_held_longer:
            return (
                f"Exited too early! Could have made {missed_profit_bps:.1f} more bps "
                f"by holding {optimal_exit_minutes} minutes longer "
                f"(exit at {optimal_price:.6f} vs {exit_price:.6f})."
            )

        if should_have_exited_earlier:
            return "Held too long - price moved against us after exit."

        if missed_profit_bps < 2.0:
            return "Exit timing was excellent - minimal missed opportunity."

        return f"Acceptable exit - missed {missed_profit_bps:.1f} bps but within tolerance."

    def _create_empty_insights(self, trade_id: int) -> PostExitInsights:
        """Create empty insights when no data available."""
        return PostExitInsights(
            trade_id=trade_id,
            price_1h_later=0.0,
            price_4h_later=0.0,
            price_24h_later=0.0,
            max_price_reached=0.0,
            max_price_time_minutes=0,
            min_price_reached=0.0,
            min_price_time_minutes=0,
            missed_profit_bps=0.0,
            optimal_exit_time_minutes=0,
            should_have_held_longer=False,
            should_have_exited_earlier=False,
            insight_summary="No future data available for analysis.",
        )

    def _store_tracking(self, insights: PostExitInsights) -> None:
        """Store post-exit tracking in database."""
        conn = self._ensure_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO post_exit_tracking (
                    trade_id, price_1h_later, price_4h_later, price_24h_later,
                    max_price_reached, max_price_time_minutes,
                    min_price_reached, min_price_time_minutes,
                    missed_profit_bps, optimal_exit_time_minutes,
                    should_have_held_longer, should_have_exited_earlier,
                    insight_summary
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    insights.trade_id,
                    insights.price_1h_later,
                    insights.price_4h_later,
                    insights.price_24h_later,
                    insights.max_price_reached,
                    insights.max_price_time_minutes,
                    insights.min_price_reached,
                    insights.min_price_time_minutes,
                    insights.missed_profit_bps,
                    insights.optimal_exit_time_minutes,
                    insights.should_have_held_longer,
                    insights.should_have_exited_earlier,
                    insights.insight_summary,
                ),
            )
            conn.commit()

    def get_exit_learning_stats(self, symbol: str, days: int = 30) -> dict:
        """Get statistics on exit timing performance."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total_exits,
                    AVG(missed_profit_bps) as avg_missed_profit_bps,
                    SUM(CASE WHEN should_have_held_longer THEN 1 ELSE 0 END) as early_exits,
                    SUM(CASE WHEN should_have_exited_earlier THEN 1 ELSE 0 END) as late_exits,
                    AVG(optimal_exit_time_minutes) as avg_optimal_hold_minutes
                FROM post_exit_tracking pet
                JOIN trade_memory tm ON pet.trade_id = tm.trade_id
                WHERE tm.symbol = %s
                  AND tm.entry_timestamp >= NOW() - INTERVAL '%s days'
                """,
                (symbol, days),
            )

            row: Optional[Dict[str, Any]] = cur.fetchone()
            if row is None:
                return {
                    "symbol": symbol,
                    "total_exits": 0,
                    "avg_missed_profit_bps": 0.0,
                    "early_exits": 0,
                    "late_exits": 0,
                    "avg_optimal_hold_minutes": 0,
                    "exit_timing_accuracy": 1.0,
                }

            return {
                "symbol": symbol,
                "total_exits": row["total_exits"] or 0,
                "avg_missed_profit_bps": float(row["avg_missed_profit_bps"] or 0.0),
                "early_exits": row["early_exits"] or 0,
                "late_exits": row["late_exits"] or 0,
                "avg_optimal_hold_minutes": int(row["avg_optimal_hold_minutes"] or 0),
                "exit_timing_accuracy": 1.0 - (
                    (row["early_exits"] + row["late_exits"]) / row["total_exits"]
                    if row["total_exits"] > 0 else 0.0
                ),
            }
