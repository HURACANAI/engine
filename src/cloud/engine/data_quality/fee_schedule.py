"""
Historical Fee Schedule Manager

Tracks exchange fee changes over time. Critical for accurate cost modeling
because fees change frequently (e.g., Binance: 10 bps → 8 bps in Jan 2023).

Without historical fee tracking, your backtests will use wrong costs and
produce unrealistic P&L estimates.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FeeSchedule:
    """Single fee schedule entry."""

    start_date: datetime
    end_date: Optional[datetime]  # None = ongoing
    maker_fee_bps: float
    taker_fee_bps: float
    exchange: str
    notes: str = ""


class HistoricalFeeManager:
    """
    Manages historical fee schedules for exchanges.

    Usage:
        fee_mgr = HistoricalFeeManager()
        fee_bps = fee_mgr.get_fee_for_date(
            exchange='binance',
            date=datetime(2023, 6, 15),
            is_maker=False
        )
    """

    def __init__(self):
        self.schedules: Dict[str, List[FeeSchedule]] = self._load_default_schedules()

        logger.info(
            "historical_fee_manager_initialized",
            exchanges=list(self.schedules.keys()),
            total_schedules=sum(len(s) for s in self.schedules.values())
        )

    def _load_default_schedules(self) -> Dict[str, List[FeeSchedule]]:
        """
        Load known historical fee schedules.

        Sources:
        - Binance: https://www.binance.com/en/fee/schedule
        - Coinbase: https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees
        """
        schedules = {}

        # Binance fee history (all dates in UTC timezone)
        schedules['binance'] = [
            FeeSchedule(
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                maker_fee_bps=10.0,
                taker_fee_bps=10.0,
                exchange='binance',
                notes="Old standard tier"
            ),
            FeeSchedule(
                start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
                maker_fee_bps=2.0,   # Maker rebate
                taker_fee_bps=8.0,
                exchange='binance',
                notes="New fee structure with maker rebates"
            ),
            FeeSchedule(
                start_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
                end_date=None,  # Ongoing
                maker_fee_bps=2.0,
                taker_fee_bps=6.0,  # Reduced taker fee
                exchange='binance',
                notes="Current tier (as of 2024)"
            ),
        ]

        # Coinbase Pro fee history (all dates in UTC timezone)
        schedules['coinbase'] = [
            FeeSchedule(
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end_date=None,
                maker_fee_bps=5.0,
                taker_fee_bps=5.0,
                exchange='coinbase',
                notes="Standard tier (< $10k volume)"
            ),
        ]

        # Kraken fee history (all dates in UTC timezone)
        schedules['kraken'] = [
            FeeSchedule(
                start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end_date=None,
                maker_fee_bps=4.0,
                taker_fee_bps=10.0,
                exchange='kraken',
                notes="Standard tier"
            ),
        ]

        return schedules

    def get_fee_for_date(
        self,
        exchange: str,
        date: datetime | int,
        is_maker: bool = False
    ) -> float:
        """
        Get fee (in bps) for a specific date.

        Args:
            exchange: Exchange name ('binance', 'coinbase', 'kraken')
            date: Date to query (datetime or timestamp in milliseconds)
            is_maker: True for maker orders, False for taker

        Returns:
            Fee in basis points
        """
        exchange = exchange.lower()
        
        # Convert timestamp (int) to datetime if needed
        if isinstance(date, int):
            # Assume milliseconds timestamp
            from datetime import timezone
            date = datetime.fromtimestamp(date / 1000, tz=timezone.utc)

        if exchange not in self.schedules:
            logger.warning(
                "unknown_exchange_using_default",
                exchange=exchange,
                default_fee_bps=8.0
            )
            return 8.0  # Conservative default

        # Ensure date is timezone-aware UTC
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        
        # Find applicable schedule
        for schedule in self.schedules[exchange]:
            # Ensure schedule dates are timezone-aware UTC
            if schedule.start_date.tzinfo is None:
                schedule.start_date = schedule.start_date.replace(tzinfo=timezone.utc)
            if schedule.end_date and schedule.end_date.tzinfo is None:
                schedule.end_date = schedule.end_date.replace(tzinfo=timezone.utc)
            
            if schedule.start_date <= date:
                if schedule.end_date is None or date < schedule.end_date:
                    fee = schedule.maker_fee_bps if is_maker else schedule.taker_fee_bps

                    logger.debug(
                        "fee_retrieved",
                        exchange=exchange,
                        date=date.date(),
                        is_maker=is_maker,
                        fee_bps=fee
                    )

                    return fee

        # Fallback
        logger.warning(
            "no_schedule_found_using_default",
            exchange=exchange,
            date=date.date()
        )
        return 8.0

    def apply_fees_to_dataframe(
        self,
        df: pl.DataFrame,
        exchange: str,
        is_maker_column: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Add fee columns to a dataframe.

        Args:
            df: DataFrame with 'timestamp' column
            exchange: Exchange name
            is_maker_column: Optional column name for maker/taker flag

        Returns:
            DataFrame with 'fee_bps' column added
        """

        # Convert to Python datetime if needed
        timestamps = df['timestamp'].to_list()

        if is_maker_column and is_maker_column in df.columns:
            is_maker_list = df[is_maker_column].to_list()
        else:
            # Default: assume taker orders (more conservative)
            is_maker_list = [False] * len(timestamps)

        # Get fees for each row
        fees = [
            self.get_fee_for_date(exchange, ts, is_maker)
            for ts, is_maker in zip(timestamps, is_maker_list)
        ]

        # Add to dataframe
        df = df.with_columns([
            pl.Series('fee_bps', fees)
        ])

        logger.info(
            "fees_applied_to_dataframe",
            rows=len(df),
            exchange=exchange,
            avg_fee_bps=sum(fees) / len(fees) if fees else 0
        )

        return df

    def add_schedule(self, schedule: FeeSchedule) -> None:
        """
        Add a custom fee schedule.

        Useful for:
        - VIP tiers with special rates
        - New exchanges not in defaults
        - Testing different fee scenarios
        """
        exchange = schedule.exchange.lower()

        if exchange not in self.schedules:
            self.schedules[exchange] = []

        self.schedules[exchange].append(schedule)

        # Sort by start_date
        self.schedules[exchange].sort(key=lambda s: s.start_date)

        logger.info(
            "custom_schedule_added",
            exchange=exchange,
            start_date=schedule.start_date.date(),
            maker_bps=schedule.maker_fee_bps,
            taker_bps=schedule.taker_fee_bps
        )

    def get_schedule_changes(
        self,
        exchange: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, float, float]]:
        """
        Get all fee changes in a date range.

        Useful for understanding if fee changes coincide with
        performance changes in your models.

        Returns:
            List of (date, maker_fee_bps, taker_fee_bps) tuples
        """
        exchange = exchange.lower()

        if exchange not in self.schedules:
            return []

        changes = []

        for schedule in self.schedules[exchange]:
            if schedule.start_date >= start_date and schedule.start_date <= end_date:
                changes.append((
                    schedule.start_date,
                    schedule.maker_fee_bps,
                    schedule.taker_fee_bps
                ))

        return changes
