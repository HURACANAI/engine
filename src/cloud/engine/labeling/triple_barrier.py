"""
Triple-Barrier Labeling System

The CORRECT way to label training data for trading models.

Problem with naive labeling:
- "Did price go up in next N candles?" → Lookahead bias
- Ignores transaction costs
- Doesn't account for stop-loss

Triple-Barrier Method (de Prado 2018):
For each potential entry, set 3 barriers:
1. Upper barrier (take-profit)
2. Lower barrier (stop-loss)
3. Time barrier (timeout)

Find which barrier hits FIRST → that's your label.
Then subtract costs → that's your META-LABEL.

This prevents lookahead because you're simulating what would actually happen.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

from .label_schemas import (
    ExitReason,
    LabelConfig,
    LabeledTrade,
    RunnerLabelConfig,
    ScalpLabelConfig
)

# Type hint for CostEstimator (forward reference to avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..costs import CostEstimator

logger = structlog.get_logger(__name__)


class TripleBarrierLabeler:
    """
    Label trades using triple-barrier method.

    Usage:
        labeler = TripleBarrierLabeler(mode='scalp')
        labels = labeler.label_dataframe(candles)

        # Get winners only
        winners = [l for l in labels if l.is_winner()]
    """

    def __init__(
        self,
        config: LabelConfig,
        cost_estimator: Optional['CostEstimator'] = None
    ):
        """
        Initialize triple-barrier labeler.

        Args:
            config: Label configuration (ScalpLabelConfig or RunnerLabelConfig)
            cost_estimator: Optional cost estimator (defaults to simple model)
        """
        self.config = config
        self.cost_estimator = cost_estimator

        logger.info(
            "triple_barrier_labeler_initialized",
            mode=config.mode_name,
            tp_bps=config.tp_bps,
            sl_bps=config.sl_bps,
            timeout_min=config.timeout_minutes
        )

    def label_dataframe(
        self,
        df: pl.DataFrame,
        symbol: str,
        max_labels: Optional[int] = None
    ) -> List[LabeledTrade]:
        """
        Label all potential entry points in a dataframe.

        Args:
            df: Candle data (must be sorted by timestamp)
            symbol: Trading symbol
            max_labels: Optional limit on number of labels (for testing)

        Returns:
            List of LabeledTrade objects
        """
        if len(df) < 10:
            logger.warning("insufficient_data_for_labeling", rows=len(df))
            return []

        # Ensure sorted
        df = df.sort('timestamp')

        labeled_trades = []

        # Label every candle as potential entry
        # (in practice, you'd filter by signal strength)
        max_idx = len(df) - 1
        last_timestamp = self._coerce_timestamp(df.row(max_idx, named=True)['timestamp'])

        for entry_idx in range(len(df) - 1):  # Leave room for exit
            if max_labels and len(labeled_trades) >= max_labels:
                break

            # Get entry candle - use proper Polars row access
            entry_row = df.row(entry_idx, named=True)
            entry_time = self._coerce_timestamp(entry_row['timestamp'])

            entry_price = entry_row['close']

            # Calculate barriers
            tp_price, sl_price, timeout_time = self._calculate_barriers(
                entry_price, entry_time, last_timestamp
            )

            # Find which barrier hits first
            exit_info = self._find_exit(
                df=df,
                entry_idx=entry_idx,
                tp_price=tp_price,
                sl_price=sl_price,
                timeout_time=timeout_time
            )

            if exit_info is None:
                # Reached end of data without hitting any barrier
                continue

            exit_idx, exit_time, exit_price, exit_reason = exit_info

            # Calculate P&L (gross, before costs)
            pnl_gross_bps = ((exit_price / entry_price) - 1) * 10000

            # Estimate costs
            costs_bps = self._estimate_costs(
                entry_row=entry_row,
                exit_time=exit_time,
                duration_minutes=(exit_time - entry_time).total_seconds() / 60
            )

            # Net P&L
            pnl_net_bps = pnl_gross_bps - costs_bps

            # Meta-label: 1 if profitable after costs
            meta_label = 1 if pnl_net_bps > 0 else 0

            # Create labeled trade
            labeled_trade = LabeledTrade(
                entry_time=entry_time,
                entry_price=entry_price,
                entry_idx=entry_idx,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                duration_minutes=(exit_time - entry_time).total_seconds() / 60,
                pnl_gross_bps=pnl_gross_bps,
                costs_bps=costs_bps,
                pnl_net_bps=pnl_net_bps,
                meta_label=meta_label,
                symbol=symbol,
                mode=self.config.mode_name
            )

            labeled_trades.append(labeled_trade)

        total = len(labeled_trades)
        winners_net = sum(1 for l in labeled_trades if l.is_winner())
        losers_net = total - winners_net
        winners_gross = sum(1 for l in labeled_trades if l.pnl_gross_bps > 0)
        losers_gross = total - winners_gross

        sample = [
            {
                "pnl_gross_bps": l.pnl_gross_bps,
                "costs_bps": l.costs_bps,
                "pnl_net_bps": l.pnl_net_bps,
                "meta_label": l.meta_label,
                "exit_reason": l.exit_reason.value,
            }
            for l in labeled_trades[:5]
        ]

        logger.info(
            "labeling_complete",
            symbol=symbol,
            total_labels=total,
            winners_net=winners_net,
            losers_net=losers_net,
            win_rate_net=winners_net / total if total else 0.0,
            winners_gross=winners_gross,
            losers_gross=losers_gross,
            win_rate_gross=winners_gross / total if total else 0.0,
            sample_labels=sample,
        )

        return labeled_trades

    def _calculate_barriers(
        self,
        entry_price: float,
        entry_time: datetime,
        last_timestamp: datetime,
    ) -> Tuple[float, float, datetime]:
        """
        Calculate TP, SL, and timeout barriers.

        Returns:
            (tp_price, sl_price, timeout_time)
        """
        # Take-profit barrier
        tp_price = entry_price * (1 + self.config.tp_bps / 10000)

        # Stop-loss barrier
        sl_price = entry_price * (1 - self.config.sl_bps / 10000)

        # Timeout barrier
        try:
            timeout_delta = timedelta(minutes=self.config.timeout_minutes)
            timeout_time = entry_time + timeout_delta
        except OverflowError:
            logger.warning(
                "triple_barrier_timeout_overflow",
                entry_time=entry_time.isoformat(),
                timeout_minutes=self.config.timeout_minutes,
            )
            timeout_time = last_timestamp

        if timeout_time > last_timestamp:
            timeout_time = last_timestamp
            logger.debug(
                "triple_barrier_timeout_clamped",
                entry_time=entry_time.isoformat(),
                timeout_time=timeout_time.isoformat(),
                last_timestamp=last_timestamp.isoformat(),
            )

        return tp_price, sl_price, timeout_time

    def _find_exit(
        self,
        df: pl.DataFrame,
        entry_idx: int,
        tp_price: float,
        sl_price: float,
        timeout_time: datetime
    ) -> Optional[Tuple[int, datetime, float, ExitReason]]:
        """
        Find which barrier hits first.

        Returns:
            (exit_idx, exit_time, exit_price, exit_reason) or None
        """
        # Look forward from entry
        for future_idx in range(entry_idx + 1, len(df)):
            future_row = df.row(future_idx, named=True)
            future_time = self._coerce_timestamp(future_row['timestamp'])

            future_high = future_row['high']
            future_low = future_row['low']
            future_close = future_row['close']

            # Check timeout FIRST (takes precedence)
            if future_time >= timeout_time:
                return (
                    future_idx,
                    timeout_time,
                    future_close,  # Exit at close
                    ExitReason.TIMEOUT
                )

            # Check take-profit
            if future_high >= tp_price:
                return (
                    future_idx,
                    future_time,
                    tp_price,  # Assume filled at TP
                    ExitReason.TAKE_PROFIT
                )

            # Check stop-loss
            if future_low <= sl_price:
                return (
                    future_idx,
                    future_time,
                    sl_price,  # Assume filled at SL
                    ExitReason.STOP_LOSS
                )

        # Reached end of data without hitting any barrier
        return None

    def _estimate_costs(
        self,
        entry_row,
        exit_time: datetime,
        duration_minutes: float
    ) -> float:
        """
        Estimate transaction costs for this trade.

        Components:
        1. Entry fee (maker or taker)
        2. Exit fee (maker or taker)
        3. Spread paid
        4. Slippage

        Returns:
            Total costs in basis points (bps)
        """
        if self.cost_estimator:
            return self.cost_estimator.estimate(entry_row, exit_time, duration_minutes)

        # Simple default cost model
        if self.config.mode_name == 'scalp':
            # Scalps often use limit orders (maker)
            fee_bps = 2.0  # Maker rebate
        else:
            # Runners often use market orders (taker)
            fee_bps = 8.0

        # Spread (from data if available, else estimate)
        spread_bps = entry_row.get('spread_bps', 3.0) if isinstance(entry_row.get('spread_bps'), (int, float)) else 3.0

        # Slippage (volatility-based)
        atr_bps = entry_row.get('atr_bps', 10.0) if isinstance(entry_row.get('atr_bps'), (int, float)) else 10.0
        slippage_bps = 0.5 * atr_bps

        # Round-trip costs (entry + exit)
        total_costs = (fee_bps * 2) + (spread_bps / 2) + slippage_bps

        return total_costs

    def _coerce_timestamp(self, value) -> datetime:
        """Convert various timestamp representations into timezone-aware datetime."""
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if hasattr(value, "to_pydatetime"):
            dt_value = value.to_pydatetime()
            if dt_value.tzinfo is None:
                dt_value = dt_value.replace(tzinfo=timezone.utc)
            return dt_value

        if isinstance(value, (int, float)):
            raw = float(value)
            # Detect units by magnitude:
            # - seconds since epoch are ~1e9
            # - milliseconds ~1e12
            # - microseconds ~1e15
            if raw > 1e14:  # microseconds
                raw /= 1_000_000
            elif raw > 1e11:  # milliseconds
                raw /= 1_000
            # else assume raw is already in seconds
            dt_value = datetime.fromtimestamp(raw, tz=timezone.utc)
            return dt_value

        # Fallback: parse ISO string
        try:
            dt_value = datetime.fromisoformat(str(value))
            if dt_value.tzinfo is None:
                dt_value = dt_value.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                "timestamp_parse_failed",
                raw_value=str(value),
                message="Falling back to current UTC time",
            )
            dt_value = datetime.now(tz=timezone.utc)

        # Sanity check: drop obviously broken timestamps (pre-2017)
        if dt_value.year < 2017:
            logger.warning(
                "timestamp_sanitized",
                original=str(dt_value),
                message="Timestamp earlier than 2017 detected, clamping to 2017-01-01 UTC",
            )
            dt_value = datetime(2017, 1, 1, tzinfo=timezone.utc)

        return dt_value

    def to_polars(self, labeled_trades: List[LabeledTrade]) -> pl.DataFrame:
        """
        Convert labeled trades to Polars DataFrame.

        Useful for saving to parquet or further analysis.
        """
        if not labeled_trades:
            return pl.DataFrame()

        records = [trade.to_dict() for trade in labeled_trades]
        return pl.DataFrame(records)

    def get_statistics(self, labeled_trades: List[LabeledTrade]) -> dict:
        """
        Calculate summary statistics from labeled trades.

        Returns:
            Dictionary with win rate, avg P&L, etc.
        """
        if not labeled_trades:
            return {}

        winners = [t for t in labeled_trades if t.is_winner()]
        losers = [t for t in labeled_trades if not t.is_winner()]

        tp_exits = [t for t in labeled_trades if t.is_tp_exit()]
        sl_exits = [t for t in labeled_trades if t.is_sl_exit()]
        timeout_exits = [t for t in labeled_trades if t.is_timeout_exit()]

        return {
            'total_trades': len(labeled_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(labeled_trades) if labeled_trades else 0,
            'avg_pnl_net_bps': np.mean([t.pnl_net_bps for t in labeled_trades]),
            'avg_winner_bps': np.mean([t.pnl_net_bps for t in winners]) if winners else 0,
            'avg_loser_bps': np.mean([t.pnl_net_bps for t in losers]) if losers else 0,
            'avg_duration_minutes': np.mean([t.duration_minutes for t in labeled_trades]),
            'avg_costs_bps': np.mean([t.costs_bps for t in labeled_trades]),
            'tp_exits': len(tp_exits),
            'sl_exits': len(sl_exits),
            'timeout_exits': len(timeout_exits),
            'exit_breakdown': {
                'tp': len(tp_exits) / len(labeled_trades) if labeled_trades else 0,
                'sl': len(sl_exits) / len(labeled_trades) if labeled_trades else 0,
                'timeout': len(timeout_exits) / len(labeled_trades) if labeled_trades else 0,
            }
        }


def print_label_statistics(stats: dict) -> None:
    """Pretty-print labeling statistics."""
    print(f"""
╔═══════════════════════════════════════════╗
║     TRIPLE-BARRIER LABELING STATS         ║
╚═══════════════════════════════════════════╝

📊 Overview
──────────────────────────────────────────────
Total Trades:      {stats['total_trades']:,}
Winners:           {stats['winners']:,} ({stats['win_rate']:.1%})
Losers:            {stats['losers']:,}

💰 Performance
──────────────────────────────────────────────
Avg P&L (net):     {stats['avg_pnl_net_bps']:+.2f} bps
Avg Winner:        {stats['avg_winner_bps']:+.2f} bps
Avg Loser:         {stats['avg_loser_bps']:+.2f} bps
Avg Duration:      {stats['avg_duration_minutes']:.1f} minutes
Avg Costs:         {stats['avg_costs_bps']:.2f} bps

🚪 Exit Breakdown
──────────────────────────────────────────────
Take-Profit:       {stats['tp_exits']:,} ({stats['exit_breakdown']['tp']:.1%})
Stop-Loss:         {stats['sl_exits']:,} ({stats['exit_breakdown']['sl']:.1%})
Timeout:           {stats['timeout_exits']:,} ({stats['exit_breakdown']['timeout']:.1%})
""")
