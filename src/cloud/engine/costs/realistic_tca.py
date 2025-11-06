"""
Realistic Transaction Cost Analysis

Models ALL costs of trading:
1. Exchange fees (from historical fee schedule)
2. Spread (bid-ask spread paid)
3. Slippage (market impact + volatility)
4. Partial fill delays (advanced)

Philosophy:
- Better to overestimate costs than underestimate
- Use historical data when available (L2 order book)
- Fall back to conservative estimates when data missing
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TCAReport:
    """Transaction Cost Analysis report for a single trade."""

    # Fee components
    fee_entry_bps: float
    fee_exit_bps: float
    fee_total_bps: float

    # Spread components
    spread_bps: float
    spread_paid_bps: float  # Usually half-spread for limit orders

    # Slippage components
    volatility_bps: float
    market_impact_bps: float
    slippage_total_bps: float

    # Total
    total_cost_bps: float

    # Metadata
    is_maker: bool
    trade_mode: str  # 'scalp' or 'runner'


class CostEstimator:
    """
    Estimate transaction costs for trades.

    Usage:
        estimator = CostEstimator(exchange='binance')
        costs = estimator.estimate(
            entry_row=candle_data,
            exit_time=exit_time,
            duration_minutes=30,
            mode='scalp'
        )
    """

    def __init__(
        self,
        exchange: str = 'binance',
        default_spread_bps: float = 3.0,
        slippage_multiplier: float = 0.5,
        use_conservative_estimates: bool = True
    ):
        """
        Initialize cost estimator.

        Args:
            exchange: Exchange name for fee lookup
            default_spread_bps: Default spread if not in data
            slippage_multiplier: Multiplier for volatility-based slippage
            use_conservative_estimates: Use pessimistic estimates when uncertain
        """
        self.exchange = exchange.lower()
        self.default_spread = default_spread_bps
        self.slippage_multiplier = slippage_multiplier
        self.conservative = use_conservative_estimates

        # Import fee manager
        from ..data_quality import HistoricalFeeManager
        self.fee_manager = HistoricalFeeManager()

        logger.info(
            "cost_estimator_initialized",
            exchange=exchange,
            default_spread_bps=default_spread_bps,
            slippage_multiplier=slippage_multiplier
        )

    def estimate(
        self,
        entry_row,
        exit_time: datetime,
        duration_minutes: float,
        mode: str = 'scalp',
        position_size_gbp: float = 1000.0
    ) -> float:
        """
        Estimate total transaction costs in basis points.

        Args:
            entry_row: Candle data at entry (Polars row or dict)
            exit_time: Exit timestamp
            duration_minutes: Trade duration
            mode: 'scalp' or 'runner'
            position_size_gbp: Position size for market impact

        Returns:
            Total costs in basis points
        """
        # Get detailed breakdown
        report = self.estimate_detailed(
            entry_row, exit_time, duration_minutes, mode, position_size_gbp
        )

        return report.total_cost_bps

    def estimate_detailed(
        self,
        entry_row,
        exit_time: datetime,
        duration_minutes: float,
        mode: str = 'scalp',
        position_size_gbp: float = 1000.0
    ) -> TCAReport:
        """
        Get detailed cost breakdown.

        Returns:
            TCAReport with all components
        """
        # Determine if maker or taker
        is_maker = self._is_maker_order(mode, duration_minutes)

        # Get entry timestamp
        if hasattr(entry_row, '__getitem__'):
            entry_time = entry_row['timestamp'][0] if hasattr(entry_row['timestamp'], '__getitem__') else entry_row['timestamp']
        else:
            entry_time = datetime.now()  # Fallback

        # 1. Fee costs
        fee_entry = self.fee_manager.get_fee_for_date(
            self.exchange, entry_time, is_maker
        )
        fee_exit = self.fee_manager.get_fee_for_date(
            self.exchange, exit_time, is_maker
        )
        fee_total = fee_entry + fee_exit

        # 2. Spread costs
        spread_bps = self._get_spread(entry_row)
        # Pay half-spread for limit orders, full spread for market
        spread_paid = spread_bps if not is_maker else (spread_bps / 2)

        # 3. Slippage costs
        volatility_bps = self._get_volatility(entry_row)
        market_impact = self._estimate_market_impact(
            position_size_gbp, entry_row
        )
        slippage_total = (volatility_bps * self.slippage_multiplier) + market_impact

        # Total round-trip costs
        total_cost = fee_total + spread_paid + slippage_total

        # Conservative adjustment
        if self.conservative:
            total_cost *= 1.1  # Add 10% buffer

        logger.debug(
            "cost_estimated",
            mode=mode,
            is_maker=is_maker,
            fee_bps=fee_total,
            spread_bps=spread_paid,
            slippage_bps=slippage_total,
            total_bps=total_cost
        )

        return TCAReport(
            fee_entry_bps=fee_entry,
            fee_exit_bps=fee_exit,
            fee_total_bps=fee_total,
            spread_bps=spread_bps,
            spread_paid_bps=spread_paid,
            volatility_bps=volatility_bps,
            market_impact_bps=market_impact,
            slippage_total_bps=slippage_total,
            total_cost_bps=total_cost,
            is_maker=is_maker,
            trade_mode=mode
        )

    def _is_maker_order(self, mode: str, duration_minutes: float) -> bool:
        """
        Determine if order is likely maker or taker.

        Heuristics:
        - Scalps with quick fills → taker
        - Runners with patient entries → maker
        - Very short duration → taker
        """
        if mode == 'scalp':
            # Scalps often use limit orders but need quick fills
            return duration_minutes > 5  # If held > 5 min, was probably maker

        # Runners are more patient
        return True

    def _get_spread(self, entry_row) -> float:
        """Get bid-ask spread from data or use default."""
        try:
            if hasattr(entry_row, '__getitem__'):
                if 'spread_bps' in entry_row:
                    spread = entry_row['spread_bps']
                    if hasattr(spread, '__getitem__'):
                        return float(spread[0])
                    return float(spread)
        except:
            pass

        return self.default_spread

    def _get_volatility(self, entry_row) -> float:
        """Get volatility from ATR or estimate."""
        try:
            if hasattr(entry_row, '__getitem__'):
                if 'atr_bps' in entry_row:
                    atr = entry_row['atr_bps']
                    if hasattr(atr, '__getitem__'):
                        return float(atr[0])
                    return float(atr)

                # Estimate from candle range
                if 'high' in entry_row and 'low' in entry_row:
                    high = entry_row['high'][0] if hasattr(entry_row['high'], '__getitem__') else entry_row['high']
                    low = entry_row['low'][0] if hasattr(entry_row['low'], '__getitem__') else entry_row['low']
                    return ((high / low) - 1) * 10000
        except:
            pass

        # Conservative default
        return 10.0  # 10 bps

    def _estimate_market_impact(
        self,
        position_size_gbp: float,
        entry_row
    ) -> float:
        """
        Estimate market impact (price movement caused by your order).

        Simple model: Impact ∝ sqrt(order_size / avg_volume)

        For small retail sizes (<£10k), impact is negligible.
        """
        if position_size_gbp < 10000:
            return 0.5  # Minimal impact

        # Get volume from data
        try:
            if hasattr(entry_row, '__getitem__'):
                if 'volume' in entry_row:
                    volume = entry_row['volume']
                    if hasattr(volume, '__getitem__'):
                        volume = float(volume[0])
                    else:
                        volume = float(volume)

                    # Simple square-root model
                    impact_factor = np.sqrt(position_size_gbp / (volume * 100))
                    return min(impact_factor * 10, 5.0)  # Cap at 5 bps
        except:
            pass

        # Conservative default for larger sizes
        return 2.0

    def estimate_from_dataframe(
        self,
        df,
        mode: str = 'scalp',
        position_size_gbp: float = 1000.0
    ) -> float:
        """
        Estimate average costs for an entire dataset.

        Useful for setting baseline expectations.
        """
        if len(df) == 0:
            return self.default_spread + 10.0  # Conservative default

        costs = []

        for i in range(min(100, len(df))):  # Sample first 100 rows
            row = df[i]
            exit_time = row['timestamp'][0] + timedelta(minutes=30)

            cost = self.estimate(
                entry_row=row,
                exit_time=exit_time,
                duration_minutes=30,
                mode=mode,
                position_size_gbp=position_size_gbp
            )
            costs.append(cost)

        avg_cost = np.mean(costs)

        logger.info(
            "average_costs_estimated",
            mode=mode,
            samples=len(costs),
            avg_cost_bps=avg_cost,
            min_cost_bps=np.min(costs),
            max_cost_bps=np.max(costs)
        )

        return avg_cost


def print_tca_report(report: TCAReport) -> None:
    """Pretty-print TCA report."""
    print(f"""
╔═══════════════════════════════════════════╗
║   TRANSACTION COST ANALYSIS (TCA)         ║
╚═══════════════════════════════════════════╝

⚙️ Trade Info
──────────────────────────────────────────────
Mode:              {report.trade_mode}
Order Type:        {'Maker' if report.is_maker else 'Taker'}

💰 Cost Breakdown
──────────────────────────────────────────────
Entry Fee:         {report.fee_entry_bps:.2f} bps
Exit Fee:          {report.fee_exit_bps:.2f} bps
Total Fees:        {report.fee_total_bps:.2f} bps

Bid-Ask Spread:    {report.spread_bps:.2f} bps
Spread Paid:       {report.spread_paid_bps:.2f} bps

Volatility:        {report.volatility_bps:.2f} bps
Market Impact:     {report.market_impact_bps:.2f} bps
Total Slippage:    {report.slippage_total_bps:.2f} bps

═══════════════════════════════════════════════
TOTAL COST:        {report.total_cost_bps:.2f} bps
═══════════════════════════════════════════════

💡 Interpretation:
   Your model must predict edge > {report.total_cost_bps:.1f} bps
   to be profitable after costs.
""")


from datetime import timedelta  # For estimate_from_dataframe
