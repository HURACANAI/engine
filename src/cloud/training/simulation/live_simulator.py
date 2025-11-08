"""
Live Simulator

Transaction fee awareness with slippage for realistic trade simulation.
Enhances Execution Simulator with live trading considerations.

Key Features:
- Transaction fee modeling (maker/taker)
- Realistic slippage simulation
- Funding rate costs
- Spread costs
- Partial fills and missed orders
- Real-time market conditions
- Cost-aware PnL calculation

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import structlog

from .execution_simulator import ExecutionSimulator, OrderType, SlippageEstimate, ExecutionResult

logger = structlog.get_logger(__name__)


class FeeType(Enum):
    """Fee type"""
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class TransactionCosts:
    """Transaction costs breakdown"""
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    funding_rate_bps: float = 0.0
    total_cost_bps: float = 0.0
    total_cost_usd: float = 0.0


@dataclass
class LiveTradeResult:
    """Live trade result with all costs"""
    trade_id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    
    # Costs
    entry_costs: TransactionCosts
    exit_costs: TransactionCosts
    funding_costs: TransactionCosts
    
    # PnL
    gross_pnl_usd: float
    net_pnl_usd: float
    net_pnl_bps: float
    
    # Execution details
    entry_filled: bool
    exit_filled: bool
    entry_fill_ratio: float  # 0.0 to 1.0
    exit_fill_ratio: float
    
    # Market conditions
    entry_spread_bps: float
    exit_spread_bps: float
    entry_liquidity_score: float
    exit_liquidity_score: float


class LiveSimulator:
    """
    Live Simulator.
    
    Simulates live trading with realistic transaction costs and slippage.
    
    Usage:
        simulator = LiveSimulator(
            maker_fee_bps=-2.0,  # Maker rebate
            taker_fee_bps=5.0,
            funding_rate_bps=1.0
        )
        
        # Simulate trade
        result = simulator.simulate_trade(
            symbol="BTC-USD",
            direction="long",
            entry_price=50000.0,
            exit_price=51000.0,
            size_usd=10000.0,
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1)
        )
    """
    
    def __init__(
        self,
        maker_fee_bps: float = -2.0,  # Negative for maker rebate
        taker_fee_bps: float = 5.0,
        funding_rate_bps: float = 1.0,  # Per 8 hours
        execution_simulator: Optional[ExecutionSimulator] = None
    ):
        """
        Initialize live simulator.
        
        Args:
            maker_fee_bps: Maker fee in basis points (negative for rebate)
            taker_fee_bps: Taker fee in basis points
            funding_rate_bps: Funding rate in basis points per 8 hours
            execution_simulator: Execution simulator instance (optional)
        """
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.funding_rate_bps = funding_rate_bps
        
        # Use provided execution simulator or create new one
        self.execution_simulator = execution_simulator or ExecutionSimulator(
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps
        )
        
        # Trade history
        self.trade_history: List[LiveTradeResult] = []
        
        logger.info(
            "live_simulator_initialized",
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            funding_rate_bps=funding_rate_bps
        )
    
    def simulate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size_usd: float,
        entry_time: datetime,
        exit_time: datetime,
        entry_order_type: OrderType = OrderType.MARKET,
        exit_order_type: OrderType = OrderType.MARKET,
        entry_spread_bps: Optional[float] = None,
        exit_spread_bps: Optional[float] = None,
        entry_liquidity_score: float = 1.0,
        exit_liquidity_score: float = 1.0,
        funding_rate: Optional[float] = None
    ) -> LiveTradeResult:
        """
        Simulate a complete trade with all costs.
        
        Args:
            symbol: Trading symbol
            direction: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            size_usd: Trade size in USD
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_order_type: Entry order type
            exit_order_type: Exit order type
            entry_spread_bps: Entry spread in bps (optional)
            exit_spread_bps: Exit spread in bps (optional)
            entry_liquidity_score: Entry liquidity score (0-1)
            exit_liquidity_score: Exit liquidity score (0-1)
            funding_rate: Funding rate (optional, uses default if not provided)
        
        Returns:
            LiveTradeResult with all costs and PnL
        """
        import uuid
        trade_id = str(uuid.uuid4())
        
        # Calculate entry costs
        entry_costs = self._calculate_entry_costs(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            order_type=entry_order_type,
            spread_bps=entry_spread_bps,
            liquidity_score=entry_liquidity_score
        )
        
        # Calculate exit costs
        exit_costs = self._calculate_exit_costs(
            symbol=symbol,
            direction=direction,
            exit_price=exit_price,
            size_usd=size_usd,
            order_type=exit_order_type,
            spread_bps=exit_spread_bps,
            liquidity_score=exit_liquidity_score
        )
        
        # Calculate funding costs
        funding_costs = self._calculate_funding_costs(
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            entry_time=entry_time,
            exit_time=exit_time,
            funding_rate=funding_rate
        )
        
        # Calculate gross PnL
        if direction == "long":
            gross_pnl_usd = (exit_price - entry_price) * (size_usd / entry_price)
        else:
            gross_pnl_usd = (entry_price - exit_price) * (size_usd / entry_price)
        
        # Calculate net PnL (after all costs)
        total_costs_usd = (
            entry_costs.total_cost_usd +
            exit_costs.total_cost_usd +
            funding_costs.total_cost_usd
        )
        net_pnl_usd = gross_pnl_usd - total_costs_usd
        
        # Calculate net PnL in bps
        net_pnl_bps = (net_pnl_usd / size_usd) * 10000
        
        # Check if orders were filled
        entry_filled = entry_costs.total_cost_bps < 1000  # Arbitrary threshold
        exit_filled = exit_costs.total_cost_bps < 1000
        
        # Calculate fill ratios (simplified)
        entry_fill_ratio = 1.0 if entry_filled else 0.0
        exit_fill_ratio = 1.0 if exit_filled else 0.0
        
        # Create result
        result = LiveTradeResult(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size_usd=size_usd,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_costs=entry_costs,
            exit_costs=exit_costs,
            funding_costs=funding_costs,
            gross_pnl_usd=gross_pnl_usd,
            net_pnl_usd=net_pnl_usd,
            net_pnl_bps=net_pnl_bps,
            entry_filled=entry_filled,
            exit_filled=exit_filled,
            entry_fill_ratio=entry_fill_ratio,
            exit_fill_ratio=exit_fill_ratio,
            entry_spread_bps=entry_spread_bps or 0.0,
            exit_spread_bps=exit_spread_bps or 0.0,
            entry_liquidity_score=entry_liquidity_score,
            exit_liquidity_score=exit_liquidity_score
        )
        
        self.trade_history.append(result)
        
        logger.info(
            "trade_simulated",
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            gross_pnl_usd=gross_pnl_usd,
            net_pnl_usd=net_pnl_usd,
            total_costs_usd=total_costs_usd
        )
        
        return result
    
    def _calculate_entry_costs(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        order_type: OrderType,
        spread_bps: Optional[float],
        liquidity_score: float
    ) -> TransactionCosts:
        """Calculate entry transaction costs"""
        # Get slippage estimate
        slippage_estimate = self.execution_simulator.estimate_slippage(
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            order_type=order_type,
            spread_bps=spread_bps
        )
        
        # Determine fee type based on order type
        if order_type == OrderType.LIMIT:
            fee_bps = self.maker_fee_bps
            fee_type = FeeType.MAKER
        else:
            fee_bps = self.taker_fee_bps
            fee_type = FeeType.TAKER
        
        # Adjust fees based on liquidity
        # Lower liquidity = higher effective fees
        adjusted_fee_bps = fee_bps / max(liquidity_score, 0.1)
        
        # Calculate spread cost
        spread_cost_bps = spread_bps or 0.0
        
        # Calculate slippage
        slippage_bps = slippage_estimate.total_slippage_bps
        
        # Total cost in bps
        total_cost_bps = abs(adjusted_fee_bps) + slippage_bps + spread_cost_bps
        
        # Total cost in USD
        total_cost_usd = (total_cost_bps / 10000) * size_usd
        
        return TransactionCosts(
            maker_fee_bps=self.maker_fee_bps if fee_type == FeeType.MAKER else 0.0,
            taker_fee_bps=self.taker_fee_bps if fee_type == FeeType.TAKER else 0.0,
            slippage_bps=slippage_bps,
            spread_bps=spread_cost_bps,
            funding_rate_bps=0.0,  # No funding on entry
            total_cost_bps=total_cost_bps,
            total_cost_usd=total_cost_usd
        )
    
    def _calculate_exit_costs(
        self,
        symbol: str,
        direction: str,
        exit_price: float,
        size_usd: float,
        order_type: OrderType,
        spread_bps: Optional[float],
        liquidity_score: float
    ) -> TransactionCosts:
        """Calculate exit transaction costs"""
        # Same as entry costs
        return self._calculate_entry_costs(
            symbol=symbol,
            direction=direction,
            entry_price=exit_price,  # Using exit price
            size_usd=size_usd,
            order_type=order_type,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score
        )
    
    def _calculate_funding_costs(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        entry_time: datetime,
        exit_time: datetime,
        funding_rate: Optional[float]
    ) -> TransactionCosts:
        """Calculate funding rate costs"""
        # Use provided funding rate or default
        rate_bps = funding_rate or self.funding_rate_bps
        
        # Calculate time in position (in 8-hour periods)
        time_diff = exit_time - entry_time
        hours = time_diff.total_seconds() / 3600
        periods = hours / 8.0
        
        # Funding costs are paid by the side that receives funding
        # Long positions pay funding when rate is positive
        # Short positions pay funding when rate is negative
        if direction == "long":
            # Long pays when rate > 0
            funding_cost_bps = rate_bps * periods if rate_bps > 0 else 0.0
        else:
            # Short pays when rate < 0
            funding_cost_bps = abs(rate_bps) * periods if rate_bps < 0 else 0.0
        
        # Total cost in USD
        total_cost_usd = (funding_cost_bps / 10000) * size_usd
        
        return TransactionCosts(
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            funding_rate_bps=funding_cost_bps,
            total_cost_bps=funding_cost_bps,
            total_cost_usd=total_cost_usd
        )
    
    def get_trade_statistics(self) -> Dict[str, float]:
        """Get statistics from trade history"""
        if not self.trade_history:
            return {}
        
        # Calculate statistics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.net_pnl_usd > 0]
        losing_trades = [t for t in self.trade_history if t.net_pnl_usd <= 0]
        
        total_gross_pnl = sum(t.gross_pnl_usd for t in self.trade_history)
        total_net_pnl = sum(t.net_pnl_usd for t in self.trade_history)
        total_costs = sum(
            t.entry_costs.total_cost_usd + t.exit_costs.total_cost_usd + t.funding_costs.total_cost_usd
            for t in self.trade_history
        )
        
        avg_gross_pnl = total_gross_pnl / total_trades if total_trades > 0 else 0.0
        avg_net_pnl = total_net_pnl / total_trades if total_trades > 0 else 0.0
        avg_costs = total_costs / total_trades if total_trades > 0 else 0.0
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_winner = np.mean([t.net_pnl_usd for t in winning_trades]) if winning_trades else 0.0
        avg_loser = np.mean([t.net_pnl_usd for t in losing_trades]) if losing_trades else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_gross_pnl_usd": total_gross_pnl,
            "total_net_pnl_usd": total_net_pnl,
            "total_costs_usd": total_costs,
            "avg_gross_pnl_usd": avg_gross_pnl,
            "avg_net_pnl_usd": avg_net_pnl,
            "avg_costs_usd": avg_costs,
            "avg_winner_usd": avg_winner,
            "avg_loser_usd": avg_loser,
            "profit_factor": abs(avg_winner / avg_loser) if avg_loser != 0 else 0.0
        }
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by type"""
        if not self.trade_history:
            return {}
        
        total_maker_fees = 0.0
        total_taker_fees = 0.0
        total_slippage = 0.0
        total_spread = 0.0
        total_funding = 0.0
        
        for trade in self.trade_history:
            total_maker_fees += trade.entry_costs.maker_fee_bps + trade.exit_costs.maker_fee_bps
            total_taker_fees += trade.entry_costs.taker_fee_bps + trade.exit_costs.taker_fee_bps
            total_slippage += trade.entry_costs.slippage_bps + trade.exit_costs.slippage_bps
            total_spread += trade.entry_costs.spread_bps + trade.exit_costs.spread_bps
            total_funding += trade.funding_costs.funding_rate_bps
        
        total_trades = len(self.trade_history)
        
        return {
            "avg_maker_fee_bps": total_maker_fees / total_trades if total_trades > 0 else 0.0,
            "avg_taker_fee_bps": total_taker_fees / total_trades if total_trades > 0 else 0.0,
            "avg_slippage_bps": total_slippage / total_trades if total_trades > 0 else 0.0,
            "avg_spread_bps": total_spread / total_trades if total_trades > 0 else 0.0,
            "avg_funding_bps": total_funding / total_trades if total_trades > 0 else 0.0
        }

