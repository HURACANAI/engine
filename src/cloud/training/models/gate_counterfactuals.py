"""
Gate Counterfactuals - Track "What Would Have Happened"

Key Problem:
Gates block trades, but we don't know if we're blocking GOOD trades or BAD trades.
Without counterfactuals, we can't prove gate value or tune thresholds.

Example Scenario:
- Cost gate blocks trade: "edge_net = +4 bps < buffer 5 bps"
- What actually happened? Price moved +20 bps in our direction
- Counterfactual: We would have made +£2 profit
- Conclusion: Gate was TOO STRICT (false negative)

Solution: Counterfactual P&L Tracking
1. Record blocked trade details (price, direction, features)
2. Wait N minutes
3. Simulate what would have happened
4. Track gate performance (value attribution)
5. Auto-tune thresholds based on false positive/negative rates

Benefits:
- Prove gate value: "Cost gate saved £150 this week"
- Find miscalibrated gates: "Meta-label blocking 40% winners"
- Auto-tune thresholds: Loosen if too many false negatives
- Per-gate economics dashboard

Usage:
    tracker = GateCounterfactualTracker()

    # When gate blocks trade
    tracker.record_blocked_trade(
        gate_name='cost_gate',
        symbol='ETH-USD',
        direction='long',
        entry_price=2000.0,
        size=0.05,
        blocked_reason='edge_net too low',
        features={'edge_hat': 8.0, 'cost': 5.0},
    )

    # Later, check what happened
    tracker.simulate_counterfactuals()  # Run periodically

    # Get gate performance
    metrics = tracker.get_gate_metrics('cost_gate')
    print(f"Saved: ${metrics.total_saved:.2f}")
    print(f"False negatives: {metrics.false_negative_rate:.1%}")

    # Auto-tune
    if metrics.false_negative_rate > 0.30:
        # Gate blocking too many winners - loosen it
        cost_gate.buffer_bps *= 0.90
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CounterfactualOutcome(Enum):
    """Outcome of counterfactual simulation."""

    PENDING = "pending"  # Not yet simulated
    SAVED_LOSS = "saved_loss"  # Gate blocked a loser (true positive)
    BLOCKED_WINNER = "blocked_winner"  # Gate blocked a winner (false negative)
    NEUTRAL = "neutral"  # Would have been breakeven


@dataclass
class BlockedTrade:
    """A trade that was blocked by a gate."""

    # Identity
    trade_id: str
    gate_name: str
    blocked_time: float

    # Trade details
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    size: float  # In asset units
    size_usd: float

    # Gate decision
    blocked_reason: str
    gate_features: Dict[str, float]  # Features at time of block

    # Counterfactual simulation (filled later)
    simulated: bool = False
    simulation_time: Optional[float] = None
    exit_price: Optional[float] = None  # Price N minutes later
    counterfactual_pnl: Optional[float] = None  # What we would have made
    outcome: CounterfactualOutcome = CounterfactualOutcome.PENDING

    # Metadata
    technique: Optional[str] = None
    regime: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class GateMetrics:
    """Performance metrics for a gate."""

    gate_name: str

    # Counts
    total_blocks: int
    simulated_blocks: int

    # Outcomes
    saved_losses: int  # True positives
    blocked_winners: int  # False negatives
    neutral: int

    # Economics
    total_saved: float  # $ saved by blocking losers
    total_missed: float  # $ missed by blocking winners
    net_value: float  # saved - missed

    # Rates
    save_rate: float  # % of blocks that were losers
    false_negative_rate: float  # % of blocks that were winners

    # Statistics
    avg_saved_per_block: float
    avg_missed_per_block: float


class GateCounterfactualTracker:
    """
    Track counterfactuals for blocked trades.

    Key Features:
    1. Record blocked trade details
    2. Simulate what would have happened
    3. Calculate gate value attribution
    4. Auto-tune gate thresholds

    Architecture:
        Gate blocks trade → Record details
        ↓
        Wait N minutes (holding period)
        ↓
        Simulate: entry → hold → exit
        ↓
        Classify: SAVED_LOSS or BLOCKED_WINNER
        ↓
        Update gate metrics
        ↓
        Auto-tune if needed
    """

    def __init__(
        self,
        hold_duration_sec: float = 300.0,  # 5 minutes default
        min_trades_for_tuning: int = 50,
        false_negative_threshold: float = 0.30,  # Tune if blocking >30% winners
        false_positive_threshold: float = 0.20,  # Tune if letting through >20% losers
    ):
        """
        Initialize counterfactual tracker.

        Args:
            hold_duration_sec: How long to hold for simulation (seconds)
            min_trades_for_tuning: Minimum blocked trades before auto-tuning
            false_negative_threshold: FN rate that triggers loosening
            false_positive_threshold: FP rate that triggers tightening
        """
        self.hold_duration = hold_duration_sec
        self.min_trades = min_trades_for_tuning
        self.fn_threshold = false_negative_threshold
        self.fp_threshold = false_positive_threshold

        # Storage
        self.blocked_trades: Dict[str, BlockedTrade] = {}
        self.next_trade_id = 0

        # Per-gate tracking
        self.gate_history: Dict[str, List[BlockedTrade]] = {}

        logger.info(
            "counterfactual_tracker_initialized",
            hold_duration_sec=hold_duration_sec,
            fn_threshold=false_negative_threshold,
        )

    def record_blocked_trade(
        self,
        gate_name: str,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        blocked_reason: str,
        gate_features: Dict[str, float],
        technique: Optional[str] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Record a blocked trade for counterfactual analysis.

        Args:
            gate_name: Name of gate that blocked trade
            symbol: Asset symbol
            direction: Trade direction ('long' or 'short')
            entry_price: Would-be entry price
            size: Position size in asset units
            blocked_reason: Why gate blocked
            gate_features: Features at time of block
            technique: Trading technique
            regime: Market regime
            confidence: Engine confidence

        Returns:
            trade_id for reference
        """
        trade_id = f"blocked_{self.next_trade_id}"
        self.next_trade_id += 1

        size_usd = size * entry_price

        blocked_trade = BlockedTrade(
            trade_id=trade_id,
            gate_name=gate_name,
            blocked_time=time.time(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            size_usd=size_usd,
            blocked_reason=blocked_reason,
            gate_features=gate_features,
            technique=technique,
            regime=regime,
            confidence=confidence,
        )

        self.blocked_trades[trade_id] = blocked_trade

        # Add to gate history
        if gate_name not in self.gate_history:
            self.gate_history[gate_name] = []
        self.gate_history[gate_name].append(blocked_trade)

        logger.info(
            "blocked_trade_recorded",
            trade_id=trade_id,
            gate=gate_name,
            symbol=symbol,
            direction=direction,
            reason=blocked_reason,
        )

        return trade_id

    def simulate_counterfactual(
        self,
        trade_id: str,
        current_price: float,
    ) -> Optional[CounterfactualOutcome]:
        """
        Simulate what would have happened for a blocked trade.

        Args:
            trade_id: Blocked trade ID
            current_price: Current price (N minutes after block)

        Returns:
            Counterfactual outcome
        """
        if trade_id not in self.blocked_trades:
            logger.warning("trade_not_found", trade_id=trade_id)
            return None

        blocked = self.blocked_trades[trade_id]

        if blocked.simulated:
            return blocked.outcome

        # Check if enough time has passed
        time_elapsed = time.time() - blocked.blocked_time
        if time_elapsed < self.hold_duration:
            return None  # Not ready yet

        # Simulate P&L
        if blocked.direction == 'long':
            pnl = (current_price - blocked.entry_price) * blocked.size
        else:  # short
            pnl = (blocked.entry_price - current_price) * blocked.size

        blocked.simulated = True
        blocked.simulation_time = time.time()
        blocked.exit_price = current_price
        blocked.counterfactual_pnl = pnl

        # Classify outcome
        if pnl < -2.0:  # Lost >$2
            blocked.outcome = CounterfactualOutcome.SAVED_LOSS
        elif pnl > 2.0:  # Would have won >$2
            blocked.outcome = CounterfactualOutcome.BLOCKED_WINNER
        else:  # Breakeven
            blocked.outcome = CounterfactualOutcome.NEUTRAL

        logger.info(
            "counterfactual_simulated",
            trade_id=trade_id,
            gate=blocked.gate_name,
            pnl=pnl,
            outcome=blocked.outcome.value,
        )

        return blocked.outcome

    def simulate_all_pending(
        self,
        price_lookup: Dict[str, float],
    ) -> int:
        """
        Simulate all pending counterfactuals.

        Args:
            price_lookup: Dict mapping symbol -> current_price

        Returns:
            Number of simulations performed
        """
        simulated_count = 0

        for trade_id, blocked in self.blocked_trades.items():
            if blocked.simulated:
                continue

            # Check if ready
            time_elapsed = time.time() - blocked.blocked_time
            if time_elapsed < self.hold_duration:
                continue

            # Get current price
            if blocked.symbol not in price_lookup:
                continue

            current_price = price_lookup[blocked.symbol]

            # Simulate
            outcome = self.simulate_counterfactual(trade_id, current_price)
            if outcome:
                simulated_count += 1

        logger.info("counterfactuals_batch_simulated", count=simulated_count)

        return simulated_count

    def get_gate_metrics(self, gate_name: str) -> GateMetrics:
        """
        Get performance metrics for a gate.

        Args:
            gate_name: Gate to analyze

        Returns:
            GateMetrics with performance stats
        """
        if gate_name not in self.gate_history:
            return GateMetrics(
                gate_name=gate_name,
                total_blocks=0,
                simulated_blocks=0,
                saved_losses=0,
                blocked_winners=0,
                neutral=0,
                total_saved=0.0,
                total_missed=0.0,
                net_value=0.0,
                save_rate=0.0,
                false_negative_rate=0.0,
                avg_saved_per_block=0.0,
                avg_missed_per_block=0.0,
            )

        history = self.gate_history[gate_name]
        simulated = [t for t in history if t.simulated]

        total_blocks = len(history)
        simulated_count = len(simulated)

        # Count outcomes
        saved_losses = sum(
            1 for t in simulated if t.outcome == CounterfactualOutcome.SAVED_LOSS
        )
        blocked_winners = sum(
            1 for t in simulated if t.outcome == CounterfactualOutcome.BLOCKED_WINNER
        )
        neutral = sum(
            1 for t in simulated if t.outcome == CounterfactualOutcome.NEUTRAL
        )

        # Calculate economics
        saved_pnls = [
            abs(t.counterfactual_pnl)
            for t in simulated
            if t.outcome == CounterfactualOutcome.SAVED_LOSS and t.counterfactual_pnl
        ]
        missed_pnls = [
            t.counterfactual_pnl
            for t in simulated
            if t.outcome == CounterfactualOutcome.BLOCKED_WINNER
            and t.counterfactual_pnl
        ]

        total_saved = sum(saved_pnls)
        total_missed = sum(missed_pnls)
        net_value = total_saved - total_missed

        # Rates
        save_rate = saved_losses / simulated_count if simulated_count > 0 else 0.0
        fn_rate = blocked_winners / simulated_count if simulated_count > 0 else 0.0

        # Averages
        avg_saved = total_saved / saved_losses if saved_losses > 0 else 0.0
        avg_missed = total_missed / blocked_winners if blocked_winners > 0 else 0.0

        return GateMetrics(
            gate_name=gate_name,
            total_blocks=total_blocks,
            simulated_blocks=simulated_count,
            saved_losses=saved_losses,
            blocked_winners=blocked_winners,
            neutral=neutral,
            total_saved=total_saved,
            total_missed=total_missed,
            net_value=net_value,
            save_rate=save_rate,
            false_negative_rate=fn_rate,
            avg_saved_per_block=avg_saved,
            avg_missed_per_block=avg_missed,
        )

    def suggest_threshold_adjustment(
        self,
        gate_name: str,
    ) -> Optional[tuple[str, float]]:
        """
        Suggest threshold adjustment based on counterfactuals.

        Args:
            gate_name: Gate to analyze

        Returns:
            (adjustment_direction, multiplier) or None
            e.g., ('loosen', 0.90) means multiply threshold by 0.90
        """
        metrics = self.get_gate_metrics(gate_name)

        if metrics.simulated_blocks < self.min_trades:
            return None  # Not enough data

        # Check false negative rate (blocking winners)
        if metrics.false_negative_rate > self.fn_threshold:
            # Blocking too many winners - LOOSEN gate
            severity = (metrics.false_negative_rate - self.fn_threshold) / 0.20
            multiplier = 1.0 - (severity * 0.10)  # Up to 10% looser

            logger.warning(
                "gate_too_strict",
                gate=gate_name,
                fn_rate=metrics.false_negative_rate,
                suggested_multiplier=multiplier,
            )

            return ('loosen', multiplier)

        # Check if we're saving losses (want high save rate)
        if metrics.save_rate < 0.50 and metrics.simulated_blocks > 100:
            # Not saving enough losses - gate might be TOO LOOSE
            # Tighten it
            multiplier = 1.05  # 5% tighter

            logger.warning(
                "gate_too_loose",
                gate=gate_name,
                save_rate=metrics.save_rate,
                suggested_multiplier=multiplier,
            )

            return ('tighten', multiplier)

        return None  # No adjustment needed

    def get_all_metrics(self) -> Dict[str, GateMetrics]:
        """Get metrics for all gates."""
        return {gate_name: self.get_gate_metrics(gate_name) for gate_name in self.gate_history}

    def get_summary(self) -> Dict:
        """Get overall summary across all gates."""
        all_metrics = self.get_all_metrics()

        total_blocks = sum(m.total_blocks for m in all_metrics.values())
        total_saved = sum(m.total_saved for m in all_metrics.values())
        total_missed = sum(m.total_missed for m in all_metrics.values())
        net_value = sum(m.net_value for m in all_metrics.values())

        return {
            'total_gates': len(self.gate_history),
            'total_blocks': total_blocks,
            'total_saved_usd': total_saved,
            'total_missed_usd': total_missed,
            'net_value_usd': net_value,
            'per_gate': {name: {
                'blocks': m.total_blocks,
                'saved': m.total_saved,
                'missed': m.total_missed,
                'net': m.net_value,
                'fn_rate': m.false_negative_rate,
            } for name, m in all_metrics.items()},
        }
