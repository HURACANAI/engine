"""
Take-Profit Ladder System

Implements scaled exits at multiple profit targets to maximize profit capture
while reducing risk progressively.

Key Problems Solved:
1. **Single TP Miss**: Price hits +190 bps but TP at +200 bps, reverses, exit at +50 bps
2. **Give-Back Risk**: Hit +300 bps, hold for +400 bps target, price reverses to +100 bps
3. **Moonshot Miss**: Exit 100% at +200 bps, price continues to +800 bps (missed 600 bps!)

Solution: Multi-Level Take-Profits with Partial Exits
- TP1 at +100 bps: Exit 30% (lock profit, reduce risk by 30%)
- TP2 at +200 bps: Exit 40% (take majority off, reduce risk to 30%)
- TP3 at +400 bps: Exit 20% (reduce to 10% remaining)
- Trail remaining 10% with wide stop for moonshots

Example Trade:
    Entry: Long BTC at $47,000 with 100 BTC

    Price → $47,470 (+100 bps):
    → TP1 hit: Exit 30 BTC at $47,470
    → Profit locked: +$14,100
    → Remaining: 70 BTC

    Price → $47,940 (+200 bps):
    → TP2 hit: Exit 40 BTC at $47,940
    → Additional profit: +$37,600
    → Total profit: $51,700
    → Remaining: 30 BTC

    Price → $48,880 (+400 bps):
    → TP3 hit: Exit 20 BTC at $48,880
    → Additional profit: +$37,760
    → Total profit: $89,460
    → Remaining: 10 BTC with trailing stop

    Price → $50,340 (+700 bps):
    → Trailing stop hit: Exit 10 BTC at $50,340
    → Additional profit: +$33,400
    → TOTAL PROFIT: $122,860 (261 bps avg!)

    VS Single TP at +200 bps:
    → Would have exited 100% at $47,940
    → Profit: $94,000 (200 bps)
    → Ladder captured EXTRA $28,860 (+30% more profit!)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ExitReason(Enum):
    """Reason for partial or full exit."""

    TP_LADDER = "tp_ladder"  # Take-profit level hit
    TRAILING_STOP = "trailing_stop"  # Final trailing stop hit
    EMERGENCY_EXIT = "emergency_exit"  # Manual or risk-based exit
    STOP_LOSS = "stop_loss"  # Stop loss hit


@dataclass
class TPLevel:
    """Single take-profit level configuration."""

    level_name: str  # "TP1", "TP2", "TP3", etc.
    profit_target_bps: float  # Profit threshold in bps
    exit_percentage: float  # % of position to exit (0-1)
    description: str


@dataclass
class LadderConfig:
    """Configuration for TP ladder."""

    levels: List[TPLevel]
    final_trail_distance_bps: float  # Trailing stop for remaining position
    final_trail_multiplier: float  # Widen trail for moonshots (e.g., 2.0x)


@dataclass
class PartialExit:
    """Record of a partial exit."""

    level_name: str
    exit_price: float
    exit_size: float  # Amount exited
    profit_bps: float
    profit_usd: float
    remaining_size: float
    timestamp: float
    reason: ExitReason


@dataclass
class LadderStatus:
    """Current status of TP ladder."""

    entry_price: float
    entry_size: float
    current_price: float
    current_pnl_bps: float

    # Execution status
    next_tp_level: Optional[TPLevel]
    levels_hit: List[str]  # Level names hit so far
    remaining_size: float
    remaining_percentage: float

    # Partial exits executed
    partial_exits: List[PartialExit]

    # Profit tracking
    total_profit_locked_bps: float
    total_profit_locked_usd: float
    unrealized_pnl_bps: float
    unrealized_pnl_usd: float

    # Trailing stop for remaining
    trailing_stop_price: Optional[float]


class TakeProfitLadder:
    """
    Manages scaled exits across multiple profit levels.

    Philosophy:
    - Lock profits progressively as price moves in favor
    - Reduce risk with each level (less size = less risk)
    - Keep small runner for moonshots (10-20% remaining)
    - Trail final position with wide stop

    Default Ladder (Conservative):
    - TP1 @ +100 bps: Exit 30% (lock early profit)
    - TP2 @ +200 bps: Exit 40% (take majority)
    - TP3 @ +400 bps: Exit 20% (reduce to 10%)
    - Trail 10% with 200 bps stop

    Aggressive Ladder (Moonshot Focused):
    - TP1 @ +150 bps: Exit 20% (minimal early exit)
    - TP2 @ +300 bps: Exit 30% (moderate reduction)
    - TP3 @ +600 bps: Exit 30% (keep 20% runner)
    - Trail 20% with 300 bps stop

    Usage:
        # Create ladder
        ladder = TakeProfitLadder.create_default_ladder()

        # Initialize position
        entry_price = 47000.0
        entry_size = 100.0  # 100 BTC
        direction = 'buy'

        # Monitor position
        while position_open:
            current_price = get_current_price()

            # Update ladder
            status = ladder.update(
                entry_price=entry_price,
                entry_size=entry_size,
                current_price=current_price,
                direction=direction,
            )

            # Check for exits
            exit_action = ladder.check_exit(status)

            if exit_action:
                execute_partial_exit(
                    size=exit_action['size'],
                    price=exit_action['price'],
                    reason=exit_action['reason'],
                )

                # Record exit
                ladder.record_exit(exit_action)
    """

    def __init__(self, config: LadderConfig):
        """
        Initialize TP ladder.

        Args:
            config: Ladder configuration with TP levels
        """
        self.config = config

        # Validate configuration
        total_exit_pct = sum(level.exit_percentage for level in config.levels)
        if total_exit_pct > 1.0:
            raise ValueError(f"Total exit percentage ({total_exit_pct:.0%}) exceeds 100%")

        # Sort levels by profit target
        self.config.levels.sort(key=lambda x: x.profit_target_bps)

        # Track state
        self.levels_hit: List[str] = []
        self.partial_exits: List[PartialExit] = []
        self.remaining_size_pct: float = 1.0  # Start with 100%
        self.highest_pnl_bps: float = 0.0  # For trailing stop
        self.trailing_stop_price: Optional[float] = None

        logger.info(
            "tp_ladder_initialized",
            num_levels=len(config.levels),
            final_trail=config.final_trail_distance_bps,
        )

    @classmethod
    def create_default_ladder(cls) -> "TakeProfitLadder":
        """Create default conservative ladder."""
        config = LadderConfig(
            levels=[
                TPLevel(
                    level_name="TP1",
                    profit_target_bps=100.0,
                    exit_percentage=0.30,
                    description="Early profit lock",
                ),
                TPLevel(
                    level_name="TP2",
                    profit_target_bps=200.0,
                    exit_percentage=0.40,
                    description="Take majority off",
                ),
                TPLevel(
                    level_name="TP3",
                    profit_target_bps=400.0,
                    exit_percentage=0.20,
                    description="Reduce to runner",
                ),
            ],
            final_trail_distance_bps=200.0,
            final_trail_multiplier=1.5,  # 300 bps trail (200 * 1.5)
        )
        return cls(config)

    @classmethod
    def create_aggressive_ladder(cls) -> "TakeProfitLadder":
        """Create aggressive moonshot-focused ladder."""
        config = LadderConfig(
            levels=[
                TPLevel(
                    level_name="TP1",
                    profit_target_bps=150.0,
                    exit_percentage=0.20,
                    description="Minimal early exit",
                ),
                TPLevel(
                    level_name="TP2",
                    profit_target_bps=300.0,
                    exit_percentage=0.30,
                    description="Moderate reduction",
                ),
                TPLevel(
                    level_name="TP3",
                    profit_target_bps=600.0,
                    exit_percentage=0.30,
                    description="Keep large runner",
                ),
            ],
            final_trail_distance_bps=300.0,
            final_trail_multiplier=2.0,  # 600 bps trail (300 * 2.0)
        )
        return cls(config)

    def update(
        self,
        entry_price: float,
        entry_size: float,
        current_price: float,
        direction: str,
    ) -> LadderStatus:
        """
        Update ladder status with current price.

        Args:
            entry_price: Original entry price
            entry_size: Original entry size
            current_price: Current market price
            direction: 'buy' or 'sell'

        Returns:
            LadderStatus with current state
        """
        # Calculate current P&L
        if direction == 'buy':
            pnl_bps = ((current_price - entry_price) / entry_price) * 10000
        else:  # sell
            pnl_bps = ((entry_price - current_price) / entry_price) * 10000

        # Update highest P&L (for trailing stop)
        if pnl_bps > self.highest_pnl_bps:
            self.highest_pnl_bps = pnl_bps

            # Update trailing stop for remaining position
            if self.levels_hit:  # Only after first TP hit
                trail_distance = self.config.final_trail_distance_bps * self.config.final_trail_multiplier

                if direction == 'buy':
                    self.trailing_stop_price = entry_price * (1 + (pnl_bps - trail_distance) / 10000)
                else:
                    self.trailing_stop_price = entry_price * (1 - (pnl_bps - trail_distance) / 10000)

        # Find next TP level
        next_tp_level = None
        for level in self.config.levels:
            if level.level_name not in self.levels_hit:
                next_tp_level = level
                break

        # Calculate remaining size
        remaining_size = entry_size * self.remaining_size_pct
        remaining_pct = self.remaining_size_pct

        # Calculate locked profits
        total_locked_bps = sum(exit.profit_bps * (exit.exit_size / entry_size) for exit in self.partial_exits)
        total_locked_usd = sum(exit.profit_usd for exit in self.partial_exits)

        # Calculate unrealized P&L on remaining
        unrealized_pnl_bps = pnl_bps * remaining_pct
        unrealized_pnl_usd = (current_price - entry_price) * remaining_size if direction == 'buy' else (entry_price - current_price) * remaining_size

        return LadderStatus(
            entry_price=entry_price,
            entry_size=entry_size,
            current_price=current_price,
            current_pnl_bps=pnl_bps,
            next_tp_level=next_tp_level,
            levels_hit=self.levels_hit.copy(),
            remaining_size=remaining_size,
            remaining_percentage=remaining_pct,
            partial_exits=self.partial_exits.copy(),
            total_profit_locked_bps=total_locked_bps,
            total_profit_locked_usd=total_locked_usd,
            unrealized_pnl_bps=unrealized_pnl_bps,
            unrealized_pnl_usd=unrealized_pnl_usd,
            trailing_stop_price=self.trailing_stop_price,
        )

    def check_exit(self, status: LadderStatus, direction: str) -> Optional[Dict[str, any]]:
        """
        Check if any exit should be triggered.

        Args:
            status: Current ladder status
            direction: 'buy' or 'sell'

        Returns:
            Dict with exit action or None
            {
                'size': exit_size,
                'price': exit_price,
                'level_name': level_name,
                'reason': ExitReason,
                'description': str,
            }
        """
        # Check if next TP level hit
        if status.next_tp_level and status.current_pnl_bps >= status.next_tp_level.profit_target_bps:
            exit_size = status.entry_size * status.next_tp_level.exit_percentage
            level_name = status.next_tp_level.level_name

            logger.info(
                "tp_level_hit",
                level=level_name,
                target_bps=status.next_tp_level.profit_target_bps,
                current_bps=status.current_pnl_bps,
                exit_pct=status.next_tp_level.exit_percentage,
            )

            return {
                'size': exit_size,
                'price': status.current_price,
                'level_name': level_name,
                'reason': ExitReason.TP_LADDER,
                'description': f"{level_name} hit at +{status.current_pnl_bps:.0f} bps",
            }

        # Check trailing stop on remaining position
        if status.trailing_stop_price is not None and status.remaining_size > 0:
            stop_hit = False

            if direction == 'buy' and status.current_price <= status.trailing_stop_price:
                stop_hit = True
            elif direction == 'sell' and status.current_price >= status.trailing_stop_price:
                stop_hit = True

            if stop_hit:
                logger.info(
                    "trailing_stop_hit",
                    stop_price=status.trailing_stop_price,
                    current_price=status.current_price,
                    remaining_size=status.remaining_size,
                )

                return {
                    'size': status.remaining_size,
                    'price': status.current_price,
                    'level_name': 'FINAL',
                    'reason': ExitReason.TRAILING_STOP,
                    'description': f"Trailing stop hit at +{status.current_pnl_bps:.0f} bps",
                }

        return None

    def record_exit(
        self,
        exit_action: Dict[str, any],
        entry_price: float,
        timestamp: float,
    ) -> None:
        """
        Record a partial exit.

        Args:
            exit_action: Exit action dict from check_exit()
            entry_price: Original entry price
            timestamp: Exit timestamp
        """
        exit_size = exit_action['size']
        exit_price = exit_action['price']
        level_name = exit_action['level_name']
        reason = exit_action['reason']

        # Calculate profit
        profit_bps = ((exit_price - entry_price) / entry_price) * 10000
        profit_usd = (exit_price - entry_price) * exit_size

        # Create partial exit record
        partial_exit = PartialExit(
            level_name=level_name,
            exit_price=exit_price,
            exit_size=exit_size,
            profit_bps=profit_bps,
            profit_usd=profit_usd,
            remaining_size=0.0,  # Will be updated
            timestamp=timestamp,
            reason=reason,
        )

        self.partial_exits.append(partial_exit)

        # Mark level as hit
        if level_name not in self.levels_hit and level_name != 'FINAL':
            self.levels_hit.append(level_name)

        # Update remaining size percentage
        for level in self.config.levels:
            if level.level_name == level_name:
                self.remaining_size_pct -= level.exit_percentage
                break

        logger.info(
            "partial_exit_recorded",
            level=level_name,
            exit_price=exit_price,
            exit_size=exit_size,
            profit_bps=profit_bps,
            remaining_pct=self.remaining_size_pct,
        )

    def get_summary(self, status: LadderStatus) -> str:
        """Get human-readable summary of ladder status."""
        lines = []
        lines.append(f"=== TP LADDER STATUS ===")
        lines.append(f"Entry: {status.entry_price:.2f} | Current: {status.current_price:.2f}")
        lines.append(f"Current P&L: +{status.current_pnl_bps:.0f} bps")
        lines.append(f"")

        lines.append(f"Levels Hit: {len(status.levels_hit)}/{len(self.config.levels)}")
        for exit in status.partial_exits:
            lines.append(f"  {exit.level_name}: Exited {exit.exit_size:.2f} @ {exit.exit_price:.2f} (+{exit.profit_bps:.0f} bps)")

        lines.append(f"")
        lines.append(f"Remaining: {status.remaining_size:.2f} ({status.remaining_percentage:.0%})")

        if status.next_tp_level:
            distance_to_next = status.next_tp_level.profit_target_bps - status.current_pnl_bps
            lines.append(f"Next: {status.next_tp_level.level_name} @ +{status.next_tp_level.profit_target_bps:.0f} bps ({distance_to_next:+.0f} bps away)")

        if status.trailing_stop_price:
            stop_distance_bps = ((status.trailing_stop_price - status.entry_price) / status.entry_price) * 10000
            lines.append(f"Trailing Stop: {status.trailing_stop_price:.2f} (+{stop_distance_bps:.0f} bps)")

        lines.append(f"")
        lines.append(f"Profit Locked: +{status.total_profit_locked_bps:.0f} bps (${status.total_profit_locked_usd:,.0f})")
        lines.append(f"Unrealized: +{status.unrealized_pnl_bps:.0f} bps (${status.unrealized_pnl_usd:,.0f})")
        lines.append(f"Total: +{status.total_profit_locked_bps + status.unrealized_pnl_bps:.0f} bps (${status.total_profit_locked_usd + status.unrealized_pnl_usd:,.0f})")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset ladder for new position."""
        self.levels_hit = []
        self.partial_exits = []
        self.remaining_size_pct = 1.0
        self.highest_pnl_bps = 0.0
        self.trailing_stop_price = None
