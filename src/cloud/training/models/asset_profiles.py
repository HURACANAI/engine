"""
Asset Mode Profiles - Dual-Mode Trading Configuration

Defines per-asset trading mode profiles:
- SHORT_HOLD: Fast scalping mode (£1-£2 net targets)
- LONG_HOLD: Swing/maximizer mode (hold through dips, maximize gains)
- BOTH: Run both modes concurrently on same asset

Configuration includes:
- Entry/exit thresholds per mode
- Risk caps per mode
- Trailing stop strategies
- Add/scale parameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class TradingMode(Enum):
    """Trading mode types."""
    SHORT_HOLD = "short_hold"  # Scalp mode
    LONG_HOLD = "long_hold"    # Swing/maximizer mode
    BOTH = "both"              # Run both concurrently


class TrailStyle(Enum):
    """Trailing stop styles for long-hold mode."""
    CHANDELIER_ATR_2 = "chandelier_atr_2"  # ATR multiplier 2x
    CHANDELIER_ATR_3 = "chandelier_atr_3"  # ATR multiplier 3x
    STRUCTURE_SWING = "structure_swing"     # Structure-based swing highs/lows
    FIXED_BPS = "fixed_bps"                # Fixed bps trail


@dataclass
class ShortHoldConfig:
    """Configuration for short-hold (scalp) mode."""

    max_book_pct: float = 0.15  # Max 15% of portfolio for short-hold book
    target_profit_bps: float = 15.0  # Target £1-£2 on £1000 base
    scratch_threshold_bps: float = -3.0  # Scratch if < -3 bps
    max_hold_minutes: int = 30  # Fast turnover
    maker_bias: bool = True  # Prefer maker fills
    aggressive_exit: bool = True  # Exit fast on adverse moves


@dataclass
class LongHoldConfig:
    """Configuration for long-hold (swing) mode."""

    max_book_pct: float = 0.35  # Max 35% of portfolio for this asset in long-hold
    add_grid_bps: List[float] = field(default_factory=lambda: [-150.0, -300.0])  # DCA ladder
    trail_style: TrailStyle = TrailStyle.CHANDELIER_ATR_3
    tp_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.8, 2.8])  # Scale-out levels
    be_lock_after_bps: float = 60.0  # Move stop to BE+costs after +60 bps
    panic_override: bool = True  # Apply special rules in PANIC regime
    min_hold_hours: int = 8  # Ignore micro noise for first 8 hours
    max_hold_days: float = 7.0  # Time stop
    max_floating_dd_bps: float = 500.0  # Max adverse move before forced reduce


@dataclass
class AssetProfile:
    """Complete profile for an asset."""

    symbol: str
    mode: TradingMode
    short_hold: Optional[ShortHoldConfig] = None
    long_hold: Optional[LongHoldConfig] = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.mode in [TradingMode.SHORT_HOLD, TradingMode.BOTH]:
            if self.short_hold is None:
                self.short_hold = ShortHoldConfig()

        if self.mode in [TradingMode.LONG_HOLD, TradingMode.BOTH]:
            if self.long_hold is None:
                self.long_hold = LongHoldConfig()


class AssetProfileManager:
    """
    Manages asset mode profiles.

    Responsibilities:
    1. Load profiles from configuration
    2. Determine which mode(s) to use for each symbol
    3. Provide mode-specific parameters
    4. Track mode performance and adapt
    """

    def __init__(self, profiles: Optional[Dict[str, AssetProfile]] = None):
        """
        Initialize profile manager.

        Args:
            profiles: Dict of symbol → AssetProfile
        """
        self.profiles = profiles or self._create_default_profiles()

        # Track per-mode performance
        self.mode_performance: Dict[str, Dict[str, List[float]]] = {}  # symbol → mode → pnl_list

        logger.info(
            "asset_profile_manager_initialized",
            num_profiles=len(self.profiles),
            symbols=list(self.profiles.keys())
        )

    def _create_default_profiles(self) -> Dict[str, AssetProfile]:
        """Create default profiles for major assets."""
        profiles = {}

        # ETH: Run both modes with swing-focused parameters
        profiles["ETH"] = AssetProfile(
            symbol="ETH",
            mode=TradingMode.BOTH,
            short_hold=ShortHoldConfig(
                max_book_pct=0.10,
                target_profit_bps=15.0,
                max_hold_minutes=30,
            ),
            long_hold=LongHoldConfig(
                max_book_pct=0.35,
                add_grid_bps=[-150.0, -300.0],
                trail_style=TrailStyle.CHANDELIER_ATR_3,
                tp_multipliers=[1.0, 1.8, 2.8],
                be_lock_after_bps=60.0,
                min_hold_hours=8,
                max_floating_dd_bps=500.0,
            ),
        )

        # SOL: Run both modes with tighter parameters
        profiles["SOL"] = AssetProfile(
            symbol="SOL",
            mode=TradingMode.BOTH,
            short_hold=ShortHoldConfig(
                max_book_pct=0.10,
                target_profit_bps=15.0,
                max_hold_minutes=30,
            ),
            long_hold=LongHoldConfig(
                max_book_pct=0.30,
                add_grid_bps=[-200.0],
                trail_style=TrailStyle.STRUCTURE_SWING,
                tp_multipliers=[1.2, 2.0, 3.0],
                be_lock_after_bps=80.0,
                min_hold_hours=8,
                max_floating_dd_bps=600.0,
            ),
        )

        # BTC: Run both modes with conservative swing parameters
        profiles["BTC"] = AssetProfile(
            symbol="BTC",
            mode=TradingMode.BOTH,
            short_hold=ShortHoldConfig(
                max_book_pct=0.10,
                target_profit_bps=15.0,
                max_hold_minutes=30,
            ),
            long_hold=LongHoldConfig(
                max_book_pct=0.40,
                add_grid_bps=[],  # No adds for BTC
                trail_style=TrailStyle.CHANDELIER_ATR_2,
                tp_multipliers=[0.8, 1.5, 2.5],
                be_lock_after_bps=50.0,
                min_hold_hours=6,
                max_floating_dd_bps=400.0,
            ),
        )

        # Default profile for other assets: SHORT_HOLD only
        default_profile = AssetProfile(
            symbol="DEFAULT",
            mode=TradingMode.SHORT_HOLD,
            short_hold=ShortHoldConfig(),
        )
        profiles["DEFAULT"] = default_profile

        return profiles

    def get_profile(self, symbol: str) -> AssetProfile:
        """
        Get profile for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            AssetProfile (returns DEFAULT if not found)
        """
        return self.profiles.get(symbol, self.profiles["DEFAULT"])

    def can_run_short_hold(self, symbol: str) -> bool:
        """Check if symbol can run short-hold mode."""
        profile = self.get_profile(symbol)
        return profile.enabled and profile.mode in [TradingMode.SHORT_HOLD, TradingMode.BOTH]

    def can_run_long_hold(self, symbol: str) -> bool:
        """Check if symbol can run long-hold mode."""
        profile = self.get_profile(symbol)
        return profile.enabled and profile.mode in [TradingMode.LONG_HOLD, TradingMode.BOTH]

    def get_max_book_allocation(self, symbol: str, mode: TradingMode) -> float:
        """Get max book allocation for symbol and mode."""
        profile = self.get_profile(symbol)

        if mode == TradingMode.SHORT_HOLD and profile.short_hold:
            return profile.short_hold.max_book_pct
        elif mode == TradingMode.LONG_HOLD and profile.long_hold:
            return profile.long_hold.max_book_pct

        return 0.0

    def update_performance(self, symbol: str, mode: TradingMode, pnl_bps: float) -> None:
        """
        Update performance tracking for a symbol+mode.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            pnl_bps: Realized PnL in basis points
        """
        if symbol not in self.mode_performance:
            self.mode_performance[symbol] = {
                TradingMode.SHORT_HOLD.value: [],
                TradingMode.LONG_HOLD.value: [],
            }

        self.mode_performance[symbol][mode.value].append(pnl_bps)

        # Keep recent history (last 100)
        if len(self.mode_performance[symbol][mode.value]) > 100:
            self.mode_performance[symbol][mode.value] = \
                self.mode_performance[symbol][mode.value][-100:]

    def get_mode_stats(self, symbol: str, mode: TradingMode) -> Dict[str, float]:
        """Get performance statistics for symbol+mode."""
        if symbol not in self.mode_performance:
            return {
                "num_trades": 0,
                "win_rate": 0.5,
                "avg_pnl_bps": 0.0,
                "sharpe": 0.0,
            }

        pnl_list = self.mode_performance[symbol].get(mode.value, [])

        if not pnl_list:
            return {
                "num_trades": 0,
                "win_rate": 0.5,
                "avg_pnl_bps": 0.0,
                "sharpe": 0.0,
            }

        import numpy as np

        pnl_array = np.array(pnl_list)
        num_wins = np.sum(pnl_array > 0)
        win_rate = num_wins / len(pnl_array)
        avg_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array) if len(pnl_array) > 1 else 1.0
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0

        return {
            "num_trades": len(pnl_array),
            "win_rate": win_rate,
            "avg_pnl_bps": avg_pnl,
            "sharpe": sharpe,
        }

    def add_profile(self, profile: AssetProfile) -> None:
        """Add or update a profile."""
        self.profiles[profile.symbol] = profile
        logger.info("asset_profile_added", symbol=profile.symbol, mode=profile.mode.value)

    def remove_profile(self, symbol: str) -> None:
        """Remove a profile."""
        if symbol in self.profiles and symbol != "DEFAULT":
            del self.profiles[symbol]
            logger.info("asset_profile_removed", symbol=symbol)

    def get_all_symbols(self, mode: Optional[TradingMode] = None) -> List[str]:
        """
        Get all symbols (optionally filtered by mode).

        Args:
            mode: Filter by mode (None = all symbols)

        Returns:
            List of symbols
        """
        if mode is None:
            return [s for s in self.profiles.keys() if s != "DEFAULT"]

        symbols = []
        for symbol, profile in self.profiles.items():
            if symbol == "DEFAULT":
                continue

            if mode == TradingMode.SHORT_HOLD and self.can_run_short_hold(symbol):
                symbols.append(symbol)
            elif mode == TradingMode.LONG_HOLD and self.can_run_long_hold(symbol):
                symbols.append(symbol)
            elif mode == TradingMode.BOTH and profile.mode == TradingMode.BOTH:
                symbols.append(symbol)

        return symbols
