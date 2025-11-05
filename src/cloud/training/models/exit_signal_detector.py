"""
Exit Signal Detection System

Monitors multiple exit conditions with priority levels to exit trades BEFORE stop loss
when danger signals appear. This prevents giving back profits or taking large losses.

Exit Priority Levels:
- P1 (DANGER): Immediate exit required - momentum reversal, regime shift to panic
- P2 (WARNING): Strong exit signal - volume climax, indicator divergence
- P3 (PROFIT): Take profit opportunity - overbought with profit target reached

Key Philosophy:
- Exit BEFORE problems become losses
- Multiple independent exit triggers
- Priority-based decision making
- Better to exit early than late

Example:
    Position: Long ETH at $2000, currently $2015 (+75 bps profit)
    Engine detects: Momentum turning negative (derivatives declining)
    → P1 DANGER signal → EXIT immediately at $2015
    → Price then drops to $1990 (saved 125 bps!)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ExitPriority(Enum):
    """Exit signal priority levels."""

    DANGER = 1  # P1: Exit immediately
    WARNING = 2  # P2: Strong exit signal
    PROFIT = 3  # P3: Take profit opportunity
    NONE = 99  # No exit signal


@dataclass
class ExitSignal:
    """Exit signal with priority and reasoning."""

    priority: ExitPriority
    reason: str  # Exit reason identifier
    description: str  # Human-readable description
    confidence: float  # 0-1, confidence in this exit signal


class ExitSignalDetector:
    """
    Detects exit signals with priority-based decision making.

    Checks multiple exit conditions:
    1. Momentum reversal (P1)
    2. Regime shift to panic with profit (P1)
    3. Volume climax/exhaustion (P2)
    4. Price vs indicator divergence (P2)
    5. Overbought + profit target (P3)
    6. Time-based exits (P3)

    Usage:
        detector = ExitSignalDetector()
        signal = detector.check_exit_signals(
            position=current_position,
            current_features=features,
            current_regime='trend',
        )

        if signal.priority == ExitPriority.DANGER:
            # Exit immediately!
            exit_position()
    """

    def __init__(
        self,
        momentum_reversal_threshold: float = -0.2,
        volume_climax_threshold: float = 2.5,
        divergence_threshold: float = 0.15,
        overbought_rsi_threshold: float = 75.0,
        oversold_rsi_threshold: float = 25.0,
        profit_target_min: float = 100.0,
    ):
        """
        Initialize exit signal detector.

        Args:
            momentum_reversal_threshold: Momentum below this = reversal
            volume_climax_threshold: Volume ratio indicating climax
            divergence_threshold: Price vs indicator divergence threshold
            overbought_rsi_threshold: RSI above this = overbought
            oversold_rsi_threshold: RSI below this = oversold
            profit_target_min: Minimum profit (bps) for profit-taking exits
        """
        self.momentum_reversal_threshold = momentum_reversal_threshold
        self.volume_climax_threshold = volume_climax_threshold
        self.divergence_threshold = divergence_threshold
        self.overbought_rsi = overbought_rsi_threshold
        self.oversold_rsi = oversold_rsi_threshold
        self.profit_target_min = profit_target_min

        logger.info("exit_signal_detector_initialized")

    def check_exit_signals(
        self,
        position_pnl_bps: float,
        position_direction: str,
        position_age_minutes: int,
        entry_regime: str,
        current_regime: str,
        current_features: Dict[str, float],
        max_hold_minutes: Optional[int] = None,
    ) -> Optional[ExitSignal]:
        """
        Check all exit conditions and return highest priority signal.

        Args:
            position_pnl_bps: Current P&L in basis points
            position_direction: 'buy' or 'sell'
            position_age_minutes: How long position has been held
            entry_regime: Regime when position was entered
            current_regime: Current market regime
            current_features: Current market features
            max_hold_minutes: Maximum hold time (optional)

        Returns:
            ExitSignal if exit condition met, None otherwise
        """
        exit_signals = []

        # P1 DANGER: Momentum Reversal
        momentum_signal = self._check_momentum_reversal(
            features=current_features,
            direction=position_direction,
        )
        if momentum_signal:
            exit_signals.append(momentum_signal)

        # P1 DANGER: Regime shift to panic with profit
        regime_panic_signal = self._check_regime_panic(
            entry_regime=entry_regime,
            current_regime=current_regime,
            pnl_bps=position_pnl_bps,
        )
        if regime_panic_signal:
            exit_signals.append(regime_panic_signal)

        # P2 WARNING: Volume climax
        volume_signal = self._check_volume_climax(
            features=current_features,
            direction=position_direction,
        )
        if volume_signal:
            exit_signals.append(volume_signal)

        # P2 WARNING: Divergence
        divergence_signal = self._check_divergence(
            features=current_features,
            direction=position_direction,
        )
        if divergence_signal:
            exit_signals.append(divergence_signal)

        # P3 PROFIT: Overbought/Oversold with profit
        extremes_signal = self._check_extremes_with_profit(
            features=current_features,
            direction=position_direction,
            pnl_bps=position_pnl_bps,
        )
        if extremes_signal:
            exit_signals.append(extremes_signal)

        # P3 PROFIT: Time-based exit
        if max_hold_minutes:
            time_signal = self._check_time_exit(
                age_minutes=position_age_minutes,
                max_minutes=max_hold_minutes,
                pnl_bps=position_pnl_bps,
            )
            if time_signal:
                exit_signals.append(time_signal)

        # Return highest priority signal
        if exit_signals:
            exit_signals.sort(key=lambda x: x.priority.value)  # Sort by priority (lower = higher priority)
            highest_priority = exit_signals[0]

            logger.info(
                "exit_signal_detected",
                priority=highest_priority.priority.name,
                reason=highest_priority.reason,
                pnl_bps=position_pnl_bps,
            )

            return highest_priority

        return None

    def _check_momentum_reversal(
        self,
        features: Dict[str, float],
        direction: str,
    ) -> Optional[ExitSignal]:
        """
        Check for momentum reversal (P1 DANGER).

        For longs: Momentum turning negative
        For shorts: Momentum turning positive
        """
        momentum = features.get('momentum_slope', 0.0)
        momentum_change = features.get('momentum_change', 0.0)  # Recent change in momentum

        if direction == 'buy':
            # Long position: exit if momentum turning bearish
            if momentum < self.momentum_reversal_threshold:
                return ExitSignal(
                    priority=ExitPriority.DANGER,
                    reason="MOMENTUM_REVERSAL",
                    description=f"Momentum turned negative ({momentum:.2f}) - exit long",
                    confidence=min(abs(momentum) / 0.5, 1.0),
                )

        elif direction == 'sell':
            # Short position: exit if momentum turning bullish
            if momentum > abs(self.momentum_reversal_threshold):
                return ExitSignal(
                    priority=ExitPriority.DANGER,
                    reason="MOMENTUM_REVERSAL",
                    description=f"Momentum turned positive ({momentum:.2f}) - exit short",
                    confidence=min(abs(momentum) / 0.5, 1.0),
                )

        return None

    def _check_regime_panic(
        self,
        entry_regime: str,
        current_regime: str,
        pnl_bps: float,
    ) -> Optional[ExitSignal]:
        """
        Check for regime shift to panic with profit (P1 DANGER).

        If regime shifts to panic and we have profit, take it immediately.
        """
        if current_regime == 'panic' and entry_regime != 'panic' and pnl_bps > 0:
            return ExitSignal(
                priority=ExitPriority.DANGER,
                reason="REGIME_PANIC_WITH_PROFIT",
                description=f"Regime shifted to PANIC with +{pnl_bps:.0f} bps profit - exit now",
                confidence=0.9,
            )

        return None

    def _check_volume_climax(
        self,
        features: Dict[str, float],
        direction: str,
    ) -> Optional[ExitSignal]:
        """
        Check for volume climax/exhaustion (P2 WARNING).

        Massive volume spike often indicates exhaustion.
        """
        volume_ratio = features.get('volume_ratio', 1.0)  # Current / avg volume

        if volume_ratio >= self.volume_climax_threshold:
            return ExitSignal(
                priority=ExitPriority.WARNING,
                reason="VOLUME_CLIMAX",
                description=f"Volume climax detected ({volume_ratio:.1f}x avg) - possible exhaustion",
                confidence=min((volume_ratio - self.volume_climax_threshold) / 2.0, 1.0),
            )

        return None

    def _check_divergence(
        self,
        features: Dict[str, float],
        direction: str,
    ) -> Optional[ExitSignal]:
        """
        Check for price vs indicator divergence (P2 WARNING).

        For longs: Price making new highs but indicators not confirming
        For shorts: Price making new lows but indicators not confirming
        """
        # Simplified divergence check using RSI and momentum
        rsi = features.get('rsi', 50.0)
        momentum = features.get('momentum_slope', 0.0)

        if direction == 'buy':
            # Long: bearish divergence (price up, indicators down)
            if rsi < 50 and momentum < 0:
                divergence_strength = (50 - rsi) / 50.0
                if divergence_strength >= self.divergence_threshold:
                    return ExitSignal(
                        priority=ExitPriority.WARNING,
                        reason="BEARISH_DIVERGENCE",
                        description=f"Bearish divergence detected (RSI {rsi:.0f}, momentum {momentum:.2f})",
                        confidence=divergence_strength,
                    )

        elif direction == 'sell':
            # Short: bullish divergence (price down, indicators up)
            if rsi > 50 and momentum > 0:
                divergence_strength = (rsi - 50) / 50.0
                if divergence_strength >= self.divergence_threshold:
                    return ExitSignal(
                        priority=ExitPriority.WARNING,
                        reason="BULLISH_DIVERGENCE",
                        description=f"Bullish divergence detected (RSI {rsi:.0f}, momentum {momentum:.2f})",
                        confidence=divergence_strength,
                    )

        return None

    def _check_extremes_with_profit(
        self,
        features: Dict[str, float],
        direction: str,
        pnl_bps: float,
    ) -> Optional[ExitSignal]:
        """
        Check for overbought/oversold with profit (P3 PROFIT).

        Take profits when at extreme levels.
        """
        rsi = features.get('rsi', 50.0)

        if pnl_bps < self.profit_target_min:
            return None  # Only relevant when in profit

        if direction == 'buy':
            # Long: overbought with profit
            if rsi >= self.overbought_rsi:
                return ExitSignal(
                    priority=ExitPriority.PROFIT,
                    reason="OVERBOUGHT_PROFIT_TAKE",
                    description=f"Overbought (RSI {rsi:.0f}) with +{pnl_bps:.0f} bps profit - take profit",
                    confidence=(rsi - self.overbought_rsi) / 25.0,
                )

        elif direction == 'sell':
            # Short: oversold with profit
            if rsi <= self.oversold_rsi:
                return ExitSignal(
                    priority=ExitPriority.PROFIT,
                    reason="OVERSOLD_PROFIT_TAKE",
                    description=f"Oversold (RSI {rsi:.0f}) with +{pnl_bps:.0f} bps profit - take profit",
                    confidence=(self.oversold_rsi - rsi) / 25.0,
                )

        return None

    def _check_time_exit(
        self,
        age_minutes: int,
        max_minutes: int,
        pnl_bps: float,
    ) -> Optional[ExitSignal]:
        """
        Check for time-based exit (P3 PROFIT).

        Exit if held too long, regardless of profit.
        """
        if age_minutes >= max_minutes:
            if pnl_bps > 0:
                return ExitSignal(
                    priority=ExitPriority.PROFIT,
                    reason="TIME_LIMIT_WITH_PROFIT",
                    description=f"Max hold time reached ({age_minutes}min) with +{pnl_bps:.0f} bps - take profit",
                    confidence=0.7,
                )
            else:
                return ExitSignal(
                    priority=ExitPriority.WARNING,
                    reason="TIME_LIMIT_CUT_LOSS",
                    description=f"Max hold time reached ({age_minutes}min) with {pnl_bps:.0f} bps - cut position",
                    confidence=0.8,
                )

        return None

    def get_exit_priority_name(self, priority: ExitPriority) -> str:
        """Get human-readable priority name."""
        names = {
            ExitPriority.DANGER: "DANGER (P1)",
            ExitPriority.WARNING: "WARNING (P2)",
            ExitPriority.PROFIT: "PROFIT (P3)",
            ExitPriority.NONE: "NONE",
        }
        return names.get(priority, "UNKNOWN")
