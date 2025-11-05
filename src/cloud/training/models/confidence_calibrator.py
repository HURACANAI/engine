"""
Confidence Calibrator

Self-adjusting confidence thresholds based on recent performance.
Prevents overconfidence during losing streaks and capitalizes during winning streaks.

Key Philosophy:
- Confidence thresholds should adapt to recent performance
- Losing streak = raise bar (be more selective)
- Winning streak = lower bar slightly (capitalize on hot hand)
- Regime-specific calibration (what works in TREND may not work in RANGE)
- Never disable learning, just adjust selectivity

The Calibration Problem:
    Static confidence threshold (e.g., 0.65) doesn't account for:
    - Market conditions changing
    - Engine temporarily miscalibrated
    - Losing/winning streaks indicating model drift

Example:
    Week 1: 8 wins, 2 losses (80% win rate) with 0.65 threshold
    → Calibrator: "Model is well-calibrated, can lower threshold to 0.62"
    → Take more trades, capture more opportunities

    Week 2: 3 wins, 7 losses (30% win rate) with 0.62 threshold
    → Calibrator: "Model is miscalibrated, raise threshold to 0.70"
    → Become more selective, avoid bad trades until recalibration
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceWindow:
    """Performance metrics over a time window."""

    wins: int
    losses: int
    total_trades: int
    win_rate: float
    avg_confidence: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float
    sharpe_estimate: float


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""

    current_threshold: float
    recommended_threshold: float
    adjustment: float
    reason: str
    performance_summary: str
    regime_specific: bool
    confidence_level: str  # 'WELL_CALIBRATED', 'OVERCONFIDENT', 'UNDERCONFIDENT'
    action: str  # 'RAISE_BAR', 'LOWER_BAR', 'MAINTAIN'


class ConfidenceCalibrator:
    """
    Self-adjusting confidence threshold system.

    The calibrator monitors recent trading performance and adjusts the
    confidence threshold to maintain optimal trade quality. It acts as
    a "thermostat" for trade selectivity.

    Calibration Logic:

    WELL CALIBRATED (60-70% win rate):
    - Current threshold is good
    - Make small adjustments based on profit factor

    OVERCONFIDENT (< 50% win rate):
    - Model is taking bad trades
    - RAISE threshold by 0.05-0.10
    - Become more selective until recalibrated

    UNDERCONFIDENT (> 75% win rate):
    - Model is too conservative
    - LOWER threshold by 0.02-0.05
    - Capture more opportunities

    Regime-Specific Calibration:
    - Each regime (TREND/RANGE/PANIC) has separate calibration
    - What works in TREND may not work in RANGE
    - Track performance separately per regime

    Usage:
        calibrator = ConfidenceCalibrator()

        # Update with recent trades
        for trade in recent_trades:
            calibrator.record_trade(
                won=trade.won,
                confidence=trade.entry_confidence,
                pnl_bps=trade.pnl_bps,
                regime=trade.entry_regime,
            )

        # Get calibrated threshold
        result = calibrator.get_calibrated_threshold(
            current_threshold=0.65,
            regime='trend',
            lookback_trades=50,
        )

        if result.action == 'RAISE_BAR':
            new_threshold = result.recommended_threshold
            logger.info("Raising confidence bar", reason=result.reason)
    """

    def __init__(
        self,
        target_win_rate: float = 0.60,
        target_profit_factor: float = 2.0,
        overconfident_threshold: float = 0.50,
        underconfident_threshold: float = 0.75,
        max_adjustment_per_calibration: float = 0.10,
        min_trades_for_calibration: int = 30,
        calibration_ema_alpha: float = 0.3,
    ):
        """
        Initialize confidence calibrator.

        Args:
            target_win_rate: Target win rate (60%)
            target_profit_factor: Target profit factor (2.0)
            overconfident_threshold: Win rate below this triggers raise
            underconfident_threshold: Win rate above this triggers lower
            max_adjustment_per_calibration: Maximum threshold adjustment
            min_trades_for_calibration: Minimum trades needed for calibration
            calibration_ema_alpha: EMA alpha for smoothing adjustments
        """
        self.target_win_rate = target_win_rate
        self.target_profit_factor = target_profit_factor
        self.overconfident_threshold = overconfident_threshold
        self.underconfident_threshold = underconfident_threshold
        self.max_adjustment = max_adjustment_per_calibration
        self.min_trades = min_trades_for_calibration
        self.ema_alpha = calibration_ema_alpha

        # Track trades by regime
        self.trade_history: Dict[str, List[Dict]] = {
            'trend': [],
            'range': [],
            'panic': [],
            'unknown': [],
        }

        # Current threshold adjustments by regime
        self.threshold_adjustments: Dict[str, float] = {
            'trend': 0.0,
            'range': 0.0,
            'panic': 0.0,
            'unknown': 0.0,
        }

        logger.info(
            "confidence_calibrator_initialized",
            target_win_rate=target_win_rate,
            target_profit_factor=target_profit_factor,
            min_trades=min_trades_for_calibration,
        )

    def record_trade(
        self,
        won: bool,
        confidence: float,
        pnl_bps: float,
        regime: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a trade for calibration analysis.

        Args:
            won: Whether trade was a winner
            confidence: Entry confidence (0-1)
            pnl_bps: P&L in basis points
            regime: Market regime during trade
            timestamp: Trade timestamp (optional)
        """
        regime_key = regime.lower() if regime else 'unknown'
        if regime_key not in self.trade_history:
            regime_key = 'unknown'

        trade_record = {
            'won': won,
            'confidence': confidence,
            'pnl_bps': pnl_bps,
            'timestamp': timestamp or 0.0,
        }

        self.trade_history[regime_key].append(trade_record)

        # Keep only last 200 trades per regime to prevent memory bloat
        if len(self.trade_history[regime_key]) > 200:
            self.trade_history[regime_key] = self.trade_history[regime_key][-200:]

    def get_calibrated_threshold(
        self,
        current_threshold: float,
        regime: Optional[str] = None,
        lookback_trades: int = 50,
    ) -> CalibrationResult:
        """
        Get calibrated confidence threshold based on recent performance.

        Args:
            current_threshold: Current confidence threshold
            regime: Specific regime to calibrate for (None = all regimes)
            lookback_trades: Number of recent trades to analyze

        Returns:
            CalibrationResult with recommended threshold adjustment
        """
        # Get performance window
        performance = self._calculate_performance_window(
            regime=regime,
            lookback_trades=lookback_trades,
        )

        # Check if we have enough data
        if performance.total_trades < self.min_trades:
            return CalibrationResult(
                current_threshold=current_threshold,
                recommended_threshold=current_threshold,
                adjustment=0.0,
                reason=f"Insufficient data ({performance.total_trades} trades, need {self.min_trades})",
                performance_summary=self._format_performance(performance),
                regime_specific=(regime is not None),
                confidence_level='UNKNOWN',
                action='MAINTAIN',
            )

        # Determine calibration state
        confidence_level = self._classify_calibration(performance.win_rate)

        # Calculate threshold adjustment
        adjustment, reason = self._calculate_adjustment(
            performance=performance,
            confidence_level=confidence_level,
            current_threshold=current_threshold,
        )

        # Apply EMA smoothing to adjustment
        regime_key = regime.lower() if regime else 'unknown'
        if regime_key in self.threshold_adjustments:
            smoothed_adjustment = (
                self.ema_alpha * adjustment +
                (1 - self.ema_alpha) * self.threshold_adjustments[regime_key]
            )
        else:
            smoothed_adjustment = adjustment

        # Cap adjustment at maximum
        smoothed_adjustment = np.clip(
            smoothed_adjustment,
            -self.max_adjustment,
            self.max_adjustment,
        )

        # Store smoothed adjustment
        if regime_key in self.threshold_adjustments:
            self.threshold_adjustments[regime_key] = smoothed_adjustment

        recommended_threshold = current_threshold + smoothed_adjustment

        # Determine action
        if abs(smoothed_adjustment) < 0.01:
            action = 'MAINTAIN'
        elif smoothed_adjustment > 0:
            action = 'RAISE_BAR'
        else:
            action = 'LOWER_BAR'

        logger.info(
            "confidence_calibrated",
            regime=regime or 'all',
            win_rate=performance.win_rate,
            confidence_level=confidence_level,
            adjustment=smoothed_adjustment,
            action=action,
        )

        return CalibrationResult(
            current_threshold=current_threshold,
            recommended_threshold=recommended_threshold,
            adjustment=smoothed_adjustment,
            reason=reason,
            performance_summary=self._format_performance(performance),
            regime_specific=(regime is not None),
            confidence_level=confidence_level,
            action=action,
        )

    def _calculate_performance_window(
        self,
        regime: Optional[str],
        lookback_trades: int,
    ) -> PerformanceWindow:
        """Calculate performance metrics over lookback window."""

        # Get relevant trades
        if regime:
            regime_key = regime.lower()
            if regime_key not in self.trade_history:
                regime_key = 'unknown'
            trades = self.trade_history[regime_key][-lookback_trades:]
        else:
            # Combine all regimes
            trades = []
            for regime_trades in self.trade_history.values():
                trades.extend(regime_trades)
            trades = trades[-lookback_trades:]

        if not trades:
            return PerformanceWindow(
                wins=0, losses=0, total_trades=0, win_rate=0.0,
                avg_confidence=0.0, avg_win_bps=0.0, avg_loss_bps=0.0,
                profit_factor=0.0, sharpe_estimate=0.0,
            )

        # Calculate metrics
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        win_rate = wins / len(trades) if len(trades) > 0 else 0.0

        avg_confidence = np.mean([t['confidence'] for t in trades])

        winners = [t['pnl_bps'] for t in trades if t['won']]
        losers = [abs(t['pnl_bps']) for t in trades if not t['won']]

        avg_win_bps = np.mean(winners) if winners else 0.0
        avg_loss_bps = np.mean(losers) if losers else 0.0

        # Profit factor
        total_wins = sum(winners) if winners else 0.0
        total_losses = sum(losers) if losers else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Sharpe estimate (simplified)
        pnls = [t['pnl_bps'] for t in trades]
        sharpe_estimate = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0

        return PerformanceWindow(
            wins=wins,
            losses=losses,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_confidence=avg_confidence,
            avg_win_bps=avg_win_bps,
            avg_loss_bps=avg_loss_bps,
            profit_factor=profit_factor,
            sharpe_estimate=sharpe_estimate,
        )

    def _classify_calibration(self, win_rate: float) -> str:
        """Classify calibration state based on win rate."""
        if win_rate < self.overconfident_threshold:
            return 'OVERCONFIDENT'
        elif win_rate > self.underconfident_threshold:
            return 'UNDERCONFIDENT'
        else:
            return 'WELL_CALIBRATED'

    def _calculate_adjustment(
        self,
        performance: PerformanceWindow,
        confidence_level: str,
        current_threshold: float,
    ) -> Tuple[float, str]:
        """Calculate threshold adjustment based on performance."""

        if confidence_level == 'OVERCONFIDENT':
            # Model taking bad trades - raise bar
            win_rate_deficit = self.target_win_rate - performance.win_rate
            adjustment = win_rate_deficit * 0.20  # Scale to adjustment

            # Extra penalty if profit factor is also bad
            if performance.profit_factor < 1.5:
                adjustment += 0.02

            reason = f"Win rate {performance.win_rate:.0%} below target {self.target_win_rate:.0%} - raising bar to be more selective"

            return adjustment, reason

        elif confidence_level == 'UNDERCONFIDENT':
            # Model too conservative - lower bar slightly
            win_rate_excess = performance.win_rate - self.target_win_rate
            adjustment = -win_rate_excess * 0.15  # Scale to adjustment (negative = lower)

            # Don't lower too much if profit factor is weak
            if performance.profit_factor < self.target_profit_factor:
                adjustment = adjustment * 0.5  # Reduce lowering

            reason = f"Win rate {performance.win_rate:.0%} above target {self.target_win_rate:.0%} - lowering bar to capture more opportunities"

            return adjustment, reason

        else:  # WELL_CALIBRATED
            # Fine-tune based on profit factor
            if performance.profit_factor < 1.5:
                # Winning enough but wins are small - raise bar slightly
                adjustment = 0.02
                reason = f"Win rate good ({performance.win_rate:.0%}) but profit factor low ({performance.profit_factor:.2f}) - raising bar slightly"
            elif performance.profit_factor > 2.5:
                # Winning with great profit factor - can lower bar slightly
                adjustment = -0.02
                reason = f"Excellent profit factor ({performance.profit_factor:.2f}) - lowering bar slightly to capture more opportunities"
            else:
                # Perfect - maintain
                adjustment = 0.0
                reason = f"Well calibrated: {performance.win_rate:.0%} win rate, {performance.profit_factor:.2f} profit factor - maintain threshold"

            return adjustment, reason

    def _format_performance(self, performance: PerformanceWindow) -> str:
        """Format performance summary."""
        return (
            f"Trades: {performance.total_trades} | "
            f"Win Rate: {performance.win_rate:.0%} | "
            f"Avg Win: +{performance.avg_win_bps:.0f} bps | "
            f"Avg Loss: -{performance.avg_loss_bps:.0f} bps | "
            f"Profit Factor: {performance.profit_factor:.2f} | "
            f"Sharpe: {performance.sharpe_estimate:.2f}"
        )

    def get_regime_calibrations(self) -> Dict[str, CalibrationResult]:
        """Get calibration for all regimes."""
        results = {}

        for regime in ['trend', 'range', 'panic']:
            if len(self.trade_history[regime]) >= self.min_trades:
                # Use base threshold of 0.65
                result = self.get_calibrated_threshold(
                    current_threshold=0.65,
                    regime=regime,
                    lookback_trades=50,
                )
                results[regime] = result

        return results

    def reset_calibration(self, regime: Optional[str] = None) -> None:
        """Reset calibration history."""
        if regime:
            regime_key = regime.lower()
            if regime_key in self.trade_history:
                self.trade_history[regime_key] = []
                self.threshold_adjustments[regime_key] = 0.0
                logger.info("calibration_reset", regime=regime)
        else:
            # Reset all
            for regime_key in self.trade_history:
                self.trade_history[regime_key] = []
                self.threshold_adjustments[regime_key] = 0.0
            logger.info("calibration_reset_all")

    def get_statistics(self) -> Dict[str, any]:
        """Get calibration statistics."""
        stats = {}

        for regime, trades in self.trade_history.items():
            if trades:
                performance = self._calculate_performance_window(
                    regime=regime,
                    lookback_trades=len(trades),
                )
                stats[regime] = {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'profit_factor': performance.profit_factor,
                    'sharpe': performance.sharpe_estimate,
                    'threshold_adjustment': self.threshold_adjustments[regime],
                }

        return stats
