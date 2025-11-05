"""
Safety Rails for Long-Hold Mode

Prevents "bag-holding forever" by applying fail-safes:
1. Per-asset max drawdown (floating) - force reduce if thesis breaks
2. Time stops - exit if thesis hasn't resolved
3. Feature drift kill - exit if signals lose predictive power
4. Event guards - clamp adds and tighten trail on vol spikes

Philosophy: Ride healthy dips, but don't hold structural breaks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .dual_book_manager import Position
from .mode_policies import SignalContext

logger = structlog.get_logger(__name__)


@dataclass
class SafetyRailConfig:
    """Configuration for safety rails."""

    # Drawdown limits
    max_floating_dd_bps: float = 500.0  # Max adverse move
    dd_regime_check: bool = True  # Check regime on DD
    dd_htf_check: bool = True  # Check HTF bias on DD

    # Time stops
    max_hold_days: float = 7.0  # Max holding period
    stall_check_hours: float = 24.0  # Check for stalled positions

    # Feature drift detection
    drift_check_enabled: bool = True
    drift_window_hours: int = 12  # Window for drift calculation
    drift_threshold: float = 0.3  # Max allowed drift

    # Event guards (volatility)
    vol_spike_threshold: float = 2.0  # Vol ratio threshold
    vol_spike_clamp_adds: bool = True
    vol_spike_tighten_trail: bool = True


@dataclass
class RailViolation:
    """Represents a safety rail violation."""

    rail_type: str  # "drawdown", "time", "drift", "event"
    severity: str  # "warning", "critical"
    message: str
    action: str  # "reduce", "close", "clamp_adds", "tighten_trail"
    timestamp: datetime


class SafetyRailsMonitor:
    """
    Monitors long-hold positions for safety rail violations.

    Responsibilities:
    1. Track floating drawdowns
    2. Detect time-based stalls
    3. Identify feature drift
    4. Monitor volatility events
    5. Recommend actions on violations
    """

    def __init__(self, config: Optional[SafetyRailConfig] = None):
        """
        Initialize safety rails monitor.

        Args:
            config: Safety rail configuration
        """
        self.config = config or SafetyRailConfig()

        # Track history for drift detection
        self.feature_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}

        # Track entry volatility for vol spike detection
        self.entry_volatility: Dict[str, float] = {}  # symbol → entry_vol

        # Track violations
        self.violations: List[RailViolation] = []

        logger.info(
            "safety_rails_monitor_initialized",
            max_dd_bps=self.config.max_floating_dd_bps,
            max_hold_days=self.config.max_hold_days,
        )

    def check_position(
        self,
        position: Position,
        context: SignalContext,
    ) -> List[RailViolation]:
        """
        Check position against all safety rails.

        Args:
            position: Position to check
            context: Current signal context

        Returns:
            List of violations
        """
        violations = []

        # Check 1: Drawdown rail
        dd_violation = self._check_drawdown_rail(position, context)
        if dd_violation:
            violations.append(dd_violation)

        # Check 2: Time rail
        time_violation = self._check_time_rail(position, context)
        if time_violation:
            violations.append(time_violation)

        # Check 3: Feature drift rail
        if self.config.drift_check_enabled:
            drift_violation = self._check_drift_rail(position, context)
            if drift_violation:
                violations.append(drift_violation)

        # Check 4: Event rail (volatility)
        event_violation = self._check_event_rail(position, context)
        if event_violation:
            violations.append(event_violation)

        # Store violations
        self.violations.extend(violations)

        # Keep recent violation history (last 1000)
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]

        return violations

    def _check_drawdown_rail(
        self,
        position: Position,
        context: SignalContext,
    ) -> Optional[RailViolation]:
        """Check max floating drawdown rail."""
        current_dd_bps = position.unrealized_pnl_bps

        # Already in profit, no DD issue
        if current_dd_bps > 0:
            return None

        dd_magnitude = abs(current_dd_bps)

        # Check magnitude
        if dd_magnitude > self.config.max_floating_dd_bps:
            # Critical: exceeded max DD
            severity = "critical"
            action = "close"
            message = f"Max DD exceeded: {current_dd_bps:.1f} bps (limit: -{self.config.max_floating_dd_bps:.1f})"

        elif dd_magnitude > self.config.max_floating_dd_bps * 0.75:
            # Warning: approaching max DD
            # Also check regime and HTF
            regime_broken = context.regime == "panic" and self.config.dd_regime_check
            htf_broken = context.htf_bias < -0.2 and self.config.dd_htf_check

            if regime_broken or htf_broken:
                severity = "critical"
                action = "reduce"
                message = f"DD {current_dd_bps:.1f} bps + regime/HTF break"
            else:
                severity = "warning"
                action = "monitor"
                message = f"Approaching max DD: {current_dd_bps:.1f} bps"
        else:
            return None

        return RailViolation(
            rail_type="drawdown",
            severity=severity,
            message=message,
            action=action,
            timestamp=context.timestamp,
        )

    def _check_time_rail(
        self,
        position: Position,
        context: SignalContext,
    ) -> Optional[RailViolation]:
        """Check time-based rails."""
        age_hours = position.age_hours(context.timestamp)
        age_days = age_hours / 24.0

        # Check 1: Max hold time
        if age_days > self.config.max_hold_days:
            # Check if thesis has resolved
            if position.unrealized_pnl_bps > 100.0:
                # In good profit, allow
                return None

            severity = "critical"
            action = "close"
            message = f"Max hold time exceeded: {age_days:.1f} days (limit: {self.config.max_hold_days})"

            return RailViolation(
                rail_type="time",
                severity=severity,
                message=message,
                action=action,
                timestamp=context.timestamp,
            )

        # Check 2: Stalled position
        if age_hours > self.config.stall_check_hours:
            # Position has been open for a while, check if it's stalled
            # Stalled = not making progress toward profit
            if -50.0 < position.unrealized_pnl_bps < 50.0:  # Stuck near break-even
                severity = "warning"
                action = "reduce"
                message = f"Position stalled: {age_hours:.1f}h at {position.unrealized_pnl_bps:.1f} bps"

                return RailViolation(
                    rail_type="time",
                    severity=severity,
                    message=message,
                    action=action,
                    timestamp=context.timestamp,
                )

        return None

    def _check_drift_rail(
        self,
        position: Position,
        context: SignalContext,
    ) -> Optional[RailViolation]:
        """Check feature drift rail (signal quality degradation)."""
        symbol = position.symbol

        # Track feature history
        if symbol not in self.feature_history:
            self.feature_history[symbol] = []

        # Add current features
        self.feature_history[symbol].append((context.timestamp, context.features))

        # Keep only recent history
        cutoff_time = context.timestamp - timedelta(hours=self.config.drift_window_hours * 2)
        self.feature_history[symbol] = [
            (ts, feats) for ts, feats in self.feature_history[symbol]
            if ts >= cutoff_time
        ]

        # Need enough history to check drift
        if len(self.feature_history[symbol]) < 10:
            return None

        # Calculate drift for key features
        history = self.feature_history[symbol]

        # Split into entry period and recent period
        entry_period = history[:len(history)//2]
        recent_period = history[len(history)//2:]

        if not entry_period or not recent_period:
            return None

        # Calculate average feature values in each period
        key_features = ["confidence", "trend_strength", "ignition_score", "micro_score"]

        drift_scores = []
        for feature_name in key_features:
            # Entry period average
            entry_values = [
                feats.get(feature_name, 0.0)
                for _, feats in entry_period
                if feature_name in feats or feature_name == "confidence"
            ]

            # Recent period average
            recent_values = [
                feats.get(feature_name, 0.0)
                for _, feats in recent_period
                if feature_name in feats or feature_name == "confidence"
            ]

            if not entry_values or not recent_values:
                continue

            entry_avg = np.mean(entry_values)
            recent_avg = np.mean(recent_values)

            # Calculate drift (normalized)
            if abs(entry_avg) > 0.1:
                drift = abs(recent_avg - entry_avg) / abs(entry_avg)
                drift_scores.append(drift)

        if not drift_scores:
            return None

        # Average drift
        avg_drift = np.mean(drift_scores)

        # Check if drift exceeds threshold
        if avg_drift > self.config.drift_threshold:
            severity = "critical" if avg_drift > self.config.drift_threshold * 1.5 else "warning"
            action = "close" if severity == "critical" else "reduce"
            message = f"Feature drift detected: {avg_drift:.2f} (threshold: {self.config.drift_threshold})"

            return RailViolation(
                rail_type="drift",
                severity=severity,
                message=message,
                action=action,
                timestamp=context.timestamp,
            )

        return None

    def _check_event_rail(
        self,
        position: Position,
        context: SignalContext,
    ) -> Optional[RailViolation]:
        """Check event rail (volatility spikes)."""
        symbol = position.symbol

        # Track entry volatility
        if symbol not in self.entry_volatility:
            # First time seeing this position, record entry vol
            entry_vol = context.features.get("volatility_bps", context.volatility_bps)
            self.entry_volatility[symbol] = entry_vol

        entry_vol = self.entry_volatility[symbol]
        current_vol = context.volatility_bps

        # Calculate vol ratio
        vol_ratio = current_vol / (entry_vol + 1e-6)

        # Check for vol spike
        if vol_ratio > self.config.vol_spike_threshold:
            severity = "warning"

            actions = []
            if self.config.vol_spike_clamp_adds:
                actions.append("clamp_adds")
            if self.config.vol_spike_tighten_trail:
                actions.append("tighten_trail")

            action = ",".join(actions) if actions else "monitor"

            message = f"Vol spike: {vol_ratio:.2f}x entry vol (entry: {entry_vol:.1f}, current: {current_vol:.1f})"

            return RailViolation(
                rail_type="event",
                severity=severity,
                message=message,
                action=action,
                timestamp=context.timestamp,
            )

        return None

    def on_position_closed(self, symbol: str) -> None:
        """Clean up tracking when position is closed."""
        if symbol in self.feature_history:
            del self.feature_history[symbol]
        if symbol in self.entry_volatility:
            del self.entry_volatility[symbol]

    def get_violation_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of recent violations.

        Args:
            hours: Lookback period in hours

        Returns:
            Violation summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_violations = [
            v for v in self.violations
            if v.timestamp >= cutoff_time
        ]

        # Count by type and severity
        summary = {
            "total": len(recent_violations),
            "by_type": {},
            "by_severity": {},
            "by_action": {},
        }

        for violation in recent_violations:
            # By type
            summary["by_type"][violation.rail_type] = \
                summary["by_type"].get(violation.rail_type, 0) + 1

            # By severity
            summary["by_severity"][violation.severity] = \
                summary["by_severity"].get(violation.severity, 0) + 1

            # By action
            summary["by_action"][violation.action] = \
                summary["by_action"].get(violation.action, 0) + 1

        return summary

    def should_clamp_adds(self, symbol: str) -> bool:
        """Check if adds should be clamped for this symbol."""
        recent_violations = [
            v for v in self.violations[-10:]  # Recent 10
            if v.rail_type == "event" and "clamp_adds" in v.action
        ]

        return len(recent_violations) > 0

    def should_tighten_trail(self, symbol: str) -> bool:
        """Check if trail should be tightened for this symbol."""
        recent_violations = [
            v for v in self.violations[-10:]  # Recent 10
            if v.rail_type in ["event", "drift"] and (
                "tighten_trail" in v.action or v.severity == "warning"
            )
        ]

        return len(recent_violations) > 0
