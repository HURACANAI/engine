"""
Risk Intelligence Systems

Bundled risk management improvements:
1. Panic/Uncertainty Action Masks
2. Triple-Barrier Labels
3. Drift & Engine-Health Penalties

These systems manage risk and adapt to changing conditions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# 1. PANIC/UNCERTAINTY ACTION MASKS
# ============================================================================


class RLAction(Enum):
    """RL policy actions."""

    HOLD = "hold"
    ENTER_SMALL = "enter_small"
    ENTER_MEDIUM = "enter_medium"
    ENTER_LARGE = "enter_large"
    ADD_TO_WINNER = "add_to_winner"
    ADD_TO_LOSER = "add_to_loser"
    EXIT_PARTIAL = "exit_partial"
    EXIT_ALL = "exit_all"


@dataclass
class ActionMask:
    """Action mask for policy."""

    allowed_actions: Set[RLAction]
    blocked_actions: Set[RLAction]
    reason: str


class PanicUncertaintyMasks:
    """
    Mask aggressive actions during PANIC or high uncertainty.

    In PANIC or uncertain states:
    - Block: ENTER_LARGE, ADD_TO_LOSER
    - Allow: ENTER_SMALL, HOLD, EXIT actions

    Usage:
        masker = PanicUncertaintyMasks()

        mask = masker.get_action_mask(
            regime='PANIC',
            uncertainty=0.85,
        )

        # Apply mask to RL policy
        for action in all_actions:
            if action in mask.blocked_actions:
                policy_logits[action] = -inf  # Block this action
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.70,
    ):
        self.uncertainty_threshold = uncertainty_threshold

    def get_action_mask(
        self,
        regime: str,
        uncertainty: float,
    ) -> ActionMask:
        """
        Get action mask based on regime and uncertainty.

        Args:
            regime: Current regime
            uncertainty: Uncertainty level (0-1)

        Returns:
            ActionMask
        """
        all_actions = set(RLAction)
        blocked = set()
        reason_parts = []

        # 1. PANIC regime masks
        if regime == 'PANIC':
            blocked.add(RLAction.ENTER_LARGE)
            blocked.add(RLAction.ENTER_MEDIUM)
            blocked.add(RLAction.ADD_TO_LOSER)
            blocked.add(RLAction.ADD_TO_WINNER)  # Risky in panic
            reason_parts.append("PANIC regime")

        # 2. High uncertainty masks
        if uncertainty >= self.uncertainty_threshold:
            blocked.add(RLAction.ENTER_LARGE)
            blocked.add(RLAction.ADD_TO_LOSER)
            reason_parts.append(f"High uncertainty ({uncertainty:.2f})")

        allowed = all_actions - blocked

        reason = "Blocked aggressive actions: " + ", ".join(reason_parts) if reason_parts else "All actions allowed"

        return ActionMask(
            allowed_actions=allowed,
            blocked_actions=blocked,
            reason=reason,
        )

    def apply_mask_to_logits(
        self,
        logits: Dict[RLAction, float],
        mask: ActionMask,
    ) -> Dict[RLAction, float]:
        """Apply mask to policy logits."""
        masked_logits = logits.copy()

        for action in mask.blocked_actions:
            if action in masked_logits:
                masked_logits[action] = -np.inf  # Block

        return masked_logits


# ============================================================================
# 2. TRIPLE-BARRIER LABELS
# ============================================================================


@dataclass
class TripleBarrierLabel:
    """Triple-barrier label result."""

    label: int  # 1 = hit TP, -1 = hit SL, 0 = hit time
    exit_reason: str  # 'take_profit', 'stop_loss', 'time_limit'
    bars_to_exit: int
    exit_return_bps: float
    hit_tp: bool
    hit_sl: bool
    hit_time: bool


class TripleBarrierLabeler:
    """
    Label trades using triple-barrier method (TP/SL/Time).

    Instead of just forward return, use:
    - Take Profit barrier (e.g., +100 bps)
    - Stop Loss barrier (e.g., -50 bps)
    - Time barrier (e.g., 50 bars)

    Label = whichever hits first.

    Usage:
        labeler = TripleBarrierLabeler(
            take_profit_bps=100,
            stop_loss_bps=50,
            time_limit_bars=50,
        )

        label = labeler.label_trade(
            entry_price=47000,
            future_prices=[47010, 47020, 47050, 47100, ...],
            direction='long',
        )

        if label.hit_tp:
            print("Hit TP in", label.bars_to_exit, "bars")
        elif label.hit_sl:
            print("Hit SL")
    """

    def __init__(
        self,
        take_profit_bps: float = 100,
        stop_loss_bps: float = 50,
        time_limit_bars: int = 50,
    ):
        self.tp_bps = take_profit_bps
        self.sl_bps = stop_loss_bps
        self.time_limit = time_limit_bars

    def label_trade(
        self,
        entry_price: float,
        future_prices: List[float],
        direction: str,  # 'long' or 'short'
    ) -> TripleBarrierLabel:
        """
        Label trade using triple barrier.

        Args:
            entry_price: Entry price
            future_prices: Future price path
            direction: Trade direction

        Returns:
            TripleBarrierLabel
        """
        for i, price in enumerate(future_prices):
            if i >= self.time_limit:
                # Hit time barrier
                exit_return = self._calculate_return(entry_price, price, direction)
                return TripleBarrierLabel(
                    label=0,
                    exit_reason='time_limit',
                    bars_to_exit=i,
                    exit_return_bps=exit_return,
                    hit_tp=False,
                    hit_sl=False,
                    hit_time=True,
                )

            # Calculate return
            ret_bps = self._calculate_return(entry_price, price, direction)

            # Check TP
            if ret_bps >= self.tp_bps:
                return TripleBarrierLabel(
                    label=1,
                    exit_reason='take_profit',
                    bars_to_exit=i,
                    exit_return_bps=ret_bps,
                    hit_tp=True,
                    hit_sl=False,
                    hit_time=False,
                )

            # Check SL
            if ret_bps <= -self.sl_bps:
                return TripleBarrierLabel(
                    label=-1,
                    exit_reason='stop_loss',
                    bars_to_exit=i,
                    exit_return_bps=ret_bps,
                    hit_tp=False,
                    hit_sl=True,
                    hit_time=False,
                )

        # Hit time limit if we get here
        final_return = self._calculate_return(entry_price, future_prices[-1], direction)
        return TripleBarrierLabel(
            label=0,
            exit_reason='time_limit',
            bars_to_exit=len(future_prices),
            exit_return_bps=final_return,
            hit_tp=False,
            hit_sl=False,
            hit_time=True,
        )

    def _calculate_return(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
    ) -> float:
        """Calculate return in bps."""
        if direction == 'long':
            return (exit_price - entry_price) / entry_price * 10000
        else:
            return (entry_price - exit_price) / entry_price * 10000


# ============================================================================
# 3. DRIFT & ENGINE-HEALTH PENALTIES
# ============================================================================


@dataclass
class EngineHealthReport:
    """Engine health assessment."""

    engine_name: str
    drift_score: float  # 0-1, higher = more drift
    psi_score: float  # Population Stability Index
    ks_score: float  # Kolmogorov-Smirnov statistic
    recent_win_rate: float
    recent_sharpe: float
    health_penalty: float  # 0-1, multiply into confidence
    is_healthy: bool
    should_freeze: bool
    reason: str


class DriftEngineHealthMonitor:
    """
    Monitor engine health and apply drift penalties.

    Monitors:
    1. Feature drift (PSI/KS on top features)
    2. Recent performance (WR, Sharpe in regime)
    3. Sudden drops in quality

    Penalties:
    - Mild drift: 0.9x confidence
    - Moderate drift: 0.7x confidence
    - Severe drift: 0.3x confidence or freeze

    Usage:
        monitor = DriftEngineHealthMonitor()

        # Update with new trades
        monitor.update_engine_performance(
            engine_name='breakout',
            won=True,
            profit_bps=120,
            regime='TREND',
            features={'vol': 1.5, 'momentum': 0.7},
        )

        # Check health before trading
        health = monitor.get_engine_health('breakout', regime='TREND')

        if health.should_freeze:
            logger.warning("Engine frozen", engine=engine_name, reason=health.reason)
            return None

        # Apply penalty
        confidence *= health.health_penalty
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,  # PSI > 0.25 = significant drift
        ks_threshold: float = 0.20,  # KS > 0.20 = distribution shift
        min_wr_threshold: float = 0.45,  # < 45% WR = unhealthy
        min_sharpe_threshold: float = 0.5,  # < 0.5 Sharpe = unhealthy
        freeze_threshold: float = 0.30,  # Health penalty < 0.3 = freeze
        lookback_trades: int = 30,
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.min_wr = min_wr_threshold
        self.min_sharpe = min_sharpe_threshold
        self.freeze_threshold = freeze_threshold
        self.lookback = lookback_trades

        # Historical data per engine
        self.engine_history: Dict[str, List[Dict]] = {}
        self.feature_baselines: Dict[str, Dict[str, np.ndarray]] = {}

    def update_engine_performance(
        self,
        engine_name: str,
        won: bool,
        profit_bps: float,
        regime: str,
        features: Dict[str, float],
    ) -> None:
        """Update engine performance history."""
        if engine_name not in self.engine_history:
            self.engine_history[engine_name] = []

        record = {
            'timestamp': time.time(),
            'won': won,
            'profit_bps': profit_bps,
            'regime': regime,
            'features': features.copy(),
        }

        self.engine_history[engine_name].append(record)

        # Keep only recent history
        if len(self.engine_history[engine_name]) > 200:
            self.engine_history[engine_name] = self.engine_history[engine_name][-200:]

    def get_engine_health(
        self,
        engine_name: str,
        regime: str,
    ) -> EngineHealthReport:
        """
        Get engine health report.

        Args:
            engine_name: Engine to check
            regime: Current regime

        Returns:
            EngineHealthReport
        """
        if engine_name not in self.engine_history:
            # No history - assume healthy but low confidence
            return EngineHealthReport(
                engine_name=engine_name,
                drift_score=0.0,
                psi_score=0.0,
                ks_score=0.0,
                recent_win_rate=0.50,
                recent_sharpe=0.0,
                health_penalty=0.8,
                is_healthy=True,
                should_freeze=False,
                reason="No history, low confidence",
            )

        history = self.engine_history[engine_name]

        # Filter by regime
        regime_history = [r for r in history if r['regime'] == regime]
        recent_history = regime_history[-self.lookback:] if len(regime_history) > self.lookback else regime_history

        if len(recent_history) < 10:
            return EngineHealthReport(
                engine_name=engine_name,
                drift_score=0.0,
                psi_score=0.0,
                ks_score=0.0,
                recent_win_rate=0.50,
                recent_sharpe=0.0,
                health_penalty=0.9,
                is_healthy=True,
                should_freeze=False,
                reason="Insufficient data for regime",
            )

        # 1. Calculate drift scores
        psi_score = self._calculate_psi(engine_name, recent_history)
        ks_score = self._calculate_ks(engine_name, recent_history)
        drift_score = max(psi_score / self.psi_threshold, ks_score / self.ks_threshold)

        # 2. Calculate recent performance
        wins = [r['won'] for r in recent_history]
        profits = [r['profit_bps'] for r in recent_history]

        recent_wr = np.mean(wins)
        if len(profits) > 1:
            recent_sharpe = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0.0
        else:
            recent_sharpe = 0.0

        # 3. Determine health
        issues = []

        if psi_score > self.psi_threshold:
            issues.append(f"PSI {psi_score:.2f} > {self.psi_threshold}")

        if ks_score > self.ks_threshold:
            issues.append(f"KS {ks_score:.2f} > {self.ks_threshold}")

        if recent_wr < self.min_wr:
            issues.append(f"WR {recent_wr:.2%} < {self.min_wr:.0%}")

        if recent_sharpe < self.min_sharpe:
            issues.append(f"Sharpe {recent_sharpe:.2f} < {self.min_sharpe}")

        # Calculate health penalty
        if not issues:
            health_penalty = 1.0
            is_healthy = True
            should_freeze = False
            reason = "Healthy"
        elif len(issues) == 1:
            health_penalty = 0.85
            is_healthy = True
            should_freeze = False
            reason = f"Mild issues: {', '.join(issues)}"
        elif len(issues) == 2:
            health_penalty = 0.65
            is_healthy = False
            should_freeze = False
            reason = f"Moderate issues: {', '.join(issues)}"
        else:
            health_penalty = 0.30
            is_healthy = False
            should_freeze = health_penalty < self.freeze_threshold
            reason = f"Severe issues: {', '.join(issues)}"

        return EngineHealthReport(
            engine_name=engine_name,
            drift_score=drift_score,
            psi_score=psi_score,
            ks_score=ks_score,
            recent_win_rate=recent_wr,
            recent_sharpe=recent_sharpe,
            health_penalty=health_penalty,
            is_healthy=is_healthy,
            should_freeze=should_freeze,
            reason=reason,
        )

    def _calculate_psi(
        self,
        engine_name: str,
        recent_history: List[Dict],
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI measures distribution shift in features.
        """
        # Simplified PSI - just check if feature means have shifted
        # In production, use proper PSI calculation with binning

        if engine_name not in self.feature_baselines:
            # Establish baseline
            self._establish_baseline(engine_name, recent_history)
            return 0.0

        baseline = self.feature_baselines[engine_name]

        # Get recent feature means
        recent_features = [r['features'] for r in recent_history]
        if not recent_features:
            return 0.0

        # Calculate shift for each feature
        shifts = []
        for feature_name in baseline.keys():
            baseline_val = baseline[feature_name]
            recent_vals = [f.get(feature_name, 0) for f in recent_features]
            recent_mean = np.mean(recent_vals)

            # Relative shift
            if baseline_val != 0:
                shift = abs(recent_mean - baseline_val) / abs(baseline_val)
            else:
                shift = abs(recent_mean)

            shifts.append(shift)

        # PSI ~ average relative shift
        psi = np.mean(shifts) if shifts else 0.0

        return min(psi, 1.0)

    def _calculate_ks(
        self,
        engine_name: str,
        recent_history: List[Dict],
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic (simplified)."""
        # Simplified KS - return 0 for now
        # In production, use scipy.stats.ks_2samp
        return 0.0

    def _establish_baseline(
        self,
        engine_name: str,
        history: List[Dict],
    ) -> None:
        """Establish feature baseline for drift detection."""
        if not history:
            return

        # Calculate baseline feature means
        all_features = [r['features'] for r in history]
        feature_names = set()
        for f in all_features:
            feature_names.update(f.keys())

        baseline = {}
        for feature_name in feature_names:
            values = [f.get(feature_name, 0) for f in all_features]
            baseline[feature_name] = np.mean(values)

        self.feature_baselines[engine_name] = baseline

        logger.info(
            "baseline_established",
            engine=engine_name,
            features=list(feature_names),
        )
