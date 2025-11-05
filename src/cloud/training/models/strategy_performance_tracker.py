"""
Strategy Performance Tracker

Monitors real-time performance of each trading strategy (technique x regime)
and automatically disables underperforming strategies.

Key Problems Solved:
1. **Blind Execution**: RANGE engine loses 70% of trades in TREND regime, but bot keeps using it
2. **No Accountability**: Can't answer "which strategies are profitable right now?"
3. **Strategy Drift**: BREAKOUT engine was 68% win rate last month, now 35% (market changed)

Solution: Real-Time Strategy Scorecards
- Track win rate, profit factor, Sharpe ratio per strategy
- Monitor recent performance (last 20/50/100 trades)
- Auto-disable strategies below threshold
- Alert when strategy degrades

Example Output:
    STRATEGY SCORECARD (Last 50 Trades):

    TREND in TREND regime:
      Win Rate: 72% ✅ (36W/14L)
      Profit Factor: 2.8 ✅
      Avg Win: +195 bps | Avg Loss: -78 bps
      Status: ENABLED
      Recommendation: STRONG - Continue using

    RANGE in TREND regime:
      Win Rate: 32% ❌ (16W/34L)
      Profit Factor: 0.9 ❌ (losing money!)
      Avg Win: +145 bps | Avg Loss: -92 bps
      Status: DISABLED (underperforming)
      Recommendation: AVOID - Mean reversion fails in trends

    BREAKOUT in RANGE regime:
      Win Rate: 48% ⚠️ (24W/26L)
      Profit Factor: 1.3 ⚠️
      Avg Win: +210 bps | Avg Loss: -105 bps
      Status: ENABLED (borderline)
      Recommendation: MONITOR - Below target but not critical
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class StrategyStatus(Enum):
    """Strategy operational status."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    MONITORING = "monitoring"  # Borderline performance


class PerformanceLevel(Enum):
    """Performance classification."""

    EXCELLENT = "excellent"  # Top performer
    GOOD = "good"  # Above target
    ACCEPTABLE = "acceptable"  # Meeting minimum
    POOR = "poor"  # Below minimum
    CRITICAL = "critical"  # Losing money


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy."""

    # Identity
    technique: str  # 'trend', 'range', 'breakout', etc.
    regime: str  # 'trend', 'range', 'panic'
    strategy_key: str  # Combined key: "trend_in_trend"

    # Trade counts
    total_trades: int
    wins: int
    losses: int
    win_rate: float

    # P&L metrics
    avg_pnl_bps: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float  # Total wins / Total losses
    sharpe_ratio: float

    # Recent performance (last 20 trades)
    recent_win_rate: float
    recent_profit_factor: float

    # Status
    status: StrategyStatus
    performance_level: PerformanceLevel
    recommendation: str
    warnings: List[str]

    # Timestamps
    first_trade_timestamp: float
    last_trade_timestamp: float


@dataclass
class StrategyAlert:
    """Alert for strategy performance change."""

    strategy_key: str
    alert_type: str  # 'DEGRADATION', 'IMPROVEMENT', 'CRITICAL'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    old_win_rate: float
    new_win_rate: float
    timestamp: float


class StrategyPerformanceTracker:
    """
    Tracks and manages performance of trading strategies.

    Each strategy is defined by:
    - Technique: TREND, RANGE, BREAKOUT, TAPE, LEADER, SWEEP
    - Regime: TREND, RANGE, PANIC

    This creates 18 strategy combinations (6 techniques × 3 regimes)

    The tracker:
    1. Records every trade by strategy
    2. Calculates performance metrics per strategy
    3. Compares against thresholds
    4. Auto-disables underperforming strategies
    5. Alerts on performance degradation

    Usage:
        tracker = StrategyPerformanceTracker()

        # Record completed trade
        tracker.record_trade(
            technique='trend',
            regime='trend',
            won=True,
            pnl_bps=185.0,
            timestamp=time.time(),
        )

        # Check if strategy should be used
        should_use = tracker.should_use_strategy(
            technique='range',
            regime='trend',
        )

        if not should_use:
            logger.warning("strategy_disabled", technique='range', regime='trend')
            skip_trade()

        # Get performance report
        metrics = tracker.get_strategy_metrics('trend', 'trend')
        print(f"TREND in TREND: {metrics.win_rate:.0%} win rate")

        # Periodic review (every 100 trades)
        if total_trades % 100 == 0:
            report = tracker.generate_report()
            logger.info("strategy_performance_report", report=report)
    """

    def __init__(
        self,
        min_win_rate: float = 0.50,
        min_profit_factor: float = 1.5,
        min_trades_to_evaluate: int = 20,
        recent_trades_window: int = 20,
        auto_disable: bool = True,
    ):
        """
        Initialize strategy performance tracker.

        Args:
            min_win_rate: Minimum acceptable win rate (default: 50%)
            min_profit_factor: Minimum acceptable profit factor (default: 1.5)
            min_trades_to_evaluate: Minimum trades before evaluation
            recent_trades_window: Window for recent performance (default: 20)
            auto_disable: Automatically disable underperforming strategies
        """
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_trades = min_trades_to_evaluate
        self.recent_window = recent_trades_window
        self.auto_disable = auto_disable

        # Trade history per strategy
        # Key: "technique_regime" (e.g., "trend_trend", "range_panic")
        # Value: List of (won, pnl_bps, timestamp) tuples
        self.strategy_trades: Dict[str, List[Tuple[bool, float, float]]] = {}

        # Strategy status
        self.strategy_status: Dict[str, StrategyStatus] = {}

        # Alerts
        self.alerts: List[StrategyAlert] = []

        # Cache metrics
        self.cached_metrics: Dict[str, StrategyMetrics] = {}

        logger.info(
            "strategy_tracker_initialized",
            min_win_rate=min_win_rate,
            min_profit_factor=min_profit_factor,
            auto_disable=auto_disable,
        )

    def record_trade(
        self,
        technique: str,
        regime: str,
        won: bool,
        pnl_bps: float,
        timestamp: float,
    ) -> None:
        """
        Record a completed trade.

        Args:
            technique: Trading technique used
            regime: Market regime during trade
            won: Whether trade was profitable
            pnl_bps: P&L in basis points
            timestamp: Trade completion timestamp
        """
        strategy_key = self._get_strategy_key(technique, regime)

        if strategy_key not in self.strategy_trades:
            self.strategy_trades[strategy_key] = []
            self.strategy_status[strategy_key] = StrategyStatus.ENABLED

        # Add trade
        self.strategy_trades[strategy_key].append((won, pnl_bps, timestamp))

        # Recalculate metrics
        old_metrics = self.cached_metrics.get(strategy_key)
        new_metrics = self._calculate_metrics(technique, regime)

        if new_metrics:
            self.cached_metrics[strategy_key] = new_metrics

            # Check for status change
            if self.auto_disable and new_metrics.total_trades >= self.min_trades:
                self._update_strategy_status(strategy_key, new_metrics, old_metrics)

        logger.debug(
            "trade_recorded",
            strategy_key=strategy_key,
            won=won,
            pnl_bps=pnl_bps,
            total_trades=len(self.strategy_trades[strategy_key]),
        )

    def should_use_strategy(
        self,
        technique: str,
        regime: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if strategy should be used.

        Args:
            technique: Trading technique
            regime: Market regime

        Returns:
            (should_use, reason) tuple
        """
        strategy_key = self._get_strategy_key(technique, regime)

        # Check status
        status = self.strategy_status.get(strategy_key, StrategyStatus.ENABLED)

        if status == StrategyStatus.DISABLED:
            metrics = self.cached_metrics.get(strategy_key)
            if metrics:
                reason = (
                    f"Strategy '{strategy_key}' disabled due to poor performance: "
                    f"{metrics.win_rate:.0%} win rate, "
                    f"{metrics.profit_factor:.2f} profit factor. "
                    f"Recommendation: {metrics.recommendation}"
                )
            else:
                reason = f"Strategy '{strategy_key}' is disabled"

            return False, reason

        return True, None

    def get_strategy_metrics(
        self,
        technique: str,
        regime: str,
    ) -> Optional[StrategyMetrics]:
        """Get performance metrics for strategy."""
        strategy_key = self._get_strategy_key(technique, regime)
        return self.cached_metrics.get(strategy_key)

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        return self.cached_metrics.copy()

    def get_top_strategies(self, n: int = 5) -> List[StrategyMetrics]:
        """Get top N performing strategies by Sharpe ratio."""
        all_metrics = list(self.cached_metrics.values())

        # Filter out strategies with insufficient trades
        valid_metrics = [m for m in all_metrics if m.total_trades >= self.min_trades]

        # Sort by Sharpe ratio
        valid_metrics.sort(key=lambda m: m.sharpe_ratio, reverse=True)

        return valid_metrics[:n]

    def get_worst_strategies(self, n: int = 5) -> List[StrategyMetrics]:
        """Get worst N performing strategies by profit factor."""
        all_metrics = list(self.cached_metrics.values())

        # Filter out strategies with insufficient trades
        valid_metrics = [m for m in all_metrics if m.total_trades >= self.min_trades]

        # Sort by profit factor
        valid_metrics.sort(key=lambda m: m.profit_factor)

        return valid_metrics[:n]

    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        lines = []
        lines.append("=" * 80)
        lines.append("STRATEGY PERFORMANCE REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Get all metrics sorted by performance
        all_metrics = sorted(
            self.cached_metrics.values(),
            key=lambda m: m.profit_factor if m.total_trades >= self.min_trades else 0,
            reverse=True,
        )

        if not all_metrics:
            lines.append("No strategy data available yet.")
            return "\n".join(lines)

        # Group by regime
        by_regime = {}
        for metrics in all_metrics:
            if metrics.regime not in by_regime:
                by_regime[metrics.regime] = []
            by_regime[metrics.regime].append(metrics)

        for regime, strategies in sorted(by_regime.items()):
            lines.append(f"\n{'='*80}")
            lines.append(f"{regime.upper()} REGIME")
            lines.append(f"{'='*80}")

            for metrics in strategies:
                if metrics.total_trades < self.min_trades:
                    continue

                # Status emoji
                if metrics.status == StrategyStatus.ENABLED:
                    status_icon = "✅"
                elif metrics.status == StrategyStatus.DISABLED:
                    status_icon = "❌"
                else:
                    status_icon = "⚠️"

                # Performance emoji
                if metrics.performance_level == PerformanceLevel.EXCELLENT:
                    perf_icon = "🌟"
                elif metrics.performance_level == PerformanceLevel.GOOD:
                    perf_icon = "✅"
                elif metrics.performance_level == PerformanceLevel.ACCEPTABLE:
                    perf_icon = "✓"
                elif metrics.performance_level == PerformanceLevel.POOR:
                    perf_icon = "⚠️"
                else:
                    perf_icon = "❌"

                lines.append(f"\n{metrics.technique.upper()} {status_icon} {perf_icon}")
                lines.append(f"  Win Rate: {metrics.win_rate:.0%} ({metrics.wins}W/{metrics.losses}L)")
                lines.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
                lines.append(f"  Avg Win: +{metrics.avg_win_bps:.0f} bps | Avg Loss: {metrics.avg_loss_bps:.0f} bps")
                lines.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
                lines.append(f"  Recent Performance: {metrics.recent_win_rate:.0%} win rate")
                lines.append(f"  Status: {metrics.status.value.upper()}")
                lines.append(f"  Recommendation: {metrics.recommendation}")

                if metrics.warnings:
                    for warning in metrics.warnings:
                        lines.append(f"  ⚠️ {warning}")

        # Summary statistics
        lines.append(f"\n{'='*80}")
        lines.append("SUMMARY")
        lines.append(f"{'='*80}")

        enabled_count = sum(1 for s in self.strategy_status.values() if s == StrategyStatus.ENABLED)
        disabled_count = sum(1 for s in self.strategy_status.values() if s == StrategyStatus.DISABLED)

        lines.append(f"Total Strategies: {len(self.strategy_status)}")
        lines.append(f"Enabled: {enabled_count}")
        lines.append(f"Disabled: {disabled_count}")

        # Top 3 strategies
        top_strategies = self.get_top_strategies(3)
        if top_strategies:
            lines.append(f"\nTop 3 Strategies:")
            for i, metrics in enumerate(top_strategies, 1):
                lines.append(f"  {i}. {metrics.strategy_key}: {metrics.win_rate:.0%} WR, {metrics.sharpe_ratio:.2f} Sharpe")

        # Worst 3 strategies
        worst_strategies = self.get_worst_strategies(3)
        if worst_strategies:
            lines.append(f"\nWorst 3 Strategies:")
            for i, metrics in enumerate(worst_strategies, 1):
                lines.append(f"  {i}. {metrics.strategy_key}: {metrics.win_rate:.0%} WR, {metrics.profit_factor:.2f} PF")

        # Recent alerts
        if self.alerts:
            recent_alerts = self.alerts[-5:]
            lines.append(f"\nRecent Alerts:")
            for alert in recent_alerts:
                alert_time = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  [{alert_time}] {alert.severity}: {alert.message}")

        return "\n".join(lines)

    def _get_strategy_key(self, technique: str, regime: str) -> str:
        """Generate strategy key."""
        return f"{technique.lower()}_in_{regime.lower()}"

    def _calculate_metrics(
        self,
        technique: str,
        regime: str,
    ) -> Optional[StrategyMetrics]:
        """Calculate performance metrics for strategy."""
        strategy_key = self._get_strategy_key(technique, regime)
        trades = self.strategy_trades.get(strategy_key, [])

        if not trades:
            return None

        # Extract data
        won_flags = [t[0] for t in trades]
        pnls = [t[1] for t in trades]
        timestamps = [t[2] for t in trades]

        # Counts
        total_trades = len(trades)
        wins = sum(won_flags)
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        avg_pnl = np.mean(pnls)
        winners = [p for p, w in zip(pnls, won_flags) if w]
        losers = [p for p, w in zip(pnls, won_flags) if not w]

        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = np.mean(losers) if losers else 0.0

        total_wins = sum(winners) if winners else 0.0
        total_losses = abs(sum(losers)) if losers else 0.0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Sharpe ratio (simplified)
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0

        # Recent performance
        recent_trades = trades[-self.recent_window:]
        recent_won = sum(t[0] for t in recent_trades)
        recent_win_rate = recent_won / len(recent_trades) if recent_trades else 0.0

        recent_pnls = [t[1] for t in recent_trades]
        recent_winners = [p for p, w in zip(recent_pnls, [t[0] for t in recent_trades]) if w]
        recent_losers = [p for p, w in zip(recent_pnls, [t[0] for t in recent_trades]) if not w]

        recent_total_wins = sum(recent_winners) if recent_winners else 0.0
        recent_total_losses = abs(sum(recent_losers)) if recent_losers else 0.0
        recent_profit_factor = recent_total_wins / recent_total_losses if recent_total_losses > 0 else 0.0

        # Determine status
        status = self.strategy_status.get(strategy_key, StrategyStatus.ENABLED)

        # Classify performance
        if profit_factor < 1.0:
            performance_level = PerformanceLevel.CRITICAL
        elif win_rate < self.min_win_rate and profit_factor < self.min_profit_factor:
            performance_level = PerformanceLevel.POOR
        elif win_rate >= self.min_win_rate and profit_factor >= self.min_profit_factor:
            if win_rate >= 0.65 and profit_factor >= 2.0:
                performance_level = PerformanceLevel.EXCELLENT
            else:
                performance_level = PerformanceLevel.GOOD
        else:
            performance_level = PerformanceLevel.ACCEPTABLE

        # Generate recommendation
        recommendation = self._generate_recommendation(performance_level, win_rate, profit_factor)

        # Generate warnings
        warnings = []
        if recent_win_rate < win_rate * 0.7:
            warnings.append(f"Recent performance degraded: {recent_win_rate:.0%} vs {win_rate:.0%} overall")
        if profit_factor < 1.0:
            warnings.append("LOSING MONEY: Profit factor below 1.0")
        if win_rate < 0.40:
            warnings.append("Very low win rate - consider disabling")

        return StrategyMetrics(
            technique=technique,
            regime=regime,
            strategy_key=strategy_key,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl_bps=avg_pnl,
            avg_win_bps=avg_win,
            avg_loss_bps=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            recent_win_rate=recent_win_rate,
            recent_profit_factor=recent_profit_factor,
            status=status,
            performance_level=performance_level,
            recommendation=recommendation,
            warnings=warnings,
            first_trade_timestamp=timestamps[0],
            last_trade_timestamp=timestamps[-1],
        )

    def _update_strategy_status(
        self,
        strategy_key: str,
        new_metrics: StrategyMetrics,
        old_metrics: Optional[StrategyMetrics],
    ) -> None:
        """Update strategy status based on performance."""
        old_status = self.strategy_status.get(strategy_key, StrategyStatus.ENABLED)

        # Determine new status
        if new_metrics.performance_level == PerformanceLevel.CRITICAL:
            new_status = StrategyStatus.DISABLED
        elif new_metrics.performance_level == PerformanceLevel.POOR:
            new_status = StrategyStatus.MONITORING if new_metrics.total_trades < 50 else StrategyStatus.DISABLED
        else:
            new_status = StrategyStatus.ENABLED

        # Update status if changed
        if new_status != old_status:
            self.strategy_status[strategy_key] = new_status

            # Create alert
            if new_status == StrategyStatus.DISABLED:
                alert = StrategyAlert(
                    strategy_key=strategy_key,
                    alert_type='CRITICAL',
                    severity='CRITICAL',
                    message=f"Strategy '{strategy_key}' DISABLED due to poor performance: {new_metrics.win_rate:.0%} WR, {new_metrics.profit_factor:.2f} PF",
                    old_win_rate=old_metrics.win_rate if old_metrics else 0.0,
                    new_win_rate=new_metrics.win_rate,
                    timestamp=new_metrics.last_trade_timestamp,
                )
            else:
                alert = StrategyAlert(
                    strategy_key=strategy_key,
                    alert_type='IMPROVEMENT',
                    severity='MEDIUM',
                    message=f"Strategy '{strategy_key}' status changed: {old_status.value} → {new_status.value}",
                    old_win_rate=old_metrics.win_rate if old_metrics else 0.0,
                    new_win_rate=new_metrics.win_rate,
                    timestamp=new_metrics.last_trade_timestamp,
                )

            self.alerts.append(alert)

            logger.warning(
                "strategy_status_changed",
                strategy_key=strategy_key,
                old_status=old_status.value,
                new_status=new_status.value,
                win_rate=new_metrics.win_rate,
                profit_factor=new_metrics.profit_factor,
            )

    def _generate_recommendation(
        self,
        performance_level: PerformanceLevel,
        win_rate: float,
        profit_factor: float,
    ) -> str:
        """Generate actionable recommendation."""
        if performance_level == PerformanceLevel.EXCELLENT:
            return f"STRONG - Excellent performance ({win_rate:.0%} WR, {profit_factor:.2f} PF) - Continue using"
        elif performance_level == PerformanceLevel.GOOD:
            return f"GOOD - Above targets ({win_rate:.0%} WR, {profit_factor:.2f} PF) - Continue using"
        elif performance_level == PerformanceLevel.ACCEPTABLE:
            return f"ACCEPTABLE - Meeting minimums ({win_rate:.0%} WR, {profit_factor:.2f} PF) - Monitor closely"
        elif performance_level == PerformanceLevel.POOR:
            return f"POOR - Below targets ({win_rate:.0%} WR, {profit_factor:.2f} PF) - Consider disabling"
        else:  # CRITICAL
            return f"CRITICAL - Losing money ({win_rate:.0%} WR, {profit_factor:.2f} PF) - DISABLE immediately"
