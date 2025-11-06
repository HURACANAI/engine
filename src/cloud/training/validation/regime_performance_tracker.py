"""
Regime-Specific Performance Tracking

Tracks performance across different market regimes.

Regimes:
1. TREND: Strong directional moves
2. RANGE: Choppy, sideways markets
3. PANIC: High volatility, fear-driven moves

Tracks:
- Win rate by regime
- Sharpe ratio by regime
- Average P&L by regime
- Trade count by regime
- Performance trends over time
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RegimePerformance:
    """Performance metrics for a single regime."""

    regime: str
    trades_count: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_bps: float
    sharpe_ratio: float
    max_drawdown: float
    best_trade_bps: float
    worst_trade_bps: float
    avg_hold_minutes: float


@dataclass
class RegimePerformanceReport:
    """Complete regime performance report."""

    overall_performance: Dict[str, float]
    regime_performance: Dict[str, RegimePerformance]
    regime_distribution: Dict[str, int]  # Regime -> trade count
    recommendations: List[str]  # Recommendations for improvement


class RegimePerformanceTracker:
    """
    Tracks performance across different market regimes.

    Regimes:
    1. TREND: Strong directional moves
    2. RANGE: Choppy, sideways markets
    3. PANIC: High volatility, fear-driven moves

    Usage:
        tracker = RegimePerformanceTracker()

        report = tracker.track_performance(
            trades=all_trades,
            model_id="model_v1",
        )

        print(f"Trend regime win rate: {report.regime_performance['TREND'].win_rate:.1%}")
    """

    def __init__(self):
        """Initialize regime performance tracker."""
        logger.info("regime_performance_tracker_initialized")

    def track_performance(
        self,
        trades: List[Dict],
        model_id: str = "model",
    ) -> RegimePerformanceReport:
        """
        Track performance across regimes.

        Args:
            trades: List of trades with regime information
            model_id: Model identifier

        Returns:
            RegimePerformanceReport with performance metrics
        """
        if not trades:
            return RegimePerformanceReport(
                overall_performance={},
                regime_performance={},
                regime_distribution={},
                recommendations=["No trades to analyze"],
            )

        # Group trades by regime
        regime_trades: Dict[str, List[Dict]] = {}

        for trade in trades:
            regime = trade.get("market_regime", "UNKNOWN")
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        # Calculate regime-specific performance
        regime_performance = {}

        for regime, regime_trade_list in regime_trades.items():
            if not regime_trade_list:
                continue

            performance = self._calculate_regime_performance(regime, regime_trade_list)
            regime_performance[regime] = performance

        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(trades)

        # Calculate regime distribution
        regime_distribution = {regime: len(trades_list) for regime, trades_list in regime_trades.items()}

        # Generate recommendations
        recommendations = self._generate_recommendations(regime_performance, overall_performance)

        report = RegimePerformanceReport(
            overall_performance=overall_performance,
            regime_performance=regime_performance,
            regime_distribution=regime_distribution,
            recommendations=recommendations,
        )

        logger.info(
            "regime_performance_tracking_complete",
            model_id=model_id,
            total_trades=len(trades),
            regimes=len(regime_performance),
        )

        return report

    def _calculate_regime_performance(
        self, regime: str, trades: List[Dict]
    ) -> RegimePerformance:
        """Calculate performance for a single regime."""
        wins = sum(1 for t in trades if t.get("is_winner", False))
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0.0

        # Calculate P&L metrics
        pnl_list = [t.get("net_profit_bps", 0.0) for t in trades]
        avg_pnl_bps = np.mean(pnl_list) if pnl_list else 0.0

        # Calculate Sharpe ratio
        sharpe_ratio = (
            (np.mean(pnl_list) / np.std(pnl_list)) if np.std(pnl_list) > 0 else 0.0
        )

        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Best and worst trades
        best_trade_bps = max(pnl_list) if pnl_list else 0.0
        worst_trade_bps = min(pnl_list) if pnl_list else 0.0

        # Average hold time
        hold_times = [t.get("hold_duration_minutes", 0) for t in trades if t.get("hold_duration_minutes")]
        avg_hold_minutes = np.mean(hold_times) if hold_times else 0.0

        return RegimePerformance(
            regime=regime,
            trades_count=len(trades),
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl_bps=avg_pnl_bps,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            best_trade_bps=best_trade_bps,
            worst_trade_bps=worst_trade_bps,
            avg_hold_minutes=avg_hold_minutes,
        )

    def _calculate_overall_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.get("is_winner", False))
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        pnl_list = [t.get("net_profit_bps", 0.0) for t in trades]
        avg_pnl_bps = np.mean(pnl_list) if pnl_list else 0.0

        sharpe_ratio = (
            (np.mean(pnl_list) / np.std(pnl_list)) if np.std(pnl_list) > 0 else 0.0
        )

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_pnl_bps": avg_pnl_bps,
            "sharpe_ratio": sharpe_ratio,
        }

    def _generate_recommendations(
        self,
        regime_performance: Dict[str, RegimePerformance],
        overall_performance: Dict[str, float],
    ) -> List[str]:
        """Generate recommendations based on regime performance."""
        recommendations = []

        # Check for weak regimes
        for regime, perf in regime_performance.items():
            if perf.win_rate < 0.50:
                recommendations.append(
                    f"⚠️ {regime} regime underperforming: {perf.win_rate:.1%} win rate (consider avoiding or adjusting strategy)"
                )

            if perf.sharpe_ratio < 0.5:
                recommendations.append(
                    f"⚠️ {regime} regime low Sharpe: {perf.sharpe_ratio:.2f} (consider risk adjustment)"
                )

        # Check for strong regimes
        for regime, perf in regime_performance.items():
            if perf.win_rate > 0.70 and perf.sharpe_ratio > 1.5:
                recommendations.append(
                    f"✅ {regime} regime performing well: {perf.win_rate:.1%} win rate, {perf.sharpe_ratio:.2f} Sharpe (consider increasing allocation)"
                )

        # Check for regime imbalance
        if len(regime_performance) > 1:
            trade_counts = [perf.trades_count for perf in regime_performance.values()]
            max_count = max(trade_counts)
            min_count = min(trade_counts)

            if max_count > min_count * 3:
                recommendations.append(
                    "⚠️ Regime imbalance detected: Consider diversifying across regimes"
                )

        return recommendations

    def get_statistics(self) -> dict:
        """Get tracker statistics."""
        return {
            'tracker_type': 'regime_performance',
        }

