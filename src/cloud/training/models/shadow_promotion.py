"""
[FUTURE/PILOT - NOT USED IN ENGINE]

Shadow A/B Promotion Criteria - Safe Production Deployments

This module is for Pilot (Local Trader) live trading deployment.
The Engine does NOT use this - Engine does shadow trading for LEARNING only.

DO NOT USE in Engine daily training pipeline.
This will be used when building Pilot component.

IMPORTANT DISTINCTION:
- Engine shadow trading = LEARNING (paper trades to train models, no deployment)
- Pilot shadow promotion = LIVE DEPLOYMENT (promoting models to production)

Key Problem:
New models/strategies look good in backtest, but:
- Might overfit to test data
- Could fail in live conditions
- No statistical validation

Solution: Shadow Mode + Promotion Criteria
1. Run new model in "shadow mode" (paper trading alongside production)
2. Compare performance statistically
3. Require MULTIPLE criteria before promotion
4. Auto-promote if all criteria pass

Promotion Criteria:
1. **Consecutive days better**: 5+ days in a row
2. **Net improvement**: +10 bps/day average
3. **Expected Shortfall**: ES(95%) > 0 (no terrible tail)
4. **Statistical significance**: t-test p < 0.05
5. **Min sample size**: 50+ trades minimum

Example Scenario:
Day 1: Shadow +15 bps, Prod +10 bps ✓
Day 2: Shadow +20 bps, Prod +18 bps ✓
Day 3: Shadow +12 bps, Prod +8 bps ✓
Day 4: Shadow +5 bps, Prod +12 bps ✗ (reset counter)
...
After meeting all criteria → AUTO-PROMOTE to production

Benefits:
- Prevents premature deployments
- Statistical confidence in improvements
- No guesswork: "Is it ready?" → Criteria tell you
- Protects against lucky backtests

Usage:
    promoter = ShadowPromotionCriteria()

    # Record daily performance
    for day in range(30):
        promoter.record_day(
            shadow_pnl_bps=shadow_performance[day],
            prod_pnl_bps=prod_performance[day],
            date=dates[day],
        )

    # Check if ready for promotion
    can_promote, reason = promoter.check_promotion()

    if can_promote:
        print(f"PROMOTE shadow to production: {reason}")
        deploy_to_production()
    else:
        print(f"NOT promoting: {reason}")
        continue_shadow_testing()
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DailyPerformance:
    """Daily performance record."""

    date: str
    shadow_pnl_bps: float
    prod_pnl_bps: float
    shadow_trades: int = 0
    prod_trades: int = 0
    shadow_win_rate: float = 0.0
    prod_win_rate: float = 0.0


@dataclass
class PromotionCriteria:
    """Promotion criteria thresholds."""

    min_consecutive_days: int = 5
    min_net_bps_improvement: float = 10.0  # 10 bps/day better
    min_expected_shortfall_95: float = 0.0  # ES(95%) must be non-negative
    max_p_value: float = 0.05  # Statistical significance
    min_sample_days: int = 10  # Minimum days of data
    min_total_trades: int = 50  # Minimum trades across all days


@dataclass
class PromotionCheck:
    """Result of promotion check."""

    can_promote: bool
    reason: str

    # Detailed metrics
    consecutive_wins: int
    net_improvement_bps: float
    shadow_es95: float
    p_value: float
    sample_days: int
    total_trades: int

    # Per-criterion results
    criteria_met: dict = field(default_factory=dict)


class ShadowPromotionCriteria:
    """
    Manage shadow A/B testing and promotion decisions.

    Workflow:
    1. Deploy new model in shadow mode (paper trading)
    2. Record daily performance for both shadow and production
    3. Check promotion criteria periodically
    4. Auto-promote when all criteria pass
    5. Track promotion history

    Architecture:
        Day 1-10: Shadow testing (collecting data)
        ↓
        Day 11: Check criteria
        - Consecutive wins: 5+ days? ✓/✗
        - Net improvement: +10 bps? ✓/✗
        - Expected Shortfall: ES(95%) > 0? ✓/✗
        - Statistical significance: p < 0.05? ✓/✗
        ↓
        If ALL pass → PROMOTE
        If ANY fail → CONTINUE testing
    """

    def __init__(
        self,
        criteria: Optional[PromotionCriteria] = None,
    ):
        """
        Initialize shadow promotion manager.

        Args:
            criteria: Promotion criteria (uses defaults if None)
        """
        self.criteria = criteria or PromotionCriteria()

        # Historical performance
        self.history: List[DailyPerformance] = []

        # Consecutive tracking
        self.current_consecutive_wins = 0
        self.max_consecutive_wins = 0

        logger.info(
            "shadow_promotion_initialized",
            min_consecutive_days=self.criteria.min_consecutive_days,
            min_net_bps=self.criteria.min_net_bps_improvement,
        )

    def record_day(
        self,
        shadow_pnl_bps: float,
        prod_pnl_bps: float,
        date: Optional[str] = None,
        shadow_trades: int = 0,
        prod_trades: int = 0,
        shadow_win_rate: float = 0.0,
        prod_win_rate: float = 0.0,
    ) -> None:
        """
        Record daily performance for shadow and production.

        Args:
            shadow_pnl_bps: Shadow model P&L in bps
            prod_pnl_bps: Production model P&L in bps
            date: Date string (YYYY-MM-DD)
            shadow_trades: Number of shadow trades
            prod_trades: Number of production trades
            shadow_win_rate: Shadow win rate
            prod_win_rate: Production win rate
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        record = DailyPerformance(
            date=date,
            shadow_pnl_bps=shadow_pnl_bps,
            prod_pnl_bps=prod_pnl_bps,
            shadow_trades=shadow_trades,
            prod_trades=prod_trades,
            shadow_win_rate=shadow_win_rate,
            prod_win_rate=prod_win_rate,
        )

        self.history.append(record)

        # Update consecutive wins
        if shadow_pnl_bps > prod_pnl_bps:
            self.current_consecutive_wins += 1
            self.max_consecutive_wins = max(
                self.max_consecutive_wins, self.current_consecutive_wins
            )
        else:
            self.current_consecutive_wins = 0

        logger.info(
            "daily_performance_recorded",
            date=date,
            shadow_bps=shadow_pnl_bps,
            prod_bps=prod_pnl_bps,
            improvement=shadow_pnl_bps - prod_pnl_bps,
            consecutive_wins=self.current_consecutive_wins,
        )

    def check_promotion(self) -> PromotionCheck:
        """
        Check if shadow model meets all promotion criteria.

        Returns:
            PromotionCheck with decision and details
        """
        if len(self.history) < self.criteria.min_sample_days:
            return PromotionCheck(
                can_promote=False,
                reason=f"Only {len(self.history)} days (need {self.criteria.min_sample_days})",
                consecutive_wins=self.current_consecutive_wins,
                net_improvement_bps=0.0,
                shadow_es95=0.0,
                p_value=1.0,
                sample_days=len(self.history),
                total_trades=0,
                criteria_met={
                    'sample_size': False,
                    'consecutive_wins': False,
                    'net_improvement': False,
                    'expected_shortfall': False,
                    'statistical_significance': False,
                },
            )

        shadow_perf = [r.shadow_pnl_bps for r in self.history]
        prod_perf = [r.prod_pnl_bps for r in self.history]

        total_shadow_trades = sum(r.shadow_trades for r in self.history)
        total_prod_trades = sum(r.prod_trades for r in self.history)
        total_trades = max(total_shadow_trades, total_prod_trades)

        criteria_met = {}

        # 1. Sample size check
        criteria_met['sample_size'] = (
            len(self.history) >= self.criteria.min_sample_days
            and total_trades >= self.criteria.min_total_trades
        )

        if not criteria_met['sample_size']:
            return PromotionCheck(
                can_promote=False,
                reason=f"Insufficient data: {len(self.history)} days, {total_trades} trades "
                f"(need {self.criteria.min_sample_days} days, {self.criteria.min_total_trades} trades)",
                consecutive_wins=self.current_consecutive_wins,
                net_improvement_bps=0.0,
                shadow_es95=0.0,
                p_value=1.0,
                sample_days=len(self.history),
                total_trades=total_trades,
                criteria_met=criteria_met,
            )

        # 2. Consecutive wins check
        criteria_met['consecutive_wins'] = (
            self.current_consecutive_wins >= self.criteria.min_consecutive_days
        )

        # 3. Net improvement check
        shadow_avg = np.mean(shadow_perf)
        prod_avg = np.mean(prod_perf)
        net_improvement = shadow_avg - prod_avg

        criteria_met['net_improvement'] = (
            net_improvement >= self.criteria.min_net_bps_improvement
        )

        # 4. Expected Shortfall check (95% ES = 5th percentile)
        shadow_es95 = np.percentile(shadow_perf, 5)
        criteria_met['expected_shortfall'] = shadow_es95 >= self.criteria.min_expected_shortfall_95

        # 5. Statistical significance check (paired t-test)
        if len(shadow_perf) >= 10:  # Need at least 10 samples for t-test
            t_stat, p_value = stats.ttest_rel(shadow_perf, prod_perf, alternative='greater')
            criteria_met['statistical_significance'] = p_value < self.criteria.max_p_value
        else:
            p_value = 1.0
            criteria_met['statistical_significance'] = False

        # Decision
        all_met = all(criteria_met.values())

        if all_met:
            reason = (
                f"PROMOTE: All criteria met - "
                f"{self.current_consecutive_wins} consecutive wins, "
                f"+{net_improvement:.1f} bps/day improvement, "
                f"ES(95%)={shadow_es95:.1f}, p={p_value:.3f}"
            )
        else:
            failed = [k for k, v in criteria_met.items() if not v]
            reason = f"NOT promoting: Failed criteria - {', '.join(failed)}"

        return PromotionCheck(
            can_promote=all_met,
            reason=reason,
            consecutive_wins=self.current_consecutive_wins,
            net_improvement_bps=net_improvement,
            shadow_es95=shadow_es95,
            p_value=p_value,
            sample_days=len(self.history),
            total_trades=total_trades,
            criteria_met=criteria_met,
        )

    def get_performance_summary(self) -> dict:
        """Get summary statistics of shadow vs production."""
        if not self.history:
            return {}

        shadow_perf = [r.shadow_pnl_bps for r in self.history]
        prod_perf = [r.prod_pnl_bps for r in self.history]

        # Daily comparisons
        improvements = [s - p for s, p in zip(shadow_perf, prod_perf)]
        days_better = sum(1 for imp in improvements if imp > 0)

        return {
            'total_days': len(self.history),
            'shadow': {
                'mean_bps': np.mean(shadow_perf),
                'median_bps': np.median(shadow_perf),
                'std_bps': np.std(shadow_perf),
                'min_bps': np.min(shadow_perf),
                'max_bps': np.max(shadow_perf),
                'es_95': np.percentile(shadow_perf, 5),
                'total_trades': sum(r.shadow_trades for r in self.history),
                'avg_win_rate': np.mean([r.shadow_win_rate for r in self.history]),
            },
            'production': {
                'mean_bps': np.mean(prod_perf),
                'median_bps': np.median(prod_perf),
                'std_bps': np.std(prod_perf),
                'min_bps': np.min(prod_perf),
                'max_bps': np.max(prod_perf),
                'es_95': np.percentile(prod_perf, 5),
                'total_trades': sum(r.prod_trades for r in self.history),
                'avg_win_rate': np.mean([r.prod_win_rate for r in self.history]),
            },
            'comparison': {
                'mean_improvement_bps': np.mean(improvements),
                'median_improvement_bps': np.median(improvements),
                'days_better': days_better,
                'days_worse': len(self.history) - days_better,
                'win_rate_pct': days_better / len(self.history),
                'current_consecutive_wins': self.current_consecutive_wins,
                'max_consecutive_wins': self.max_consecutive_wins,
            },
        }

    def reset(self) -> None:
        """Reset tracking (e.g., after promotion)."""
        self.history.clear()
        self.current_consecutive_wins = 0
        self.max_consecutive_wins = 0

        logger.info("shadow_promotion_reset")


def run_promotion_check_example():
    """Example usage of shadow promotion criteria."""
    # Initialize
    promoter = ShadowPromotionCriteria(
        criteria=PromotionCriteria(
            min_consecutive_days=5,
            min_net_bps_improvement=10.0,
            min_expected_shortfall_95=0.0,
            max_p_value=0.05,
            min_sample_days=10,
            min_total_trades=50,
        )
    )

    # Simulate 20 days of performance
    np.random.seed(42)

    for day in range(20):
        # Shadow is better on average (+12 bps improvement)
        prod_bps = np.random.normal(15, 10)
        shadow_bps = prod_bps + np.random.normal(12, 5)

        promoter.record_day(
            shadow_pnl_bps=shadow_bps,
            prod_pnl_bps=prod_bps,
            date=f"2025-01-{day + 1:02d}",
            shadow_trades=np.random.randint(5, 15),
            prod_trades=np.random.randint(5, 15),
            shadow_win_rate=0.75,
            prod_win_rate=0.68,
        )

    # Check promotion
    check = promoter.check_promotion()

    print(f"\n{'='*60}")
    print(f"PROMOTION CHECK RESULT")
    print(f"{'='*60}")
    print(f"Decision: {check.reason}")
    print(f"\nMetrics:")
    print(f"  Consecutive wins: {check.consecutive_wins} (need {promoter.criteria.min_consecutive_days})")
    print(f"  Net improvement: {check.net_improvement_bps:.1f} bps (need {promoter.criteria.min_net_bps_improvement})")
    print(f"  Expected Shortfall: {check.shadow_es95:.1f} bps (need {promoter.criteria.min_expected_shortfall_95})")
    print(f"  P-value: {check.p_value:.3f} (need <{promoter.criteria.max_p_value})")
    print(f"\nCriteria met:")
    for criterion, met in check.criteria_met.items():
        status = "✓" if met else "✗"
        print(f"  {status} {criterion}")

    # Get summary
    summary = promoter.get_performance_summary()
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Shadow: {summary['shadow']['mean_bps']:.1f} bps/day (ES95={summary['shadow']['es_95']:.1f})")
    print(f"Production: {summary['production']['mean_bps']:.1f} bps/day (ES95={summary['production']['es_95']:.1f})")
    print(f"Improvement: {summary['comparison']['mean_improvement_bps']:.1f} bps/day")
    print(f"Days better: {summary['comparison']['days_better']}/{summary['total_days']}")


if __name__ == "__main__":
    run_promotion_check_example()
