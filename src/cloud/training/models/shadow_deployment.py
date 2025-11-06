"""
[FUTURE/PILOT - NOT USED IN ENGINE]

Shadow Deployment Framework - Safe Production Validation

This module is for Pilot (Local Trader) live trading deployment.
The Engine does NOT use this - Engine does shadow trading for LEARNING only.

DO NOT USE in Engine daily training pipeline.
This will be used when building Pilot component.

IMPORTANT DISTINCTION:
- Engine shadow trading = LEARNING (paper trades to train models, no deployment)
- Pilot shadow deployment = LIVE DEPLOYMENT (testing new models before production)

Purpose:
Run the new dual-mode system in parallel with existing system without
risking real capital. Compare performance before full deployment.

Key Features:
1. Parallel Execution
   - Shadow system runs alongside production
   - Records what trades IT would have taken
   - Tracks hypothetical P&L
   - No real orders placed

2. Performance Comparison
   - Win rate: Shadow vs Production
   - P&L: Shadow vs Production
   - Risk metrics: Drawdown, volatility
   - Statistical significance testing

3. Safety Checks
   - Max shadow trades per day
   - Kill switch if shadow performs badly
   - Alerts for anomalies
   - Automatic rollback triggers

4. Gradual Rollout
   - Phase 1: 100% shadow (0% live)
   - Phase 2: 10% live (90% shadow)
   - Phase 3: 50% live (50% shadow)
   - Phase 4: 100% live (0% shadow)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DeploymentPhase(Enum):
    """Deployment phase."""
    SHADOW_ONLY = "shadow_only"  # 0% live
    PHASE_2 = "phase_2"  # 10% live
    PHASE_3 = "phase_3"  # 50% live
    FULL_LIVE = "full_live"  # 100% live


@dataclass
class TradeDecision:
    """Decision to trade or not."""

    should_trade: bool
    symbol: str
    direction: str
    size_usd: float
    confidence: float
    edge_hat_bps: float
    book: Optional[str] = None  # 'scalp' or 'runner' for dual-mode
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ShadowTrade:
    """A hypothetical trade from shadow system."""

    trade_id: int
    timestamp: float
    decision: TradeDecision
    simulated_outcome_bps: float  # What actually happened
    simulated_pnl_usd: float
    won: bool


@dataclass
class ComparisonMetrics:
    """Comparison between shadow and production."""

    # Win rates
    shadow_win_rate: float
    production_win_rate: float
    wr_difference: float

    # P&L
    shadow_total_pnl: float
    production_total_pnl: float
    pnl_difference: float

    # Volume
    shadow_trades: int
    production_trades: int

    # Risk
    shadow_max_drawdown: float
    production_max_drawdown: float
    shadow_sharpe: float
    production_sharpe: float

    # Statistical significance
    p_value: float  # Is difference significant?
    is_significant: bool

    # Safety checks
    passes_safety_checks: bool
    safety_failures: List[str]


@dataclass
class ComparisonReport:
    """Full comparison report."""

    phase: DeploymentPhase
    start_time: float
    end_time: float
    days_elapsed: float

    metrics: ComparisonMetrics

    shadow_is_better: bool
    ready_for_next_phase: bool
    recommendation: str

    # Detailed breakdowns
    shadow_by_mode: Optional[Dict] = None  # For dual-mode systems
    shadow_by_regime: Optional[Dict] = None


class ShadowDeployment:
    """
    Shadow deployment manager.

    Runs new system in parallel with production, comparing performance
    before switching over.

    Architecture:
        Signal → Both Systems → Production Executes, Shadow Records
                       ↓
        Compare Performance → Safety Checks → Promote Decision
    """

    def __init__(
        self,
        shadow_system,  # New system (dual-mode coordinator)
        production_system,  # Old system
        min_days_before_promote: int = 7,
        min_trades_before_promote: int = 100,
        required_wr_improvement: float = 0.02,  # +2%
        required_pnl_improvement: float = 0.10,  # +10%
        max_drawdown_tolerance: float = 0.15,  # -15%
        significance_level: float = 0.05,  # p < 0.05
    ):
        """
        Initialize shadow deployment.

        Args:
            shadow_system: New system to test
            production_system: Current production system
            min_days_before_promote: Min days to observe before promotion
            min_trades_before_promote: Min trades before promotion
            required_wr_improvement: Required WR improvement for promotion
            required_pnl_improvement: Required P&L improvement for promotion
            max_drawdown_tolerance: Max allowed drawdown
            significance_level: Statistical significance threshold
        """
        self.shadow_system = shadow_system
        self.production_system = production_system

        self.min_days = min_days_before_promote
        self.min_trades = min_trades_before_promote
        self.required_wr_improvement = required_wr_improvement
        self.required_pnl_improvement = required_pnl_improvement
        self.max_drawdown = max_drawdown_tolerance
        self.significance_level = significance_level

        # Current phase
        self.phase = DeploymentPhase.SHADOW_ONLY
        self.live_traffic_pct = 0.0

        # Trade history
        self.shadow_trades: List[ShadowTrade] = []
        self.production_trades: List[Dict] = []

        # Metrics tracking
        self.start_time = time.time()
        self.last_evaluation_time = time.time()

        # Safety monitoring
        self.anomalies_detected: List[str] = []
        self.kill_switch_triggered = False

        logger.info(
            "shadow_deployment_initialized",
            phase=self.phase.value,
            min_days=min_days_before_promote,
            required_wr_improvement=required_wr_improvement,
        )

    def process_signal(
        self,
        symbol: str,
        price: float,
        features: Dict,
        regime: str,
    ) -> Tuple[TradeDecision, TradeDecision]:
        """
        Process signal through both systems.

        Args:
            symbol: Asset symbol
            price: Current price
            features: Signal features
            regime: Market regime

        Returns:
            (production_decision, shadow_decision)
        """
        # Check kill switch
        if self.kill_switch_triggered:
            logger.warning("shadow_deployment_kill_switch_active")
            return (
                TradeDecision(
                    should_trade=False,
                    symbol=symbol,
                    direction='long',
                    size_usd=0.0,
                    confidence=0.0,
                    edge_hat_bps=0.0,
                    reason="Kill switch active",
                ),
                TradeDecision(
                    should_trade=False,
                    symbol=symbol,
                    direction='long',
                    size_usd=0.0,
                    confidence=0.0,
                    edge_hat_bps=0.0,
                    reason="Kill switch active",
                ),
            )

        # Get production decision
        prod_result = self.production_system.process_signal(
            symbol=symbol,
            price=price,
            features=features,
            regime=regime,
        )

        prod_decision = TradeDecision(
            should_trade=prod_result.should_trade,
            symbol=symbol,
            direction='long',  # Simplified
            size_usd=100.0,
            confidence=prod_result.confidence,
            edge_hat_bps=prod_result.edge_hat_bps,
            gates_passed=prod_result.gates_passed,
            gates_failed=prod_result.gates_failed,
            reason=prod_result.reason,
        )

        # Get shadow decision
        shadow_result = self.shadow_system.process_signal(
            symbol=symbol,
            price=price,
            features=features,
            regime=regime,
        )

        shadow_decision = TradeDecision(
            should_trade=shadow_result.should_trade,
            symbol=symbol,
            direction='long',
            size_usd=100.0,
            confidence=shadow_result.confidence,
            edge_hat_bps=shadow_result.edge_hat_bps,
            book=getattr(shadow_result, 'book', None),
            gates_passed=shadow_result.gates_passed,
            gates_failed=shadow_result.gates_failed,
            reason=shadow_result.reason,
        )

        # Determine which to execute based on phase
        if self.phase == DeploymentPhase.SHADOW_ONLY:
            # Only production executes
            pass
        elif self.phase == DeploymentPhase.PHASE_2:
            # 10% shadow goes live
            if np.random.random() < 0.10:
                # Use shadow decision
                return (shadow_decision, shadow_decision)
        elif self.phase == DeploymentPhase.PHASE_3:
            # 50% shadow goes live
            if np.random.random() < 0.50:
                return (shadow_decision, shadow_decision)
        elif self.phase == DeploymentPhase.FULL_LIVE:
            # 100% shadow (it's now production)
            return (shadow_decision, shadow_decision)

        return (prod_decision, shadow_decision)

    def record_shadow_trade(
        self,
        decision: TradeDecision,
        actual_outcome_bps: float,
    ) -> None:
        """
        Record hypothetical shadow trade.

        Args:
            decision: Shadow system's decision
            actual_outcome_bps: What actually happened in market
        """
        if not decision.should_trade:
            return

        # Simulate P&L
        # Assume execution costs of ~8 bps total (entry + exit)
        net_outcome_bps = actual_outcome_bps - 8.0
        pnl_usd = net_outcome_bps / 10000 * decision.size_usd

        trade = ShadowTrade(
            trade_id=len(self.shadow_trades) + 1,
            timestamp=time.time(),
            decision=decision,
            simulated_outcome_bps=net_outcome_bps,
            simulated_pnl_usd=pnl_usd,
            won=(net_outcome_bps > 0),
        )

        self.shadow_trades.append(trade)

        logger.debug(
            "shadow_trade_recorded",
            trade_id=trade.trade_id,
            won=trade.won,
            pnl_bps=net_outcome_bps,
        )

    def record_production_trade(
        self,
        symbol: str,
        pnl_bps: float,
        pnl_usd: float,
        won: bool,
    ) -> None:
        """Record production trade for comparison."""
        trade = {
            'timestamp': time.time(),
            'symbol': symbol,
            'pnl_bps': pnl_bps,
            'pnl_usd': pnl_usd,
            'won': won,
        }

        self.production_trades.append(trade)

    def generate_comparison_report(self) -> ComparisonReport:
        """
        Generate comprehensive comparison report.

        Returns:
            ComparisonReport with metrics and recommendation
        """
        days_elapsed = (time.time() - self.start_time) / 86400

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Determine if shadow is better
        shadow_better = (
            metrics.wr_difference > self.required_wr_improvement
            and metrics.pnl_difference > self.required_pnl_improvement
            and metrics.is_significant
        )

        # Check if ready for next phase
        ready = (
            days_elapsed >= self.min_days
            and len(self.shadow_trades) >= self.min_trades
            and shadow_better
            and metrics.passes_safety_checks
        )

        # Generate recommendation
        if not ready:
            if days_elapsed < self.min_days:
                recommendation = f"Continue shadow testing (Day {days_elapsed:.1f}/{self.min_days})"
            elif len(self.shadow_trades) < self.min_trades:
                recommendation = f"Need more trades ({len(self.shadow_trades)}/{self.min_trades})"
            elif not shadow_better:
                recommendation = "Shadow not outperforming production yet"
            elif not metrics.passes_safety_checks:
                recommendation = f"Safety check failures: {', '.join(metrics.safety_failures)}"
            else:
                recommendation = "Continue monitoring"
        else:
            if self.phase == DeploymentPhase.SHADOW_ONLY:
                recommendation = "✓ Ready for Phase 2: 10% live traffic"
            elif self.phase == DeploymentPhase.PHASE_2:
                recommendation = "✓ Ready for Phase 3: 50% live traffic"
            elif self.phase == DeploymentPhase.PHASE_3:
                recommendation = "✓ Ready for full deployment: 100% live"
            else:
                recommendation = "Fully deployed"

        return ComparisonReport(
            phase=self.phase,
            start_time=self.start_time,
            end_time=time.time(),
            days_elapsed=days_elapsed,
            metrics=metrics,
            shadow_is_better=shadow_better,
            ready_for_next_phase=ready,
            recommendation=recommendation,
        )

    def _calculate_metrics(self) -> ComparisonMetrics:
        """Calculate comparison metrics."""
        # Shadow metrics
        shadow_wins = sum(1 for t in self.shadow_trades if t.won)
        shadow_wr = shadow_wins / len(self.shadow_trades) if self.shadow_trades else 0.0
        shadow_pnl = sum(t.simulated_pnl_usd for t in self.shadow_trades)

        # Production metrics
        prod_wins = sum(1 for t in self.production_trades if t['won'])
        prod_wr = prod_wins / len(self.production_trades) if self.production_trades else 0.0
        prod_pnl = sum(t['pnl_usd'] for t in self.production_trades)

        # Differences
        wr_diff = shadow_wr - prod_wr
        pnl_diff_pct = (shadow_pnl - prod_pnl) / abs(prod_pnl) if prod_pnl != 0 else 0.0

        # Statistical significance (simplified t-test)
        if len(self.shadow_trades) > 30 and len(self.production_trades) > 30:
            shadow_returns = [t.simulated_pnl_usd for t in self.shadow_trades]
            prod_returns = [t['pnl_usd'] for t in self.production_trades]

            # Two-sample t-test
            mean_diff = np.mean(shadow_returns) - np.mean(prod_returns)
            pooled_std = np.sqrt(
                (np.std(shadow_returns) ** 2 / len(shadow_returns))
                + (np.std(prod_returns) ** 2 / len(prod_returns))
            )

            t_stat = mean_diff / pooled_std if pooled_std > 0 else 0.0
            # Simplified p-value (assumes large samples)
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat / np.sqrt(2))))
            is_significant = p_value < self.significance_level
        else:
            p_value = 1.0
            is_significant = False

        # Risk metrics
        if self.shadow_trades:
            shadow_equity = np.cumsum([t.simulated_pnl_usd for t in self.shadow_trades])
            shadow_running_max = np.maximum.accumulate(shadow_equity)
            shadow_dd = np.min((shadow_equity - shadow_running_max) / np.maximum(shadow_running_max, 1.0))
            shadow_sharpe = (
                np.mean([t.simulated_pnl_usd for t in self.shadow_trades])
                / np.std([t.simulated_pnl_usd for t in self.shadow_trades])
                * np.sqrt(252)
                if len(self.shadow_trades) > 1
                else 0.0
            )
        else:
            shadow_dd = 0.0
            shadow_sharpe = 0.0

        if self.production_trades:
            prod_equity = np.cumsum([t['pnl_usd'] for t in self.production_trades])
            prod_running_max = np.maximum.accumulate(prod_equity)
            prod_dd = np.min((prod_equity - prod_running_max) / np.maximum(prod_running_max, 1.0))
            prod_sharpe = (
                np.mean([t['pnl_usd'] for t in self.production_trades])
                / np.std([t['pnl_usd'] for t in self.production_trades])
                * np.sqrt(252)
                if len(self.production_trades) > 1
                else 0.0
            )
        else:
            prod_dd = 0.0
            prod_sharpe = 0.0

        # Safety checks
        safety_failures = []
        passes_safety = True

        if shadow_dd < -self.max_drawdown:
            safety_failures.append(f"Shadow drawdown too large: {shadow_dd:.1%}")
            passes_safety = False

        if shadow_pnl < 0 and prod_pnl > 0:
            safety_failures.append("Shadow losing money while production profitable")
            passes_safety = False

        if len(self.shadow_trades) > len(self.production_trades) * 3:
            safety_failures.append("Shadow trading too frequently")
            passes_safety = False

        return ComparisonMetrics(
            shadow_win_rate=shadow_wr,
            production_win_rate=prod_wr,
            wr_difference=wr_diff,
            shadow_total_pnl=shadow_pnl,
            production_total_pnl=prod_pnl,
            pnl_difference=pnl_diff_pct,
            shadow_trades=len(self.shadow_trades),
            production_trades=len(self.production_trades),
            shadow_max_drawdown=shadow_dd,
            production_max_drawdown=prod_dd,
            shadow_sharpe=shadow_sharpe,
            production_sharpe=prod_sharpe,
            p_value=p_value,
            is_significant=is_significant,
            passes_safety_checks=passes_safety,
            safety_failures=safety_failures,
        )

    def promote_to_next_phase(self) -> bool:
        """
        Promote to next deployment phase.

        Returns:
            True if promotion successful
        """
        report = self.generate_comparison_report()

        if not report.ready_for_next_phase:
            logger.warning(
                "shadow_deployment_promotion_blocked",
                reason=report.recommendation,
            )
            return False

        # Promote
        if self.phase == DeploymentPhase.SHADOW_ONLY:
            self.phase = DeploymentPhase.PHASE_2
            self.live_traffic_pct = 0.10
        elif self.phase == DeploymentPhase.PHASE_2:
            self.phase = DeploymentPhase.PHASE_3
            self.live_traffic_pct = 0.50
        elif self.phase == DeploymentPhase.PHASE_3:
            self.phase = DeploymentPhase.FULL_LIVE
            self.live_traffic_pct = 1.00
        else:
            logger.info("shadow_deployment_already_full_live")
            return False

        # Reset monitoring for new phase
        self.start_time = time.time()
        self.shadow_trades = []
        self.production_trades = []

        logger.info(
            "shadow_deployment_promoted",
            new_phase=self.phase.value,
            live_traffic_pct=self.live_traffic_pct,
        )

        return True

    def trigger_kill_switch(self, reason: str) -> None:
        """
        Trigger kill switch - stop all shadow testing immediately.

        Args:
            reason: Why kill switch was triggered
        """
        self.kill_switch_triggered = True
        self.anomalies_detected.append(f"{time.time()}: {reason}")

        logger.error(
            "shadow_deployment_kill_switch_triggered",
            reason=reason,
        )

    def print_report(self, report: ComparisonReport) -> None:
        """Print human-readable report."""
        print("=" * 70)
        print("SHADOW DEPLOYMENT REPORT")
        print("=" * 70)
        print(f"Phase: {report.phase.value}")
        print(f"Days Elapsed: {report.days_elapsed:.1f}")
        print()

        print("PERFORMANCE COMPARISON")
        print("-" * 70)
        m = report.metrics

        print(f"Win Rate:")
        print(f"  Shadow: {m.shadow_win_rate:.1%}")
        print(f"  Production: {m.production_win_rate:.1%}")
        print(f"  Difference: {m.wr_difference:+.1%} {'✓' if m.wr_difference > 0 else '✗'}")
        print()

        print(f"P&L:")
        print(f"  Shadow: ${m.shadow_total_pnl:,.2f}")
        print(f"  Production: ${m.production_total_pnl:,.2f}")
        print(f"  Difference: {m.pnl_difference:+.1%} {'✓' if m.pnl_difference > 0 else '✗'}")
        print()

        print(f"Volume:")
        print(f"  Shadow Trades: {m.shadow_trades}")
        print(f"  Production Trades: {m.production_trades}")
        print()

        print(f"Risk Metrics:")
        print(f"  Shadow Sharpe: {m.shadow_sharpe:.2f}")
        print(f"  Production Sharpe: {m.production_sharpe:.2f}")
        print(f"  Shadow Max DD: {m.shadow_max_drawdown:.1%}")
        print(f"  Production Max DD: {m.production_max_drawdown:.1%}")
        print()

        print(f"Statistical Significance:")
        print(f"  p-value: {m.p_value:.4f}")
        print(f"  Significant: {m.is_significant} (p < {self.significance_level})")
        print()

        print("SAFETY CHECKS")
        print("-" * 70)
        if m.passes_safety_checks:
            print("✓ All safety checks passed")
        else:
            print("✗ Safety check failures:")
            for failure in m.safety_failures:
                print(f"  - {failure}")
        print()

        print("RECOMMENDATION")
        print("-" * 70)
        print(f"{report.recommendation}")
        print()


def run_shadow_deployment_example():
    """Example usage of shadow deployment."""
    from gate_calibration import generate_synthetic_trades

    print("=" * 70)
    print("SHADOW DEPLOYMENT SIMULATION")
    print("=" * 70)
    print()

    # Mock systems
    class MockProductionSystem:
        def process_signal(self, **kwargs):
            class Result:
                should_trade = np.random.random() < 0.30
                confidence = np.random.beta(3, 3)
                edge_hat_bps = np.random.normal(8, 5)
                gates_passed = ['cost']
                gates_failed = []
                reason = "Production decision"

            return Result()

    class MockShadowSystem:
        def process_signal(self, **kwargs):
            class Result:
                should_trade = np.random.random() < 0.20  # More selective
                confidence = np.random.beta(4, 2)  # Higher confidence
                edge_hat_bps = np.random.normal(12, 5)  # Better edge
                book = 'scalp' if np.random.random() < 0.70 else 'runner'
                gates_passed = ['cost', 'meta_label']
                gates_failed = []
                reason = "Shadow decision"

            return Result()

    # Initialize shadow deployment
    shadow = ShadowDeployment(
        shadow_system=MockShadowSystem(),
        production_system=MockProductionSystem(),
        min_days_before_promote=0.01,  # 15 minutes for demo
        min_trades_before_promote=50,
    )

    # Simulate 200 signals
    for i in range(200):
        # Generate signal
        symbol = 'BTC-USD'
        price = 50000 + np.random.normal(0, 100)
        features = {'confidence': np.random.beta(3, 2)}
        regime = np.random.choice(['TREND', 'RANGE', 'PANIC'])

        # Process through both systems
        prod_decision, shadow_decision = shadow.process_signal(
            symbol=symbol,
            price=price,
            features=features,
            regime=regime,
        )

        # Simulate outcomes
        actual_outcome = np.random.normal(10, 15)

        # Record
        if prod_decision.should_trade:
            shadow.record_production_trade(
                symbol=symbol,
                pnl_bps=actual_outcome - 8,
                pnl_usd=(actual_outcome - 8) / 10000 * 100,
                won=(actual_outcome > 8),
            )

        if shadow_decision.should_trade:
            shadow.record_shadow_trade(
                decision=shadow_decision,
                actual_outcome_bps=actual_outcome,
            )

    # Generate report
    report = shadow.generate_comparison_report()
    shadow.print_report(report)


if __name__ == '__main__':
    run_shadow_deployment_example()
