"""
Extended Paper Trading Validation Framework

Validates models through extended paper trading (2-4 weeks minimum).

Features:
1. Extended paper trading (2-4 weeks minimum)
2. Regime-specific performance tracking
3. Performance degradation detection
4. Comparison with backtest results
5. Automatic pass/fail determination

All models must pass extended paper trading before live deployment.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PaperTradingWindow:
    """Single paper trading window."""

    start_date: datetime
    end_date: datetime
    duration_days: int
    trades_count: int
    win_rate: float
    sharpe_ratio: float
    avg_pnl_bps: float
    max_drawdown: float
    regime_distribution: Dict[str, int]  # Regime -> trade count


@dataclass
class PaperTradingResult:
    """Complete paper trading result."""

    passed: bool  # True if all checks passed
    windows: List[PaperTradingWindow]
    total_trades: int
    overall_win_rate: float
    overall_sharpe: float
    overall_avg_pnl_bps: float
    overall_max_drawdown: float
    regime_performance: Dict[str, Dict[str, float]]  # Regime -> metrics
    backtest_comparison: Dict[str, float]  # Comparison with backtest
    blocking_issues: List[str]  # Issues that block deployment
    recommendation: str


class ExtendedPaperTradingValidator:
    """
    Extended paper trading validation framework.

    Validates models through extended paper trading (2-4 weeks minimum).

    Requirements:
    1. Minimum 2-4 weeks of paper trading
    2. Minimum 100 trades
    3. Win rate > 55%
    4. Sharpe ratio > 1.0
    5. Performance within 20% of backtest
    6. No significant degradation over time

    Usage:
        validator = ExtendedPaperTradingValidator(
            min_duration_days=14,
            min_trades=100,
            min_win_rate=0.55,
            min_sharpe=1.0,
        )

        result = validator.validate(
            paper_trades=paper_trades,
            backtest_results=backtest_results,
            model_id="model_v1",
        )

        if not result.passed:
            raise ValidationError(f"Paper trading failed: {result.blocking_issues}")
    """

    def __init__(
        self,
        min_duration_days: int = 14,  # Minimum 2 weeks
        min_trades: int = 100,
        min_win_rate: float = 0.55,
        min_sharpe: float = 1.0,
        max_backtest_deviation: float = 0.20,  # Max 20% deviation from backtest
        max_degradation_rate: float = -0.10,  # Max 10% degradation per week
    ):
        """
        Initialize extended paper trading validator.

        Args:
            min_duration_days: Minimum paper trading duration in days
            min_trades: Minimum number of trades required
            min_win_rate: Minimum acceptable win rate
            min_sharpe: Minimum acceptable Sharpe ratio
            max_backtest_deviation: Maximum acceptable deviation from backtest
            max_degradation_rate: Maximum acceptable performance degradation rate
        """
        self.min_duration_days = min_duration_days
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.max_backtest_deviation = max_backtest_deviation
        self.max_degradation_rate = max_degradation_rate

        logger.info(
            "extended_paper_trading_validator_initialized",
            min_duration_days=min_duration_days,
            min_trades=min_trades,
        )

    def validate(
        self,
        paper_trades: List[Dict],
        backtest_results: Optional[Dict] = None,
        model_id: str = "model",
    ) -> PaperTradingResult:
        """
        Validate through extended paper trading.

        Args:
            paper_trades: List of paper trades
            backtest_results: Backtest results for comparison (optional)
            model_id: Model identifier

        Returns:
            PaperTradingResult with validation results

        Raises:
            ValueError: If validation fails
        """
        if not paper_trades:
            raise ValueError("No paper trades provided")

        # Calculate overall metrics
        total_trades = len(paper_trades)
        wins = sum(1 for t in paper_trades if t.get("is_winner", False))
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Calculate P&L metrics
        pnl_list = [t.get("net_profit_bps", 0.0) for t in paper_trades]
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

        # Calculate duration
        start_date = min(t.get("entry_timestamp", datetime.now()) for t in paper_trades)
        end_date = max(t.get("entry_timestamp", datetime.now()) for t in paper_trades)
        duration_days = (end_date - start_date).days

        # Calculate regime-specific performance
        regime_performance = self._calculate_regime_performance(paper_trades)

        # Compare with backtest
        backtest_comparison = {}
        if backtest_results:
            backtest_comparison = self._compare_with_backtest(
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                avg_pnl_bps=avg_pnl_bps,
                backtest_results=backtest_results,
            )

        # Check for degradation over time
        degradation_detected = self._detect_degradation(paper_trades)

        # Validate requirements
        blocking_issues = []
        checks = []

        # Check 1: Minimum duration
        if duration_days < self.min_duration_days:
            blocking_issues.append(
                f"Duration {duration_days} days < {self.min_duration_days} days (insufficient!)"
            )
            checks.append(False)
        else:
            checks.append(True)

        # Check 2: Minimum trades
        if total_trades < self.min_trades:
            blocking_issues.append(
                f"Trades {total_trades} < {self.min_trades} (insufficient sample!)"
            )
            checks.append(False)
        else:
            checks.append(True)

        # Check 3: Minimum win rate
        if win_rate < self.min_win_rate:
            blocking_issues.append(
                f"Win rate {win_rate:.1%} < {self.min_win_rate:.1%} (below threshold!)"
            )
            checks.append(False)
        else:
            checks.append(True)

        # Check 4: Minimum Sharpe
        if sharpe_ratio < self.min_sharpe:
            blocking_issues.append(
                f"Sharpe {sharpe_ratio:.2f} < {self.min_sharpe:.2f} (below threshold!)"
            )
            checks.append(False)
        else:
            checks.append(True)

        # Check 5: Backtest comparison
        if backtest_comparison:
            deviation = backtest_comparison.get("deviation", 0.0)
            if abs(deviation) > self.max_backtest_deviation:
                blocking_issues.append(
                    f"Backtest deviation {deviation:.1%} > {self.max_backtest_deviation:.1%} (too different!)"
                )
                checks.append(False)
            else:
                checks.append(True)

        # Check 6: Degradation
        if degradation_detected:
            blocking_issues.append("Performance degradation detected over time!")
            checks.append(False)
        else:
            checks.append(True)

        # Determine pass/fail
        all_passed = all(checks)

        # Generate recommendation
        if all_passed:
            recommendation = "✅ Model passed extended paper trading. Safe to deploy."
        else:
            recommendation = f"❌ Model FAILED extended paper trading. {len(blocking_issues)} blocking issue(s). DO NOT DEPLOY."

        result = PaperTradingResult(
            passed=all_passed,
            windows=[],  # Would calculate windows if needed
            total_trades=total_trades,
            overall_win_rate=win_rate,
            overall_sharpe=sharpe_ratio,
            overall_avg_pnl_bps=avg_pnl_bps,
            overall_max_drawdown=max_drawdown,
            regime_performance=regime_performance,
            backtest_comparison=backtest_comparison,
            blocking_issues=blocking_issues,
            recommendation=recommendation,
        )

        logger.info(
            "extended_paper_trading_validation_complete",
            model_id=model_id,
            passed=all_passed,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            blocking_issues=len(blocking_issues),
        )

        # HARD BLOCK: Raise error if validation fails
        if not all_passed:
            error_msg = f"Model {model_id} FAILED extended paper trading validation:\n"
            error_msg += "\n".join(f"  - {issue}" for issue in blocking_issues)
            error_msg += "\n\nDO NOT DEPLOY THIS MODEL!"
            raise ValueError(error_msg)

        return result

    def _calculate_regime_performance(
        self, paper_trades: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate regime-specific performance."""
        regime_trades: Dict[str, List[Dict]] = {}

        for trade in paper_trades:
            regime = trade.get("market_regime", "unknown")
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)

        regime_performance = {}

        for regime, trades in regime_trades.items():
            if not trades:
                continue

            wins = sum(1 for t in trades if t.get("is_winner", False))
            win_rate = wins / len(trades) if trades else 0.0

            pnl_list = [t.get("net_profit_bps", 0.0) for t in trades]
            avg_pnl_bps = np.mean(pnl_list) if pnl_list else 0.0

            sharpe_ratio = (
                (np.mean(pnl_list) / np.std(pnl_list)) if np.std(pnl_list) > 0 else 0.0
            )

            regime_performance[regime] = {
                "trades": len(trades),
                "win_rate": win_rate,
                "avg_pnl_bps": avg_pnl_bps,
                "sharpe_ratio": sharpe_ratio,
            }

        return regime_performance

    def _compare_with_backtest(
        self,
        win_rate: float,
        sharpe_ratio: float,
        avg_pnl_bps: float,
        backtest_results: Dict,
    ) -> Dict[str, float]:
        """Compare paper trading with backtest results."""
        backtest_wr = backtest_results.get("test_win_rate", 0.0)
        backtest_sharpe = backtest_results.get("test_sharpe", 0.0)
        backtest_pnl = backtest_results.get("test_avg_pnl_bps", 0.0)

        wr_deviation = (win_rate - backtest_wr) / backtest_wr if backtest_wr > 0 else 0.0
        sharpe_deviation = (
            (sharpe_ratio - backtest_sharpe) / backtest_sharpe if backtest_sharpe > 0 else 0.0
        )
        pnl_deviation = (
            (avg_pnl_bps - backtest_pnl) / backtest_pnl if backtest_pnl > 0 else 0.0
        )

        return {
            "win_rate_deviation": wr_deviation,
            "sharpe_deviation": sharpe_deviation,
            "pnl_deviation": pnl_deviation,
            "deviation": max(abs(wr_deviation), abs(sharpe_deviation), abs(pnl_deviation)),
        }

    def _detect_degradation(self, paper_trades: List[Dict]) -> bool:
        """Detect performance degradation over time."""
        if len(paper_trades) < 20:
            return False  # Not enough data

        # Sort by timestamp
        sorted_trades = sorted(paper_trades, key=lambda t: t.get("entry_timestamp", datetime.now()))

        # Split into early and late periods
        mid_point = len(sorted_trades) // 2
        early_trades = sorted_trades[:mid_point]
        late_trades = sorted_trades[mid_point:]

        # Calculate metrics for each period
        early_wr = sum(1 for t in early_trades if t.get("is_winner", False)) / len(early_trades)
        late_wr = sum(1 for t in late_trades if t.get("is_winner", False)) / len(late_trades)

        early_pnl = np.mean([t.get("net_profit_bps", 0.0) for t in early_trades])
        late_pnl = np.mean([t.get("net_profit_bps", 0.0) for t in late_trades])

        # Check for degradation
        wr_degradation = (late_wr - early_wr) / early_wr if early_wr > 0 else 0.0
        pnl_degradation = (late_pnl - early_pnl) / early_pnl if early_pnl > 0 else 0.0

        # Degradation if both metrics decline significantly
        degradation = (
            wr_degradation < self.max_degradation_rate
            or pnl_degradation < self.max_degradation_rate
        )

        return degradation

    def get_statistics(self) -> dict:
        """Get validator statistics."""
        return {
            'min_duration_days': self.min_duration_days,
            'min_trades': self.min_trades,
            'min_win_rate': self.min_win_rate,
            'min_sharpe': self.min_sharpe,
            'max_backtest_deviation': self.max_backtest_deviation,
            'max_degradation_rate': self.max_degradation_rate,
        }

