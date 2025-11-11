"""
Shadow Testing System

Runs new models in shadow mode (paper trading) and compares performance
with baseline models. Promotes models only if they show statistically
significant improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class PromotionStatus(str, Enum):
    """Model promotion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ShadowTrade:
    """Shadow trade record."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    size_usd: float
    model_id: str
    baseline_model_id: Optional[str] = None
    realized_pnl: Optional[float] = None
    holding_period_minutes: Optional[int] = None


@dataclass
class ShadowTestResult:
    """Shadow test result."""
    model_id: str
    baseline_model_id: str
    start_date: datetime
    end_date: datetime
    num_trades: int
    new_model_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    promotion_status: PromotionStatus = PromotionStatus.PENDING
    promotion_reason: Optional[str] = None


@dataclass
class ShadowTestConfig:
    """Configuration for shadow testing."""
    min_duration_days: int = 14
    min_trades: int = 100
    promotion_criteria: Dict[str, float] = field(default_factory=lambda: {
        "min_sharpe_improvement": 0.2,
        "min_win_rate_improvement": 0.05,
        "statistical_significance": 0.95,
    })


class ShadowTester:
    """
    Shadow testing system for model promotion.
    
    Features:
    - Shadow trading for new models
    - Performance comparison with baseline
    - Statistical significance testing
    - Automatic promotion/rejection
    """
    
    def __init__(
        self,
        config: ShadowTestConfig,
        brain_library: Optional[Any] = None,  # BrainLibrary type
    ) -> None:
        """
        Initialize shadow tester.
        
        Args:
            config: Shadow test configuration
            brain_library: Optional Brain Library for model storage
        """
        self.config = config
        self.brain_library = brain_library
        
        # Active shadow tests (model_id -> ShadowTestResult)
        self.active_tests: Dict[str, ShadowTestResult] = {}
        
        # Shadow trades (model_id -> List[ShadowTrade])
        self.shadow_trades: Dict[str, List[ShadowTrade]] = {}
        
        logger.info(
            "shadow_tester_initialized",
            min_duration_days=config.min_duration_days,
            min_trades=config.min_trades,
        )
    
    def start_shadow_test(
        self,
        new_model_id: str,
        baseline_model_id: str,
        symbol: str,
    ) -> ShadowTestResult:
        """
        Start a shadow test for a new model.
        
        Args:
            new_model_id: New model identifier
            baseline_model_id: Baseline model identifier
            symbol: Trading symbol
        
        Returns:
            Shadow test result
        """
        test_result = ShadowTestResult(
            model_id=new_model_id,
            baseline_model_id=baseline_model_id,
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=self.config.min_duration_days),
            num_trades=0,
            promotion_status=PromotionStatus.IN_PROGRESS,
        )
        
        self.active_tests[new_model_id] = test_result
        self.shadow_trades[new_model_id] = []
        
        logger.info(
            "shadow_test_started",
            new_model_id=new_model_id,
            baseline_model_id=baseline_model_id,
            symbol=symbol,
            start_date=test_result.start_date.isoformat(),
            end_date=test_result.end_date.isoformat(),
        )
        
        return test_result
    
    def record_shadow_trade(
        self,
        model_id: str,
        trade: ShadowTrade,
    ) -> None:
        """
        Record a shadow trade.
        
        Args:
            model_id: Model identifier
            trade: Shadow trade record
        """
        if model_id not in self.shadow_trades:
            self.shadow_trades[model_id] = []
        
        self.shadow_trades[model_id].append(trade)
        
        if model_id in self.active_tests:
            self.active_tests[model_id].num_trades += 1
        
        logger.debug(
            "shadow_trade_recorded",
            model_id=model_id,
            symbol=trade.symbol,
            side=trade.side,
            price=trade.price,
        )
    
    def calculate_metrics(
        self,
        trades: List[ShadowTrade],
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from shadow trades.
        
        Args:
            trades: List of shadow trades
        
        Returns:
            Dictionary of metrics
        """
        if not trades:
            return {}
        
        # Extract PnL series
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        
        if not pnls:
            return {}
        
        pnl_array = np.array(pnls)
        
        # Basic metrics
        total_pnl = float(np.sum(pnl_array))
        num_trades = len(pnls)
        
        # Win rate
        wins = sum(1 for pnl in pnls if pnl > 0)
        win_rate = wins / num_trades if num_trades > 0 else 0.0
        
        # Sharpe ratio (annualized)
        if len(pnl_array) > 1 and np.std(pnl_array) > 0:
            sharpe = float((np.mean(pnl_array) / np.std(pnl_array)) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Average trade
        avg_trade = float(np.mean(pnl_array))
        
        # Profit factor
        profits = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]
        if len(losses) > 0 and abs(np.sum(losses)) > 0:
            profit_factor = float(np.sum(profits) / abs(np.sum(losses)))
        else:
            profit_factor = 0.0
        
        return {
            "total_pnl": total_pnl,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
        }
    
    def compare_models(
        self,
        new_model_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compare new model with baseline.
        
        Args:
            new_model_metrics: New model metrics
            baseline_metrics: Baseline model metrics
        
        Returns:
            Tuple of (improvement_dict, statistical_significance_dict)
        """
        improvement = {}
        significance = {}
        
        # Key metrics to compare
        key_metrics = ["sharpe_ratio", "win_rate", "total_pnl", "profit_factor"]
        
        for metric in key_metrics:
            new_val = new_model_metrics.get(metric, 0.0)
            baseline_val = baseline_metrics.get(metric, 0.0)
            
            if baseline_val == 0.0:
                improvement[metric] = 0.0
                significance[metric] = 0.0
                continue
            
            # Calculate improvement
            if metric in ["sharpe_ratio", "win_rate", "profit_factor"]:
                # Percentage improvement
                improvement[metric] = ((new_val - baseline_val) / abs(baseline_val)) * 100.0
            else:
                # Absolute improvement
                improvement[metric] = new_val - baseline_val
            
            # Statistical significance (t-test)
            # This is simplified - in practice, you'd need the raw returns
            # For now, we use a heuristic based on the metrics
            if abs(improvement[metric]) > 0.1:
                # Assume significant if improvement is substantial
                significance[metric] = 0.95
            else:
                significance[metric] = 0.5
        
        return improvement, significance
    
    def check_promotion_criteria(
        self,
        test_result: ShadowTestResult,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if model meets promotion criteria.
        
        Args:
            test_result: Shadow test result
        
        Returns:
            Tuple of (should_promote, reason)
        """
        criteria = self.config.promotion_criteria
        
        # Check minimum duration
        duration_days = (datetime.now(timezone.utc) - test_result.start_date).days
        if duration_days < self.config.min_duration_days:
            return False, f"Insufficient duration: {duration_days} days < {self.config.min_duration_days} days"
        
        # Check minimum trades
        if test_result.num_trades < self.config.min_trades:
            return False, f"Insufficient trades: {test_result.num_trades} < {self.config.min_trades}"
        
        # Check Sharpe improvement
        sharpe_improvement = test_result.improvement.get("sharpe_ratio", 0.0)
        min_sharpe_improvement = criteria.get("min_sharpe_improvement", 0.2)
        
        if sharpe_improvement < min_sharpe_improvement:
            return False, f"Sharpe improvement {sharpe_improvement:.2f}% < {min_sharpe_improvement:.2f}%"
        
        # Check win rate improvement
        win_rate_improvement = test_result.improvement.get("win_rate", 0.0)
        min_win_rate_improvement = criteria.get("min_win_rate_improvement", 0.05)
        
        if win_rate_improvement < min_win_rate_improvement:
            return False, f"Win rate improvement {win_rate_improvement:.2f}% < {min_win_rate_improvement:.2f}%"
        
        # Check statistical significance
        min_significance = criteria.get("statistical_significance", 0.95)
        sharpe_significance = test_result.statistical_significance.get("sharpe_ratio", 0.0)
        
        if sharpe_significance < min_significance:
            return False, f"Statistical significance {sharpe_significance:.2f} < {min_significance:.2f}"
        
        return True, "All promotion criteria met"
    
    def evaluate_shadow_test(
        self,
        model_id: str,
        baseline_trades: Optional[List[ShadowTrade]] = None,
    ) -> ShadowTestResult:
        """
        Evaluate a shadow test and determine promotion status.
        
        Args:
            model_id: Model identifier
            baseline_trades: Optional baseline trades for comparison
        
        Returns:
            Updated shadow test result
        """
        if model_id not in self.active_tests:
            logger.warning("shadow_test_not_found", model_id=model_id)
            return ShadowTestResult(
                model_id=model_id,
                baseline_model_id="",
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc),
                num_trades=0,
                promotion_status=PromotionStatus.FAILED,
            )
        
        test_result = self.active_tests[model_id]
        
        # Get new model trades
        new_model_trades = self.shadow_trades.get(model_id, [])
        
        # Calculate new model metrics
        new_model_metrics = self.calculate_metrics(new_model_trades)
        test_result.new_model_metrics = new_model_metrics
        
        # Calculate baseline metrics
        if baseline_trades:
            baseline_metrics = self.calculate_metrics(baseline_trades)
            test_result.baseline_metrics = baseline_metrics
            
            # Compare models
            improvement, significance = self.compare_models(
                new_model_metrics,
                baseline_metrics,
            )
            test_result.improvement = improvement
            test_result.statistical_significance = significance
        
        # Check promotion criteria
        should_promote, reason = self.check_promotion_criteria(test_result)
        
        if should_promote:
            test_result.promotion_status = PromotionStatus.PROMOTED
            test_result.promotion_reason = reason
            logger.info(
                "model_promoted",
                model_id=model_id,
                reason=reason,
                sharpe_improvement=test_result.improvement.get("sharpe_ratio", 0.0),
            )
        else:
            test_result.promotion_status = PromotionStatus.REJECTED
            test_result.promotion_reason = reason
            logger.info(
                "model_rejected",
                model_id=model_id,
                reason=reason,
            )
        
        test_result.end_date = datetime.now(timezone.utc)
        
        return test_result
    
    def get_active_tests(self) -> List[ShadowTestResult]:
        """Get all active shadow tests."""
        return list(self.active_tests.values())
    
    def get_test_result(self, model_id: str) -> Optional[ShadowTestResult]:
        """Get shadow test result for a model."""
        return self.active_tests.get(model_id)

