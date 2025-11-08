"""
Trust Index System

Calculates a confidence weight blending model accuracy, drawdown duration,
and recovery slope to decide when to reduce capital allocation.

Key Features:
- Model accuracy tracking
- Drawdown duration monitoring
- Recovery slope calculation
- Trust index calculation (0-1)
- Capital allocation adjustment
- Integration with Council voting

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrustIndexMetrics:
    """Trust index metrics for a model"""
    model_id: str
    trust_index: float  # 0-1, higher is better
    accuracy_score: float  # Model prediction accuracy
    drawdown_duration_days: int  # Days in drawdown
    recovery_slope: float  # Recovery rate (positive = recovering)
    recent_performance: float  # Recent performance score
    capital_allocation: float  # Recommended capital allocation (0-1)
    last_updated: datetime
    recommendation: str  # "INCREASE", "MAINTAIN", "REDUCE", "PAUSE"


class TrustIndexCalculator:
    """
    Calculates trust index for models based on performance metrics.
    
    Trust Index Components:
    1. Model Accuracy: How well the model predicts
    2. Drawdown Duration: How long the model has been in drawdown
    3. Recovery Slope: How quickly the model is recovering
    4. Recent Performance: Performance over recent period
    
    Usage:
        calculator = TrustIndexCalculator()
        metrics = calculator.calculate_trust_index(
            model_id="model_1",
            accuracy=0.75,
            drawdown_duration=5,
            recovery_slope=0.02,
            recent_performance=0.15
        )
    """
    
    def __init__(
        self,
        accuracy_weight: float = 0.3,
        drawdown_weight: float = 0.25,
        recovery_weight: float = 0.25,
        recent_performance_weight: float = 0.2,
        max_drawdown_days: int = 30,  # Maximum acceptable drawdown duration
        min_trust_threshold: float = 0.5,  # Minimum trust to maintain allocation
    ):
        """
        Initialize trust index calculator.
        
        Args:
            accuracy_weight: Weight for accuracy component
            drawdown_weight: Weight for drawdown component
            recovery_weight: Weight for recovery component
            recent_performance_weight: Weight for recent performance
            max_drawdown_days: Maximum acceptable drawdown duration
            min_trust_threshold: Minimum trust threshold
        """
        self.accuracy_weight = accuracy_weight
        self.drawdown_weight = drawdown_weight
        self.recovery_weight = recovery_weight
        self.recent_performance_weight = recent_performance_weight
        self.max_drawdown_days = max_drawdown_days
        self.min_trust_threshold = min_trust_threshold
        
        # Validate weights sum to 1.0
        total_weight = (
            accuracy_weight + drawdown_weight + 
            recovery_weight + recent_performance_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        logger.info(
            "trust_index_calculator_initialized",
            accuracy_weight=accuracy_weight,
            drawdown_weight=drawdown_weight,
            recovery_weight=recovery_weight,
            recent_performance_weight=recent_performance_weight,
            max_drawdown_days=max_drawdown_days
        )
    
    def calculate_trust_index(
        self,
        model_id: str,
        accuracy: float,  # 0-1, prediction accuracy
        drawdown_duration_days: int,  # Days in drawdown
        recovery_slope: float,  # Recovery rate (positive = recovering)
        recent_performance: float,  # Recent performance (e.g., Sharpe ratio)
        current_allocation: float = 1.0  # Current capital allocation
    ) -> TrustIndexMetrics:
        """
        Calculate trust index for a model.
        
        Args:
            model_id: Model identifier
            accuracy: Model accuracy (0-1)
            drawdown_duration_days: Days in drawdown
            recovery_slope: Recovery slope (positive = recovering)
            recent_performance: Recent performance metric
            current_allocation: Current capital allocation
        
        Returns:
            TrustIndexMetrics
        """
        # Normalize components to 0-1 scale
        accuracy_score = self._normalize_accuracy(accuracy)
        drawdown_score = self._normalize_drawdown(drawdown_duration_days)
        recovery_score = self._normalize_recovery(recovery_slope)
        performance_score = self._normalize_performance(recent_performance)
        
        # Calculate weighted trust index
        trust_index = (
            accuracy_score * self.accuracy_weight +
            drawdown_score * self.drawdown_weight +
            recovery_score * self.recovery_weight +
            performance_score * self.recent_performance_weight
        )
        
        # Calculate recommended capital allocation
        capital_allocation = self._calculate_capital_allocation(
            trust_index,
            drawdown_duration_days,
            recovery_slope
        )
        
        # Get recommendation
        recommendation = self._get_recommendation(
            trust_index,
            drawdown_duration_days,
            recovery_slope,
            current_allocation,
            capital_allocation
        )
        
        metrics = TrustIndexMetrics(
            model_id=model_id,
            trust_index=trust_index,
            accuracy_score=accuracy_score,
            drawdown_duration_days=drawdown_duration_days,
            recovery_slope=recovery_slope,
            recent_performance=recent_performance,
            capital_allocation=capital_allocation,
            last_updated=datetime.now(timezone.utc),
            recommendation=recommendation
        )
        
        logger.info(
            "trust_index_calculated",
            model_id=model_id,
            trust_index=trust_index,
            capital_allocation=capital_allocation,
            recommendation=recommendation
        )
        
        return metrics
    
    def _normalize_accuracy(self, accuracy: float) -> float:
        """Normalize accuracy to 0-1 scale"""
        # Accuracy is already 0-1, but we can apply a curve
        # Higher accuracy gets more weight (exponential)
        return min(1.0, accuracy ** 0.8)
    
    def _normalize_drawdown(self, drawdown_duration_days: int) -> float:
        """Normalize drawdown duration to 0-1 scale (higher is worse)"""
        # Longer drawdown = lower score
        if drawdown_duration_days <= 0:
            return 1.0
        
        # Exponential decay: score decreases faster as drawdown increases
        score = np.exp(-drawdown_duration_days / self.max_drawdown_days)
        return max(0.0, min(1.0, score))
    
    def _normalize_recovery(self, recovery_slope: float) -> float:
        """Normalize recovery slope to 0-1 scale"""
        # Positive slope = recovering = good
        # Negative slope = worsening = bad
        # Normalize to 0-1 using sigmoid
        normalized = 1.0 / (1.0 + np.exp(-recovery_slope * 10))
        return normalized
    
    def _normalize_performance(self, recent_performance: float) -> float:
        """Normalize recent performance to 0-1 scale"""
        # Assume performance is Sharpe ratio or similar
        # Normalize using sigmoid: 0 Sharpe = 0.5, positive = higher, negative = lower
        normalized = 1.0 / (1.0 + np.exp(-recent_performance * 2))
        return normalized
    
    def _calculate_capital_allocation(
        self,
        trust_index: float,
        drawdown_duration_days: int,
        recovery_slope: float
    ) -> float:
        """Calculate recommended capital allocation"""
        # Start with trust index as base allocation
        base_allocation = trust_index
        
        # Reduce allocation if drawdown is too long
        if drawdown_duration_days > self.max_drawdown_days:
            reduction_factor = 1.0 - (drawdown_duration_days - self.max_drawdown_days) / self.max_drawdown_days
            base_allocation *= max(0.1, reduction_factor)
        
        # Increase allocation if recovering quickly
        if recovery_slope > 0:
            recovery_boost = min(0.2, recovery_slope * 2)
            base_allocation = min(1.0, base_allocation + recovery_boost)
        
        # Ensure allocation is within bounds
        allocation = max(0.0, min(1.0, base_allocation))
        
        return allocation
    
    def _get_recommendation(
        self,
        trust_index: float,
        drawdown_duration_days: int,
        recovery_slope: float,
        current_allocation: float,
        recommended_allocation: float
    ) -> str:
        """Get recommendation based on trust index"""
        # Pause if trust is very low or drawdown is extreme
        if trust_index < 0.3 or drawdown_duration_days > self.max_drawdown_days * 2:
            return "PAUSE"
        
        # Reduce if trust is below threshold
        if trust_index < self.min_trust_threshold:
            return "REDUCE"
        
        # Increase if trust is high and recovering
        if trust_index > 0.7 and recovery_slope > 0 and recommended_allocation > current_allocation:
            return "INCREASE"
        
        # Maintain otherwise
        return "MAINTAIN"
    
    def update_trust_index_from_trades(
        self,
        model_id: str,
        trades: List[Dict[str, float]],  # [{"return": 0.01, "prediction": "buy", "actual": "buy"}, ...]
        lookback_days: int = 30
    ) -> TrustIndexMetrics:
        """
        Calculate trust index from trade history.
        
        Args:
            model_id: Model identifier
            trades: List of trade results
            lookback_days: Number of days to look back
        
        Returns:
            TrustIndexMetrics
        """
        if not trades:
            return TrustIndexMetrics(
                model_id=model_id,
                trust_index=0.0,
                accuracy_score=0.0,
                drawdown_duration_days=0,
                recovery_slope=0.0,
                recent_performance=0.0,
                capital_allocation=0.0,
                last_updated=datetime.now(timezone.utc),
                recommendation="PAUSE"
            )
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(trades)
        
        # Calculate drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(trades, lookback_days)
        
        # Calculate recovery slope
        recovery_slope = self._calculate_recovery_slope(trades, lookback_days)
        
        # Calculate recent performance
        recent_performance = self._calculate_recent_performance(trades, lookback_days)
        
        # Calculate trust index
        return self.calculate_trust_index(
            model_id=model_id,
            accuracy=accuracy,
            drawdown_duration_days=drawdown_duration,
            recovery_slope=recovery_slope,
            recent_performance=recent_performance
        )
    
    def _calculate_accuracy(self, trades: List[Dict[str, float]]) -> float:
        """Calculate prediction accuracy from trades"""
        if not trades:
            return 0.0
        
        correct = 0
        total = 0
        
        for trade in trades:
            prediction = trade.get("prediction")
            actual = trade.get("actual")
            
            if prediction is not None and actual is not None:
                if prediction == actual:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_drawdown_duration(
        self,
        trades: List[Dict[str, float]],
        lookback_days: int
    ) -> int:
        """Calculate drawdown duration in days"""
        if not trades:
            return 0
        
        # Get returns
        returns = [t.get("return", 0.0) for t in trades]
        
        # Calculate cumulative returns
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        
        # Count days in drawdown (simplified - would need timestamps for accurate days)
        days_in_drawdown = sum(1 for dd in drawdowns if dd > 0)
        
        return days_in_drawdown
    
    def _calculate_recovery_slope(
        self,
        trades: List[Dict[str, float]],
        lookback_days: int
    ) -> float:
        """Calculate recovery slope (positive = recovering)"""
        if len(trades) < 2:
            return 0.0
        
        # Get recent returns
        recent_returns = [t.get("return", 0.0) for t in trades[-lookback_days:]]
        
        if len(recent_returns) < 2:
            return 0.0
        
        # Calculate slope of returns (trend)
        x = np.arange(len(recent_returns))
        slope = np.polyfit(x, recent_returns, 1)[0]
        
        return float(slope)
    
    def _calculate_recent_performance(
        self,
        trades: List[Dict[str, float]],
        lookback_days: int
    ) -> float:
        """Calculate recent performance (e.g., Sharpe ratio)"""
        if not trades:
            return 0.0
        
        # Get recent returns
        recent_returns = [t.get("return", 0.0) for t in trades[-lookback_days:]]
        
        if len(recent_returns) < 2:
            return 0.0
        
        # Calculate Sharpe ratio
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        return float(sharpe)

