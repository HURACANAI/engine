"""
Adaptive Loss Function - Regime-Aware Loss Switching

Tracks which loss functions produce stable forecasts and adapts based on market regime.
- For trend markets → MAE (Mean Absolute Error)
- For range markets → MSE (Mean Squared Error)
- Tracks MSE, MAE, and directional accuracy over time

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class LossType(Enum):
    """Loss function types."""
    MSE = "mse"  # Mean Squared Error
    MAE = "mae"  # Mean Absolute Error
    HUBER = "huber"  # Huber loss
    QUANTILE = "quantile"  # Quantile loss
    DIRECTIONAL = "directional"  # Directional accuracy


@dataclass
class LossMetrics:
    """Loss function performance metrics."""
    loss_type: LossType
    mse: float
    mae: float
    directional_accuracy: float
    stability_score: float  # How stable the loss is over time
    last_updated: datetime


class AdaptiveLoss:
    """
    Adaptive loss function that switches based on market regime.
    
    Usage:
        adaptive_loss = AdaptiveLoss()
        
        # Track loss performance
        adaptive_loss.update_metrics(
            predictions=y_pred,
            actuals=y_true,
            regime="trending"
        )
        
        # Get best loss for current regime
        best_loss = adaptive_loss.get_best_loss("trending")
        
        # Use in training
        loss_fn = adaptive_loss.get_loss_function(best_loss)
    """
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize adaptive loss manager.
        
        Args:
            lookback_periods: Number of periods to track for stability
        """
        self.lookback_periods = lookback_periods
        self.regime_metrics: Dict[str, Dict[LossType, LossMetrics]] = {}
        self.loss_history: Dict[str, Dict[LossType, List[float]]] = {}
        
        logger.info(
            "adaptive_loss_initialized",
            lookback_periods=lookback_periods
        )
    
    def update_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        regime: str,
        loss_type: Optional[LossType] = None
    ) -> Dict[str, float]:
        """
        Update loss metrics for a regime.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            regime: Market regime
            loss_type: Specific loss type to evaluate (None = all)
        
        Returns:
            Dictionary of calculated metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Initialize regime if needed
        if regime not in self.regime_metrics:
            self.regime_metrics[regime] = {}
            self.loss_history[regime] = {}
        
        # Calculate all loss types
        loss_types = [loss_type] if loss_type else list(LossType)
        
        metrics = {}
        for lt in loss_types:
            if lt == LossType.MSE:
                loss_value = self._mse(predictions, actuals)
            elif lt == LossType.MAE:
                loss_value = self._mae(predictions, actuals)
            elif lt == LossType.HUBER:
                loss_value = self._huber(predictions, actuals)
            elif lt == LossType.QUANTILE:
                loss_value = self._quantile(predictions, actuals)
            elif lt == LossType.DIRECTIONAL:
                loss_value = 1.0 - self._directional_accuracy(predictions, actuals)
            else:
                continue
            
            # Update history
            if lt not in self.loss_history[regime]:
                self.loss_history[regime][lt] = []
            
            self.loss_history[regime][lt].append(loss_value)
            
            # Keep only lookback period
            if len(self.loss_history[regime][lt]) > self.lookback_periods:
                self.loss_history[regime][lt] = self.loss_history[regime][lt][-self.lookback_periods:]
            
            # Calculate stability (lower std = more stable)
            history = self.loss_history[regime][lt]
            stability = 1.0 / (1.0 + np.std(history)) if len(history) > 1 else 1.0
            
            # Calculate all metrics
            mse = self._mse(predictions, actuals)
            mae = self._mae(predictions, actuals)
            directional_acc = self._directional_accuracy(predictions, actuals)
            
            # Store metrics
            self.regime_metrics[regime][lt] = LossMetrics(
                loss_type=lt,
                mse=mse,
                mae=mae,
                directional_accuracy=directional_acc,
                stability_score=stability,
                last_updated=datetime.now()
            )
            
            metrics[lt.value] = {
                'loss': loss_value,
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_acc,
                'stability': stability
            }
        
        logger.debug(
            "loss_metrics_updated",
            regime=regime,
            metrics=metrics
        )
        
        return metrics
    
    def get_best_loss(self, regime: str) -> LossType:
        """
        Get best loss function for a regime.
        
        Selection criteria:
        - For trending: Prefer MAE (less sensitive to outliers)
        - For ranging: Prefer MSE (penalizes large errors)
        - Otherwise: Choose based on stability and accuracy
        
        Args:
            regime: Market regime
        
        Returns:
            Best LossType for the regime
        """
        if regime not in self.regime_metrics:
            # Default based on regime type
            if "trend" in regime.lower():
                return LossType.MAE
            elif "range" in regime.lower():
                return LossType.MSE
            else:
                return LossType.MSE
        
        metrics = self.regime_metrics[regime]
        
        if not metrics:
            return LossType.MSE
        
        # Score each loss type
        scores = {}
        for loss_type, loss_metrics in metrics.items():
            # Composite score: stability * (1 - normalized_loss) * directional_accuracy
            normalized_loss = min(1.0, loss_metrics.mae / (loss_metrics.mae + 1e-8))
            score = (
                loss_metrics.stability_score *
                (1.0 - normalized_loss) *
                loss_metrics.directional_accuracy
            )
            scores[loss_type] = score
        
        # Return best scoring loss
        best_loss = max(scores.items(), key=lambda x: x[1])[0]
        
        logger.info(
            "best_loss_selected",
            regime=regime,
            best_loss=best_loss.value,
            score=scores[best_loss]
        )
        
        return best_loss
    
    def get_loss_function(self, loss_type: LossType):
        """
        Get loss function callable.
        
        Args:
            loss_type: Type of loss function
        
        Returns:
            Loss function callable
        """
        if loss_type == LossType.MSE:
            return self._mse
        elif loss_type == LossType.MAE:
            return self._mae
        elif loss_type == LossType.HUBER:
            return self._huber
        elif loss_type == LossType.QUANTILE:
            return self._quantile
        elif loss_type == LossType.DIRECTIONAL:
            return lambda p, a: 1.0 - self._directional_accuracy(p, a)
        else:
            return self._mse
    
    def _mse(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((predictions - actuals) ** 2))
    
    def _mae(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(predictions - actuals)))
    
    def _huber(self, predictions: np.ndarray, actuals: np.ndarray, delta: float = 1.0) -> float:
        """Huber loss (smooth MAE/MSE)."""
        error = predictions - actuals
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return float(np.mean(0.5 * quadratic ** 2 + delta * linear))
    
    def _quantile(self, predictions: np.ndarray, actuals: np.ndarray, quantile: float = 0.5) -> float:
        """Quantile loss."""
        error = actuals - predictions
        return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))
    
    def _directional_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Directional accuracy (percentage of correct direction predictions)."""
        if len(predictions) < 2:
            return 0.0
        
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        
        correct = np.sum(pred_direction == actual_direction)
        return float(correct / len(pred_direction))
    
    def get_regime_metrics(self, regime: str) -> Dict[str, LossMetrics]:
        """Get all loss metrics for a regime."""
        return self.regime_metrics.get(regime, {})
    
    def get_stability_report(self) -> Dict[str, Dict[str, float]]:
        """Get stability report for all regimes."""
        report = {}
        for regime, metrics in self.regime_metrics.items():
            report[regime] = {
                loss_type.value: {
                    'stability': m.stability_score,
                    'mse': m.mse,
                    'mae': m.mae,
                    'directional_accuracy': m.directional_accuracy
                }
                for loss_type, m in metrics.items()
            }
        return report

