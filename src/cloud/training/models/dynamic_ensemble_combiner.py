"""
Dynamic Ensemble Combiner - Regime-Aware Model Weighting

Dynamically weights models based on:
- Recent performance
- Market regime
- Model agreement
- Historical performance patterns

Usage:
    combiner = DynamicEnsembleCombiner()
    
    # Update with recent performance
    combiner.update_performance('xgboost', won=True, profit_bps=150, regime='TREND')
    
    # Get dynamic weights
    weights = combiner.get_weights(current_regime='TREND')
    
    # Combine predictions
    ensemble_pred = combiner.combine(predictions, weights)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    wins: int = 0
    losses: int = 0
    total_profit_bps: float = 0.0
    recent_profit_bps: List[float] = None  # Last N trades
    win_rate: float = 0.0
    avg_profit_bps: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.recent_profit_bps is None:
            self.recent_profit_bps = []


class DynamicEnsembleCombiner:
    """
    Dynamically combines model predictions based on performance and regime.
    
    Features:
    - Regime-specific weighting
    - Recent performance tracking
    - Model agreement scoring
    - Automatic weight adjustment
    """

    def __init__(
        self,
        lookback_trades: int = 50,  # Number of recent trades to consider
        min_trades_for_weighting: int = 10,  # Minimum trades before weighting
        agreement_weight: float = 0.3,  # Weight for model agreement
        performance_weight: float = 0.7,  # Weight for performance
        decay_factor: float = 0.95,  # Decay for older performance
    ):
        """
        Initialize dynamic ensemble combiner.
        
        Args:
            lookback_trades: Number of recent trades to consider
            min_trades_for_weighting: Minimum trades before using performance weighting
            agreement_weight: Weight for model agreement in final weighting
            performance_weight: Weight for performance in final weighting
            decay_factor: Decay factor for older performance (0.95 = 5% decay per trade)
        """
        self.lookback_trades = lookback_trades
        self.min_trades_for_weighting = min_trades_for_weighting
        self.agreement_weight = agreement_weight
        self.performance_weight = performance_weight
        self.decay_factor = decay_factor
        
        # Performance tracking by model and regime
        self.performance: Dict[str, Dict[str, ModelPerformance]] = defaultdict(
            lambda: defaultdict(ModelPerformance)
        )
        
        # Model agreement tracking
        self.agreement_history: List[Dict[str, float]] = []
        
        logger.info(
            "dynamic_ensemble_combiner_initialized",
            lookback_trades=lookback_trades,
            min_trades=min_trades_for_weighting,
        )

    def update_performance(
        self,
        model_name: str,
        won: bool,
        profit_bps: float,
        regime: str,
    ):
        """
        Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            won: Whether the trade was a winner
            profit_bps: Profit/loss in basis points
            regime: Market regime
        """
        perf = self.performance[model_name][regime]
        
        if won:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.total_profit_bps += profit_bps
        
        # Track recent profits
        perf.recent_profit_bps.append(profit_bps)
        if len(perf.recent_profit_bps) > self.lookback_trades:
            perf.recent_profit_bps.pop(0)
        
        # Update metrics
        total_trades = perf.wins + perf.losses
        if total_trades > 0:
            perf.win_rate = perf.wins / total_trades
            perf.avg_profit_bps = perf.total_profit_bps / total_trades
        
        # Calculate Sharpe ratio from recent trades
        if len(perf.recent_profit_bps) > 1:
            recent_profits = np.array(perf.recent_profit_bps)
            mean_profit = np.mean(recent_profits)
            std_profit = np.std(recent_profits)
            if std_profit > 0:
                perf.sharpe_ratio = mean_profit / std_profit
            else:
                perf.sharpe_ratio = 0.0
        
        perf.last_updated = datetime.utcnow()
        
        logger.debug(
            "performance_updated",
            model=model_name,
            regime=regime,
            win_rate=perf.win_rate,
            avg_profit=perf.avg_profit_bps,
        )

    def get_weights(
        self,
        current_regime: str,
        model_names: List[str],
        predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Get dynamic weights for models based on performance and agreement.
        
        Args:
            current_regime: Current market regime
            model_names: List of model names
            predictions: Model predictions (for agreement calculation)
            
        Returns:
            Dictionary of weights for each model
        """
        # 1. Performance-based weights
        perf_weights = self._get_performance_weights(model_names, current_regime)
        
        # 2. Agreement-based weights (if predictions provided)
        if predictions:
            agreement_weights = self._get_agreement_weights(model_names, predictions)
        else:
            agreement_weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # 3. Combine weights
        final_weights = {}
        for name in model_names:
            perf_w = perf_weights.get(name, 0.0)
            agree_w = agreement_weights.get(name, 0.0)
            
            # Weighted combination
            final_weight = (
                self.performance_weight * perf_w +
                self.agreement_weight * agree_w
            )
            final_weights[name] = final_weight
        
        # Normalize
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            # Equal weights if all zero
            equal_weight = 1.0 / len(model_names)
            final_weights = {k: equal_weight for k in model_names}
        
        logger.debug(
            "dynamic_weights_calculated",
            regime=current_regime,
            weights=final_weights,
        )
        
        return final_weights

    def _get_performance_weights(
        self,
        model_names: List[str],
        regime: str,
    ) -> Dict[str, float]:
        """Get weights based on model performance."""
        weights = {}
        scores = {}
        
        for name in model_names:
            perf = self.performance[name][regime]
            total_trades = perf.wins + perf.losses
            
            if total_trades < self.min_trades_for_weighting:
                # Not enough data - use equal weight
                scores[name] = 0.5
            else:
                # Score based on win rate and Sharpe ratio
                win_rate_score = perf.win_rate
                sharpe_score = max(0.0, perf.sharpe_ratio / 2.0)  # Normalize Sharpe
                scores[name] = 0.6 * win_rate_score + 0.4 * sharpe_score
        
        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            weights = {k: v / total_score for k, v in scores.items()}
        else:
            equal_weight = 1.0 / len(model_names)
            weights = {k: equal_weight for k in model_names}
        
        return weights

    def _get_agreement_weights(
        self,
        model_names: List[str],
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Get weights based on model agreement."""
        if len(predictions) < 2:
            # No agreement to calculate
            equal_weight = 1.0 / len(model_names)
            return {name: equal_weight for name in model_names}
        
        # Calculate pairwise agreement
        agreement_scores = {}
        pred_array = np.array([predictions[name] for name in model_names])
        
        for i, name in enumerate(model_names):
            # Agreement = inverse of distance from mean
            pred = pred_array[i]
            mean_pred = pred_array.mean(axis=0)
            distance = np.abs(pred - mean_pred).mean()
            
            # Lower distance = higher agreement = higher weight
            agreement_scores[name] = 1.0 / (1.0 + distance)
        
        # Normalize
        total_agreement = sum(agreement_scores.values())
        if total_agreement > 0:
            weights = {k: v / total_agreement for k, v in agreement_scores.items()}
        else:
            equal_weight = 1.0 / len(model_names)
            weights = {k: equal_weight for k in model_names}
        
        return weights

    def combine(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Combine predictions using weights.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of weights for each model
            
        Returns:
            Combined prediction
        """
        weighted_sum = None
        
        for name, pred in predictions.items():
            weight = weights.get(name, 0.0)
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
        
        return weighted_sum

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all models."""
        summary = {}
        
        for model_name, regime_perf in self.performance.items():
            summary[model_name] = {}
            for regime, perf in regime_perf.items():
                summary[model_name][regime] = {
                    'win_rate': perf.win_rate,
                    'avg_profit_bps': perf.avg_profit_bps,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'total_trades': perf.wins + perf.losses,
                }
        
        return summary

    def reset_performance(self, model_name: Optional[str] = None):
        """Reset performance tracking."""
        if model_name:
            if model_name in self.performance:
                del self.performance[model_name]
        else:
            self.performance.clear()
        
        logger.info("performance_reset", model=model_name or "all")

