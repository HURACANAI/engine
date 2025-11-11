"""
Enhanced Meta Combiner

Enhanced meta combiner with adaptive weighting, hyperparameter tuning, and performance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..engines.enhanced_engine_interface import (
    BaseEnhancedEngine,
    EnhancedEngineInput,
    EnhancedEngineOutput,
    TradingHorizon,
)

logger = structlog.get_logger(__name__)


@dataclass
class EnginePerformance:
    """Engine performance metrics."""
    engine_id: str
    accuracy: float  # Hit rate (0.0 to 1.0)
    net_edge_bps: float  # Net edge in basis points
    sharpe_ratio: float  # Sharpe ratio
    win_rate: float  # Win rate (0.0 to 1.0)
    avg_win_bps: float  # Average win in basis points
    avg_loss_bps: float  # Average loss in basis points
    total_trades: int  # Total number of trades
    last_updated: datetime
    # Performance decay
    accuracy_decay: float = 0.0  # Accuracy decay rate
    edge_decay: float = 0.0  # Edge decay rate
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineWeight:
    """Enhanced engine weight with hyperparameters."""
    engine_id: str
    weight: float  # Weight (0.0 to 1.0)
    accuracy: float  # Recent accuracy (0.0 to 1.0)
    net_edge_bps: float  # Recent net edge in basis points
    last_updated: datetime
    # Hyperparameters
    ema_alpha: float = 0.1  # EMA smoothing factor
    confidence_multiplier: float = 1.0  # Confidence multiplier
    edge_clip_min: float = -10.0  # Minimum edge clipping
    edge_clip_max: float = 10.0  # Maximum edge clipping
    # Performance tracking
    performance_window_days: int = 7  # Performance window in days
    min_trades_for_weight: int = 10  # Minimum trades for weight calculation
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedMetaOutput:
    """Enhanced meta combiner output."""
    direction: str  # buy, sell, wait, hold
    edge_bps_before_costs: float  # Combined edge in basis points
    confidence_0_1: float  # Combined confidence (0.0 to 1.0)
    horizon_minutes: int  # Prediction horizon in minutes
    horizon_type: TradingHorizon  # Trading horizon type
    engine_weights: Dict[str, float]  # Engine weights used
    contributing_engines: Dict[str, float]  # Engine contributions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for adaptive tuning."""
    # EMA alpha range
    ema_alpha_min: float = 0.05
    ema_alpha_max: float = 0.3
    ema_alpha_step: float = 0.05
    # Confidence multiplier range
    confidence_multiplier_min: float = 0.5
    confidence_multiplier_max: float = 2.0
    confidence_multiplier_step: float = 0.1
    # Edge clipping range
    edge_clip_min: float = -20.0
    edge_clip_max: float = 20.0
    # Tuning frequency
    tuning_frequency_days: int = 7  # Tune every 7 days
    # Performance window
    performance_window_days: int = 30  # 30 days performance window
    min_trades_for_tuning: int = 50  # Minimum trades for hyperparameter tuning


class EnhancedMetaCombiner:
    """Enhanced meta combiner with adaptive weighting and hyperparameter tuning."""
    
    def __init__(
        self,
        symbol: str,
        ema_alpha: float = 0.1,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
        accuracy_threshold: float = 0.45,
        hyperparameter_config: Optional[HyperparameterConfig] = None,
    ):
        """Initialize enhanced meta combiner.
        
        Args:
            symbol: Trading symbol
            ema_alpha: EMA smoothing factor (0.0 to 1.0)
            min_weight: Minimum weight for engines
            max_weight: Maximum weight for engines
            accuracy_threshold: Minimum accuracy to include engine
            hyperparameter_config: Hyperparameter configuration
        """
        self.symbol = symbol
        self.ema_alpha = ema_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.accuracy_threshold = accuracy_threshold
        self.hyperparameter_config = hyperparameter_config or HyperparameterConfig()
        
        self.engine_weights: Dict[str, EngineWeight] = {}
        self.engine_performances: Dict[str, EnginePerformance] = {}
        self.last_tuning_date: Optional[datetime] = None
        
        logger.info(
            "enhanced_meta_combiner_initialized",
            symbol=symbol,
            ema_alpha=ema_alpha,
            accuracy_threshold=accuracy_threshold,
        )
    
    def combine(
        self,
        engine_outputs: List[EnhancedEngineOutput],
        engine_ids: List[str],
        regime: str,
        horizon: TradingHorizon,
    ) -> EnhancedMetaOutput:
        """Combine engine outputs using adaptive weighted average.
        
        Args:
            engine_outputs: List of engine outputs
            engine_ids: List of engine IDs (corresponding to outputs)
            regime: Market regime
            horizon: Trading horizon
            
        Returns:
            Enhanced meta combiner output
        """
        if not engine_outputs or not engine_ids:
            return EnhancedMetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                horizon_type=horizon,
                engine_weights={},
                contributing_engines={},
                metadata={"reason": "no_engine_outputs"},
            )
        
        # Get weights for each engine
        weights = self._get_adaptive_weights(engine_ids, regime, horizon)
        
        # Filter engines by minimum weight and accuracy threshold
        filtered_outputs = []
        filtered_weights = {}
        
        for engine_output, engine_id in zip(engine_outputs, engine_ids):
            weight = weights.get(engine_id, 0.0)
            performance = self.engine_performances.get(engine_id)
            
            if weight >= self.min_weight:
                if performance is None or performance.accuracy >= self.accuracy_threshold:
                    filtered_outputs.append(engine_output)
                    filtered_weights[engine_id] = weight
        
        if not filtered_outputs:
            return EnhancedMetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                horizon_type=horizon,
                engine_weights=weights,
                contributing_engines={},
                metadata={"reason": "no_engines_pass_threshold"},
            )
        
        # Combine outputs with adaptive weighting
        total_weight = sum(filtered_weights.values())
        
        if total_weight == 0:
            return EnhancedMetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                horizon_type=horizon,
                engine_weights=weights,
                contributing_engines={},
                metadata={"reason": "total_weight_zero"},
            )
        
        # Weighted average with clipping and confidence adjustment
        weighted_edge = 0.0
        weighted_confidence = 0.0
        contributing_engines = {}
        
        for engine_output, engine_id in zip(filtered_outputs, filtered_weights.keys()):
            weight = filtered_weights[engine_id]
            weight_obj = self.engine_weights.get(engine_id)
            
            # Clip edge based on engine hyperparameters
            clipped_edge = engine_output.edge_bps_before_costs
            if weight_obj:
                clipped_edge = max(
                    weight_obj.edge_clip_min,
                    min(weight_obj.edge_clip_max, clipped_edge),
                )
            
            # Adjust confidence based on engine hyperparameters
            adjusted_confidence = engine_output.confidence_0_1
            if weight_obj:
                adjusted_confidence = adjusted_confidence * weight_obj.confidence_multiplier
                adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))
            
            # Accumulate weighted values
            weighted_edge += clipped_edge * weight
            weighted_confidence += adjusted_confidence * weight
            contributing_engines[engine_id] = weight
        
        # Normalize
        weighted_edge = weighted_edge / total_weight
        weighted_confidence = weighted_confidence / total_weight
        
        # Average horizon
        avg_horizon = int(sum(
            output.horizon_minutes for output in filtered_outputs
        ) / len(filtered_outputs))
        
        # Determine direction based on weighted edge
        direction = "wait"
        if weighted_edge > 5.0:
            direction = "buy"
        elif weighted_edge < -5.0:
            direction = "sell"
        
        # Check if we should hold (for swing trades)
        if horizon in [TradingHorizon.SWING, TradingHorizon.POSITION, TradingHorizon.CORE]:
            if weighted_confidence > 0.7 and abs(weighted_edge) > 3.0:
                # Strong signal for swing trade
                pass
            elif weighted_confidence < 0.5:
                # Weak signal, wait
                direction = "wait"
        
        return EnhancedMetaOutput(
            direction=direction,
            edge_bps_before_costs=weighted_edge,
            confidence_0_1=weighted_confidence,
            horizon_minutes=avg_horizon,
            horizon_type=horizon,
            engine_weights=weights,
            contributing_engines=contributing_engines,
            metadata={
                "engine_count": len(filtered_outputs),
                "total_weight": total_weight,
                "regime": regime,
                "horizon": horizon.value,
            },
        )
    
    def _get_adaptive_weights(
        self,
        engine_ids: List[str],
        regime: str,
        horizon: TradingHorizon,
    ) -> Dict[str, float]:
        """Get adaptive weights for engines.
        
        Args:
            engine_ids: List of engine IDs
            regime: Market regime
            horizon: Trading horizon
            
        Returns:
            Dictionary of engine_id to weight
        """
        weights = {}
        
        for engine_id in engine_ids:
            weight_obj = self.engine_weights.get(engine_id)
            performance = self.engine_performances.get(engine_id)
            
            if weight_obj and performance:
                # Use existing weight with performance adjustment
                weight = weight_obj.weight
                
                # Adjust weight based on performance decay
                if performance.accuracy_decay > 0:
                    weight = weight * (1.0 - performance.accuracy_decay)
                
                if performance.edge_decay > 0:
                    weight = weight * (1.0 - performance.edge_decay)
                
            elif weight_obj:
                # Use existing weight
                weight = weight_obj.weight
            else:
                # Initialize with default weight
                weight = 1.0 / len(engine_ids) if engine_ids else 0.0
            
            # Clip to min/max
            weight = max(self.min_weight, min(self.max_weight, weight))
            weights[engine_id] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def update_performance(
        self,
        engine_id: str,
        accuracy: float,
        net_edge_bps: float,
        sharpe_ratio: float,
        win_rate: float,
        avg_win_bps: float,
        avg_loss_bps: float,
        total_trades: int,
        timestamp: datetime,
    ) -> None:
        """Update engine performance metrics.
        
        Args:
            engine_id: Engine identifier
            accuracy: Hit rate (0.0 to 1.0)
            net_edge_bps: Net edge in basis points
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate (0.0 to 1.0)
            avg_win_bps: Average win in basis points
            avg_loss_bps: Average loss in basis points
            total_trades: Total number of trades
            timestamp: Timestamp for update
        """
        # Calculate performance decay
        existing_performance = self.engine_performances.get(engine_id)
        
        accuracy_decay = 0.0
        edge_decay = 0.0
        
        if existing_performance:
            # Calculate decay based on performance change
            accuracy_change = accuracy - existing_performance.accuracy
            edge_change = net_edge_bps - existing_performance.net_edge_bps
            
            # Decay if performance is worsening
            if accuracy_change < 0:
                accuracy_decay = abs(accuracy_change) * 0.5
            if edge_change < 0:
                edge_decay = abs(edge_change) / 100.0 * 0.1
        
        # Update performance
        self.engine_performances[engine_id] = EnginePerformance(
            engine_id=engine_id,
            accuracy=accuracy,
            net_edge_bps=net_edge_bps,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win_bps=avg_win_bps,
            avg_loss_bps=avg_loss_bps,
            total_trades=total_trades,
            last_updated=timestamp,
            accuracy_decay=accuracy_decay,
            edge_decay=edge_decay,
        )
        
        # Update weight using EMA
        self._update_weight(engine_id, accuracy, net_edge_bps, timestamp)
        
        logger.debug(
            "performance_updated",
            symbol=self.symbol,
            engine_id=engine_id,
            accuracy=accuracy,
            net_edge_bps=net_edge_bps,
            sharpe_ratio=sharpe_ratio,
        )
    
    def _update_weight(
        self,
        engine_id: str,
        accuracy: float,
        net_edge_bps: float,
        timestamp: datetime,
    ) -> None:
        """Update engine weight using EMA.
        
        Args:
            engine_id: Engine identifier
            accuracy: Recent accuracy (0.0 to 1.0)
            net_edge_bps: Recent net edge in basis points
            timestamp: Timestamp for update
        """
        weight_obj = self.engine_weights.get(engine_id)
        
        # Get EMA alpha from weight object or use default
        ema_alpha = weight_obj.ema_alpha if weight_obj else self.ema_alpha
        
        # Calculate new weight based on accuracy and net edge
        # Weight = accuracy * (1 + net_edge_bps / 100) * confidence_multiplier
        performance = self.engine_performances.get(engine_id)
        confidence_multiplier = weight_obj.confidence_multiplier if weight_obj else 1.0
        
        base_weight = accuracy * (1.0 + max(0.0, net_edge_bps / 100.0)) * confidence_multiplier
        
        # Get existing weight
        existing_weight = weight_obj.weight if weight_obj else (1.0 / 10.0)  # Default weight
        
        # Update using EMA
        new_weight = ema_alpha * base_weight + (1 - ema_alpha) * existing_weight
        
        # Clip to min/max
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        # Update or create weight object
        if weight_obj:
            weight_obj.weight = new_weight
            weight_obj.accuracy = accuracy
            weight_obj.net_edge_bps = net_edge_bps
            weight_obj.last_updated = timestamp
        else:
            self.engine_weights[engine_id] = EngineWeight(
                engine_id=engine_id,
                weight=new_weight,
                accuracy=accuracy,
                net_edge_bps=net_edge_bps,
                last_updated=timestamp,
                ema_alpha=ema_alpha,
                confidence_multiplier=confidence_multiplier,
            )
    
    def tune_hyperparameters(
        self,
        engine_id: str,
        performance_history: List[EnginePerformance],
    ) -> Dict[str, float]:
        """Tune hyperparameters for an engine.
        
        Args:
            engine_id: Engine identifier
            performance_history: Performance history for tuning
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        if len(performance_history) < self.hyperparameter_config.min_trades_for_tuning:
            logger.warning(
                "insufficient_trades_for_tuning",
                engine_id=engine_id,
                trades=len(performance_history),
                min_trades=self.hyperparameter_config.min_trades_for_tuning,
            )
            return {}
        
        # Grid search for optimal hyperparameters
        best_score = -float('inf')
        best_params = {}
        
        for ema_alpha in self._generate_range(
            self.hyperparameter_config.ema_alpha_min,
            self.hyperparameter_config.ema_alpha_max,
            self.hyperparameter_config.ema_alpha_step,
        ):
            for confidence_multiplier in self._generate_range(
                self.hyperparameter_config.confidence_multiplier_min,
                self.hyperparameter_config.confidence_multiplier_max,
                self.hyperparameter_config.confidence_multiplier_step,
            ):
                # Evaluate hyperparameters
                score = self._evaluate_hyperparameters(
                    performance_history,
                    ema_alpha,
                    confidence_multiplier,
                )
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "ema_alpha": ema_alpha,
                        "confidence_multiplier": confidence_multiplier,
                    }
        
        # Update weight object with optimized hyperparameters
        if best_params and engine_id in self.engine_weights:
            weight_obj = self.engine_weights[engine_id]
            weight_obj.ema_alpha = best_params["ema_alpha"]
            weight_obj.confidence_multiplier = best_params["confidence_multiplier"]
        
        logger.info(
            "hyperparameters_tuned",
            symbol=self.symbol,
            engine_id=engine_id,
            best_params=best_params,
            best_score=best_score,
        )
        
        return best_params
    
    def _evaluate_hyperparameters(
        self,
        performance_history: List[EnginePerformance],
        ema_alpha: float,
        confidence_multiplier: float,
    ) -> float:
        """Evaluate hyperparameters using performance history.
        
        Args:
            performance_history: Performance history
            ema_alpha: EMA alpha to evaluate
            confidence_multiplier: Confidence multiplier to evaluate
            
        Returns:
            Evaluation score (higher is better)
        """
        # Calculate weighted performance score
        total_score = 0.0
        total_weight = 0.0
        
        for performance in performance_history:
            # Calculate weight using EMA
            weight = ema_alpha * performance.accuracy + (1 - ema_alpha) * 0.5
            
            # Adjust by confidence multiplier
            adjusted_weight = weight * confidence_multiplier
            
            # Score = accuracy * net_edge * sharpe
            score = performance.accuracy * (1.0 + performance.net_edge_bps / 100.0) * performance.sharpe_ratio
            
            total_score += score * adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _generate_range(self, min_val: float, max_val: float, step: float) -> List[float]:
        """Generate range of values.
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            step: Step size
            
        Returns:
            List of values
        """
        values = []
        current = min_val
        while current <= max_val:
            values.append(current)
            current += step
        return values
    
    def should_tune_hyperparameters(self) -> bool:
        """Check if hyperparameters should be tuned.
        
        Returns:
            True if should tune
        """
        if self.last_tuning_date is None:
            return True
        
        days_since_tuning = (datetime.now() - self.last_tuning_date).days
        return days_since_tuning >= self.hyperparameter_config.tuning_frequency_days
    
    def get_top_engines(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N engines by weight.
        
        Args:
            n: Number of top engines to return
            
        Returns:
            List of (engine_id, weight) tuples sorted by weight
        """
        sorted_engines = sorted(
            self.engine_weights.items(),
            key=lambda x: x[1].weight,
            reverse=True,
        )
        
        return [(engine_id, weight_obj.weight) for engine_id, weight_obj in sorted_engines[:n]]


