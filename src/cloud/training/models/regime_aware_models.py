"""
Regime-Aware Models

Separate sub-models per regime with top-level classifier.
Label regimes (bull, bear, sideways, high volatility, low volatility).

Key Features:
- Regime classification
- Separate sub-models per regime
- Top-level classifier for regime selection
- Time-decay weighting for recent data
- Regime transition detection

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timezone

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class RegimeType(Enum):
    """Regime type"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class VolatilityBucket(Enum):
    """Volatility bucket"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TrendBucket(Enum):
    """Trend bucket"""
    UPWARD = "upward"
    DOWNWARD = "downward"
    NEUTRAL = "neutral"


@dataclass
class RegimeLabel:
    """Regime label"""
    timestamp: datetime
    symbol: str
    regime_type: RegimeType
    volatility_bucket: VolatilityBucket
    trend_bucket: TrendBucket
    confidence: float = 1.0


@dataclass
class RegimeAwareModel:
    """Regime-aware model"""
    model_id: str
    regime_type: RegimeType
    model: Any  # The actual model
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_samples: int = 0
    last_trained: Optional[datetime] = None


class RegimeClassifier:
    """
    Regime Classifier.
    
    Classifies market regimes (bull, bear, sideways, volatility levels).
    
    Usage:
        classifier = RegimeClassifier()
        
        # Classify regime
        regime = classifier.classify(features)
        
        # Get regime label
        label = RegimeLabel(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            regime_type=regime,
            volatility_bucket=VolatilityBucket.HIGH,
            trend_bucket=TrendBucket.UPWARD
        )
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        volatility_percentile_high: float = 0.75,
        volatility_percentile_low: float = 0.25,
        trend_threshold: float = 0.02  # 2% trend threshold
    ):
        """
        Initialize regime classifier.
        
        Args:
            volatility_window: Window size for volatility calculation
            trend_window: Window size for trend calculation
            volatility_percentile_high: Percentile for high volatility
            volatility_percentile_low: Percentile for low volatility
            trend_threshold: Threshold for trend detection
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volatility_percentile_high = volatility_percentile_high
        self.volatility_percentile_low = volatility_percentile_low
        self.trend_threshold = trend_threshold
        
        logger.info("regime_classifier_initialized")
    
    def classify(
        self,
        data: pl.DataFrame,
        idx: int
    ) -> RegimeLabel:
        """
        Classify regime at given index.
        
        Args:
            data: Historical data
            idx: Current index
        
        Returns:
            RegimeLabel
        """
        # Get window data (only past data, no future)
        window_start = max(0, idx - self.trend_window)
        window_data = data[window_start:idx]
        
        if len(window_data) < self.volatility_window:
            # Default to neutral if insufficient data
            return RegimeLabel(
                timestamp=data[idx, "timestamp"] if "timestamp" in data.columns else datetime.now(timezone.utc),
                symbol=data[idx, "symbol"] if "symbol" in data.columns else "UNKNOWN",
                regime_type=RegimeType.SIDEWAYS,
                volatility_bucket=VolatilityBucket.MEDIUM,
                trend_bucket=TrendBucket.NEUTRAL,
                confidence=0.5
            )
        
        # Calculate volatility
        if "close" in window_data.columns:
            returns = window_data["close"].pct_change().drop_nulls()
            volatility = returns.std()
        else:
            volatility = 0.0
        
        # Classify volatility bucket
        if volatility > np.percentile(returns.to_numpy(), self.volatility_percentile_high * 100):
            vol_bucket = VolatilityBucket.HIGH
        elif volatility < np.percentile(returns.to_numpy(), self.volatility_percentile_low * 100):
            vol_bucket = VolatilityBucket.LOW
        else:
            vol_bucket = VolatilityBucket.MEDIUM
        
        # Calculate trend
        if "close" in window_data.columns:
            prices = window_data["close"].to_numpy()
            trend_slope = (prices[-1] - prices[0]) / prices[0] if len(prices) > 0 else 0.0
        else:
            trend_slope = 0.0
        
        # Classify trend bucket
        if trend_slope > self.trend_threshold:
            trend_bucket = TrendBucket.UPWARD
        elif trend_slope < -self.trend_threshold:
            trend_bucket = TrendBucket.DOWNWARD
        else:
            trend_bucket = TrendBucket.NEUTRAL
        
        # Classify regime type
        if trend_bucket == TrendBucket.UPWARD and vol_bucket == VolatilityBucket.LOW:
            regime_type = RegimeType.BULL
        elif trend_bucket == TrendBucket.DOWNWARD and vol_bucket == VolatilityBucket.HIGH:
            regime_type = RegimeType.BEAR
        elif trend_bucket == TrendBucket.NEUTRAL:
            regime_type = RegimeType.SIDEWAYS
        elif vol_bucket == VolatilityBucket.HIGH:
            regime_type = RegimeType.HIGH_VOLATILITY
        elif vol_bucket == VolatilityBucket.LOW:
            regime_type = RegimeType.LOW_VOLATILITY
        else:
            regime_type = RegimeType.SIDEWAYS
        
        return RegimeLabel(
            timestamp=data[idx, "timestamp"] if "timestamp" in data.columns else datetime.now(timezone.utc),
            symbol=data[idx, "symbol"] if "symbol" in data.columns else "UNKNOWN",
            regime_type=regime_type,
            volatility_bucket=vol_bucket,
            trend_bucket=trend_bucket,
            confidence=1.0
        )


class RegimeAwareModelSystem:
    """
    Regime-Aware Model System.
    
    Maintains separate sub-models per regime with top-level classifier.
    
    Usage:
        system = RegimeAwareModelSystem(
            train_fn=my_train_function,
            predict_fn=my_predict_function
        )
        
        # Train sub-models
        system.train_sub_models(data, regime_labels)
        
        # Predict
        prediction = system.predict(features, current_regime)
    """
    
    def __init__(
        self,
        train_fn: Callable[[pl.DataFrame, Dict[str, Any]], Any],
        predict_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        time_decay_weight: Optional[float] = None
    ):
        """
        Initialize regime-aware model system.
        
        Args:
            train_fn: Training function (data, config) -> model
            predict_fn: Prediction function (model, features) -> prediction
            time_decay_weight: Time decay weight for recent samples
        """
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.time_decay_weight = time_decay_weight
        
        # Sub-models per regime
        self.regime_models: Dict[RegimeType, RegimeAwareModel] = {}
        self.regime_classifier = RegimeClassifier()
        
        logger.info("regime_aware_model_system_initialized")
    
    def train_sub_models(
        self,
        data: pl.DataFrame,
        regime_labels: List[RegimeLabel],
        min_samples_per_regime: int = 100
    ) -> None:
        """
        Train sub-models for each regime.
        
        Args:
            data: Training data
            regime_labels: Regime labels for each data point
            min_samples_per_regime: Minimum samples per regime
        """
        # Group data by regime
        regime_data: Dict[RegimeType, List[int]] = {}
        for idx, regime_label in enumerate(regime_labels):
            if regime_label.regime_type not in regime_data:
                regime_data[regime_label.regime_type] = []
            regime_data[regime_label.regime_type].append(idx)
        
        # Train sub-model for each regime
        for regime_type, indices in regime_data.items():
            if len(indices) < min_samples_per_regime:
                logger.warning(
                    "insufficient_samples_for_regime",
                    regime=regime_type.value,
                    samples=len(indices),
                    min_required=min_samples_per_regime
                )
                continue
            
            # Get regime data
            regime_data_subset = data[indices]
            
            # Apply time decay if enabled
            if self.time_decay_weight:
                regime_data_subset = self._apply_time_decay(regime_data_subset, self.time_decay_weight)
            
            # Train model
            try:
                model = self.train_fn(regime_data_subset, {
                    "regime": regime_type.value,
                    "samples": len(indices)
                })
                
                # Create regime-aware model
                regime_model = RegimeAwareModel(
                    model_id=f"model_{regime_type.value}",
                    regime_type=regime_type,
                    model=model,
                    training_samples=len(indices),
                    last_trained=datetime.now(timezone.utc)
                )
                
                self.regime_models[regime_type] = regime_model
                
                logger.info(
                    "regime_model_trained",
                    regime=regime_type.value,
                    samples=len(indices)
                )
                
            except Exception as e:
                logger.error(
                    "regime_model_training_failed",
                    regime=regime_type.value,
                    error=str(e)
                )
    
    def predict(
        self,
        features: Dict[str, float],
        current_regime: RegimeLabel
    ) -> Dict[str, Any]:
        """
        Predict using regime-appropriate model.
        
        Args:
            features: Feature dictionary
            current_regime: Current regime label
        
        Returns:
            Prediction dictionary
        """
        # Get regime model
        regime_model = self.regime_models.get(current_regime.regime_type)
        
        if not regime_model:
            logger.warning(
                "regime_model_not_found",
                regime=current_regime.regime_type.value
            )
            return {
                "prediction": 0.0,
                "confidence": 0.0,
                "regime": current_regime.regime_type.value
            }
        
        # Make prediction
        try:
            prediction = self.predict_fn(regime_model.model, {
                "features": features,
                "regime": current_regime.regime_type.value
            })
            
            # Add regime information
            prediction["regime"] = current_regime.regime_type.value
            prediction["volatility_bucket"] = current_regime.volatility_bucket.value
            prediction["trend_bucket"] = current_regime.trend_bucket.value
            
            return prediction
            
        except Exception as e:
            logger.error(
                "regime_prediction_failed",
                regime=current_regime.regime_type.value,
                error=str(e)
            )
            return {
                "prediction": 0.0,
                "confidence": 0.0,
                "regime": current_regime.regime_type.value
            }
    
    def _apply_time_decay(self, data: pl.DataFrame, decay_rate: float) -> pl.DataFrame:
        """Apply time decay weighting"""
        # Placeholder - would implement actual time decay
        return data
    
    def get_regime_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each regime model"""
        performance = {}
        for regime_type, model in self.regime_models.items():
            performance[regime_type.value] = model.performance_metrics
        return performance
    
    def update_regime_model_performance(
        self,
        regime_type: RegimeType,
        metrics: Dict[str, float]
    ) -> None:
        """Update performance metrics for a regime model"""
        if regime_type in self.regime_models:
            self.regime_models[regime_type].performance_metrics.update(metrics)

