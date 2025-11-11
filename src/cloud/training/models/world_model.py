"""
World Model - State-Space Market Regime Prediction

Instead of predicting next price directly, predicts latent state transitions
(e.g., "market regime shifts"). This provides higher-order market context
for all trading logic.

Key Concept: Intelligence as "understanding the world through models"
- Builds a state-space model that summarizes the last N days into a compressed "world state"
- Predicts the next state vector instead of raw price
- Uses this vector as context for all trading logic

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WorldState:
    """Compressed world state vector representing market context."""
    timestamp: datetime
    state_vector: np.ndarray  # Compressed representation of last N days
    regime: str  # Current market regime (trending, ranging, volatile, etc.)
    volatility_level: float  # Normalized volatility (0-1)
    trend_strength: float  # Normalized trend strength (0-1)
    liquidity_score: float  # Normalized liquidity (0-1)
    metadata: Dict[str, float]  # Additional state features


class WorldModel:
    """
    World Model that predicts latent state transitions instead of raw prices.
    
    Usage:
        world_model = WorldModel(state_dim=32, lookback_days=30)
        
        # Build state from historical data
        state = world_model.build_state(data, symbol="BTC/USDT")
        
        # Predict next state
        next_state = world_model.predict_next_state(current_state)
        
        # Use state for trading decisions
        if next_state.regime == "trending" and next_state.trend_strength > 0.7:
            # Use trend-following strategy
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        lookback_days: int = 30,
        compression_method: str = "pca"
    ):
        """
        Initialize World Model.
        
        Args:
            state_dim: Dimension of compressed state vector (default: 32)
            lookback_days: Number of days to compress into state (default: 30)
            compression_method: Method for compression ('pca', 'autoencoder', 'lstm')
        """
        self.state_dim = state_dim
        self.lookback_days = lookback_days
        self.compression_method = compression_method
        
        # State transition model (to be trained)
        self.transition_model = None
        self.state_encoder = None
        
        logger.info(
            "world_model_initialized",
            state_dim=state_dim,
            lookback_days=lookback_days,
            compression_method=compression_method
        )
    
    def build_state(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: str,
        timestamp_column: str = 'timestamp'
    ) -> WorldState:
        """
        Build world state from historical data.
        
        Summarizes the last N days into a compressed "world state" vector.
        
        Args:
            data: Historical price/feature data
            symbol: Trading symbol
            timestamp_column: Name of timestamp column
        
        Returns:
            WorldState object with compressed state vector
        """
        # Convert to pandas if needed
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        # Sort by timestamp
        if timestamp_column in df.columns:
            df = df.sort_values(timestamp_column)
            if len(df) == 0:
                raise ValueError(f"DataFrame is empty for symbol {symbol}")
            latest_timestamp = df[timestamp_column].iloc[-1]
        else:
            latest_timestamp = datetime.now()
        
        if len(df) == 0:
            raise ValueError(f"DataFrame is empty for symbol {symbol}")
        
        # Extract last N days of features
        lookback_data = df.tail(self.lookback_days * 24).copy()  # Assuming hourly data
        
        # Extract key features for state representation
        features = self._extract_state_features(lookback_data)
        
        # Compress into state vector
        if self.state_encoder is None:
            # Use simple PCA-like compression (can be replaced with trained autoencoder)
            state_vector = self._compress_features_simple(features)
        else:
            state_vector = self.state_encoder.transform(features.reshape(1, -1))[0]
        
        # Extract regime and other metadata
        regime = self._classify_regime(lookback_data)
        volatility_level = self._calculate_volatility_level(lookback_data)
        trend_strength = self._calculate_trend_strength(lookback_data)
        liquidity_score = self._calculate_liquidity_score(lookback_data)
        
        # Additional metadata
        metadata = {
            'volume_trend': float(self._calculate_volume_trend(lookback_data)),
            'price_momentum': float(self._calculate_price_momentum(lookback_data)),
            'volatility_regime': float(self._classify_volatility_regime(lookback_data))
        }
        
        state = WorldState(
            timestamp=latest_timestamp,
            state_vector=state_vector,
            regime=regime,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            metadata=metadata
        )
        
        logger.debug(
            "world_state_built",
            symbol=symbol,
            regime=regime,
            volatility_level=volatility_level,
            trend_strength=trend_strength
        )
        
        return state
    
    def predict_next_state(
        self,
        current_state: WorldState,
        horizon_days: int = 1
    ) -> WorldState:
        """
        Predict next state vector.
        
        Instead of predicting raw price, predicts the next state transition.
        
        Args:
            current_state: Current world state
            horizon_days: Prediction horizon in days
        
        Returns:
            Predicted next WorldState
        """
        if self.transition_model is None:
            # Simple linear prediction (can be replaced with trained model)
            next_state_vector = self._predict_state_simple(current_state.state_vector)
        else:
            next_state_vector = self.transition_model.predict(
                current_state.state_vector.reshape(1, -1)
            )[0]
        
        # Predict regime transition (simplified)
        next_regime = self._predict_regime_transition(
            current_state.regime,
            current_state.state_vector
        )
        
        # Predict other state components
        next_volatility = self._predict_volatility(current_state, horizon_days)
        next_trend = self._predict_trend_strength(current_state, horizon_days)
        next_liquidity = self._predict_liquidity(current_state, horizon_days)
        
        next_state = WorldState(
            timestamp=current_state.timestamp,  # Would be updated with actual timestamp
            state_vector=next_state_vector,
            regime=next_regime,
            volatility_level=next_volatility,
            trend_strength=next_trend,
            liquidity_score=next_liquidity,
            metadata=current_state.metadata.copy()
        )
        
        logger.debug(
            "next_state_predicted",
            current_regime=current_state.regime,
            predicted_regime=next_regime,
            horizon_days=horizon_days
        )
        
        return next_state
    
    def _extract_state_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for state representation."""
        features = []
        
        # Price features
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            features.extend([
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis()
            ])
        
        # Volume features
        if 'volume' in data.columns:
            volume = data['volume']
            features.extend([
                volume.mean(),
                volume.std(),
                (volume / volume.shift(1)).mean()  # Volume trend
            ])
        
        # Volatility features
        if 'high' in data.columns and 'low' in data.columns:
            volatility = ((data['high'] - data['low']) / data['close']).mean()
            features.append(volatility)
        
        # Trend features
        if 'close' in data.columns:
            prices = data['close']
            sma_short = prices.tail(5).mean()
            sma_long = prices.tail(20).mean()
            features.append((sma_short - sma_long) / sma_long)
        
        return np.array(features)
    
    def _compress_features_simple(self, features: np.ndarray) -> np.ndarray:
        """Simple compression using PCA-like approach."""
        # Normalize features
        features_norm = (features - features.mean()) / (features.std() + 1e-8)
        
        # Simple dimensionality reduction (can be replaced with trained autoencoder)
        if len(features_norm) > self.state_dim:
            # Use first N principal components (simplified)
            # In production, use actual PCA or autoencoder
            state_vector = features_norm[:self.state_dim]
        else:
            # Pad with zeros if needed
            state_vector = np.pad(
                features_norm,
                (0, max(0, self.state_dim - len(features_norm))),
                mode='constant'
            )
        
        return state_vector
    
    def _classify_regime(self, data: pd.DataFrame) -> str:
        """Classify current market regime."""
        if 'close' not in data.columns or len(data) == 0:
            return "unknown"
        
        prices = data['close']
        returns = prices.pct_change().dropna()
        
        if len(returns) == 0:
            return "unknown"
        
        # Simple regime classification
        volatility = returns.std()
        
        # Protect against division by zero
        if prices.iloc[0] == 0 or abs(prices.iloc[0]) < 1e-8:
            trend = 0.0
        else:
            trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        if abs(trend) > 0.05 and volatility < 0.02:
            return "trending"
        elif volatility > 0.03:
            return "volatile"
        elif abs(trend) < 0.01:
            return "ranging"
        else:
            return "mixed"
    
    def _calculate_volatility_level(self, data: pd.DataFrame) -> float:
        """Calculate normalized volatility level (0-1)."""
        if 'close' not in data.columns:
            return 0.5
        
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Normalize to 0-1 (assuming typical range 0-0.1)
        return min(1.0, max(0.0, volatility * 10))
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate normalized trend strength (0-1)."""
        if 'close' not in data.columns or len(data) == 0:
            return 0.5
        
        prices = data['close']
        
        # Protect against division by zero
        if prices.iloc[0] == 0 or abs(prices.iloc[0]) < 1e-8:
            trend = 0.0
        else:
            trend = abs((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])
        
        # Normalize to 0-1
        return min(1.0, max(0.0, trend * 10))
    
    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate normalized liquidity score (0-1)."""
        if 'volume' not in data.columns:
            return 0.5
        
        volume = data['volume']
        avg_volume = volume.mean()
        
        # Normalize based on recent volume vs historical average
        # Simplified - in production, compare to longer-term average
        return min(1.0, max(0.0, avg_volume / (avg_volume + 1e6)))
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """Calculate volume trend."""
        if 'volume' not in data.columns:
            return 0.0
        
        volume = data['volume']
        recent_avg = volume.tail(5).mean()
        older_avg = volume.head(len(volume) - 5).mean() if len(volume) > 5 else recent_avg
        
        return (recent_avg - older_avg) / (older_avg + 1e-8)
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum."""
        if 'close' not in data.columns or len(data) == 0:
            return 0.0
        
        prices = data['close']
        
        # Protect against division by zero
        if prices.iloc[0] == 0 or abs(prices.iloc[0]) < 1e-8:
            return 0.0
        
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    
    def _classify_volatility_regime(self, data: pd.DataFrame) -> float:
        """Classify volatility regime (0=low, 1=high)."""
        return self._calculate_volatility_level(data)
    
    def _predict_state_simple(self, current_state: np.ndarray) -> np.ndarray:
        """Simple state prediction (linear extrapolation)."""
        # In production, use trained transition model
        return current_state * 0.95  # Slight decay
    
    def _predict_regime_transition(
        self,
        current_regime: str,
        state_vector: np.ndarray
    ) -> str:
        """Predict regime transition."""
        # Simplified - in production, use trained classifier
        # For now, assume regime persistence with some probability
        return current_regime
    
    def _predict_volatility(
        self,
        current_state: WorldState,
        horizon_days: int
    ) -> float:
        """Predict future volatility level."""
        # Simplified - assume mean reversion
        return current_state.volatility_level * 0.9
    
    def _predict_trend_strength(
        self,
        current_state: WorldState,
        horizon_days: int
    ) -> float:
        """Predict future trend strength."""
        # Simplified - assume decay
        return current_state.trend_strength * 0.95
    
    def _predict_liquidity(
        self,
        current_state: WorldState,
        horizon_days: int
    ) -> float:
        """Predict future liquidity."""
        # Simplified - assume persistence
        return current_state.liquidity_score

