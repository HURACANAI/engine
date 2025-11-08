"""
Adapter for AI-Generated Engines

Converts between the AI engine interface (symbol, df, regime, meta) 
and the standard engine interface (features, current_regime).
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from cloud.training.models.alpha_engines import AlphaSignal
except ImportError:
    # Fallback for relative import when loaded as part of package
    from ..alpha_engines import AlphaSignal


class AIGeneratedEngineAdapter:
    """
    Adapter to make AI-generated engines compatible with AlphaEngineCoordinator.
    
    AI engines use: generate_signal(symbol, df, regime, meta)
    Standard engines use: generate_signal(features, current_regime)
    
    This adapter:
    1. Converts features dict to a minimal DataFrame
    2. Extracts symbol and meta from context
    3. Calls the AI engine with the correct interface
    4. Returns the signal
    """
    
    def __init__(self, ai_engine, symbol: Optional[str] = None, technique=None):
        """
        Initialize adapter.
        
        Args:
            ai_engine: The AI-generated engine instance
            symbol: Optional symbol (if known)
            technique: Optional TradingTechnique enum
        """
        self.ai_engine = ai_engine
        self.symbol = symbol or "UNKNOWN"
        self.technique = technique
        
    def generate_signal(
        self, 
        features: Dict[str, float], 
        current_regime: str
    ) -> AlphaSignal:
        """
        Generate signal using AI engine with adapter.
        
        Args:
            features: Feature dictionary
            current_regime: Current market regime
            
        Returns:
            AlphaSignal
        """
        # Convert features dict to DataFrame
        # Create a single-row DataFrame with features as columns
        df = self._features_to_dataframe(features)
        
        # Create meta dict
        meta = {
            "features": features,
            "regime": current_regime,
        }
        
        # Normalize regime name (AI engines might expect uppercase)
        regime = current_regime.upper() if current_regime else "RANGE"
        
        # Call AI engine
        try:
            signal = self.ai_engine.generate_signal(
                symbol=self.symbol,
                df=df,
                regime=regime,
                meta=meta
            )
            return signal
        except Exception as e:
            # Fallback: create hold signal on error
            return AlphaSignal(
                technique=self.technique or self._infer_technique(),
                direction="hold",
                confidence=0.0,
                reasoning=f"AI engine error: {str(e)}",
                key_features={},
                regime_affinity=0.0
            )
    
    def _features_to_dataframe(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Convert features dict to a minimal DataFrame.
        
        Creates a single-row DataFrame where each feature becomes a column.
        For features that look like indicators (e.g., 'rsi_14'), we create
        a minimal history by repeating the value.
        """
        # Create a single row with all features
        row = features.copy()
        
        # Ensure we have basic OHLCV columns if they exist in features
        # If not, create dummy values
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in row:
                # Try to infer from other features
                if col == 'close' and 'price' in features:
                    row['close'] = features['price']
                elif col == 'volume' and 'volume' not in row:
                    row['volume'] = 1000.0  # Default volume
                elif col in ['open', 'high', 'low'] and 'close' in row:
                    # Use close as approximation
                    row[col] = row.get('close', 100.0)
        
        # Create DataFrame with single row
        df = pd.DataFrame([row])
        
        # For indicators that need history, create minimal history
        # by repeating the current value (simple approach)
        indicator_cols = [col for col in df.columns if any(ind in col.lower() 
                          for ind in ['rsi', 'macd', 'ema', 'sma', 'bb', 'atr'])]
        
        # Create a small history window (e.g., 20 rows) by repeating
        # This is a simple approach - in production, you'd want actual historical data
        if len(df) == 1 and indicator_cols:
            # Repeat the row to create minimal history
            history_length = 20
            df_repeated = pd.concat([df] * history_length, ignore_index=True)
            
            # For time series indicators, you might want to add slight variation
            # For now, just repeat (this is a limitation when we don't have real history)
            df = df_repeated
        
        return df
    
    def _infer_technique(self):
        """Infer TradingTechnique from engine metadata."""
        from ..alpha_engines import TradingTechnique
        
        if hasattr(self.ai_engine, 'METADATA'):
            strategy_type = self.ai_engine.METADATA.get('strategy_type', '').lower()
            if 'trend' in strategy_type:
                return TradingTechnique.TREND
            elif 'reversal' in strategy_type or 'range' in strategy_type:
                return TradingTechnique.RANGE
            elif 'breakout' in strategy_type:
                return TradingTechnique.BREAKOUT
        
        return TradingTechnique.RANGE  # Default

