"""
AI-Generated AlphaEngine: Rsi Reversal

Auto-generated from backtest by Strategy Translator.
Strategy Type: reversal
"""

try:
    from cloud.training.models.alpha_engines import AlphaSignal, TradingTechnique
except ImportError:
    # Fallback for relative import when loaded as part of package
    from ..alpha_engines import AlphaSignal, TradingTechnique
import pandas as pd


class AIGeneratedEngine_RsiReversal:
    """
    Rsi Reversal Strategy

    Entry: RSI < 30
    Exit: RSI > 70
    Indicators: RSI
    """

    METADATA = {
        "source": "rbi_agent",
        "generation_date": "2025-11-08",
        "strategy_type": "reversal",
        "status": "testing",
        "description": "Rsi Reversal strategy"
    }

    def __init__(self):
        self.name = "rsi_reversal"

    def calculate_features(self, df):
        """Calculate required indicators"""
        # RSI calculation (assuming it's in df already from FeatureRecipe)
        # If not, we'd calculate it here
        return df

    def generate_signal(self, symbol, df, regime, meta):
        """
        Generate trading signal based on RSI reversal logic.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            df: DataFrame with OHLCV + features
            regime: Current market regime (TREND/RANGE/PANIC)
            meta: Metadata dict

        Returns:
            AlphaSignal with technique, direction, confidence, reasoning, key_features, regime_affinity
        """
        # Check if we have enough data
        if len(df) < 14:
            return AlphaSignal(
                technique=TradingTechnique.RANGE,
                direction="hold",
                confidence=0.0,
                reasoning="Insufficient data",
                key_features={},
                regime_affinity=0.0
            )

        # Get RSI value (assuming feature name is 'rsi_14')
        if 'rsi_14' not in df.columns:
            return AlphaSignal(
                technique=TradingTechnique.RANGE,
                direction="hold",
                confidence=0.0,
                reasoning="RSI feature not found",
                key_features={},
                regime_affinity=0.0
            )

        current_rsi = df['rsi_14'].iloc[-1]

        # Entry signal: RSI < 30 (oversold)
        if current_rsi < 30:
            confidence = 0.65  # Base confidence
            regime_affinity = 0.7  # Works well in RANGE regime
            # Increase confidence in RANGE regime
            if regime == "RANGE":
                confidence += 0.10
                regime_affinity = 0.9
            return AlphaSignal(
                technique=TradingTechnique.RANGE,
                direction="buy",
                confidence=min(confidence, 0.95),
                reasoning=f"RSI oversold at {current_rsi:.1f} < 30",
                key_features={"rsi_14": current_rsi},
                regime_affinity=regime_affinity
            )

        # Exit signal: RSI > 70 (overbought)
        elif current_rsi > 70:
            confidence = 0.60
            regime_affinity = 0.7
            # Increase confidence in RANGE regime
            if regime == "RANGE":
                confidence += 0.10
                regime_affinity = 0.9
            return AlphaSignal(
                technique=TradingTechnique.RANGE,
                direction="sell",
                confidence=min(confidence, 0.95),
                reasoning=f"RSI overbought at {current_rsi:.1f} > 70",
                key_features={"rsi_14": current_rsi},
                regime_affinity=regime_affinity
            )

        # No signal
        return AlphaSignal(
            technique=TradingTechnique.RANGE,
            direction="hold",
            confidence=0.0,
            reasoning=f"RSI neutral at {current_rsi:.1f}",
            key_features={"rsi_14": current_rsi},
            regime_affinity=0.5
        )
