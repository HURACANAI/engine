"""
Sentiment Gate - Fear & Greed Index Gate

Blocks trades based on extreme market sentiment:
- Blocks new longs in extreme greed (overbought)
- Blocks new shorts in extreme fear (oversold but risky)

Source: CoinMarketCap Fear & Greed Index
Expected Impact: Better risk management, prevents bad entries
"""

from dataclasses import dataclass
from typing import Optional
import structlog  # type: ignore

from src.cloud.training.analysis.fear_greed_index import FearGreedIndex, FearGreedData

logger = structlog.get_logger(__name__)


@dataclass
class SentimentGateResult:
    """Result of sentiment gate evaluation."""
    passed: bool
    reason: str
    fear_greed_value: int
    fear_greed_level: str
    adjustment_applied: Optional[str] = None


class SentimentGate:
    """
    Gate that blocks trades based on Fear & Greed Index.
    
    Strategy:
    - Extreme Greed (75-100): Block new longs (overbought)
    - Extreme Fear (0-24): Block new shorts (oversold but risky)
    - Normal sentiment: Pass
    """

    def __init__(
        self,
        block_extreme_greed: bool = True,
        block_extreme_fear: bool = True,
        extreme_greed_threshold: int = 80,
        extreme_fear_threshold: int = 20,
    ):
        """
        Initialize sentiment gate.
        
        Args:
            block_extreme_greed: Whether to block trades in extreme greed
            block_extreme_fear: Whether to block trades in extreme fear
            extreme_greed_threshold: Greed threshold (default: 80)
            extreme_fear_threshold: Fear threshold (default: 20)
        """
        self.block_extreme_greed = block_extreme_greed
        self.block_extreme_fear = block_extreme_fear
        self.extreme_greed_threshold = extreme_greed_threshold
        self.extreme_fear_threshold = extreme_fear_threshold
        
        self.fear_greed_index = FearGreedIndex()
        
        logger.info(
            "sentiment_gate_initialized",
            block_extreme_greed=block_extreme_greed,
            block_extreme_fear=block_extreme_fear,
        )

    def evaluate(
        self,
        direction: str,  # 'buy' or 'sell'
        confidence: float,
    ) -> SentimentGateResult:
        """
        Evaluate sentiment gate.
        
        Args:
            direction: Trade direction ('buy' or 'sell')
            confidence: Trade confidence (0-1)
            
        Returns:
            SentimentGateResult
        """
        # Get Fear & Greed Index
        try:
            fear_greed_data = self.fear_greed_index.get_current_index()
        except Exception as e:
            logger.warning("fear_greed_index_fetch_failed", error=str(e))
            # Pass gate if we can't fetch (don't block trades)
            return SentimentGateResult(
                passed=True,
                reason="Fear & Greed Index unavailable - passing gate",
                fear_greed_value=50,
                fear_greed_level="neutral",
            )
        
        value = fear_greed_data.value
        level = fear_greed_data.level.value
        
        # Check if should block
        should_block, reason = self.fear_greed_index.should_block_trade(
            direction, fear_greed_data
        )
        
        if should_block:
            return SentimentGateResult(
                passed=False,
                reason=reason,
                fear_greed_value=value,
                fear_greed_level=level,
            )
        
        # Pass gate
        return SentimentGateResult(
            passed=True,
            reason=f"Sentiment OK ({fear_greed_data.classification})",
            fear_greed_value=value,
            fear_greed_level=level,
        )

