"""
Confidence-Based Position Scaling

Scale position sizes based on model confidence:
- Confidence > 0.80: 2x base size (max £200 on £1000 capital)
- Confidence 0.60-0.80: 1x base size (£100)
- Confidence 0.50-0.60: 0.5x base size (£50)
- Confidence < 0.50: Skip trade

Also considers:
- Regime confidence
- Recent performance
- Circuit breaker limits

Usage:
    scaler = ConfidenceBasedPositionScaler(base_size_gbp=100.0, capital_gbp=1000.0)
    
    # Get scaled position size
    scaled_size = scaler.scale_position(
        confidence=0.75,
        regime='TREND',
        regime_confidence=0.80,
    )
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionScalingResult:
    """Position scaling result"""
    original_size: float
    scaled_size: float
    scale_factor: float
    reason: str
    confidence_used: float


class ConfidenceBasedPositionScaler:
    """
    Scale position sizes based on confidence levels.
    
    Higher confidence = larger positions
    Lower confidence = smaller positions or skip
    """

    def __init__(
        self,
        base_size_gbp: float = 100.0,
        capital_gbp: Optional[float] = None,  # None = unlimited (shadow trading)
        max_position_pct: float = 0.20,  # Max 20% of capital per position
        min_position_pct: float = 0.05,  # Min 5% of capital per position
        high_confidence_threshold: float = 0.80,
        medium_confidence_threshold: float = 0.60,
        low_confidence_threshold: float = 0.50,
        high_confidence_multiplier: float = 2.0,
        medium_confidence_multiplier: float = 1.0,
        low_confidence_multiplier: float = 0.5,
        shadow_trading_mode: bool = True,  # True = unlimited capital for shadow trading
    ):
        """
        Initialize position scaler.
        
        Args:
            base_size_gbp: Base position size
            capital_gbp: Total capital (None = unlimited for shadow trading)
            max_position_pct: Maximum position as % of capital
            min_position_pct: Minimum position as % of capital
            high_confidence_threshold: Confidence threshold for high confidence
            medium_confidence_threshold: Confidence threshold for medium confidence
            low_confidence_threshold: Confidence threshold for low confidence
            high_confidence_multiplier: Multiplier for high confidence
            medium_confidence_multiplier: Multiplier for medium confidence
            low_confidence_multiplier: Multiplier for low confidence
            shadow_trading_mode: If True, capital is unlimited (shadow trading only)
        """
        self.shadow_trading_mode = shadow_trading_mode
        
        if shadow_trading_mode or capital_gbp is None:
            # Shadow trading: unlimited capital
            self.capital_gbp = None  # Unlimited
            self.unlimited_mode = True
            # Use virtual capital for limit calculations (for reporting only)
            self.virtual_capital_gbp = 1000.0
            # No limits in shadow trading mode
            self.max_position_gbp = float('inf')  # Unlimited
            self.min_position_gbp = 0.0  # No minimum
        else:
            # Real trading: use actual capital
            self.capital_gbp = capital_gbp
            self.unlimited_mode = False
            self.virtual_capital_gbp = capital_gbp
            self.max_position_gbp = capital_gbp * max_position_pct
            self.min_position_gbp = capital_gbp * min_position_pct
        
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        
        self.high_confidence_multiplier = high_confidence_multiplier
        self.medium_confidence_multiplier = medium_confidence_multiplier
        self.low_confidence_multiplier = low_confidence_multiplier
        
        logger.info(
            "confidence_position_scaler_initialized",
            base_size=base_size_gbp,
            capital=capital_gbp,
            max_position=self.max_position_gbp,
            min_position=self.min_position_gbp,
        )

    def scale_position(
        self,
        confidence: float,
        regime: Optional[str] = None,
        regime_confidence: Optional[float] = None,
        recent_performance: Optional[float] = None,  # Recent win rate
        circuit_breaker_active: bool = False,
    ) -> Optional[PositionScalingResult]:
        """
        Scale position size based on confidence.
        
        Args:
            confidence: Model confidence (0-1)
            regime: Market regime (optional)
            regime_confidence: Confidence in regime detection (optional)
            recent_performance: Recent win rate (optional)
            circuit_breaker_active: Whether circuit breaker is active
            
        Returns:
            PositionScalingResult or None if trade should be skipped
        """
        # Circuit breaker check
        if circuit_breaker_active:
            logger.warning("position_scaling_skipped", reason="circuit_breaker_active")
            return None
        
        # Confidence too low - skip trade
        if confidence < self.low_confidence_threshold:
            logger.debug(
                "position_scaling_skipped",
                reason="confidence_too_low",
                confidence=confidence,
                threshold=self.low_confidence_threshold,
            )
            return None
        
        # Calculate combined confidence
        combined_confidence = confidence
        
        # Adjust for regime confidence
        if regime_confidence is not None:
            # Weight regime confidence at 20%
            combined_confidence = 0.8 * confidence + 0.2 * regime_confidence
        
        # Adjust for recent performance
        if recent_performance is not None:
            # If recent performance is poor, reduce confidence
            if recent_performance < 0.60:
                combined_confidence *= 0.9  # Reduce by 10%
            elif recent_performance > 0.80:
                combined_confidence *= 1.05  # Increase by 5% (cap at 1.0)
                combined_confidence = min(combined_confidence, 1.0)
        
        # Determine scale factor based on combined confidence
        if combined_confidence >= self.high_confidence_threshold:
            scale_factor = self.high_confidence_multiplier
            reason = f"High confidence ({combined_confidence:.1%})"
        elif combined_confidence >= self.medium_confidence_threshold:
            scale_factor = self.medium_confidence_multiplier
            reason = f"Medium confidence ({combined_confidence:.1%})"
        else:
            scale_factor = self.low_confidence_multiplier
            reason = f"Low confidence ({combined_confidence:.1%})"
        
        # Calculate scaled size
        scaled_size = self.base_size_gbp * scale_factor
        
        # Apply limits (only in real trading mode)
        if not self.unlimited_mode:
            scaled_size = max(scaled_size, self.min_position_gbp)
            scaled_size = min(scaled_size, self.max_position_gbp)
        # Shadow trading: no limits, but still track for reporting
        
        # Round to 2 decimal places
        scaled_size = round(scaled_size, 2)
        
        result = PositionScalingResult(
            original_size=self.base_size_gbp,
            scaled_size=scaled_size,
            scale_factor=scale_factor,
            reason=reason,
            confidence_used=combined_confidence,
        )
        
        logger.debug(
            "position_scaled",
            original_size=self.base_size_gbp,
            scaled_size=scaled_size,
            scale_factor=scale_factor,
            confidence=combined_confidence,
            reason=reason,
        )
        
        return result

    def get_scaling_info(self) -> Dict[str, Any]:
        """Get scaling configuration info"""
        return {
            'base_size_gbp': self.base_size_gbp,
            'capital_gbp': self.capital_gbp if not self.unlimited_mode else None,
            'unlimited_mode': self.unlimited_mode,
            'shadow_trading_mode': self.shadow_trading_mode,
            'max_position_gbp': self.max_position_gbp if not self.unlimited_mode else float('inf'),
            'min_position_gbp': self.min_position_gbp,
            'high_confidence_threshold': self.high_confidence_threshold,
            'high_confidence_multiplier': self.high_confidence_multiplier,
            'medium_confidence_threshold': self.medium_confidence_threshold,
            'medium_confidence_multiplier': self.medium_confidence_multiplier,
            'low_confidence_threshold': self.low_confidence_threshold,
            'low_confidence_multiplier': self.low_confidence_multiplier,
        }

