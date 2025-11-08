"""
Event Fade Engine

Trades only around spikes. Fades liquidation cascades and funding flips
when regime is range.

Key Features:
- Event detection (spikes, liquidations, funding flips)
- Fade strategy in range regimes
- Spike detection with volatility thresholds
- Liquidation cascade detection
- Funding flip detection
- Regime-specific trading logic

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

from .alpha_engines import AlphaSignal, TradingTechnique

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Event type"""
    SPIKE = "spike"  # Price spike
    LIQUIDATION_CASCADE = "liquidation_cascade"  # Liquidation cascade
    FUNDING_FLIP = "funding_flip"  # Funding rate flip
    VOLATILITY_EXPANSION = "volatility_expansion"  # Volatility expansion
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"  # Order book imbalance


@dataclass
class Event:
    """Detected event"""
    event_type: EventType
    timestamp: float
    price_change_pct: float
    volume_spike: float
    intensity: float  # 0-1, higher is more intense
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class EventFadeSignal:
    """Event fade signal"""
    event: Event
    fade_direction: str  # "buy" or "sell"
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str


class EventFadeEngine:
    """
    Event Fade Engine - Trades only around spikes.
    
    Detects events (spikes, liquidations, funding flips) and fades them
    when regime is range.
    
    Usage:
        engine = EventFadeEngine()
        
        signal = engine.generate_signal(
            features={...},
            current_regime="range",
            price_history=[...],
            volume_history=[...]
        )
        
        if signal:
            # Execute fade trade
            pass
    """
    
    def __init__(
        self,
        spike_threshold_pct: float = 2.0,  # 2% price move = spike
        volume_spike_multiplier: float = 2.0,  # 2x volume = spike
        volatility_expansion_threshold: float = 1.5,  # 1.5x volatility = expansion
        min_event_intensity: float = 0.6,  # Minimum intensity to trade
        fade_in_range_only: bool = True,  # Only fade in range regimes
        target_reversion_pct: float = 0.5,  # Target 50% reversion
        stop_loss_pct: float = 1.0  # 1% stop loss
    ):
        """
        Initialize event fade engine.
        
        Args:
            spike_threshold_pct: Price change threshold for spike detection
            volume_spike_multiplier: Volume multiplier for spike detection
            volatility_expansion_threshold: Volatility expansion threshold
            min_event_intensity: Minimum event intensity to trade
            fade_in_range_only: Only fade in range regimes
            target_reversion_pct: Target reversion percentage
            stop_loss_pct: Stop loss percentage
        """
        self.spike_threshold_pct = spike_threshold_pct
        self.volume_spike_multiplier = volume_spike_multiplier
        self.volatility_expansion_threshold = volatility_expansion_threshold
        self.min_event_intensity = min_event_intensity
        self.fade_in_range_only = fade_in_range_only
        self.target_reversion_pct = target_reversion_pct
        self.stop_loss_pct = stop_loss_pct
        
        # Event history for pattern detection
        self.event_history: List[Event] = []
        
        logger.info(
            "event_fade_engine_initialized",
            spike_threshold_pct=spike_threshold_pct,
            volume_spike_multiplier=volume_spike_multiplier,
            fade_in_range_only=fade_in_range_only
        )
    
    def generate_signal(
        self,
        features: Dict[str, float],
        current_regime: str,
        price_history: Optional[List[float]] = None,
        volume_history: Optional[List[float]] = None,
        funding_rate: Optional[float] = None,
        funding_rate_history: Optional[List[float]] = None
    ) -> Optional[AlphaSignal]:
        """
        Generate event fade signal.
        
        Args:
            features: Feature dictionary
            current_regime: Current market regime
            price_history: Recent price history
            volume_history: Recent volume history
            funding_rate: Current funding rate
            funding_rate_history: Funding rate history
        
        Returns:
            AlphaSignal or None if no event detected
        """
        # Only trade in range regimes if fade_in_range_only is True
        if self.fade_in_range_only and current_regime != "range":
            return None
        
        # Detect events
        events = self._detect_events(
            features=features,
            price_history=price_history,
            volume_history=volume_history,
            funding_rate=funding_rate,
            funding_rate_history=funding_rate_history
        )
        
        if not events:
            return None
        
        # Find strongest event
        strongest_event = max(events, key=lambda e: e.intensity)
        
        # Check if event is strong enough
        if strongest_event.intensity < self.min_event_intensity:
            return None
        
        # Generate fade signal
        signal = self._generate_fade_signal(
            event=strongest_event,
            features=features,
            current_price=features.get("close", 0.0)
        )
        
        if signal:
            # Convert to AlphaSignal
            return AlphaSignal(
                technique=TradingTechnique.RANGE,  # Use RANGE technique
                direction=signal.fade_direction,
                confidence=signal.confidence,
                reasoning=f"Event fade: {signal.reasoning}",
                key_features={
                    "event_type": strongest_event.event_type.value,
                    "event_intensity": strongest_event.intensity,
                    "price_change_pct": strongest_event.price_change_pct,
                    "volume_spike": strongest_event.volume_spike
                },
                regime_affinity=1.0 if current_regime == "range" else 0.5
            )
        
        return None
    
    def _detect_events(
        self,
        features: Dict[str, float],
        price_history: Optional[List[float]],
        volume_history: Optional[List[float]],
        funding_rate: Optional[float],
        funding_rate_history: Optional[List[float]]
    ) -> List[Event]:
        """Detect events in market data"""
        events = []
        
        if price_history and len(price_history) >= 2:
            # Detect price spike
            spike_event = self._detect_spike(price_history, volume_history, features)
            if spike_event:
                events.append(spike_event)
            
            # Detect liquidation cascade (simplified - would use liquidation data)
            liquidation_event = self._detect_liquidation_cascade(price_history, volume_history, features)
            if liquidation_event:
                events.append(liquidation_event)
        
        # Detect funding flip
        if funding_rate is not None and funding_rate_history:
            funding_event = self._detect_funding_flip(funding_rate, funding_rate_history, features)
            if funding_event:
                events.append(funding_event)
        
        # Detect volatility expansion
        volatility_event = self._detect_volatility_expansion(features)
        if volatility_event:
            events.append(volatility_event)
        
        return events
    
    def _detect_spike(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]],
        features: Dict[str, float]
    ) -> Optional[Event]:
        """Detect price spike"""
        if len(price_history) < 2:
            return None
        
        # Calculate recent price change
        recent_price = price_history[-1]
        previous_price = price_history[-2] if len(price_history) >= 2 else price_history[0]
        
        price_change_pct = abs((recent_price - previous_price) / previous_price) * 100
        
        # Check if spike
        if price_change_pct < self.spike_threshold_pct:
            return None
        
        # Calculate volume spike
        volume_spike = 1.0
        if volume_history and len(volume_history) >= 2:
            recent_volume = volume_history[-1]
            avg_volume = np.mean(volume_history[:-1])
            if avg_volume > 0:
                volume_spike = recent_volume / avg_volume
        
        # Calculate intensity
        price_intensity = min(1.0, price_change_pct / (self.spike_threshold_pct * 2))
        volume_intensity = min(1.0, volume_spike / self.volume_spike_multiplier)
        intensity = (price_intensity * 0.7 + volume_intensity * 0.3)
        
        return Event(
            event_type=EventType.SPIKE,
            timestamp=features.get("timestamp", 0.0),
            price_change_pct=price_change_pct,
            volume_spike=volume_spike,
            intensity=intensity,
            metadata={
                "direction": "up" if recent_price > previous_price else "down"
            }
        )
    
    def _detect_liquidation_cascade(
        self,
        price_history: List[float],
        volume_history: Optional[List[float]],
        features: Dict[str, float]
    ) -> Optional[Event]:
        """Detect liquidation cascade (simplified)"""
        # In production, would use actual liquidation data
        # For now, detect rapid price moves with high volume
        
        if len(price_history) < 5:
            return None
        
        # Calculate recent price acceleration
        recent_changes = [abs(price_history[i] - price_history[i-1]) / price_history[i-1] 
                         for i in range(1, len(price_history))]
        
        if len(recent_changes) < 2:
            return None
        
        # Check for acceleration (increasing price changes)
        acceleration = recent_changes[-1] - recent_changes[-2] if len(recent_changes) >= 2 else 0
        
        # High acceleration + high volume = potential liquidation cascade
        volume_spike = 1.0
        if volume_history and len(volume_history) >= 2:
            recent_volume = volume_history[-1]
            avg_volume = np.mean(volume_history[:-1])
            if avg_volume > 0:
                volume_spike = recent_volume / avg_volume
        
        if acceleration > 0.005 and volume_spike > 1.5:  # 0.5% acceleration + 1.5x volume
            intensity = min(1.0, (acceleration * 100 + volume_spike) / 3.0)
            
            return Event(
                event_type=EventType.LIQUIDATION_CASCADE,
                timestamp=features.get("timestamp", 0.0),
                price_change_pct=recent_changes[-1] * 100,
                volume_spike=volume_spike,
                intensity=intensity,
                metadata={
                    "acceleration": acceleration
                }
            )
        
        return None
    
    def _detect_funding_flip(
        self,
        funding_rate: float,
        funding_rate_history: List[float],
        features: Dict[str, float]
    ) -> Optional[Event]:
        """Detect funding rate flip"""
        if len(funding_rate_history) < 2:
            return None
        
        # Check if funding rate flipped sign
        previous_rate = funding_rate_history[-2]
        
        if (funding_rate > 0 and previous_rate < 0) or (funding_rate < 0 and previous_rate > 0):
            # Funding flip detected
            flip_magnitude = abs(funding_rate - previous_rate)
            intensity = min(1.0, flip_magnitude * 100)  # Normalize
            
            return Event(
                event_type=EventType.FUNDING_FLIP,
                timestamp=features.get("timestamp", 0.0),
                price_change_pct=0.0,
                volume_spike=1.0,
                intensity=intensity,
                metadata={
                    "funding_rate": funding_rate,
                    "previous_rate": previous_rate,
                    "flip_magnitude": flip_magnitude
                }
            )
        
        return None
    
    def _detect_volatility_expansion(
        self,
        features: Dict[str, float]
    ) -> Optional[Event]:
        """Detect volatility expansion"""
        # Check volatility regime feature
        volatility_regime = features.get("volatility_regime", 1.0)
        
        if volatility_regime > self.volatility_expansion_threshold:
            intensity = min(1.0, (volatility_regime - 1.0) / 1.0)  # Normalize
            
            return Event(
                event_type=EventType.VOLATILITY_EXPANSION,
                timestamp=features.get("timestamp", 0.0),
                price_change_pct=0.0,
                volume_spike=1.0,
                intensity=intensity,
                metadata={
                    "volatility_regime": volatility_regime
                }
            )
        
        return None
    
    def _generate_fade_signal(
        self,
        event: Event,
        features: Dict[str, float],
        current_price: float
    ) -> Optional[EventFadeSignal]:
        """Generate fade signal for event"""
        if current_price == 0:
            return None
        
        # Determine fade direction (opposite of spike direction)
        if event.event_type == EventType.SPIKE:
            spike_direction = event.metadata.get("direction", "up")
            fade_direction = "sell" if spike_direction == "up" else "buy"
        elif event.event_type == EventType.LIQUIDATION_CASCADE:
            # Fade the cascade (assume it's a sell-off, so fade = buy)
            fade_direction = "buy"
        elif event.event_type == EventType.FUNDING_FLIP:
            # Fade based on funding flip direction
            funding_rate = event.metadata.get("funding_rate", 0.0)
            fade_direction = "buy" if funding_rate > 0 else "sell"
        else:
            # Default: fade based on price change
            fade_direction = "sell" if event.price_change_pct > 0 else "buy"
        
        # Calculate target and stop loss
        if fade_direction == "buy":
            # Buying fade: expect price to revert up
            target_price = current_price * (1 + self.target_reversion_pct / 100)
            stop_loss = current_price * (1 - self.stop_loss_pct / 100)
        else:
            # Selling fade: expect price to revert down
            target_price = current_price * (1 - self.target_reversion_pct / 100)
            stop_loss = current_price * (1 + self.stop_loss_pct / 100)
        
        # Calculate confidence based on event intensity
        confidence = event.intensity
        
        # Build reasoning
        reasoning = f"Fading {event.event_type.value} event (intensity: {event.intensity:.2f})"
        
        return EventFadeSignal(
            event=event,
            fade_direction=fade_direction,
            confidence=confidence,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasoning
        )

