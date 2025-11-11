"""
Enhanced Engine Interface

Supports long horizons (hours to days/weeks) for swing trading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class TradingHorizon(Enum):
    """Trading horizon types."""
    SCALP = "scalp"  # Minutes to hours
    SWING = "swing"  # Hours to days
    POSITION = "position"  # Days to weeks
    CORE = "core"  # Long-term holds (weeks to months)


class Direction(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"
    HOLD = "hold"  # Hold existing position (for swing/core)


@dataclass
class EnhancedEngineInput:
    """Enhanced input to engine inference with swing trading support."""
    symbol: str
    timestamp: datetime
    features: Dict[str, float]  # Feature values for this bar
    regime: str  # Market regime (TREND, RANGE, PANIC, ILLIQUID)
    costs: Dict[str, float]  # Costs per symbol: fees_bps, spread_bps, slippage_bps, funding_bps
    metadata: Optional[Dict[str, Any]] = None
    # Swing trading context
    current_position: Optional[Dict[str, Any]] = None  # Current position state
    holding_duration_hours: float = 0.0  # How long position has been held
    portfolio_allocation: Optional[Dict[str, float]] = None  # Portfolio allocation by horizon


@dataclass
class EnhancedEngineOutput:
    """Enhanced output from engine inference with swing trading support."""
    direction: Direction  # buy, sell, wait, hold
    edge_bps_before_costs: float  # Expected edge in basis points (before costs)
    confidence_0_1: float  # Confidence score (0.0 to 1.0)
    horizon_minutes: int  # Prediction horizon in minutes (can be 60 * 24 * 7 for weeks)
    horizon_type: TradingHorizon  # Horizon type (scalp, swing, position, core)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Swing trading specific
    stop_loss_bps: Optional[float] = None  # Stop loss in basis points
    take_profit_bps: Optional[float] = None  # Take profit in basis points
    trailing_stop_bps: Optional[float] = None  # Trailing stop in basis points
    position_size_multiplier: float = 1.0  # Position size multiplier (0.0 to 1.0)
    hold_until: Optional[datetime] = None  # Hold until this timestamp (for swing trades)
    # Risk management
    max_holding_hours: Optional[float] = None  # Maximum holding time in hours
    funding_cost_estimate_bps: float = 0.0  # Estimated funding cost over holding period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "edge_bps_before_costs": self.edge_bps_before_costs,
            "confidence_0_1": self.confidence_0_1,
            "horizon_minutes": self.horizon_minutes,
            "horizon_type": self.horizon_type.value,
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trailing_stop_bps": self.trailing_stop_bps,
            "position_size_multiplier": self.position_size_multiplier,
            "hold_until": self.hold_until.isoformat() if self.hold_until else None,
            "max_holding_hours": self.max_holding_hours,
            "funding_cost_estimate_bps": self.funding_cost_estimate_bps,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnhancedEngineOutput:
        """Create from dictionary."""
        return cls(
            direction=Direction(data["direction"]),
            edge_bps_before_costs=data["edge_bps_before_costs"],
            confidence_0_1=data["confidence_0_1"],
            horizon_minutes=data["horizon_minutes"],
            horizon_type=TradingHorizon(data.get("horizon_type", "scalp")),
            stop_loss_bps=data.get("stop_loss_bps"),
            take_profit_bps=data.get("take_profit_bps"),
            trailing_stop_bps=data.get("trailing_stop_bps"),
            position_size_multiplier=data.get("position_size_multiplier", 1.0),
            hold_until=datetime.fromisoformat(data["hold_until"]) if data.get("hold_until") else None,
            max_holding_hours=data.get("max_holding_hours"),
            funding_cost_estimate_bps=data.get("funding_cost_estimate_bps", 0.0),
            metadata=data.get("metadata", {}),
        )
    
    def is_swing_trade(self) -> bool:
        """Check if this is a swing trade."""
        return self.horizon_type in [TradingHorizon.SWING, TradingHorizon.POSITION, TradingHorizon.CORE]
    
    def get_holding_duration_days(self) -> float:
        """Get holding duration in days."""
        return self.horizon_minutes / (60.0 * 24.0)


class BaseEnhancedEngine(ABC):
    """Base enhanced engine interface for all engines with swing trading support."""
    
    def __init__(
        self,
        engine_id: str,
        name: str,
        supported_regimes: List[str],
        supported_horizons: List[TradingHorizon],
        default_horizon: TradingHorizon = TradingHorizon.SCALP,
    ):
        """Initialize engine.
        
        Args:
            engine_id: Unique engine identifier
            name: Engine name
            supported_regimes: List of supported market regimes
            supported_horizons: List of supported trading horizons
            default_horizon: Default trading horizon
        """
        self.engine_id = engine_id
        self.name = name
        self.supported_regimes = supported_regimes
        self.supported_horizons = supported_horizons
        self.default_horizon = default_horizon
        logger.info(
            "enhanced_engine_initialized",
            engine_id=engine_id,
            name=name,
            supported_regimes=supported_regimes,
            supported_horizons=[h.value for h in supported_horizons],
        )
    
    @abstractmethod
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        """Run inference on input data.
        
        Args:
            input_data: Enhanced engine input data
            
        Returns:
            Enhanced engine output with direction, edge, confidence, horizon, stops
        """
        pass
    
    def is_supported_regime(self, regime: str) -> bool:
        """Check if engine supports the given regime.
        
        Args:
            regime: Market regime
            
        Returns:
            True if engine supports the regime
        """
        return regime in self.supported_regimes
    
    def is_supported_horizon(self, horizon: TradingHorizon) -> bool:
        """Check if engine supports the given horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            True if engine supports the horizon
        """
        return horizon in self.supported_horizons
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information.
        
        Returns:
            Engine information dictionary
        """
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "supported_regimes": self.supported_regimes,
            "supported_horizons": [h.value for h in self.supported_horizons],
            "default_horizon": self.default_horizon.value,
        }


class EnhancedEngineRegistry:
    """Enhanced registry for all engines with swing trading support."""
    
    def __init__(self):
        """Initialize enhanced engine registry."""
        self.engines: Dict[str, BaseEnhancedEngine] = {}
        logger.info("enhanced_engine_registry_initialized")
    
    def register(self, engine: BaseEnhancedEngine) -> None:
        """Register an engine.
        
        Args:
            engine: Engine instance
        """
        self.engines[engine.engine_id] = engine
        logger.info("enhanced_engine_registered", engine_id=engine.engine_id, name=engine.name)
    
    def get_engine(self, engine_id: str) -> Optional[BaseEnhancedEngine]:
        """Get engine by ID.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Engine instance, or None if not found
        """
        return self.engines.get(engine_id)
    
    def get_engines_for_regime(self, regime: str) -> List[BaseEnhancedEngine]:
        """Get all engines that support a given regime.
        
        Args:
            regime: Market regime
            
        Returns:
            List of engines that support the regime
        """
        return [engine for engine in self.engines.values() if engine.is_supported_regime(regime)]
    
    def get_engines_for_horizon(self, horizon: TradingHorizon) -> List[BaseEnhancedEngine]:
        """Get all engines that support a given horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            List of engines that support the horizon
        """
        return [engine for engine in self.engines.values() if engine.is_supported_horizon(horizon)]
    
    def get_engines_for_regime_and_horizon(
        self,
        regime: str,
        horizon: TradingHorizon,
    ) -> List[BaseEnhancedEngine]:
        """Get all engines that support both regime and horizon.
        
        Args:
            regime: Market regime
            horizon: Trading horizon
            
        Returns:
            List of engines that support both
        """
        return [
            engine
            for engine in self.engines.values()
            if engine.is_supported_regime(regime) and engine.is_supported_horizon(horizon)
        ]
    
    def get_all_engines(self) -> List[BaseEnhancedEngine]:
        """Get all registered engines.
        
        Returns:
            List of all engines
        """
        return list(self.engines.values())
    
    def get_engine_count(self) -> int:
        """Get total number of registered engines.
        
        Returns:
            Number of engines
        """
        return len(self.engines)


# Utility functions for horizon conversion
def minutes_to_horizon_type(minutes: int) -> TradingHorizon:
    """Convert minutes to horizon type.
    
    Args:
        minutes: Horizon in minutes
        
    Returns:
        Trading horizon type
    """
    if minutes <= 60:  # <= 1 hour
        return TradingHorizon.SCALP
    elif minutes <= 24 * 60:  # <= 1 day
        return TradingHorizon.SWING
    elif minutes <= 7 * 24 * 60:  # <= 1 week
        return TradingHorizon.POSITION
    else:  # > 1 week
        return TradingHorizon.CORE


def horizon_type_to_minutes(horizon: TradingHorizon, default: int = 60) -> int:
    """Convert horizon type to default minutes.
    
    Args:
        horizon: Trading horizon type
        default: Default minutes for the horizon
        
    Returns:
        Horizon in minutes
    """
    horizon_defaults = {
        TradingHorizon.SCALP: 60,  # 1 hour
        TradingHorizon.SWING: 24 * 60,  # 1 day
        TradingHorizon.POSITION: 7 * 24 * 60,  # 1 week
        TradingHorizon.CORE: 30 * 24 * 60,  # 1 month
    }
    return horizon_defaults.get(horizon, default)

