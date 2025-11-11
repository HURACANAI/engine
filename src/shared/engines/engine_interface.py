"""
Engine Interface

One unified interface for all 23 engines. Same inputs and outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class Direction(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"


@dataclass
class EngineInput:
    """Input to engine inference."""
    symbol: str
    timestamp: datetime
    features: Dict[str, float]  # Feature values for this bar
    regime: str  # Market regime (TREND, RANGE, PANIC, ILLIQUID)
    costs: Dict[str, float]  # Costs per symbol: fees_bps, spread_bps, slippage_bps
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EngineOutput:
    """Output from engine inference."""
    direction: Direction  # buy, sell, wait
    edge_bps_before_costs: float  # Expected edge in basis points (before costs)
    confidence_0_1: float  # Confidence score (0.0 to 1.0)
    horizon_minutes: int  # Prediction horizon in minutes
    metadata: Dict[str, Any]  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "edge_bps_before_costs": self.edge_bps_before_costs,
            "confidence_0_1": self.confidence_0_1,
            "horizon_minutes": self.horizon_minutes,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EngineOutput:
        """Create from dictionary."""
        return cls(
            direction=Direction(data["direction"]),
            edge_bps_before_costs=data["edge_bps_before_costs"],
            confidence_0_1=data["confidence_0_1"],
            horizon_minutes=data["horizon_minutes"],
            metadata=data.get("metadata", {}),
        )


class BaseEngine(ABC):
    """Base engine interface for all 23 engines."""
    
    def __init__(self, engine_id: str, name: str, supported_regimes: List[str]):
        """Initialize engine.
        
        Args:
            engine_id: Unique engine identifier
            name: Engine name
            supported_regimes: List of supported market regimes
        """
        self.engine_id = engine_id
        self.name = name
        self.supported_regimes = supported_regimes
        logger.info("engine_initialized", engine_id=engine_id, name=name, supported_regimes=supported_regimes)
    
    @abstractmethod
    def infer(self, input_data: EngineInput) -> EngineOutput:
        """Run inference on input data.
        
        Args:
            input_data: Engine input data
            
        Returns:
            Engine output with direction, edge, confidence, horizon
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information.
        
        Returns:
            Engine information dictionary
        """
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "supported_regimes": self.supported_regimes,
        }


class EngineRegistry:
    """Registry for all 23 engines."""
    
    def __init__(self):
        """Initialize engine registry."""
        self.engines: Dict[str, BaseEngine] = {}
        logger.info("engine_registry_initialized")
    
    def register(self, engine: BaseEngine) -> None:
        """Register an engine.
        
        Args:
            engine: Engine instance
        """
        self.engines[engine.engine_id] = engine
        logger.info("engine_registered", engine_id=engine.engine_id, name=engine.name)
    
    def get_engine(self, engine_id: str) -> Optional[BaseEngine]:
        """Get engine by ID.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Engine instance, or None if not found
        """
        return self.engines.get(engine_id)
    
    def get_engines_for_regime(self, regime: str) -> List[BaseEngine]:
        """Get all engines that support a given regime.
        
        Args:
            regime: Market regime
            
        Returns:
            List of engines that support the regime
        """
        return [engine for engine in self.engines.values() if engine.is_supported_regime(regime)]
    
    def get_all_engines(self) -> List[BaseEngine]:
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

