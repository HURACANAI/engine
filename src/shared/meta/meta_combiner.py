"""
Meta Combiner

Per coin meta combiner. EMA weights by recent accuracy and net edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import structlog

from ..engines.engine_interface import BaseEngine, EngineOutput

logger = structlog.get_logger(__name__)


@dataclass
class EngineWeight:
    """Engine weight for meta combination."""
    engine_id: str
    weight: float  # Weight (0.0 to 1.0)
    accuracy: float  # Recent accuracy (0.0 to 1.0)
    net_edge_bps: float  # Recent net edge in basis points
    last_updated: datetime


@dataclass
class MetaOutput:
    """Meta combiner output."""
    direction: str  # buy, sell, wait
    edge_bps_before_costs: float  # Combined edge in basis points
    confidence_0_1: float  # Combined confidence (0.0 to 1.0)
    horizon_minutes: int  # Prediction horizon in minutes
    engine_weights: Dict[str, float]  # Engine weights used
    metadata: Dict[str, Any]  # Additional metadata


class MetaCombiner:
    """Meta combiner per coin with EMA weights."""
    
    def __init__(
        self,
        symbol: str,
        ema_alpha: float = 0.1,  # EMA smoothing factor
        min_weight: float = 0.01,  # Minimum weight
        max_weight: float = 1.0,  # Maximum weight
        accuracy_threshold: float = 0.45,  # Minimum accuracy to include
    ):
        """Initialize meta combiner.
        
        Args:
            symbol: Trading symbol
            ema_alpha: EMA smoothing factor (0.0 to 1.0)
            min_weight: Minimum weight for engines
            max_weight: Maximum weight for engines
            accuracy_threshold: Minimum accuracy to include engine
        """
        self.symbol = symbol
        self.ema_alpha = ema_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.accuracy_threshold = accuracy_threshold
        
        self.engine_weights: Dict[str, EngineWeight] = {}
        logger.info("meta_combiner_initialized", symbol=symbol, ema_alpha=ema_alpha)
    
    def combine(
        self,
        engine_outputs: Dict[str, EngineOutput],
        regime: str,
    ) -> MetaOutput:
        """Combine engine outputs using weighted average.
        
        Args:
            engine_outputs: Dictionary of engine_id to EngineOutput
            regime: Market regime
            
        Returns:
            Meta combiner output
        """
        if not engine_outputs:
            # Default to wait if no outputs
            return MetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                engine_weights={},
                metadata={"reason": "no_engine_outputs"},
            )
        
        # Get weights for each engine
        weights = self._get_weights(engine_outputs.keys())
        
        # Filter engines by minimum weight
        filtered_outputs = {
            engine_id: output
            for engine_id, output in engine_outputs.items()
            if weights.get(engine_id, 0.0) >= self.min_weight
        }
        
        if not filtered_outputs:
            # Default to wait if no engines pass weight threshold
            return MetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                engine_weights=weights,
                metadata={"reason": "no_engines_pass_weight_threshold"},
            )
        
        # Combine outputs
        total_weight = sum(weights.get(engine_id, 0.0) for engine_id in filtered_outputs.keys())
        
        if total_weight == 0:
            # Default to wait if total weight is zero
            return MetaOutput(
                direction="wait",
                edge_bps_before_costs=0.0,
                confidence_0_1=0.0,
                horizon_minutes=60,
                engine_weights=weights,
                metadata={"reason": "total_weight_zero"},
            )
        
        # Weighted average of edges
        weighted_edge = sum(
            output.edge_bps_before_costs * weights.get(engine_id, 0.0)
            for engine_id, output in filtered_outputs.items()
        ) / total_weight
        
        # Weighted average of confidences
        weighted_confidence = sum(
            output.confidence_0_1 * weights.get(engine_id, 0.0)
            for engine_id, output in filtered_outputs.items()
        ) / total_weight
        
        # Average horizon
        avg_horizon = int(sum(
            output.horizon_minutes
            for output in filtered_outputs.values()
        ) / len(filtered_outputs))
        
        # Determine direction based on weighted edge
        if weighted_edge > 5.0:  # Threshold for buy
            direction = "buy"
        elif weighted_edge < -5.0:  # Threshold for sell
            direction = "sell"
        else:
            direction = "wait"
        
        return MetaOutput(
            direction=direction,
            edge_bps_before_costs=weighted_edge,
            confidence_0_1=weighted_confidence,
            horizon_minutes=avg_horizon,
            engine_weights=weights,
            metadata={
                "engine_count": len(filtered_outputs),
                "total_weight": total_weight,
                "regime": regime,
            },
        )
    
    def _get_weights(self, engine_ids: List[str]) -> Dict[str, float]:
        """Get weights for engines.
        
        Args:
            engine_ids: List of engine IDs
            
        Returns:
            Dictionary of engine_id to weight
        """
        weights = {}
        
        for engine_id in engine_ids:
            weight_obj = self.engine_weights.get(engine_id)
            
            if weight_obj:
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
    
    def update_weights(
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
        # Calculate new weight based on accuracy and net edge
        # Weight = accuracy * (1 + net_edge_bps / 100)
        base_weight = accuracy * (1.0 + max(0.0, net_edge_bps / 100.0))
        
        # Get existing weight
        existing_weight = self.engine_weights.get(engine_id)
        
        if existing_weight:
            # Update using EMA
            new_weight = self.ema_alpha * base_weight + (1 - self.ema_alpha) * existing_weight.weight
        else:
            # Initialize with base weight
            new_weight = base_weight
        
        # Clip to min/max
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        
        # Update weight object
        self.engine_weights[engine_id] = EngineWeight(
            engine_id=engine_id,
            weight=new_weight,
            accuracy=accuracy,
            net_edge_bps=net_edge_bps,
            last_updated=timestamp,
        )
        
        logger.debug("weight_updated", 
                    symbol=self.symbol,
                    engine_id=engine_id,
                    weight=new_weight,
                    accuracy=accuracy,
                    net_edge_bps=net_edge_bps)
    
    def get_top_engines(self, n: int = 10) -> List[str]:
        """Get top N engines by weight.
        
        Args:
            n: Number of top engines to return
            
        Returns:
            List of engine IDs sorted by weight
        """
        sorted_engines = sorted(
            self.engine_weights.items(),
            key=lambda x: x[1].weight,
            reverse=True,
        )
        
        return [engine_id for engine_id, _ in sorted_engines[:n]]

