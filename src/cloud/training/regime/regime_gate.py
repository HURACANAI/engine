"""
Regime Gate Service for Training Architecture

Enables/disables engines per regime based on historical performance.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class RegimeType(Enum):
    """Regime type."""
    TREND = "trend"
    RANGE = "range"
    PANIC = "panic"
    ILLIQUID = "illiquid"


@dataclass
class RegimePerformance:
    """Regime performance metrics."""
    regime: RegimeType
    engine_type: str
    win_rate: float
    sharpe_ratio: float
    avg_return_bps: float
    sample_size: int
    last_updated: float = field(default_factory=time.time)


@dataclass
class RegimeGateConfig:
    """Regime gate configuration."""
    min_win_rate: float = 0.55
    min_sharpe: float = 1.0
    min_sample_size: int = 50
    enable_all_by_default: bool = True


class RegimeGate:
    """
    Regime gate for enabling/disabling engines per regime.
    
    Features:
    - Per-regime engine enabling
    - Performance-based gating
    - Dynamic updates
    - Default fallback behavior
    """
    
    def __init__(self, config: RegimeGateConfig):
        """
        Initialize regime gate.
        
        Args:
            config: Regime gate configuration
        """
        self.config = config
        self.regime_performance: Dict[tuple[RegimeType, str], RegimePerformance] = {}
        self.enabled_engines: Dict[RegimeType, Set[str]] = {
            regime: set() for regime in RegimeType
        }
        self.default_engines: Set[str] = set()
        
        logger.info(
            "regime_gate_initialized",
            min_win_rate=config.min_win_rate,
            min_sharpe=config.min_sharpe,
            min_sample_size=config.min_sample_size,
        )
    
    def update_performance(
        self,
        regime: RegimeType,
        engine_type: str,
        win_rate: float,
        sharpe_ratio: float,
        avg_return_bps: float,
        sample_size: int,
    ) -> None:
        """
        Update performance for an engine in a regime.
        
        Args:
            regime: Regime type
            engine_type: Engine type
            win_rate: Win rate
            sharpe_ratio: Sharpe ratio
            avg_return_bps: Average return in basis points
            sample_size: Sample size
        """
        key = (regime, engine_type)
        self.regime_performance[key] = RegimePerformance(
            regime=regime,
            engine_type=engine_type,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            avg_return_bps=avg_return_bps,
            sample_size=sample_size,
            last_updated=time.time(),
        )
        
        # Update enabled engines
        self._update_enabled_engines(regime, engine_type)
        
        logger.debug(
            "regime_performance_updated",
            regime=regime.value,
            engine_type=engine_type,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
        )
    
    def _update_enabled_engines(self, regime: RegimeType, engine_type: str) -> None:
        """Update enabled engines for a regime."""
        key = (regime, engine_type)
        performance = self.regime_performance.get(key)
        
        if not performance:
            return
        
        # Check if engine meets criteria
        meets_criteria = (
            performance.sample_size >= self.config.min_sample_size and
            performance.win_rate >= self.config.min_win_rate and
            performance.sharpe_ratio >= self.config.min_sharpe
        )
        
        if meets_criteria:
            self.enabled_engines[regime].add(engine_type)
        else:
            self.enabled_engines[regime].discard(engine_type)
        
        logger.debug(
            "engine_gating_updated",
            regime=regime.value,
            engine_type=engine_type,
            enabled=meets_criteria,
        )
    
    def is_engine_enabled(self, regime: RegimeType, engine_type: str) -> bool:
        """
        Check if an engine is enabled for a regime.
        
        Args:
            regime: Regime type
            engine_type: Engine type
        
        Returns:
            True if engine is enabled
        """
        # Check if engine is explicitly enabled
        if engine_type in self.enabled_engines[regime]:
            return True
        
        # Check default behavior
        if self.config.enable_all_by_default:
            # If no performance data, enable by default
            key = (regime, engine_type)
            if key not in self.regime_performance:
                return True
            
            # If performance data exists but doesn't meet criteria, disable
            return False
        else:
            # Disable by default if not explicitly enabled
            return False
    
    def get_enabled_engines(self, regime: RegimeType) -> Set[str]:
        """
        Get enabled engines for a regime.
        
        Args:
            regime: Regime type
        
        Returns:
            Set of enabled engine types
        """
        return self.enabled_engines[regime].copy()
    
    def get_all_enabled_engines(self) -> Dict[RegimeType, Set[str]]:
        """Get all enabled engines per regime."""
        return {
            regime: engines.copy()
            for regime, engines in self.enabled_engines.items()
        }
    
    def get_regime_map(self, coin: str) -> Dict[str, Any]:
        """
        Get regime map for a coin.
        
        Args:
            coin: Coin symbol
        
        Returns:
            Dictionary with regime information
        """
        return {
            "coin": coin,
            "regimes": {
                regime.value: {
                    "enabled_engines": list(self.enabled_engines[regime]),
                    "engine_count": len(self.enabled_engines[regime]),
                }
                for regime in RegimeType
            },
            "performance": {
                f"{regime.value}_{engine}": {
                    "win_rate": perf.win_rate,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "avg_return_bps": perf.avg_return_bps,
                    "sample_size": perf.sample_size,
                }
                for (regime, engine), perf in self.regime_performance.items()
            },
        }
    
    def set_default_engines(self, engines: Set[str]) -> None:
        """Set default engines for all regimes."""
        self.default_engines = engines.copy()
        logger.info("default_engines_set", engines=list(engines))
    
    def enable_engine_for_regime(self, regime: RegimeType, engine_type: str) -> None:
        """Manually enable an engine for a regime."""
        self.enabled_engines[regime].add(engine_type)
        logger.info("engine_enabled", regime=regime.value, engine_type=engine_type)
    
    def disable_engine_for_regime(self, regime: RegimeType, engine_type: str) -> None:
        """Manually disable an engine for a regime."""
        self.enabled_engines[regime].discard(engine_type)
        logger.info("engine_disabled", regime=regime.value, engine_type=engine_type)
