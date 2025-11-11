"""
Hard Regime Gates - Strict engine enablement based on regime.

Only enables engines approved for the current regime.
Maintains per-regime leaderboards with weekly refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional
import structlog

logger = structlog.get_logger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    TREND = "trend"
    RANGE = "range"
    PANIC = "panic"
    ILLIQUID = "illiquid"
    UNKNOWN = "unknown"


@dataclass
class RegimeLeaderboard:
    """Per-regime engine leaderboard."""
    regime: RegimeType
    engines: List[Dict[str, any]]  # Sorted by performance
    last_refresh: datetime
    refresh_interval_days: int = 7


@dataclass
class EngineRegimeApproval:
    """Engine approval status per regime."""
    engine_id: str
    approved_regimes: Set[RegimeType]
    performance_by_regime: Dict[RegimeType, float]  # Performance score per regime


class HardRegimeGates:
    """
    Hard regime gates - only enable approved engines per regime.
    
    Features:
    - Regime classification (TREND, RANGE, PANIC, ILLIQUID)
    - Per-regime engine approval list
    - Weekly leaderboard refresh
    - Automatic engine enablement/disablement
    """
    
    def __init__(
        self,
        leaderboard_refresh_days: int = 7,
        min_performance_threshold: float = 0.0,
    ) -> None:
        """
        Initialize hard regime gates.
        
        Args:
            leaderboard_refresh_days: Days between leaderboard refreshes (default: 7)
            min_performance_threshold: Minimum performance to approve engine (default: 0.0)
        """
        self.leaderboard_refresh_days = leaderboard_refresh_days
        self.min_performance_threshold = min_performance_threshold
        
        # Per-regime leaderboards
        self.leaderboards: Dict[RegimeType, RegimeLeaderboard] = {}
        
        # Engine approvals per regime
        self.engine_approvals: Dict[str, EngineRegimeApproval] = {}
        
        # Default engine-regime mappings (can be overridden)
        self._initialize_default_approvals()
        
        logger.info(
            "hard_regime_gates_initialized",
            refresh_days=leaderboard_refresh_days,
            min_threshold=min_performance_threshold
        )
    
    def _initialize_default_approvals(self) -> None:
        """Initialize default engine-regime approvals."""
        # Price-action engines
        self._set_approval("trend_engine", {RegimeType.TREND})
        self._set_approval("range_engine", {RegimeType.RANGE})
        self._set_approval("breakout_engine", {RegimeType.TREND})
        self._set_approval("tape_engine", {RegimeType.TREND, RegimeType.RANGE})  # Works in most regimes
        self._set_approval("leader_engine", {RegimeType.TREND})
        self._set_approval("sweep_engine", {RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC})
        self._set_approval("scalper_engine", {RegimeType.TREND, RegimeType.RANGE})
        
        # Cross-asset engines
        self._set_approval("correlation_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("funding_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("arbitrage_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("volatility_engine", {RegimeType.TREND, RegimeType.PANIC})
        
        # Learning engines (work in all regimes)
        self._set_approval("adaptive_meta_engine", {
            RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC, RegimeType.ILLIQUID
        })
        self._set_approval("evolutionary_engine", {
            RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC
        })
        self._set_approval("risk_engine", {
            RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC, RegimeType.ILLIQUID
        })
        
        # Advanced engines
        self._set_approval("flow_prediction_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("latency_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("market_maker_engine", {RegimeType.RANGE})
        self._set_approval("anomaly_engine", {
            RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC
        })
        self._set_approval("regime_engine", {
            RegimeType.TREND, RegimeType.RANGE, RegimeType.PANIC, RegimeType.ILLIQUID
        })
        
        # Pattern engines
        self._set_approval("momentum_reversal_engine", {RegimeType.TREND, RegimeType.PANIC})
        self._set_approval("divergence_engine", {RegimeType.TREND, RegimeType.RANGE})
        self._set_approval("support_resistance_engine", {RegimeType.RANGE})
        self._set_approval("pattern_engine", {RegimeType.TREND, RegimeType.RANGE})
    
    def _set_approval(
        self,
        engine_id: str,
        approved_regimes: Set[RegimeType]
    ) -> None:
        """Set engine approval for regimes."""
        self.engine_approvals[engine_id] = EngineRegimeApproval(
            engine_id=engine_id,
            approved_regimes=approved_regimes,
            performance_by_regime={}
        )
    
    def classify_regime(
        self,
        features: Dict[str, float],
        current_regime: Optional[RegimeType] = None
    ) -> RegimeType:
        """
        Classify current market regime.
        
        Args:
            features: Market features (volatility, trend_strength, etc.)
            current_regime: Current regime (for persistence)
        
        Returns:
            Classified regime type
        """
        # Extract key features
        volatility = features.get('volatility', 0.0)
        trend_strength = features.get('trend_strength', 0.0)
        adx = features.get('adx', 0.0)
        volume = features.get('volume', 0.0)
        spread = features.get('spread_bps', 0.0)
        
        # Classification logic
        # PANIC: High volatility, negative trend, high volume
        if volatility > 0.05 and trend_strength < -0.5 and volume > 1.5:
            return RegimeType.PANIC
        
        # ILLIQUID: Low volume, wide spreads
        if volume < 0.3 or spread > 50:  # 50 bps spread
            return RegimeType.ILLIQUID
        
        # TREND: Strong trend (ADX > 25)
        if adx > 25:
            return RegimeType.TREND
        
        # RANGE: Low trend strength, moderate volatility
        if adx < 20 and 0.01 < volatility < 0.04:
            return RegimeType.RANGE
        
        # Default to current regime or UNKNOWN
        return current_regime or RegimeType.UNKNOWN
    
    def get_approved_engines(self, regime: RegimeType) -> List[str]:
        """
        Get list of approved engines for a regime.
        
        Args:
            regime: Current market regime
        
        Returns:
            List of approved engine IDs
        """
        approved = []
        for engine_id, approval in self.engine_approvals.items():
            if regime in approval.approved_regimes:
                approved.append(engine_id)
        
        logger.debug(
            "approved_engines_for_regime",
            regime=regime.value,
            count=len(approved),
            engines=approved
        )
        
        return approved
    
    def is_engine_approved(
        self,
        engine_id: str,
        regime: RegimeType
    ) -> bool:
        """
        Check if engine is approved for regime.
        
        Args:
            engine_id: Engine identifier
            regime: Current regime
        
        Returns:
            True if approved
        """
        if engine_id not in self.engine_approvals:
            return False
        
        return regime in self.engine_approvals[engine_id].approved_regimes
    
    def update_leaderboard(
        self,
        regime: RegimeType,
        engine_performance: Dict[str, Dict[str, float]],
        refresh_date: Optional[datetime] = None
    ) -> None:
        """
        Update per-regime leaderboard.
        
        Args:
            regime: Regime type
            engine_performance: Dictionary of engine_id -> performance metrics
            refresh_date: Refresh date (default: now)
        """
        if refresh_date is None:
            refresh_date = datetime.now()
        
        # Filter to approved engines only
        approved_engines = self.get_approved_engines(regime)
        filtered_performance = {
            engine_id: metrics
            for engine_id, metrics in engine_performance.items()
            if engine_id in approved_engines
        }
        
        # Sort by performance (e.g., Sharpe ratio or win rate)
        sorted_engines = sorted(
            filtered_performance.items(),
            key=lambda x: x[1].get('sharpe_ratio', x[1].get('win_rate', 0.0)),
            reverse=True
        )
        
        # Create leaderboard entry
        leaderboard_entries = [
            {
                'engine_id': engine_id,
                'performance': metrics,
                'rank': i + 1
            }
            for i, (engine_id, metrics) in enumerate(sorted_engines)
        ]
        
        # Update leaderboard
        self.leaderboards[regime] = RegimeLeaderboard(
            regime=regime,
            engines=leaderboard_entries,
            last_refresh=refresh_date,
            refresh_interval_days=self.leaderboard_refresh_days
        )
        
        logger.info(
            "leaderboard_updated",
            regime=regime.value,
            engines=len(leaderboard_entries),
            refresh_date=refresh_date.isoformat()
        )
    
    def should_refresh_leaderboard(self, regime: RegimeType) -> bool:
        """
        Check if leaderboard should be refreshed.
        
        Args:
            regime: Regime type
        
        Returns:
            True if refresh needed
        """
        if regime not in self.leaderboards:
            return True
        
        leaderboard = self.leaderboards[regime]
        days_since_refresh = (datetime.now() - leaderboard.last_refresh).days
        
        return days_since_refresh >= leaderboard.refresh_interval_days
    
    def get_leaderboard(self, regime: RegimeType) -> Optional[RegimeLeaderboard]:
        """
        Get leaderboard for regime.
        
        Args:
            regime: Regime type
        
        Returns:
            Leaderboard or None if not available
        """
        return self.leaderboards.get(regime)
    
    def filter_engines_by_regime(
        self,
        engine_ids: List[str],
        regime: RegimeType
    ) -> List[str]:
        """
        Filter engine list to only approved engines for regime.
        
        Args:
            engine_ids: List of engine IDs
            regime: Current regime
        
        Returns:
            Filtered list of approved engines
        """
        approved = self.get_approved_engines(regime)
        filtered = [eid for eid in engine_ids if eid in approved]
        
        if len(filtered) < len(engine_ids):
            logger.info(
                "engines_filtered_by_regime",
                regime=regime.value,
                original_count=len(engine_ids),
                filtered_count=len(filtered),
                removed=set(engine_ids) - set(filtered)
            )
        
        return filtered

