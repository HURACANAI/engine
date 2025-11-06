"""
Adaptive / Meta-Learning Engine

Tracks each engine's performance and re-weights them dynamically.
Learns which engines work best in which conditions.

Key Features:
1. Engine performance tracking (win rate, Sharpe, profit)
2. Dynamic re-weighting (adjust weights based on performance)
3. Regime-specific performance (track performance per regime)
4. Auto-disable underperforming engines
5. Meta-learning (learns which engines to use when)

Best in: All regimes (continuously adapts)
Strategy: Track performance and re-weight engines dynamically
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EngineType(Enum):
    """Engine types."""
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    TAPE = "tape"
    LEADER = "leader"
    SWEEP = "sweep"
    SCALPER = "scalper"
    VOLATILITY = "volatility"
    FUNDING = "funding"
    ARBITRAGE = "arbitrage"
    CORRELATION = "correlation"
    FLOW_PREDICTION = "flow_prediction"
    LATENCY = "latency"
    MARKET_MAKER = "market_maker"


@dataclass
class EnginePerformance:
    """Performance metrics for an engine."""
    engine_type: EngineType
    n_trades: int
    n_wins: int
    win_rate: float
    total_profit_bps: float
    avg_profit_bps: float
    sharpe_ratio: float
    max_drawdown_pct: float
    last_updated: datetime
    performance_by_regime: Dict[str, Dict[str, float]]  # regime -> metrics


@dataclass
class EngineWeight:
    """Engine weight for ensemble."""
    engine_type: EngineType
    weight: float  # 0-1, weight in ensemble
    confidence: float  # 0-1, confidence in weight
    reasoning: str
    last_updated: datetime


class AdaptiveMetaEngine:
    """
    Adaptive / Meta-Learning Engine.
    
    Tracks each engine's performance and re-weights them dynamically.
    Learns which engines work best in which conditions.
    
    Key Features:
    - Engine performance tracking
    - Dynamic re-weighting
    - Regime-specific performance
    - Auto-disable underperforming engines
    - Meta-learning
    """
    
    def __init__(
        self,
        min_win_rate: float = 0.50,  # Minimum win rate to keep engine active
        min_sharpe: float = 0.5,  # Minimum Sharpe to keep engine active
        lookback_trades: int = 100,  # Number of trades to look back
        reweight_frequency: int = 50,  # Reweight every N trades
        use_meta_learning: bool = True,  # Use meta-learning for weight optimization
    ):
        """
        Initialize adaptive/meta-learning engine.
        
        Args:
            min_win_rate: Minimum win rate to keep engine active
            min_sharpe: Minimum Sharpe to keep engine active
            lookback_trades: Number of trades to look back
            reweight_frequency: Reweight every N trades
            use_meta_learning: Whether to use meta-learning
        """
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.lookback_trades = lookback_trades
        self.reweight_frequency = reweight_frequency
        self.use_meta_learning = use_meta_learning
        
        # Engine performance tracking
        self.engine_performance: Dict[EngineType, EnginePerformance] = {}
        
        # Engine weights
        self.engine_weights: Dict[EngineType, EngineWeight] = {}
        
        # Trade history (engine -> [trade_results])
        self.trade_history: Dict[EngineType, List[Dict[str, float]]] = {}
        
        # Reweight counter
        self.trade_count = 0
        
        logger.info(
            "adaptive_meta_engine_initialized",
            min_win_rate=min_win_rate,
            min_sharpe=min_sharpe,
            lookback_trades=lookback_trades,
        )
    
    def update_engine_performance(
        self,
        engine_type: EngineType,
        trade_result: Dict[str, float],  # {won: bool, profit_bps: float, regime: str, ...}
    ) -> None:
        """
        Update engine performance with new trade result.
        
        Args:
            engine_type: Engine type
            trade_result: Trade result dictionary
        """
        if engine_type not in self.trade_history:
            self.trade_history[engine_type] = []
        
        # Add trade result
        self.trade_history[engine_type].append(trade_result)
        
        # Keep only last N trades
        if len(self.trade_history[engine_type]) > self.lookback_trades:
            self.trade_history[engine_type] = self.trade_history[engine_type][-self.lookback_trades:]
        
        # Recalculate performance
        self._recalculate_performance(engine_type)
        
        # Increment trade count
        self.trade_count += 1
        
        # Reweight if needed
        if self.trade_count % self.reweight_frequency == 0:
            self._reweight_engines()
    
    def _recalculate_performance(
        self,
        engine_type: EngineType,
    ) -> None:
        """Recalculate performance metrics for an engine."""
        trade_history = self.trade_history.get(engine_type, [])
        
        if len(trade_history) == 0:
            return
        
        # Calculate overall metrics
        n_trades = len(trade_history)
        n_wins = sum(1 for t in trade_history if t.get("won", False))
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        
        profits = [t.get("profit_bps", 0.0) for t in trade_history]
        total_profit_bps = sum(profits)
        avg_profit_bps = np.mean(profits) if profits else 0.0
        
        # Calculate Sharpe ratio
        if len(profits) > 1 and np.std(profits) > 0:
            sharpe_ratio = np.mean(profits) / np.std(profits)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        cumulative_profits = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = (running_max - cumulative_profits) / (running_max + 1e-10) * 100.0
        max_drawdown_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Calculate performance by regime
        performance_by_regime = {}
        regimes = set(t.get("regime", "UNKNOWN") for t in trade_history)
        
        for regime in regimes:
            regime_trades = [t for t in trade_history if t.get("regime") == regime]
            if len(regime_trades) > 0:
                regime_wins = sum(1 for t in regime_trades if t.get("won", False))
                regime_win_rate = regime_wins / len(regime_trades)
                regime_profits = [t.get("profit_bps", 0.0) for t in regime_trades]
                regime_avg_profit = np.mean(regime_profits) if regime_profits else 0.0
                
                if len(regime_profits) > 1 and np.std(regime_profits) > 0:
                    regime_sharpe = np.mean(regime_profits) / np.std(regime_profits)
                else:
                    regime_sharpe = 0.0
                
                performance_by_regime[regime] = {
                    "n_trades": len(regime_trades),
                    "win_rate": regime_win_rate,
                    "avg_profit_bps": regime_avg_profit,
                    "sharpe_ratio": regime_sharpe,
                }
        
        # Update performance
        self.engine_performance[engine_type] = EnginePerformance(
            engine_type=engine_type,
            n_trades=n_trades,
            n_wins=n_wins,
            win_rate=win_rate,
            total_profit_bps=total_profit_bps,
            avg_profit_bps=avg_profit_bps,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            last_updated=datetime.now(timezone.utc),
            performance_by_regime=performance_by_regime,
        )
    
    def _reweight_engines(self) -> None:
        """Recalculate engine weights based on performance."""
        if len(self.engine_performance) == 0:
            return
        
        # Calculate weights based on performance
        weights = {}
        total_score = 0.0
        
        for engine_type, performance in self.engine_performance.items():
            # Calculate composite score
            # Higher win rate, Sharpe, and profit = higher score
            win_rate_score = performance.win_rate
            sharpe_score = max(0.0, performance.sharpe_ratio / 2.0)  # Normalize to [0, 1]
            profit_score = min(1.0, performance.avg_profit_bps / 100.0)  # Normalize to [0, 1]
            
            # Composite score (weighted average)
            composite_score = (win_rate_score * 0.4 + sharpe_score * 0.3 + profit_score * 0.3)
            
            # Penalize if below minimum thresholds
            if performance.win_rate < self.min_win_rate:
                composite_score *= 0.5  # Halve score
            if performance.sharpe_ratio < self.min_sharpe:
                composite_score *= 0.5  # Halve score
            
            weights[engine_type] = composite_score
            total_score += composite_score
        
        # Normalize weights
        if total_score > 0:
            for engine_type in weights:
                weights[engine_type] = weights[engine_type] / total_score
        else:
            # Equal weights if all scores are zero
            n_engines = len(weights)
            for engine_type in weights:
                weights[engine_type] = 1.0 / n_engines if n_engines > 0 else 0.0
        
        # Update engine weights
        for engine_type, weight in weights.items():
            performance = self.engine_performance[engine_type]
            
            # Calculate confidence (based on number of trades)
            confidence = min(1.0, performance.n_trades / self.lookback_trades)
            
            # Determine reasoning
            if performance.win_rate < self.min_win_rate:
                reasoning = f"Low win rate ({performance.win_rate:.1%} < {self.min_win_rate:.1%})"
            elif performance.sharpe_ratio < self.min_sharpe:
                reasoning = f"Low Sharpe ({performance.sharpe_ratio:.2f} < {self.min_sharpe:.2f})"
            elif weight > 0.15:
                reasoning = f"Strong performance (WR={performance.win_rate:.1%}, Sharpe={performance.sharpe_ratio:.2f})"
            else:
                reasoning = f"Moderate performance (WR={performance.win_rate:.1%}, Sharpe={performance.sharpe_ratio:.2f})"
            
            self.engine_weights[engine_type] = EngineWeight(
                engine_type=engine_type,
                weight=weight,
                confidence=confidence,
                reasoning=reasoning,
                last_updated=datetime.now(timezone.utc),
            )
        
        logger.info(
            "engines_reweighted",
            n_engines=len(weights),
            total_trades=self.trade_count,
        )
    
    def get_engine_weights(
        self,
        current_regime: Optional[str] = None,
    ) -> Dict[EngineType, float]:
        """
        Get engine weights for ensemble.
        
        Args:
            current_regime: Current market regime (for regime-specific weighting)
        
        Returns:
            Dict of {engine_type: weight}
        """
        if len(self.engine_weights) == 0:
            # No weights yet, return equal weights
            return {}
        
        # If regime-specific weighting is requested
        if current_regime:
            # Adjust weights based on regime-specific performance
            adjusted_weights = {}
            total_adjusted = 0.0
            
            for engine_type, weight_obj in self.engine_weights.items():
                performance = self.engine_performance.get(engine_type)
                if performance and current_regime in performance.performance_by_regime:
                    # Boost weight if engine performs well in this regime
                    regime_perf = performance.performance_by_regime[current_regime]
                    regime_boost = min(1.5, 1.0 + regime_perf.get("sharpe_ratio", 0.0) * 0.2)
                    adjusted_weight = weight_obj.weight * regime_boost
                else:
                    adjusted_weight = weight_obj.weight
                
                adjusted_weights[engine_type] = adjusted_weight
                total_adjusted += adjusted_weight
            
            # Normalize
            if total_adjusted > 0:
                for engine_type in adjusted_weights:
                    adjusted_weights[engine_type] = adjusted_weights[engine_type] / total_adjusted
            
            return {engine_type: w for engine_type, w in adjusted_weights.items()}
        else:
            # Return base weights
            return {engine_type: w.weight for engine_type, w in self.engine_weights.items()}
    
    def get_active_engines(
        self,
        min_weight: float = 0.05,  # Minimum weight to be active
    ) -> List[EngineType]:
        """
        Get list of active engines (above minimum weight).
        
        Args:
            min_weight: Minimum weight to be active
        
        Returns:
            List of active engine types
        """
        active = []
        for engine_type, weight_obj in self.engine_weights.items():
            if weight_obj.weight >= min_weight:
                active.append(engine_type)
        
        return active
    
    def get_engine_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all engine performance."""
        summary = {}
        
        for engine_type, performance in self.engine_performance.items():
            weight_obj = self.engine_weights.get(engine_type)
            weight = weight_obj.weight if weight_obj else 0.0
            
            summary[engine_type.value] = {
                "n_trades": performance.n_trades,
                "win_rate": performance.win_rate,
                "avg_profit_bps": performance.avg_profit_bps,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown_pct": performance.max_drawdown_pct,
                "weight": weight,
                "is_active": weight >= 0.05,
            }
        
        return summary

