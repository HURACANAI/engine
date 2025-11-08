"""
Meta Engine

Learns which base engines perform best in different regimes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EnginePerformance:
    """Performance record for a base engine"""
    engine_name: str
    regime: str
    timestamp: datetime

    sharpe: float
    pnl_bps: float
    win_rate: float
    num_trades: int


@dataclass
class EngineSelection:
    """Engine selection result"""
    selected_engine: str
    confidence: float  # [0-1]
    regime: str

    # Alternative engines with weights
    engine_weights: Dict[str, float]

    # Reasoning
    reason: str


class MetaEngine:
    """
    Adaptive Meta Engine

    Learns which base trading engines perform best in different regimes
    and adaptively selects/blends them.

    Example:
        meta = MetaEngine(
            base_engines=["trend", "mean_reversion", "breakout"]
        )

        # Track performance
        meta.update_performance(
            engine_name="trend",
            regime="trending",
            sharpe=2.0,
            pnl_bps=300,
            win_rate=0.6,
            num_trades=50
        )

        # Select best engine
        selection = meta.select_engine(
            current_regime="trending",
            lookback_days=30
        )

        print(f"Use: {selection.selected_engine}")
        print(f"Confidence: {selection.confidence:.0%}")
    """

    def __init__(
        self,
        base_engines: List[str],
        decay_halflife_days: int = 30,
        min_trades_for_confidence: int = 20
    ):
        """
        Initialize meta engine

        Args:
            base_engines: List of base engine names
            decay_halflife_days: Half-life for performance decay (recent > old)
            min_trades_for_confidence: Min trades needed for high confidence
        """
        self.base_engines = base_engines
        self.decay_halflife_days = decay_halflife_days
        self.min_trades_for_confidence = min_trades_for_confidence

        # Performance history
        self.performance_history: List[EnginePerformance] = []

        # Cached regime-specific stats
        self._regime_stats_cache: Dict[str, Dict] = {}
        self._cache_updated: Optional[datetime] = None

    def update_performance(
        self,
        engine_name: str,
        regime: str,
        sharpe: float,
        pnl_bps: float,
        win_rate: float,
        num_trades: int,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update engine performance

        Args:
            engine_name: Engine name
            regime: Market regime
            sharpe: Sharpe ratio
            pnl_bps: PnL in basis points
            win_rate: Win rate [0-1]
            num_trades: Number of trades
            timestamp: Timestamp (defaults to now)
        """
        if engine_name not in self.base_engines:
            logger.warning(
                "unknown_engine_ignored",
                engine_name=engine_name,
                known_engines=self.base_engines
            )
            return

        if timestamp is None:
            timestamp = datetime.utcnow()

        record = EnginePerformance(
            engine_name=engine_name,
            regime=regime,
            timestamp=timestamp,
            sharpe=sharpe,
            pnl_bps=pnl_bps,
            win_rate=win_rate,
            num_trades=num_trades
        )

        self.performance_history.append(record)

        # Invalidate cache
        self._cache_updated = None

        logger.debug(
            "engine_performance_updated",
            engine_name=engine_name,
            regime=regime,
            sharpe=sharpe,
            pnl_bps=pnl_bps
        )

    def select_engine(
        self,
        current_regime: str,
        lookback_days: int = 30,
        metric: str = "sharpe"
    ) -> EngineSelection:
        """
        Select best engine for current regime

        Args:
            current_regime: Current market regime
            lookback_days: Days to look back for performance
            metric: Metric to optimize ("sharpe", "pnl", "win_rate")

        Returns:
            EngineSelection
        """
        # Get regime stats
        regime_stats = self._get_regime_stats(
            regime=current_regime,
            lookback_days=lookback_days
        )

        if len(regime_stats) == 0:
            # No data for this regime - use best overall
            logger.warning(
                "no_regime_data_using_fallback",
                regime=current_regime
            )
            return self._select_fallback_engine(current_regime)

        # Find best engine by metric
        best_engine = None
        best_value = float('-inf')

        for engine_name, stats in regime_stats.items():
            value = stats.get(metric, 0.0)

            if value > best_value:
                best_value = value
                best_engine = engine_name

        # Calculate confidence based on number of trades
        total_trades = regime_stats.get(best_engine, {}).get('total_trades', 0)
        confidence = min(1.0, total_trades / self.min_trades_for_confidence)

        # Calculate engine weights (softmax of metric values)
        engine_weights = self._calculate_softmax_weights(regime_stats, metric)

        selection = EngineSelection(
            selected_engine=best_engine,
            confidence=confidence,
            regime=current_regime,
            engine_weights=engine_weights,
            reason=(
                f"Best {metric}={best_value:.2f} in {current_regime} "
                f"({total_trades} trades)"
            )
        )

        logger.info(
            "engine_selected",
            regime=current_regime,
            selected_engine=best_engine,
            confidence=confidence,
            metric=metric,
            value=best_value
        )

        return selection

    def get_engine_weights(
        self,
        regime: str,
        lookback_days: int = 30,
        metric: str = "sharpe"
    ) -> Dict[str, float]:
        """
        Get engine weights for blending

        Args:
            regime: Market regime
            lookback_days: Lookback period
            metric: Metric to weight by

        Returns:
            Dict of {engine_name: weight}
        """
        regime_stats = self._get_regime_stats(regime, lookback_days)

        if len(regime_stats) == 0:
            # Equal weights if no data
            return {engine: 1.0 / len(self.base_engines) for engine in self.base_engines}

        weights = self._calculate_softmax_weights(regime_stats, metric)

        return weights

    def _get_regime_stats(
        self,
        regime: str,
        lookback_days: int
    ) -> Dict[str, Dict]:
        """
        Get performance statistics by regime

        Args:
            regime: Regime name
            lookback_days: Days to look back

        Returns:
            Dict of {engine_name: stats}
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        # Filter relevant records
        relevant = [
            r for r in self.performance_history
            if r.regime == regime and r.timestamp >= cutoff
        ]

        if len(relevant) == 0:
            return {}

        # Group by engine
        engine_stats = {}

        for engine_name in self.base_engines:
            engine_records = [r for r in relevant if r.engine_name == engine_name]

            if len(engine_records) == 0:
                continue

            # Calculate weighted statistics (recent > old)
            weights = self._calculate_time_weights(engine_records)

            weighted_sharpe = sum(
                r.sharpe * w for r, w in zip(engine_records, weights)
            ) / sum(weights)

            weighted_pnl = sum(
                r.pnl_bps * w for r, w in zip(engine_records, weights)
            ) / sum(weights)

            weighted_win_rate = sum(
                r.win_rate * w for r, w in zip(engine_records, weights)
            ) / sum(weights)

            total_trades = sum(r.num_trades for r in engine_records)

            engine_stats[engine_name] = {
                "sharpe": weighted_sharpe,
                "pnl": weighted_pnl,
                "win_rate": weighted_win_rate,
                "total_trades": total_trades,
                "num_records": len(engine_records)
            }

        return engine_stats

    def _calculate_time_weights(
        self,
        records: List[EnginePerformance]
    ) -> np.ndarray:
        """
        Calculate exponential time decay weights

        Recent records get higher weight.

        Args:
            records: Performance records

        Returns:
            Array of weights
        """
        if len(records) == 0:
            return np.array([])

        # Calculate days ago for each record
        now = datetime.utcnow()
        days_ago = np.array([
            (now - r.timestamp).total_seconds() / 86400
            for r in records
        ])

        # Exponential decay: weight = 0.5^(days_ago / halflife)
        decay_lambda = np.log(2) / self.decay_halflife_days
        weights = np.exp(-decay_lambda * days_ago)

        return weights

    def _calculate_softmax_weights(
        self,
        regime_stats: Dict[str, Dict],
        metric: str,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate softmax weights from metric values

        Args:
            regime_stats: Engine statistics
            metric: Metric to use
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Dict of {engine_name: weight}
        """
        if len(regime_stats) == 0:
            return {}

        # Get metric values
        values = np.array([
            stats.get(metric, 0.0)
            for stats in regime_stats.values()
        ])

        # Softmax
        exp_values = np.exp(values / temperature)
        softmax = exp_values / exp_values.sum()

        # Map back to engine names
        weights = {
            engine: float(weight)
            for engine, weight in zip(regime_stats.keys(), softmax)
        }

        return weights

    def _select_fallback_engine(self, regime: str) -> EngineSelection:
        """
        Select fallback engine when no regime-specific data

        Args:
            regime: Regime name

        Returns:
            EngineSelection
        """
        # Use best overall engine across all regimes
        all_stats = {}

        for engine_name in self.base_engines:
            engine_records = [
                r for r in self.performance_history
                if r.engine_name == engine_name
            ]

            if len(engine_records) == 0:
                all_stats[engine_name] = {"sharpe": 0.0}
            else:
                avg_sharpe = np.mean([r.sharpe for r in engine_records])
                all_stats[engine_name] = {"sharpe": avg_sharpe}

        # Find best
        best_engine = max(all_stats, key=lambda e: all_stats[e]["sharpe"])
        best_sharpe = all_stats[best_engine]["sharpe"]

        # Equal weights as fallback
        equal_weight = 1.0 / len(self.base_engines)
        engine_weights = {engine: equal_weight for engine in self.base_engines}

        return EngineSelection(
            selected_engine=best_engine,
            confidence=0.5,  # Low confidence without regime data
            regime=regime,
            engine_weights=engine_weights,
            reason=f"Fallback: best overall Sharpe={best_sharpe:.2f}"
        )

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary DataFrame

        Returns:
            DataFrame with performance by engine and regime
        """
        if len(self.performance_history) == 0:
            return pd.DataFrame()

        records = []

        for record in self.performance_history:
            records.append({
                "timestamp": record.timestamp,
                "engine": record.engine_name,
                "regime": record.regime,
                "sharpe": record.sharpe,
                "pnl_bps": record.pnl_bps,
                "win_rate": record.win_rate,
                "num_trades": record.num_trades
            })

        return pd.DataFrame(records)
