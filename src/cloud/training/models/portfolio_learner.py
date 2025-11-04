"""
Portfolio-Level Learning Module

Learns from portfolio outcomes rather than just individual trades:
1. Portfolio-wide feature importance
2. Cross-symbol pattern transfer learning
3. Portfolio regime detection
4. Global win/loss patterns across all assets

Goes beyond single-trade analysis to understand portfolio dynamics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class PortfolioOutcome:
    """Outcome of a portfolio state."""

    timestamp: datetime
    symbols: List[str]
    total_pnl_gbp: float
    win_rate: float  # % of winning positions
    num_positions: int
    portfolio_regime: str  # Regime during this outcome
    features_snapshot: Dict[str, float]  # Portfolio-level features


@dataclass
class CrossSymbolPattern:
    """Pattern that appears across multiple symbols."""

    pattern_signature: str  # Unique identifier
    symbols: Set[str]  # Symbols where pattern appears
    avg_win_rate: float
    sample_count: int
    last_seen: datetime


class PortfolioLearner:
    """
    Learns from portfolio-level outcomes.

    Key capabilities:
    1. Track which features predict portfolio success
    2. Learn patterns that work across multiple symbols
    3. Detect portfolio-level regimes
    4. Transfer learning between correlated symbols
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        cross_symbol_threshold: int = 3,  # Pattern must appear in 3+ symbols
    ):
        """
        Initialize portfolio learner.

        Args:
            ema_alpha: Learning rate for EMA updates
            cross_symbol_threshold: Min symbols for cross-symbol pattern
        """
        self.ema_alpha = ema_alpha
        self.cross_symbol_threshold = cross_symbol_threshold

        # Portfolio-level feature importance
        self.portfolio_feature_importance: Dict[str, float] = {}

        # Cross-symbol patterns
        self.cross_symbol_patterns: Dict[str, CrossSymbolPattern] = {}

        # Portfolio outcomes history
        self.portfolio_outcomes: List[PortfolioOutcome] = []

        # Global statistics
        self.total_portfolio_sessions = 0
        self.successful_sessions = 0  # Sessions with net profit

        logger.info(
            "portfolio_learner_initialized",
            ema_alpha=ema_alpha,
            cross_symbol_threshold=cross_symbol_threshold,
        )

    def update_portfolio_outcome(
        self,
        timestamp: datetime,
        symbols: List[str],
        total_pnl_gbp: float,
        win_rate: float,
        num_positions: int,
        portfolio_regime: str,
        features: Dict[str, float],
    ) -> None:
        """
        Update learning with a portfolio outcome.

        Args:
            timestamp: When outcome occurred
            symbols: Symbols in portfolio
            total_pnl_gbp: Total P&L
            win_rate: Win rate across all positions
            num_positions: Number of positions
            portfolio_regime: Portfolio-level regime
            features: Portfolio-level features (aggregated)
        """
        # Store outcome
        outcome = PortfolioOutcome(
            timestamp=timestamp,
            symbols=symbols,
            total_pnl_gbp=total_pnl_gbp,
            win_rate=win_rate,
            num_positions=num_positions,
            portfolio_regime=portfolio_regime,
            features_snapshot=features,
        )

        self.portfolio_outcomes.append(outcome)

        # Update global stats
        self.total_portfolio_sessions += 1
        if total_pnl_gbp > 0:
            self.successful_sessions += 1

        # Update portfolio-level feature importance
        is_successful = total_pnl_gbp > 0
        self._update_portfolio_feature_importance(features, is_successful)

        # Keep only recent outcomes (last 100)
        if len(self.portfolio_outcomes) > 100:
            self.portfolio_outcomes = self.portfolio_outcomes[-100:]

        logger.debug(
            "portfolio_outcome_recorded",
            pnl_gbp=total_pnl_gbp,
            win_rate=win_rate,
            num_symbols=len(symbols),
            total_sessions=self.total_portfolio_sessions,
        )

    def _update_portfolio_feature_importance(
        self, features: Dict[str, float], is_successful: bool
    ) -> None:
        """
        Update portfolio-level feature importance using EMA.

        Args:
            features: Portfolio features
            is_successful: Whether portfolio outcome was successful
        """
        success_signal = 1.0 if is_successful else -1.0

        for feature_name, feature_value in features.items():
            if not isinstance(feature_value, (int, float)):
                continue

            if np.isnan(feature_value) or np.isinf(feature_value):
                continue

            # Normalize feature (simple tanh normalization)
            normalized_value = np.tanh(feature_value / 100.0)

            # Update importance using EMA
            correlation_update = normalized_value * success_signal

            if feature_name not in self.portfolio_feature_importance:
                self.portfolio_feature_importance[feature_name] = correlation_update
            else:
                old_importance = self.portfolio_feature_importance[feature_name]
                self.portfolio_feature_importance[feature_name] = (
                    1 - self.ema_alpha
                ) * old_importance + self.ema_alpha * correlation_update

    def record_cross_symbol_pattern(
        self,
        pattern_signature: str,
        symbol: str,
        is_winner: bool,
        timestamp: datetime,
    ) -> None:
        """
        Record a pattern that appears on a symbol.

        Args:
            pattern_signature: Unique pattern identifier
            symbol: Symbol where pattern appeared
            is_winner: Whether trade was successful
            timestamp: When pattern occurred
        """
        if pattern_signature not in self.cross_symbol_patterns:
            self.cross_symbol_patterns[pattern_signature] = CrossSymbolPattern(
                pattern_signature=pattern_signature,
                symbols=set(),
                avg_win_rate=0.5,
                sample_count=0,
                last_seen=timestamp,
            )

        pattern = self.cross_symbol_patterns[pattern_signature]

        # Add symbol to pattern's symbol set
        pattern.symbols.add(symbol)

        # Update win rate using EMA
        win_signal = 1.0 if is_winner else 0.0
        pattern.avg_win_rate = (
            1 - self.ema_alpha
        ) * pattern.avg_win_rate + self.ema_alpha * win_signal

        pattern.sample_count += 1
        pattern.last_seen = timestamp

    def get_cross_symbol_patterns(
        self, min_symbols: Optional[int] = None
    ) -> List[CrossSymbolPattern]:
        """
        Get patterns that appear across multiple symbols.

        Args:
            min_symbols: Minimum number of symbols (default: cross_symbol_threshold)

        Returns:
            List of cross-symbol patterns sorted by win rate
        """
        threshold = min_symbols or self.cross_symbol_threshold

        patterns = [
            p
            for p in self.cross_symbol_patterns.values()
            if len(p.symbols) >= threshold
        ]

        # Sort by win rate
        patterns.sort(key=lambda x: x.avg_win_rate, reverse=True)

        return patterns

    def should_apply_pattern_to_symbol(
        self, pattern_signature: str, target_symbol: str, correlated_symbols: Set[str]
    ) -> tuple[bool, float]:
        """
        Determine if a pattern should be applied to a target symbol.

        Uses transfer learning: if pattern works on correlated symbols,
        it might work on target symbol.

        Args:
            pattern_signature: Pattern to evaluate
            target_symbol: Symbol to apply pattern to
            correlated_symbols: Symbols correlated with target

        Returns:
            (should_apply, confidence) tuple
        """
        if pattern_signature not in self.cross_symbol_patterns:
            return (False, 0.0)

        pattern = self.cross_symbol_patterns[pattern_signature]

        # Check if pattern has been tested on target symbol
        if target_symbol in pattern.symbols:
            # Already tested, use actual win rate
            confidence = pattern.avg_win_rate
            return (pattern.avg_win_rate > 0.55, confidence)

        # Check overlap with correlated symbols
        overlap = pattern.symbols & correlated_symbols

        if len(overlap) == 0:
            # No correlation data
            return (False, 0.0)

        # Pattern works on correlated symbols, try it here
        # Confidence based on pattern win rate and number of correlated symbols
        base_confidence = pattern.avg_win_rate
        correlation_boost = min(0.1, len(overlap) * 0.03)  # Up to +10% boost
        confidence = base_confidence + correlation_boost

        should_apply = confidence > 0.6 and pattern.sample_count >= 10

        return (should_apply, confidence)

    def get_portfolio_success_rate(self) -> float:
        """Get overall portfolio success rate."""
        if self.total_portfolio_sessions == 0:
            return 0.5

        return self.successful_sessions / self.total_portfolio_sessions

    def get_portfolio_feature_importance(self, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Get top portfolio-level features.

        Args:
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        sorted_features = sorted(
            self.portfolio_feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return sorted_features[:top_k]

    def get_best_regime_for_portfolio(self) -> tuple[str, float]:
        """
        Determine which regime has best portfolio outcomes.

        Returns:
            (best_regime, win_rate) tuple
        """
        if not self.portfolio_outcomes:
            return ("unknown", 0.5)

        # Group outcomes by regime
        regime_outcomes: Dict[str, List[float]] = {}

        for outcome in self.portfolio_outcomes:
            regime = outcome.portfolio_regime
            if regime not in regime_outcomes:
                regime_outcomes[regime] = []

            # Classify as win if pnl > 0
            regime_outcomes[regime].append(1.0 if outcome.total_pnl_gbp > 0 else 0.0)

        # Calculate win rate for each regime
        regime_win_rates = {
            regime: np.mean(outcomes) for regime, outcomes in regime_outcomes.items()
        }

        # Find best regime
        best_regime = max(regime_win_rates.items(), key=lambda x: x[1])

        return best_regime

    def get_optimal_portfolio_size(self) -> tuple[int, float]:
        """
        Determine optimal number of concurrent positions.

        Returns:
            (optimal_size, win_rate) tuple
        """
        if not self.portfolio_outcomes:
            return (3, 0.5)  # Default to 3 positions

        # Group outcomes by portfolio size
        size_outcomes: Dict[int, List[float]] = {}

        for outcome in self.portfolio_outcomes:
            size = outcome.num_positions
            if size not in size_outcomes:
                size_outcomes[size] = []

            size_outcomes[size].append(1.0 if outcome.total_pnl_gbp > 0 else 0.0)

        # Calculate win rate for each size
        size_win_rates = {
            size: np.mean(outcomes) for size, outcomes in size_outcomes.items()
        }

        # Find best size
        best_size = max(size_win_rates.items(), key=lambda x: x[1])

        return best_size

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "portfolio_feature_importance": self.portfolio_feature_importance.copy(),
            "cross_symbol_patterns": {
                sig: {
                    "symbols": list(p.symbols),
                    "avg_win_rate": p.avg_win_rate,
                    "sample_count": p.sample_count,
                    "last_seen": p.last_seen.isoformat(),
                }
                for sig, p in self.cross_symbol_patterns.items()
            },
            "total_portfolio_sessions": self.total_portfolio_sessions,
            "successful_sessions": self.successful_sessions,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        self.portfolio_feature_importance = state.get("portfolio_feature_importance", {})

        # Load cross-symbol patterns
        self.cross_symbol_patterns = {}
        for sig, data in state.get("cross_symbol_patterns", {}).items():
            self.cross_symbol_patterns[sig] = CrossSymbolPattern(
                pattern_signature=sig,
                symbols=set(data["symbols"]),
                avg_win_rate=data["avg_win_rate"],
                sample_count=data["sample_count"],
                last_seen=datetime.fromisoformat(data["last_seen"]),
            )

        self.total_portfolio_sessions = state.get("total_portfolio_sessions", 0)
        self.successful_sessions = state.get("successful_sessions", 0)

        logger.info(
            "portfolio_learner_state_loaded",
            total_sessions=self.total_portfolio_sessions,
            num_patterns=len(self.cross_symbol_patterns),
        )
