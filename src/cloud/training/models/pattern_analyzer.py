"""
Win/Loss Pattern Analyzer

Analyzes historical trades to identify WHY trades won or lost, then generates
actionable avoidance rules and success signatures.

Key Problems Solved:
1. **Repeated Mistakes**: Bot makes same mistake 20 times (e.g., losing RANGE trades when BTC vol > 5%)
2. **Unknown Success Factors**: Bot wins but doesn't know WHY → can't repeat success
3. **Pattern Blindness**: Can't see that "Monday morning + PANIC regime = 80% loss rate"

Example Output:
    FAILURE PATTERN DETECTED:
    "RANGE trades in TREND regime when ADX > 30"
    - 23 trades matching pattern
    - 17% win rate (terrible!)
    - Avg loss: -95 bps
    - Generated Rule: AVOID RANGE when regime=TREND AND adx>30

    SUCCESS SIGNATURE DETECTED:
    "TREND trades with ADX>30 + RSI 50-60 + volume >1.5x"
    - 45 trades matching signature
    - 78% win rate (excellent!)
    - Avg win: +185 bps
    - Recommendation: SEEK these conditions for TREND trades
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class PatternType(Enum):
    """Type of pattern discovered."""

    FAILURE = "failure"  # Pattern that leads to losses
    SUCCESS = "success"  # Pattern that leads to wins
    NEUTRAL = "neutral"  # No clear edge


@dataclass
class TradeRecord:
    """Record of a completed trade for analysis."""

    trade_id: str
    symbol: str
    technique: str  # 'trend', 'range', 'breakout', etc.
    direction: str  # 'buy', 'sell'
    regime: str  # 'trend', 'range', 'panic'
    entry_confidence: float
    pnl_bps: float
    won: bool

    # Entry conditions
    entry_features: Dict[str, float]
    entry_timestamp: float

    # Trade duration
    hold_minutes: int

    # Additional context
    day_of_week: int  # 0=Monday, 6=Sunday
    hour_of_day: int  # 0-23
    volatility: float


@dataclass
class PatternSignature:
    """A discovered pattern (success or failure)."""

    pattern_type: PatternType
    name: str
    description: str
    conditions: Dict[str, Tuple[float, float]]  # feature: (min, max)
    technique: Optional[str]  # Specific to this technique or all
    regime: Optional[str]  # Specific to this regime or all

    # Statistics
    matching_trades: int
    win_rate: float
    avg_pnl_bps: float
    avg_win_bps: float
    avg_loss_bps: float
    confidence: float  # 0-1, how confident in this pattern

    # Actionable recommendation
    recommendation: str  # "AVOID", "SEEK", "NEUTRAL"
    rule: str  # Human-readable rule


@dataclass
class PatternAnalysisResult:
    """Result of pattern analysis."""

    total_trades_analyzed: int
    failure_patterns: List[PatternSignature]
    success_patterns: List[PatternSignature]
    neutral_patterns: List[PatternSignature]
    avoidance_rules: List[str]  # Auto-generated rules to avoid failures
    success_signatures: List[str]  # Signatures to seek out


class WinLossPatternAnalyzer:
    """
    Analyzes trade history to discover patterns in wins and losses.

    Three main functions:
    1. **Failure Pattern Detection**: Identify repeated losing patterns
    2. **Success Signature Extraction**: Find conditions that lead to wins
    3. **Avoidance Rule Generation**: Create actionable rules to prevent losses

    Approach:
    - Clustering: Group similar trades together
    - Statistical Analysis: Identify clusters with extreme win/loss rates
    - Feature Importance: Determine which features define the pattern

    Usage:
        analyzer = WinLossPatternAnalyzer()

        # Add completed trades
        for trade in trade_history:
            analyzer.add_trade(trade)

        # Analyze patterns (run periodically, e.g., every 100 trades)
        result = analyzer.analyze_patterns(min_cluster_size=10)

        # Review failure patterns
        for pattern in result.failure_patterns:
            if pattern.win_rate < 0.30:  # Less than 30% win rate
                logger.warning("failure_pattern_detected",
                              name=pattern.name,
                              win_rate=pattern.win_rate,
                              rule=pattern.rule)

                # Auto-apply avoidance rule
                add_avoidance_rule(pattern.rule)

        # Review success patterns
        for pattern in result.success_patterns:
            if pattern.win_rate > 0.70:  # Over 70% win rate
                logger.info("success_signature_found",
                           name=pattern.name,
                           win_rate=pattern.win_rate,
                           recommendation=pattern.recommendation)
    """

    def __init__(
        self,
        failure_win_rate_threshold: float = 0.35,
        success_win_rate_threshold: float = 0.70,
        min_pattern_size: int = 10,
        min_confidence: float = 0.65,
        max_trades_stored: int = 5000,
    ):
        """
        Initialize pattern analyzer.

        Args:
            failure_win_rate_threshold: Below this = failure pattern
            success_win_rate_threshold: Above this = success pattern
            min_pattern_size: Minimum trades in pattern to be valid
            min_confidence: Minimum confidence to report pattern
            max_trades_stored: Maximum trade history to keep
        """
        self.failure_threshold = failure_win_rate_threshold
        self.success_threshold = success_win_rate_threshold
        self.min_pattern_size = min_pattern_size
        self.min_confidence = min_confidence
        self.max_trades_stored = max_trades_stored

        # Trade history
        self.trades: List[TradeRecord] = []

        # Discovered patterns (cached)
        self.failure_patterns: List[PatternSignature] = []
        self.success_patterns: List[PatternSignature] = []

        logger.info(
            "pattern_analyzer_initialized",
            failure_threshold=failure_win_rate_threshold,
            success_threshold=success_win_rate_threshold,
            min_pattern_size=min_pattern_size,
        )

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a completed trade to history."""
        self.trades.append(trade)

        # Keep only max_trades_stored
        if len(self.trades) > self.max_trades_stored:
            self.trades = self.trades[-self.max_trades_stored:]

        logger.debug(
            "trade_added",
            trade_id=trade.trade_id,
            won=trade.won,
            pnl_bps=trade.pnl_bps,
            total_trades=len(self.trades),
        )

    def analyze_patterns(
        self,
        min_cluster_size: int = 10,
        recent_only: bool = False,
        recent_trades: int = 500,
    ) -> PatternAnalysisResult:
        """
        Analyze trade history to discover patterns.

        Args:
            min_cluster_size: Minimum trades per cluster
            recent_only: Only analyze recent trades
            recent_trades: Number of recent trades if recent_only=True

        Returns:
            PatternAnalysisResult with discovered patterns
        """
        # Select trades to analyze
        if recent_only:
            trades_to_analyze = self.trades[-recent_trades:]
        else:
            trades_to_analyze = self.trades

        if len(trades_to_analyze) < min_cluster_size * 2:
            logger.warning(
                "insufficient_trades_for_analysis",
                total_trades=len(trades_to_analyze),
                min_required=min_cluster_size * 2,
            )
            return PatternAnalysisResult(
                total_trades_analyzed=len(trades_to_analyze),
                failure_patterns=[],
                success_patterns=[],
                neutral_patterns=[],
                avoidance_rules=[],
                success_signatures=[],
            )

        # Extract features for clustering
        feature_matrix, feature_names = self._extract_features_matrix(trades_to_analyze)

        # Normalize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Cluster trades using DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        cluster_labels = clusterer.fit_predict(feature_matrix_scaled)

        # Analyze each cluster
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Remove noise cluster

        failure_patterns = []
        success_patterns = []
        neutral_patterns = []

        for cluster_id in unique_clusters:
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_trades = [trades_to_analyze[i] for i in cluster_indices]

            if len(cluster_trades) < self.min_pattern_size:
                continue

            # Analyze cluster
            pattern = self._analyze_cluster(
                cluster_trades,
                feature_matrix[cluster_indices],
                feature_names,
            )

            if pattern and pattern.confidence >= self.min_confidence:
                if pattern.pattern_type == PatternType.FAILURE:
                    failure_patterns.append(pattern)
                elif pattern.pattern_type == PatternType.SUCCESS:
                    success_patterns.append(pattern)
                else:
                    neutral_patterns.append(pattern)

        # Sort by importance (win rate deviation * confidence * size)
        failure_patterns.sort(
            key=lambda p: (self.failure_threshold - p.win_rate) * p.confidence * p.matching_trades,
            reverse=True,
        )
        success_patterns.sort(
            key=lambda p: (p.win_rate - self.success_threshold) * p.confidence * p.matching_trades,
            reverse=True,
        )

        # Generate avoidance rules and success signatures
        avoidance_rules = [p.rule for p in failure_patterns[:5]]  # Top 5 failure patterns
        success_signatures = [p.rule for p in success_patterns[:5]]  # Top 5 success patterns

        # Cache patterns
        self.failure_patterns = failure_patterns
        self.success_patterns = success_patterns

        logger.info(
            "pattern_analysis_complete",
            total_trades=len(trades_to_analyze),
            failure_patterns=len(failure_patterns),
            success_patterns=len(success_patterns),
            neutral_patterns=len(neutral_patterns),
        )

        return PatternAnalysisResult(
            total_trades_analyzed=len(trades_to_analyze),
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            neutral_patterns=neutral_patterns,
            avoidance_rules=avoidance_rules,
            success_signatures=success_signatures,
        )

    def check_trade_against_patterns(
        self,
        proposed_trade: Dict[str, any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if proposed trade matches any failure patterns.

        Args:
            proposed_trade: Dict with trade attributes
                {
                    'technique': 'range',
                    'regime': 'trend',
                    'entry_features': {'adx': 35.0, 'rsi': 45.0, ...},
                    ...
                }

        Returns:
            (should_avoid, reason) tuple
                should_avoid: True if matches failure pattern
                reason: Explanation if should avoid
        """
        if not self.failure_patterns:
            return False, None

        # Check against each failure pattern
        for pattern in self.failure_patterns:
            if self._trade_matches_pattern(proposed_trade, pattern):
                reason = (
                    f"AVOID: Matches failure pattern '{pattern.name}'. "
                    f"Historical win rate: {pattern.win_rate:.0%} "
                    f"({pattern.matching_trades} trades). "
                    f"Rule: {pattern.rule}"
                )
                logger.warning(
                    "trade_matches_failure_pattern",
                    pattern=pattern.name,
                    win_rate=pattern.win_rate,
                )
                return True, reason

        return False, None

    def get_pattern_statistics(self) -> Dict[str, any]:
        """Get summary statistics of discovered patterns."""
        return {
            'total_trades': len(self.trades),
            'failure_patterns_count': len(self.failure_patterns),
            'success_patterns_count': len(self.success_patterns),
            'worst_failure_pattern': self.failure_patterns[0] if self.failure_patterns else None,
            'best_success_pattern': self.success_patterns[0] if self.success_patterns else None,
        }

    def _extract_features_matrix(
        self,
        trades: List[TradeRecord],
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract feature matrix for clustering."""
        # Feature categories to extract
        feature_names = []
        feature_vectors = []

        # Get common entry features across all trades
        all_feature_keys = set()
        for trade in trades:
            all_feature_keys.update(trade.entry_features.keys())

        feature_names = sorted(list(all_feature_keys))

        # Build feature matrix
        for trade in trades:
            features = []

            # Entry features
            for key in feature_names:
                features.append(trade.entry_features.get(key, 0.0))

            # Technique one-hot (simple: use hash)
            features.append(hash(trade.technique) % 10)

            # Regime one-hot
            features.append(hash(trade.regime) % 10)

            # Confidence
            features.append(trade.entry_confidence)

            # Volatility
            features.append(trade.volatility)

            # Day of week
            features.append(trade.day_of_week)

            # Hour of day
            features.append(trade.hour_of_day / 24.0)  # Normalize

            feature_vectors.append(features)

        # Add meta features to names
        feature_names.extend([
            'technique_hash',
            'regime_hash',
            'entry_confidence',
            'volatility',
            'day_of_week',
            'hour_of_day_normalized',
        ])

        return np.array(feature_vectors), feature_names

    def _analyze_cluster(
        self,
        cluster_trades: List[TradeRecord],
        cluster_features: np.ndarray,
        feature_names: List[str],
    ) -> Optional[PatternSignature]:
        """Analyze a single cluster to extract pattern."""
        if len(cluster_trades) < self.min_pattern_size:
            return None

        # Calculate cluster statistics
        wins = sum(1 for t in cluster_trades if t.won)
        losses = len(cluster_trades) - wins
        win_rate = wins / len(cluster_trades)

        pnls = [t.pnl_bps for t in cluster_trades]
        avg_pnl = np.mean(pnls)

        winners = [t.pnl_bps for t in cluster_trades if t.won]
        losers = [t.pnl_bps for t in cluster_trades if not t.won]

        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = np.mean(losers) if losers else 0.0

        # Determine pattern type
        if win_rate < self.failure_threshold:
            pattern_type = PatternType.FAILURE
            recommendation = "AVOID"
        elif win_rate > self.success_threshold:
            pattern_type = PatternType.SUCCESS
            recommendation = "SEEK"
        else:
            pattern_type = PatternType.NEUTRAL
            recommendation = "NEUTRAL"

        # Extract defining conditions (features that are consistent across cluster)
        conditions = {}
        for i, feature_name in enumerate(feature_names):
            feature_values = cluster_features[:, i]
            feature_min = np.min(feature_values)
            feature_max = np.max(feature_values)
            feature_range = feature_max - feature_min

            # If feature is fairly consistent (low range), include it
            if feature_range < 0.3 * (np.max(cluster_features[:, i]) + 0.01):  # Avoid div by zero
                conditions[feature_name] = (feature_min, feature_max)

        # Identify common technique and regime
        techniques = [t.technique for t in cluster_trades]
        regimes = [t.regime for t in cluster_trades]

        most_common_technique = max(set(techniques), key=techniques.count) if techniques else None
        most_common_regime = max(set(regimes), key=regimes.count) if regimes else None

        technique_purity = techniques.count(most_common_technique) / len(techniques) if most_common_technique else 0
        regime_purity = regimes.count(most_common_regime) / len(regimes) if most_common_regime else 0

        # Use technique/regime if dominant in cluster
        pattern_technique = most_common_technique if technique_purity > 0.7 else None
        pattern_regime = most_common_regime if regime_purity > 0.7 else None

        # Generate name and description
        name = self._generate_pattern_name(
            pattern_type,
            pattern_technique,
            pattern_regime,
            conditions,
        )

        description = self._generate_pattern_description(
            pattern_type,
            cluster_trades,
            win_rate,
            avg_pnl,
        )

        # Generate actionable rule
        rule = self._generate_pattern_rule(
            pattern_type,
            pattern_technique,
            pattern_regime,
            conditions,
            win_rate,
        )

        # Calculate confidence based on sample size and consistency
        confidence = min(1.0, len(cluster_trades) / 50.0)  # Max confidence at 50+ trades

        return PatternSignature(
            pattern_type=pattern_type,
            name=name,
            description=description,
            conditions=conditions,
            technique=pattern_technique,
            regime=pattern_regime,
            matching_trades=len(cluster_trades),
            win_rate=win_rate,
            avg_pnl_bps=avg_pnl,
            avg_win_bps=avg_win,
            avg_loss_bps=avg_loss,
            confidence=confidence,
            recommendation=recommendation,
            rule=rule,
        )

    def _generate_pattern_name(
        self,
        pattern_type: PatternType,
        technique: Optional[str],
        regime: Optional[str],
        conditions: Dict[str, Tuple[float, float]],
    ) -> str:
        """Generate human-readable pattern name."""
        parts = []

        if technique:
            parts.append(technique.upper())

        if regime:
            parts.append(f"in {regime.upper()} regime")

        # Add key conditions (top 2 most restrictive)
        sorted_conditions = sorted(
            conditions.items(),
            key=lambda x: x[1][1] - x[1][0],  # Sort by range (smaller = more specific)
        )[:2]

        for feature, (min_val, max_val) in sorted_conditions:
            if 'hash' not in feature:  # Skip hash features
                parts.append(f"{feature}={min_val:.1f}-{max_val:.1f}")

        name = " ".join(parts) if parts else "General pattern"

        prefix = "❌ FAILURE:" if pattern_type == PatternType.FAILURE else "✅ SUCCESS:" if pattern_type == PatternType.SUCCESS else "⚪ NEUTRAL:"

        return f"{prefix} {name}"

    def _generate_pattern_description(
        self,
        pattern_type: PatternType,
        trades: List[TradeRecord],
        win_rate: float,
        avg_pnl: float,
    ) -> str:
        """Generate pattern description."""
        return (
            f"{len(trades)} trades with {win_rate:.0%} win rate, "
            f"avg P&L: {avg_pnl:+.0f} bps"
        )

    def _generate_pattern_rule(
        self,
        pattern_type: PatternType,
        technique: Optional[str],
        regime: Optional[str],
        conditions: Dict[str, Tuple[float, float]],
        win_rate: float,
    ) -> str:
        """Generate actionable rule."""
        if pattern_type == PatternType.FAILURE:
            prefix = "AVOID"
        elif pattern_type == PatternType.SUCCESS:
            prefix = "SEEK"
        else:
            prefix = "MONITOR"

        parts = []

        if technique:
            parts.append(f"{technique.upper()} trades")
        else:
            parts.append("trades")

        conditions_str = []
        for feature, (min_val, max_val) in list(conditions.items())[:3]:
            if 'hash' not in feature:
                conditions_str.append(f"{feature}={min_val:.1f}-{max_val:.1f}")

        if regime:
            conditions_str.insert(0, f"regime={regime.upper()}")

        if conditions_str:
            parts.append(f"WHEN {' AND '.join(conditions_str)}")

        rule = f"{prefix}: {' '.join(parts)} (historical win rate: {win_rate:.0%})"

        return rule

    def _trade_matches_pattern(
        self,
        trade: Dict[str, any],
        pattern: PatternSignature,
    ) -> bool:
        """Check if trade matches pattern conditions."""
        # Check technique
        if pattern.technique and trade.get('technique') != pattern.technique:
            return False

        # Check regime
        if pattern.regime and trade.get('regime') != pattern.regime:
            return False

        # Check conditions
        entry_features = trade.get('entry_features', {})

        for feature, (min_val, max_val) in pattern.conditions.items():
            if 'hash' in feature:  # Skip hash features
                continue

            feature_value = entry_features.get(feature)
            if feature_value is None:
                continue

            # Check if within range (with some tolerance)
            tolerance = (max_val - min_val) * 0.2  # 20% tolerance
            if not (min_val - tolerance <= feature_value <= max_val + tolerance):
                return False

        return True
