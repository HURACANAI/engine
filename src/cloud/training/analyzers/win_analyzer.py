"""Deep analysis of winning trades to understand what drives success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from scipy import stats
import structlog

from ..memory.store import MemoryStore

logger = structlog.get_logger(__name__)


@dataclass
class WinInsights:
    """Insights extracted from a winning trade."""
    trade_id: int
    top_features: Dict[str, float]  # Feature name -> importance score
    pattern_frequency: int
    historical_win_rate: float
    skill_vs_luck_score: float  # 0=pure luck, 1=pure skill
    entry_quality: str  # 'EXCELLENT', 'GOOD', 'FAIR'
    exit_quality: str  # 'OPTIMAL', 'EARLY', 'ACCEPTABLE'
    pattern_strength: float
    confidence_for_future: float
    insights_text: str


class WinAnalyzer:
    """Analyzes winning trades to extract actionable insights."""

    def __init__(self, dsn: str, memory_store: MemoryStore):
        self._dsn = dsn
        self._memory = memory_store
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def analyze_win(
        self,
        trade_id: int,
        entry_features: Dict[str, Any],
        entry_embedding: np.ndarray,
        profit_gbp: float,
        profit_bps: float,
        missed_profit_gbp: float,
        symbol: str,
        market_regime: str,
    ) -> WinInsights:
        """
        Perform deep analysis on a winning trade.

        Args:
            trade_id: Trade identifier
            entry_features: All features at entry time
            entry_embedding: Vector embedding of features
            profit_gbp: Realized profit
            profit_bps: Profit in basis points
            missed_profit_gbp: How much more we could have made
            symbol: Trading symbol
            market_regime: Market regime at entry

        Returns:
            WinInsights with actionable learnings
        """
        logger.info("analyzing_win", trade_id=trade_id, profit_gbp=profit_gbp)

        # 1. Find similar historical patterns
        similar_patterns = self._memory.find_similar_patterns(
            embedding=entry_embedding,
            symbol=symbol,
            market_regime=market_regime,
            top_k=30,
            min_similarity=0.6,
        )

        pattern_stats = self._memory.get_pattern_stats(similar_patterns)

        # 2. Feature importance analysis
        top_features = self._identify_contributing_features(entry_features)

        # 3. Skill vs luck assessment
        skill_score = self._assess_skill_vs_luck(
            profit_bps=profit_bps,
            pattern_stats=pattern_stats,
            symbol=symbol,
        )

        # 4. Entry quality
        entry_quality = self._assess_entry_quality(
            entry_features=entry_features,
            pattern_stats=pattern_stats,
        )

        # 5. Exit quality
        exit_quality = self._assess_exit_quality(
            profit_gbp=profit_gbp,
            missed_profit_gbp=missed_profit_gbp,
        )

        # 6. Pattern strength
        pattern_strength = self._calculate_pattern_strength(
            pattern_stats=pattern_stats,
            profit_bps=profit_bps,
        )

        # 7. Future confidence
        confidence = self._calculate_future_confidence(
            pattern_stats=pattern_stats,
            skill_score=skill_score,
            pattern_strength=pattern_strength,
        )

        # 8. Generate insights text
        insights_text = self._generate_insights_text(
            profit_gbp=profit_gbp,
            pattern_stats=pattern_stats,
            top_features=top_features,
            exit_quality=exit_quality,
            skill_score=skill_score,
        )

        insights = WinInsights(
            trade_id=trade_id,
            top_features=top_features,
            pattern_frequency=pattern_stats.total_occurrences,
            historical_win_rate=pattern_stats.win_rate,
            skill_vs_luck_score=skill_score,
            entry_quality=entry_quality,
            exit_quality=exit_quality,
            pattern_strength=pattern_strength,
            confidence_for_future=confidence,
            insights_text=insights_text,
        )

        # Store in database
        self._store_analysis(insights)

        logger.info(
            "win_analyzed",
            trade_id=trade_id,
            skill_score=skill_score,
            confidence=confidence,
        )

        return insights

    def _identify_contributing_features(
        self,
        entry_features: Dict[str, Any],
        top_n: int = 10,
    ) -> Dict[str, float]:
        """
        Identify which features likely contributed most to the win.

        Uses simple heuristic: features with extreme values are likely important.
        """
        feature_scores = {}

        for name, value in entry_features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Score based on absolute deviation from typical values
                # In production, use SHAP values from trained model
                score = abs(value)
                feature_scores[name] = score

        # Return top N
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])

    def _assess_skill_vs_luck(
        self,
        profit_bps: float,
        pattern_stats: Any,
        symbol: str,
    ) -> float:
        """
        Determine if win was due to skill or luck.

        Returns value between 0 (pure luck) and 1 (pure skill).
        """
        # Factor 1: Pattern has strong historical performance
        historical_factor = pattern_stats.win_rate if pattern_stats.total_occurrences >= 10 else 0.3

        # Factor 2: Profit magnitude vs expected
        if pattern_stats.avg_profit_gbp > 0:
            profit_factor = min(1.0, profit_bps / (pattern_stats.avg_profit_gbp * 100))
        else:
            profit_factor = 0.5

        # Factor 3: Statistical significance
        if pattern_stats.total_occurrences >= 30:
            # Enough samples to be confident
            significance_factor = 0.9
        elif pattern_stats.total_occurrences >= 10:
            significance_factor = 0.6
        else:
            significance_factor = 0.3

        # Weighted combination
        skill_score = (
            0.5 * historical_factor +
            0.2 * profit_factor +
            0.3 * significance_factor
        )

        return min(1.0, max(0.0, skill_score))

    def _assess_entry_quality(
        self,
        entry_features: Dict[str, Any],
        pattern_stats: Any,
    ) -> str:
        """Assess quality of entry timing."""
        # Check if pattern is strong
        if pattern_stats.win_rate > 0.65 and pattern_stats.total_occurrences >= 20:
            return "EXCELLENT"
        elif pattern_stats.win_rate > 0.55 and pattern_stats.total_occurrences >= 10:
            return "GOOD"
        else:
            return "FAIR"

    def _assess_exit_quality(
        self,
        profit_gbp: float,
        missed_profit_gbp: float,
    ) -> str:
        """Assess quality of exit timing."""
        if missed_profit_gbp < 0.25:  # Missed less than £0.25
            return "OPTIMAL"
        elif missed_profit_gbp < 1.0:  # Missed less than £1
            return "ACCEPTABLE"
        else:
            return "EARLY"  # Left significant money on table

    def _calculate_pattern_strength(
        self,
        pattern_stats: Any,
        profit_bps: float,
    ) -> float:
        """Calculate how strong/reliable this pattern is."""
        if pattern_stats.total_occurrences == 0:
            return 0.3

        # Combine win rate, sample size, and sharpe
        win_rate_component = pattern_stats.win_rate
        sample_size_component = min(1.0, pattern_stats.total_occurrences / 50.0)
        sharpe_component = min(1.0, max(0.0, pattern_stats.sharpe_ratio / 3.0))

        strength = (
            0.5 * win_rate_component +
            0.3 * sample_size_component +
            0.2 * sharpe_component
        )

        return strength

    def _calculate_future_confidence(
        self,
        pattern_stats: Any,
        skill_score: float,
        pattern_strength: float,
    ) -> float:
        """Calculate confidence for taking similar trades in future."""
        # High confidence if: strong pattern, skill-based, good stats
        confidence = (
            0.4 * pattern_strength +
            0.3 * skill_score +
            0.3 * pattern_stats.reliability_score
        )

        return min(1.0, max(0.0, confidence))

    def _generate_insights_text(
        self,
        profit_gbp: float,
        pattern_stats: Any,
        top_features: Dict[str, float],
        exit_quality: str,
        skill_score: float,
    ) -> str:
        """Generate human-readable insights."""
        lines = [
            f"Won £{profit_gbp:.2f} with {int(pattern_stats.win_rate * 100)}% historical win rate.",
        ]

        if skill_score > 0.7:
            lines.append("This appears to be a skill-based edge, not luck.")
        elif skill_score < 0.4:
            lines.append("This win may have been partially luck - be cautious with similar setups.")

        if exit_quality == "EARLY":
            lines.append("Exited too early - could have held for more profit.")
        elif exit_quality == "OPTIMAL":
            lines.append("Exit timing was excellent.")

        if top_features:
            top_3 = list(top_features.keys())[:3]
            lines.append(f"Key features: {', '.join(top_3)}.")

        if pattern_stats.total_occurrences >= 30:
            lines.append(f"Strong pattern with {pattern_stats.total_occurrences} historical occurrences.")

        return " ".join(lines)

    def _store_analysis(self, insights: WinInsights) -> None:
        """Store analysis in database."""
        self.connect()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO win_analysis (
                    trade_id, top_contributing_features, pattern_frequency,
                    historical_win_rate, statistical_significance, skill_vs_luck_score,
                    entry_quality, exit_quality, pattern_strength, confidence_for_future,
                    insights
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    insights.trade_id,
                    Json(insights.top_features),
                    insights.pattern_frequency,
                    insights.historical_win_rate,
                    0.95,  # Placeholder for statistical significance
                    insights.skill_vs_luck_score,
                    insights.entry_quality,
                    insights.exit_quality,
                    insights.pattern_strength,
                    insights.confidence_for_future,
                    insights.insights_text,
                ),
            )
            self._conn.commit()
