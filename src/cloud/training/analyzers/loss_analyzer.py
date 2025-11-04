"""Root cause analysis of losing trades to prevent future failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import structlog

from ..memory.store import MemoryStore

logger = structlog.get_logger(__name__)


@dataclass
class LossInsights:
    """Insights from analyzing a losing trade."""
    trade_id: int
    primary_failure_reason: str
    misleading_features: Dict[str, float]
    regime_changed: bool
    stop_too_tight: bool
    stop_too_wide: bool
    adverse_selection: bool
    insufficient_confirmation: bool
    news_event_impact: bool
    loss_severity: str  # 'MINOR', 'MODERATE', 'SEVERE'
    preventable: bool
    corrective_action: str
    pattern_to_avoid: Dict[str, Any]
    confidence_penalty: float  # How much to reduce confidence for similar setups
    insights_text: str


class LossAnalyzer:
    """Analyzes losing trades to understand root causes and prevent recurrence."""

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

    def analyze_loss(
        self,
        trade_id: int,
        entry_features: Dict[str, Any],
        entry_embedding: np.ndarray,
        loss_gbp: float,
        loss_bps: float,
        entry_spread_bps: float,
        exit_spread_bps: float,
        entry_volatility_bps: float,
        exit_volatility_bps: float,
        symbol: str,
        market_regime_entry: str,
        market_regime_exit: str,
        hold_duration_minutes: int,
        stop_loss_bps: float,
    ) -> LossInsights:
        """
        Perform root cause analysis on a losing trade.

        Args:
            trade_id: Trade identifier
            entry_features: Features at entry
            entry_embedding: Vector embedding
            loss_gbp: Loss in GBP
            loss_bps: Loss in basis points
            entry_spread_bps: Spread at entry
            exit_spread_bps: Spread at exit
            entry_volatility_bps: Volatility at entry
            exit_volatility_bps: Volatility at exit
            symbol: Trading symbol
            market_regime_entry: Regime at entry
            market_regime_exit: Regime at exit
            hold_duration_minutes: How long held
            stop_loss_bps: Stop loss threshold used

        Returns:
            LossInsights with root cause analysis
        """
        logger.info("analyzing_loss", trade_id=trade_id, loss_gbp=loss_gbp)

        # 1. Find similar historical patterns
        similar_patterns = self._memory.find_similar_patterns(
            embedding=entry_embedding,
            symbol=symbol,
            market_regime=market_regime_entry,
            top_k=30,
            min_similarity=0.6,
        )

        pattern_stats = self._memory.get_pattern_stats(similar_patterns)

        # 2. Identify misleading features
        misleading_features = self._identify_misleading_features(
            entry_features=entry_features,
            pattern_stats=pattern_stats,
        )

        # 3. Check if market regime changed mid-trade
        regime_changed = market_regime_entry != market_regime_exit

        # 4. Assess stop loss sizing
        stop_too_tight = self._is_stop_too_tight(
            stop_loss_bps=stop_loss_bps,
            volatility_bps=entry_volatility_bps,
            loss_bps=abs(loss_bps),
        )

        stop_too_wide = self._is_stop_too_wide(
            stop_loss_bps=stop_loss_bps,
            loss_bps=abs(loss_bps),
        )

        # 5. Check for adverse selection (bad fill price)
        adverse_selection = self._detect_adverse_selection(
            entry_spread_bps=entry_spread_bps,
            loss_bps=abs(loss_bps),
        )

        # 6. Check if had insufficient confirmation
        insufficient_confirmation = self._check_insufficient_confirmation(
            entry_features=entry_features,
            pattern_stats=pattern_stats,
        )

        # 7. News event detection (simplified - would integrate news API)
        news_event_impact = self._detect_news_event(
            volatility_change=exit_volatility_bps - entry_volatility_bps,
        )

        # 8. Determine primary failure reason
        primary_reason = self._determine_primary_reason(
            regime_changed=regime_changed,
            stop_too_tight=stop_too_tight,
            adverse_selection=adverse_selection,
            insufficient_confirmation=insufficient_confirmation,
            news_event_impact=news_event_impact,
            pattern_stats=pattern_stats,
        )

        # 9. Assess severity
        loss_severity = self._assess_loss_severity(loss_gbp)

        # 10. Determine if preventable
        preventable = self._is_preventable(
            primary_reason=primary_reason,
            insufficient_confirmation=insufficient_confirmation,
            pattern_stats=pattern_stats,
        )

        # 11. Generate corrective action
        corrective_action = self._generate_corrective_action(
            primary_reason=primary_reason,
            stop_too_tight=stop_too_tight,
            insufficient_confirmation=insufficient_confirmation,
        )

        # 12. Pattern to avoid
        pattern_to_avoid = self._build_pattern_to_avoid(
            entry_features=entry_features,
            market_regime=market_regime_entry,
        )

        # 13. Calculate confidence penalty
        confidence_penalty = self._calculate_confidence_penalty(
            loss_severity=loss_severity,
            preventable=preventable,
            pattern_stats=pattern_stats,
        )

        # 14. Generate insights text
        insights_text = self._generate_insights_text(
            loss_gbp=loss_gbp,
            primary_reason=primary_reason,
            preventable=preventable,
            corrective_action=corrective_action,
        )

        insights = LossInsights(
            trade_id=trade_id,
            primary_failure_reason=primary_reason,
            misleading_features=misleading_features,
            regime_changed=regime_changed,
            stop_too_tight=stop_too_tight,
            stop_too_wide=stop_too_wide,
            adverse_selection=adverse_selection,
            insufficient_confirmation=insufficient_confirmation,
            news_event_impact=news_event_impact,
            loss_severity=loss_severity,
            preventable=preventable,
            corrective_action=corrective_action,
            pattern_to_avoid=pattern_to_avoid,
            confidence_penalty=confidence_penalty,
            insights_text=insights_text,
        )

        # Store in database
        self._store_analysis(insights)

        logger.info(
            "loss_analyzed",
            trade_id=trade_id,
            primary_reason=primary_reason,
            preventable=preventable,
        )

        return insights

    def _identify_misleading_features(
        self,
        entry_features: Dict[str, Any],
        pattern_stats: Any,
        top_n: int = 5,
    ) -> Dict[str, float]:
        """Identify features that gave false signal."""
        # Simplified: in production, compare to winning trades
        # Return features that had extreme values
        misleading = {}

        for name, value in entry_features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Features with extreme values that didn't work out
                if abs(value) > 2.0:  # Simplified threshold
                    misleading[name] = value

        sorted_misleading = sorted(misleading.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_misleading[:top_n])

    def _is_stop_too_tight(
        self,
        stop_loss_bps: float,
        volatility_bps: float,
        loss_bps: float,
    ) -> bool:
        """Check if stop was too tight relative to volatility."""
        # Stop should be at least 2x volatility
        recommended_stop = volatility_bps * 2.0
        return stop_loss_bps < recommended_stop and loss_bps >= stop_loss_bps * 0.9

    def _is_stop_too_wide(
        self,
        stop_loss_bps: float,
        loss_bps: float,
    ) -> bool:
        """Check if stop was unnecessarily wide."""
        # Loss should be close to stop if stop was appropriate
        return loss_bps < stop_loss_bps * 0.5

    def _detect_adverse_selection(
        self,
        entry_spread_bps: float,
        loss_bps: float,
    ) -> bool:
        """Check if we got filled at a bad price."""
        # If loss is mostly just the spread, we got adversely selected
        return entry_spread_bps > 5.0 and loss_bps < entry_spread_bps * 1.5

    def _check_insufficient_confirmation(
        self,
        entry_features: Dict[str, Any],
        pattern_stats: Any,
    ) -> bool:
        """Check if we entered without enough confirmation."""
        # Pattern has poor historical performance
        if pattern_stats.total_occurrences > 10 and pattern_stats.win_rate < 0.45:
            return True

        # Not enough historical data
        if pattern_stats.total_occurrences < 5:
            return True

        return False

    def _detect_news_event(self, volatility_change: float) -> bool:
        """Detect if a news event impacted the trade."""
        # Sudden volatility spike suggests news
        return volatility_change > 20.0  # Volatility increased by 20+ bps

    def _determine_primary_reason(
        self,
        regime_changed: bool,
        stop_too_tight: bool,
        adverse_selection: bool,
        insufficient_confirmation: bool,
        news_event_impact: bool,
        pattern_stats: Any,
    ) -> str:
        """Determine the main reason for the loss."""
        if news_event_impact:
            return "NEWS_EVENT"
        if regime_changed:
            return "REGIME_CHANGE"
        if adverse_selection:
            return "ADVERSE_SELECTION"
        if insufficient_confirmation:
            return "WEAK_PATTERN"
        if stop_too_tight:
            return "STOP_TOO_TIGHT"
        if pattern_stats.win_rate < 0.40:
            return "POOR_HISTORICAL_PATTERN"

        return "NORMAL_LOSS"

    def _assess_loss_severity(self, loss_gbp: float) -> str:
        """Categorize loss severity."""
        if abs(loss_gbp) < 1.0:
            return "MINOR"
        elif abs(loss_gbp) < 3.0:
            return "MODERATE"
        else:
            return "SEVERE"

    def _is_preventable(
        self,
        primary_reason: str,
        insufficient_confirmation: bool,
        pattern_stats: Any,
    ) -> bool:
        """Determine if loss was preventable."""
        preventable_reasons = {
            "WEAK_PATTERN",
            "POOR_HISTORICAL_PATTERN",
            "STOP_TOO_TIGHT",
            "ADVERSE_SELECTION",
        }

        if primary_reason in preventable_reasons:
            return True

        if insufficient_confirmation and pattern_stats.win_rate < 0.50:
            return True

        return False

    def _generate_corrective_action(
        self,
        primary_reason: str,
        stop_too_tight: bool,
        insufficient_confirmation: bool,
    ) -> str:
        """Generate specific corrective action."""
        actions = {
            "WEAK_PATTERN": "Require stronger historical win rate (>55%) before entering similar setups.",
            "STOP_TOO_TIGHT": "Widen stop loss to 2x volatility minimum.",
            "ADVERSE_SELECTION": "Use limit orders only; avoid market orders in wide spreads.",
            "REGIME_CHANGE": "Exit immediately when regime shifts mid-trade.",
            "NEWS_EVENT": "Implement news blackout periods; avoid trading 30min pre/post major events.",
            "POOR_HISTORICAL_PATTERN": "Blacklist this pattern - historical win rate too low.",
            "NORMAL_LOSS": "Acceptable loss - part of normal variance.",
        }

        return actions.get(primary_reason, "Continue monitoring pattern performance.")

    def _build_pattern_to_avoid(
        self,
        entry_features: Dict[str, Any],
        market_regime: str,
    ) -> Dict[str, Any]:
        """Build pattern signature to avoid in future."""
        # Simplified: store key features and regime
        return {
            "market_regime": market_regime,
            "feature_snapshot": {k: v for k, v in list(entry_features.items())[:10]},
        }

    def _calculate_confidence_penalty(
        self,
        loss_severity: str,
        preventable: bool,
        pattern_stats: Any,
    ) -> float:
        """Calculate how much to penalize confidence for similar patterns."""
        base_penalty = {
            "MINOR": 0.05,
            "MODERATE": 0.10,
            "SEVERE": 0.20,
        }[loss_severity]

        if preventable:
            base_penalty *= 1.5

        # Larger penalty if pattern already has poor track record
        if pattern_stats.win_rate < 0.45:
            base_penalty *= 1.3

        return min(0.5, base_penalty)  # Cap at 50% penalty

    def _generate_insights_text(
        self,
        loss_gbp: float,
        primary_reason: str,
        preventable: bool,
        corrective_action: str,
    ) -> str:
        """Generate human-readable insights."""
        lines = [
            f"Lost £{abs(loss_gbp):.2f}. Primary reason: {primary_reason.replace('_', ' ').lower()}.",
        ]

        if preventable:
            lines.append("This loss was PREVENTABLE.")
        else:
            lines.append("This was an acceptable loss within normal variance.")

        lines.append(f"Action: {corrective_action}")

        return " ".join(lines)

    def _store_analysis(self, insights: LossInsights) -> None:
        """Store analysis in database."""
        self.connect()
        conn = self._conn
        if conn is None:
            raise RuntimeError("Database connection is not available")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO loss_analysis (
                    trade_id, primary_failure_reason, misleading_features,
                    regime_changed, stop_too_tight, stop_too_wide,
                    adverse_selection, insufficient_confirmation, news_event_impact,
                    loss_severity, preventable, corrective_action,
                    pattern_to_avoid, confidence_penalty, insights
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    insights.trade_id,
                    insights.primary_failure_reason,
                    Json(insights.misleading_features),
                    insights.regime_changed,
                    insights.stop_too_tight,
                    insights.stop_too_wide,
                    insights.adverse_selection,
                    insights.insufficient_confirmation,
                    insights.news_event_impact,
                    insights.loss_severity,
                    insights.preventable,
                    insights.corrective_action,
                    Json(insights.pattern_to_avoid),
                    insights.confidence_penalty,
                    insights.insights_text,
                ),
            )
            conn.commit()
