"""
Multi-Timeframe Analysis System

Analyzes trading signals across multiple timeframes to ensure confluence
before executing trades. This prevents false signals from single timeframe noise.

Key Features:
- Analyzes 3 timeframes simultaneously (5m, 15m, 1h by default)
- Calculates confluence score (agreement between timeframes)
- Only allows trades when 2+ timeframes agree
- Prevents trading on conflicting signals

Example:
    5m says BUY, 15m says BUY, 1h says SELL
    → Confluence score = 0.67 (2/3 agree)
    → If min_confluence = 0.67, ALLOW trade

    5m says BUY, 15m says SELL, 1h says SELL
    → Confluence score = 0.33 (1/3 agree)
    → BLOCK trade (conflicting signals)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: str  # '5m', '15m', '1h', etc.
    direction: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    trend_strength: float  # -1 to 1
    momentum: float  # -1 to 1


@dataclass
class ConfluenceResult:
    """Result of multi-timeframe confluence analysis."""

    confluence_score: float  # 0-1, higher = more agreement
    agreed_direction: str  # 'buy', 'sell', 'hold'
    num_agreeing: int  # Number of timeframes in agreement
    total_timeframes: int  # Total timeframes analyzed
    conflicting: bool  # True if signals conflict
    reasoning: str  # Human-readable explanation
    timeframe_breakdown: Dict[str, str]  # {timeframe: direction}


class MultiTimeframeAnalyzer:
    """
    Analyzes trading signals across multiple timeframes.

    Philosophy:
    - Single timeframe = noise + signal
    - Multiple timeframes = signal confirmation
    - Conflict = uncertainty = don't trade

    Usage:
        analyzer = MultiTimeframeAnalyzer(timeframes=['5m', '15m', '1h'])
        result = analyzer.analyze_confluence(features_dict, regime)

        if result.confluence_score >= 0.67:
            # 2+ timeframes agree, safe to trade
            trade(result.agreed_direction)
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        min_confluence: float = 0.67,
        trend_threshold: float = 0.3,
        momentum_threshold: float = 0.2,
    ):
        """
        Initialize multi-timeframe analyzer.

        Args:
            timeframes: List of timeframe strings (e.g., ['5m', '15m', '1h'])
            min_confluence: Minimum agreement score to allow trade (0-1)
            trend_threshold: Minimum trend strength to consider directional
            momentum_threshold: Minimum momentum to consider directional
        """
        self.timeframes = timeframes or ['5m', '15m', '1h']
        self.min_confluence = min_confluence
        self.trend_threshold = trend_threshold
        self.momentum_threshold = momentum_threshold

        logger.info(
            "multi_timeframe_analyzer_initialized",
            timeframes=self.timeframes,
            min_confluence=min_confluence,
        )

    def analyze_confluence(
        self,
        features_dict: Dict[str, Dict[str, float]],
        regime: str,
    ) -> ConfluenceResult:
        """
        Analyze confluence across multiple timeframes.

        Args:
            features_dict: {timeframe: features} dictionary
                Example: {
                    '5m': {'trend_strength': 0.7, 'momentum': 0.6, ...},
                    '15m': {'trend_strength': 0.8, 'momentum': 0.5, ...},
                    '1h': {'trend_strength': 0.6, 'momentum': 0.4, ...},
                }
            regime: Current market regime

        Returns:
            ConfluenceResult with agreement analysis
        """
        # Extract signals from each timeframe
        timeframe_signals = []
        timeframe_breakdown = {}

        for tf in self.timeframes:
            if tf not in features_dict:
                logger.warning("missing_timeframe_data", timeframe=tf)
                continue

            signal = self._extract_timeframe_signal(
                timeframe=tf,
                features=features_dict[tf],
                regime=regime,
            )

            timeframe_signals.append(signal)
            timeframe_breakdown[tf] = signal.direction

        if not timeframe_signals:
            logger.error("no_timeframe_signals", features_dict=features_dict)
            return self._create_empty_result("No timeframe data available")

        # Calculate confluence
        confluence_score, agreed_direction, num_agreeing = self._calculate_confluence(
            timeframe_signals
        )

        # Determine if conflicting
        conflicting = self._has_conflicting_signals(timeframe_signals)

        # Build reasoning
        reasoning = self._build_reasoning(
            confluence_score=confluence_score,
            agreed_direction=agreed_direction,
            num_agreeing=num_agreeing,
            total=len(timeframe_signals),
            conflicting=conflicting,
        )

        result = ConfluenceResult(
            confluence_score=confluence_score,
            agreed_direction=agreed_direction,
            num_agreeing=num_agreeing,
            total_timeframes=len(timeframe_signals),
            conflicting=conflicting,
            reasoning=reasoning,
            timeframe_breakdown=timeframe_breakdown,
        )

        logger.debug(
            "confluence_analyzed",
            score=confluence_score,
            direction=agreed_direction,
            agreeing=num_agreeing,
            total=len(timeframe_signals),
        )

        return result

    def _extract_timeframe_signal(
        self,
        timeframe: str,
        features: Dict[str, float],
        regime: str,
    ) -> TimeframeSignal:
        """Extract signal from single timeframe features."""
        # Extract key features
        trend_strength = features.get('trend_strength', 0.0)
        momentum = features.get('momentum_slope', 0.0)
        ema_slope = features.get('ema_slope', 0.0)
        adx = features.get('adx', 0.0)

        # Normalize confidence (0-1)
        confidence = min(abs(trend_strength) + (adx / 50.0), 1.0)

        # Determine direction
        if trend_strength > self.trend_threshold and momentum > self.momentum_threshold:
            direction = 'buy'
        elif trend_strength < -self.trend_threshold and momentum < -self.momentum_threshold:
            direction = 'sell'
        else:
            direction = 'hold'

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            trend_strength=trend_strength,
            momentum=momentum,
        )

    def _calculate_confluence(
        self,
        signals: List[TimeframeSignal],
    ) -> Tuple[float, str, int]:
        """
        Calculate confluence score and agreed direction.

        Returns:
            (confluence_score, agreed_direction, num_agreeing)
        """
        if not signals:
            return 0.0, 'hold', 0

        # Count votes for each direction
        buy_votes = sum(1 for s in signals if s.direction == 'buy')
        sell_votes = sum(1 for s in signals if s.direction == 'sell')
        hold_votes = sum(1 for s in signals if s.direction == 'hold')

        total_signals = len(signals)

        # Find majority direction
        max_votes = max(buy_votes, sell_votes, hold_votes)

        if max_votes == buy_votes:
            agreed_direction = 'buy'
            num_agreeing = buy_votes
        elif max_votes == sell_votes:
            agreed_direction = 'sell'
            num_agreeing = sell_votes
        else:
            agreed_direction = 'hold'
            num_agreeing = hold_votes

        # Calculate confluence score (0-1)
        confluence_score = num_agreeing / total_signals

        return confluence_score, agreed_direction, num_agreeing

    def _has_conflicting_signals(self, signals: List[TimeframeSignal]) -> bool:
        """Check if signals are conflicting (buy vs sell)."""
        has_buy = any(s.direction == 'buy' for s in signals)
        has_sell = any(s.direction == 'sell' for s in signals)

        # Conflicting if both buy and sell signals present
        return has_buy and has_sell

    def _build_reasoning(
        self,
        confluence_score: float,
        agreed_direction: str,
        num_agreeing: int,
        total: int,
        conflicting: bool,
    ) -> str:
        """Build human-readable reasoning."""
        if conflicting:
            return f"CONFLICTING signals: {num_agreeing}/{total} agree on {agreed_direction.upper()}, but other timeframes disagree"

        if confluence_score >= 0.8:
            return f"STRONG confluence: {num_agreeing}/{total} timeframes agree on {agreed_direction.upper()}"
        elif confluence_score >= self.min_confluence:
            return f"MODERATE confluence: {num_agreeing}/{total} timeframes agree on {agreed_direction.upper()}"
        else:
            return f"WEAK confluence: Only {num_agreeing}/{total} timeframes agree on {agreed_direction.upper()}"

    def _create_empty_result(self, reason: str) -> ConfluenceResult:
        """Create empty result when no data available."""
        return ConfluenceResult(
            confluence_score=0.0,
            agreed_direction='hold',
            num_agreeing=0,
            total_timeframes=0,
            conflicting=False,
            reasoning=reason,
            timeframe_breakdown={},
        )

    def meets_minimum_confluence(self, result: ConfluenceResult) -> bool:
        """Check if confluence result meets minimum threshold."""
        return result.confluence_score >= self.min_confluence

    def get_confidence_multiplier(self, result: ConfluenceResult) -> float:
        """
        Get confidence multiplier based on confluence strength.

        Returns:
            Multiplier (0.5 - 1.2) to adjust signal confidence
            - Strong confluence (>0.8) → 1.2x boost
            - Moderate confluence (0.67-0.8) → 1.0x (no change)
            - Weak confluence (<0.67) → 0.5x penalty
        """
        if result.conflicting:
            return 0.3  # Heavy penalty for conflicting signals

        if result.confluence_score >= 0.8:
            return 1.2  # Boost for strong agreement
        elif result.confluence_score >= self.min_confluence:
            return 1.0  # No change for moderate agreement
        else:
            return 0.5  # Penalty for weak agreement
