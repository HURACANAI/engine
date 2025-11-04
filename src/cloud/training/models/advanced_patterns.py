"""
Advanced Pattern Recognition Module

Detects sophisticated chart patterns:
1. Reversal patterns (V-bottoms, double-tops, head-and-shoulders)
2. Breakout patterns (range expansion, volume breakouts)
3. Momentum continuation patterns
4. Pattern quality scoring

Based on technical analysis and statistical pattern recognition.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger()


class PatternType(Enum):
    """Types of chart patterns."""

    # Reversal patterns
    V_BOTTOM = "v_bottom"
    V_TOP = "v_top"
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"

    # Breakout patterns
    RANGE_BREAKOUT_UP = "range_breakout_up"
    RANGE_BREAKOUT_DOWN = "range_breakout_down"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    VOLUME_BREAKOUT = "volume_breakout"

    # Continuation patterns
    MOMENTUM_CONTINUATION = "momentum_continuation"
    TREND_ACCELERATION = "trend_acceleration"
    PULLBACK_CONTINUATION = "pullback_continuation"

    # Other
    UNKNOWN = "unknown"


@dataclass
class PatternDetection:
    """Result of pattern detection."""

    pattern_type: PatternType
    quality_score: float  # 0-1, where 1 = perfect pattern
    confidence: float  # 0-1, confidence in detection
    start_idx: int  # Where pattern starts
    end_idx: int  # Where pattern ends
    key_levels: dict  # Important price levels (support, resistance, etc.)
    expected_direction: str  # "up", "down", or "neutral"
    expected_move_pct: float  # Expected price move percentage
    description: str


class AdvancedPatternDetector:
    """
    Detects advanced chart patterns in price data.

    Uses statistical methods and technical analysis rules.
    """

    def __init__(
        self,
        min_pattern_quality: float = 0.6,
        lookback_periods: int = 50,
    ):
        """
        Initialize pattern detector.

        Args:
            min_pattern_quality: Minimum quality score to report pattern
            lookback_periods: How many periods to analyze for patterns
        """
        self.min_quality = min_pattern_quality
        self.lookback = lookback_periods

        logger.info(
            "advanced_pattern_detector_initialized",
            min_quality=min_pattern_quality,
            lookback=lookback_periods,
        )

    def detect_patterns(
        self, df: pl.DataFrame, current_idx: int
    ) -> List[PatternDetection]:
        """
        Detect all patterns in recent price data.

        Args:
            df: DataFrame with OHLCV data
            current_idx: Current index in dataframe

        Returns:
            List of detected patterns
        """
        patterns = []

        # Extract recent data
        start_idx = max(0, current_idx - self.lookback)
        recent_data = df[start_idx : current_idx + 1]

        if recent_data.height < 10:
            return []  # Not enough data

        # Detect each pattern type
        patterns.extend(self._detect_reversal_patterns(recent_data, start_idx))
        patterns.extend(self._detect_breakout_patterns(recent_data, start_idx))
        patterns.extend(self._detect_continuation_patterns(recent_data, start_idx))

        # Filter by quality
        patterns = [p for p in patterns if p.quality_score >= self.min_quality]

        return patterns

    def _detect_reversal_patterns(
        self, df: pl.DataFrame, offset: int
    ) -> List[PatternDetection]:
        """Detect reversal patterns (V-bottoms, double-tops, etc.)."""
        patterns = []

        closes = df["close"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()

        # V-Bottom detection
        v_bottom = self._detect_v_bottom(closes, lows, offset)
        if v_bottom:
            patterns.append(v_bottom)

        # V-Top detection
        v_top = self._detect_v_top(closes, highs, offset)
        if v_top:
            patterns.append(v_top)

        # Double bottom
        double_bottom = self._detect_double_bottom(closes, lows, offset)
        if double_bottom:
            patterns.append(double_bottom)

        # Double top
        double_top = self._detect_double_top(closes, highs, offset)
        if double_top:
            patterns.append(double_top)

        return patterns

    def _detect_v_bottom(
        self, closes: np.ndarray, lows: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """
        Detect V-bottom pattern: Sharp decline followed by sharp recovery.
        """
        if len(closes) < 20:
            return None

        # Find the lowest point in recent history
        low_idx = np.argmin(lows[-20:])
        low_price = lows[-20 + low_idx]

        # Check for decline before low
        if low_idx < 5:
            return None  # Need at least 5 bars decline

        pre_low_prices = closes[-20 : -20 + low_idx]
        decline_pct = (pre_low_prices[0] - low_price) / pre_low_prices[0]

        # Check for recovery after low
        post_low_prices = closes[-20 + low_idx :]
        if len(post_low_prices) < 5:
            return None

        recovery_pct = (post_low_prices[-1] - low_price) / low_price

        # V-bottom criteria:
        # 1. Significant decline (> 3%)
        # 2. Significant recovery (> 2%)
        # 3. Sharp (not gradual)
        if decline_pct > 0.03 and recovery_pct > 0.02:
            # Calculate quality based on symmetry
            symmetry = min(recovery_pct / decline_pct, decline_pct / recovery_pct)
            quality = float(np.clip(symmetry, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.V_BOTTOM,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"support": float(low_price)},
                expected_direction="up",
                expected_move_pct=recovery_pct * 0.5,  # Expect 50% more of current recovery
                description=f"V-Bottom: {decline_pct:.1%} decline, {recovery_pct:.1%} recovery",
            )

        return None

    def _detect_v_top(
        self, closes: np.ndarray, highs: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """
        Detect V-top pattern: Sharp rally followed by sharp decline.
        """
        if len(closes) < 20:
            return None

        # Find the highest point
        high_idx = np.argmax(highs[-20:])
        high_price = highs[-20 + high_idx]

        # Check for rally before high
        if high_idx < 5:
            return None

        pre_high_prices = closes[-20 : -20 + high_idx]
        rally_pct = (high_price - pre_high_prices[0]) / pre_high_prices[0]

        # Check for decline after high
        post_high_prices = closes[-20 + high_idx :]
        if len(post_high_prices) < 5:
            return None

        decline_pct = (high_price - post_high_prices[-1]) / high_price

        # V-top criteria
        if rally_pct > 0.03 and decline_pct > 0.02:
            symmetry = min(rally_pct / decline_pct, decline_pct / rally_pct)
            quality = float(np.clip(symmetry, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.V_TOP,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"resistance": float(high_price)},
                expected_direction="down",
                expected_move_pct=-decline_pct * 0.5,
                description=f"V-Top: {rally_pct:.1%} rally, {decline_pct:.1%} decline",
            )

        return None

    def _detect_double_bottom(
        self, closes: np.ndarray, lows: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect double bottom pattern."""
        if len(closes) < 30:
            return None

        # Find two lowest points
        recent_lows = lows[-30:]
        sorted_indices = np.argsort(recent_lows)

        first_low_idx = sorted_indices[0]
        second_low_idx = sorted_indices[1]

        # They should be separated by at least 5 bars
        if abs(first_low_idx - second_low_idx) < 5:
            return None

        first_low = recent_lows[first_low_idx]
        second_low = recent_lows[second_low_idx]

        # Lows should be similar (within 2%)
        if abs(first_low - second_low) / first_low > 0.02:
            return None

        # Price should be above both lows now
        current_price = closes[-1]
        if current_price <= max(first_low, second_low):
            return None

        # Calculate quality
        low_similarity = 1.0 - abs(first_low - second_low) / first_low
        quality = float(np.clip(low_similarity, 0.0, 1.0))

        return PatternDetection(
            pattern_type=PatternType.DOUBLE_BOTTOM,
            quality_score=quality,
            confidence=0.75,
            start_idx=offset + len(closes) - 30,
            end_idx=offset + len(closes) - 1,
            key_levels={"support": float((first_low + second_low) / 2)},
            expected_direction="up",
            expected_move_pct=0.05,  # Expect 5% move up
            description=f"Double Bottom at {first_low:.2f}",
        )

    def _detect_double_top(
        self, closes: np.ndarray, highs: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect double top pattern."""
        if len(closes) < 30:
            return None

        recent_highs = highs[-30:]
        sorted_indices = np.argsort(recent_highs)[::-1]  # Descending

        first_high_idx = sorted_indices[0]
        second_high_idx = sorted_indices[1]

        if abs(first_high_idx - second_high_idx) < 5:
            return None

        first_high = recent_highs[first_high_idx]
        second_high = recent_highs[second_high_idx]

        # Highs should be similar
        if abs(first_high - second_high) / first_high > 0.02:
            return None

        # Price should be below both highs
        current_price = closes[-1]
        if current_price >= min(first_high, second_high):
            return None

        high_similarity = 1.0 - abs(first_high - second_high) / first_high
        quality = float(np.clip(high_similarity, 0.0, 1.0))

        return PatternDetection(
            pattern_type=PatternType.DOUBLE_TOP,
            quality_score=quality,
            confidence=0.75,
            start_idx=offset + len(closes) - 30,
            end_idx=offset + len(closes) - 1,
            key_levels={"resistance": float((first_high + second_high) / 2)},
            expected_direction="down",
            expected_move_pct=-0.05,
            description=f"Double Top at {first_high:.2f}",
        )

    def _detect_breakout_patterns(
        self, df: pl.DataFrame, offset: int
    ) -> List[PatternDetection]:
        """Detect breakout patterns."""
        patterns = []

        closes = df["close"].to_numpy()
        volumes = df["volume"].to_numpy()

        # Range breakout
        range_breakout = self._detect_range_breakout(closes, offset)
        if range_breakout:
            patterns.append(range_breakout)

        # Volatility contraction
        vol_contraction = self._detect_volatility_contraction(closes, offset)
        if vol_contraction:
            patterns.append(vol_contraction)

        # Volume breakout
        if len(volumes) > 0:
            volume_breakout = self._detect_volume_breakout(closes, volumes, offset)
            if volume_breakout:
                patterns.append(volume_breakout)

        return patterns

    def _detect_range_breakout(
        self, closes: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect range breakout."""
        if len(closes) < 20:
            return None

        # Calculate range for first 15 bars
        range_closes = closes[-20:-5]
        range_high = np.max(range_closes)
        range_low = np.min(range_closes)
        range_size = range_high - range_low

        # Current price
        current_price = closes[-1]

        # Check for breakout
        if current_price > range_high:
            # Upward breakout
            breakout_strength = (current_price - range_high) / range_size
            quality = float(np.clip(breakout_strength, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.RANGE_BREAKOUT_UP,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"breakout_level": float(range_high), "support": float(range_low)},
                expected_direction="up",
                expected_move_pct=range_size / range_low,  # Expect move equal to range size
                description=f"Range Breakout Up from {range_low:.2f}-{range_high:.2f}",
            )

        elif current_price < range_low:
            # Downward breakout
            breakout_strength = (range_low - current_price) / range_size
            quality = float(np.clip(breakout_strength, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.RANGE_BREAKOUT_DOWN,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"breakout_level": float(range_low), "resistance": float(range_high)},
                expected_direction="down",
                expected_move_pct=-(range_size / range_high),
                description=f"Range Breakout Down from {range_low:.2f}-{range_high:.2f}",
            )

        return None

    def _detect_volatility_contraction(
        self, closes: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect volatility contraction (squeeze)."""
        if len(closes) < 30:
            return None

        # Calculate rolling volatility
        returns = np.diff(closes) / closes[:-1]
        recent_vol = np.std(returns[-10:])
        historical_vol = np.std(returns[-30:-10])

        # Volatility contraction if recent vol < 60% of historical
        if recent_vol < 0.6 * historical_vol:
            quality = float(1.0 - (recent_vol / historical_vol))

            return PatternDetection(
                pattern_type=PatternType.VOLATILITY_CONTRACTION,
                quality_score=quality,
                confidence=0.65,
                start_idx=offset + len(closes) - 30,
                end_idx=offset + len(closes) - 1,
                key_levels={"current_price": float(closes[-1])},
                expected_direction="neutral",  # Could go either way
                expected_move_pct=0.03,  # Expect significant move (3%)
                description=f"Volatility Squeeze: {recent_vol:.4f} vs {historical_vol:.4f}",
            )

        return None

    def _detect_volume_breakout(
        self, closes: np.ndarray, volumes: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect volume breakout."""
        if len(volumes) < 20:
            return None

        avg_volume = np.mean(volumes[-20:-1])
        current_volume = volumes[-1]

        # Volume breakout if current > 2x average
        if current_volume > 2.0 * avg_volume:
            # Determine direction based on price change
            price_change = (closes[-1] - closes[-2]) / closes[-2]

            direction = "up" if price_change > 0 else "down"
            quality = float(np.clip(current_volume / (3.0 * avg_volume), 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.VOLUME_BREAKOUT,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"volume_surge": float(current_volume / avg_volume)},
                expected_direction=direction,
                expected_move_pct=abs(price_change) * 2.0,  # Expect 2x the current move
                description=f"Volume Surge: {current_volume / avg_volume:.1f}x average",
            )

        return None

    def _detect_continuation_patterns(
        self, df: pl.DataFrame, offset: int
    ) -> List[PatternDetection]:
        """Detect momentum continuation patterns."""
        patterns = []

        closes = df["close"].to_numpy()

        # Momentum continuation
        momentum_cont = self._detect_momentum_continuation(closes, offset)
        if momentum_cont:
            patterns.append(momentum_cont)

        return patterns

    def _detect_momentum_continuation(
        self, closes: np.ndarray, offset: int
    ) -> Optional[PatternDetection]:
        """Detect momentum continuation pattern."""
        if len(closes) < 20:
            return None

        # Calculate momentum (rate of change over 10 periods)
        prev_momentum = (closes[-11] - closes[-21]) / closes[-21]
        current_momentum = (closes[-1] - closes[-11]) / closes[-11]

        # Continuation if momentum in same direction and accelerating
        if prev_momentum > 0.02 and current_momentum > prev_momentum:
            quality = float(np.clip(current_momentum / prev_momentum - 1, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.MOMENTUM_CONTINUATION,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"momentum": float(current_momentum)},
                expected_direction="up",
                expected_move_pct=current_momentum * 0.5,
                description=f"Momentum Continuation: {current_momentum:.1%}",
            )

        elif prev_momentum < -0.02 and current_momentum < prev_momentum:
            quality = float(np.clip(abs(current_momentum / prev_momentum) - 1, 0.0, 1.0))

            return PatternDetection(
                pattern_type=PatternType.MOMENTUM_CONTINUATION,
                quality_score=quality,
                confidence=0.7,
                start_idx=offset + len(closes) - 20,
                end_idx=offset + len(closes) - 1,
                key_levels={"momentum": float(current_momentum)},
                expected_direction="down",
                expected_move_pct=current_momentum * 0.5,
                description=f"Momentum Continuation: {current_momentum:.1%}",
            )

        return None
