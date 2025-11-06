"""
Alpha Engines - Revuelto Integration

The 6 specialized trading engines that exploit different market conditions:

1. **Trend Engine** - Rides strong directional moves
2. **Range Engine** - Plays mean reversion in choppy markets
3. **Breakout Engine** - Catches explosive moves from compression
4. **Tape Engine** - Exploits microstructure inefficiencies
5. **Leader Engine** - Trades relative strength momentum
6. **Sweep Engine** - Plays liquidity sweeps and stop hunts

Each engine:
- Has its own feature affinity (which features it trusts)
- Generates independent signals
- Has its own confidence scoring
- Operates best in specific market regimes
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class TradingTechnique(Enum):
    """Trading technique types."""

    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    TAPE = "tape"  # Microstructure
    LEADER = "leader"  # Relative strength
    SWEEP = "sweep"  # Liquidity


@dataclass
class AlphaSignal:
    """Signal from an alpha engine."""

    technique: TradingTechnique
    direction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    reasoning: str
    key_features: Dict[str, float]  # Features that triggered this signal
    regime_affinity: float  # How well current regime matches this technique


class TrendEngine:
    """
    Trend Engine - Rides strong directional moves.

    Best in: TREND regime
    Key features: trend_strength, ema_slope, momentum_slope, htf_bias, adx
    Strategy: Enter when aligned multi-timeframe trend, exit on weakness
    
    Enhanced with Moving Average Crossover Strategy (verified):
    - Golden Cross: SMA50 crosses above SMA200 → BUY signal
    - Death Cross: SMA50 crosses below SMA200 → SELL signal
    - Works best in trending markets
    """

    def __init__(
        self,
        min_trend_strength: float = 0.6,
        min_adx: float = 25.0,
        min_confidence: float = 0.55,
        use_ma_crossover: bool = True,
    ):
        self.min_trend_strength = min_trend_strength
        self.min_adx = min_adx
        self.min_confidence = min_confidence
        self.use_ma_crossover = use_ma_crossover

        # Feature weights for this technique
        self.feature_weights = {
            "trend_strength": 0.25,
            "ema_slope": 0.20,
            "momentum_slope": 0.20,
            "htf_bias": 0.20,
            "adx": 0.15,
        }

    def detect_ma_crossover(
        self, features: Dict[str, float]
    ) -> Optional[Tuple[str, float]]:
        """
        Detect moving average crossover (Golden Cross / Death Cross).
        
        Based on verified trend following strategy.
        
        Args:
            features: Feature dictionary (should contain sma50, sma200, prev_sma50, prev_sma200)
        
        Returns:
            (direction, confidence) tuple if crossover detected, None otherwise
        """
        if not self.use_ma_crossover:
            return None

        # Get MA values
        sma50 = features.get("sma50", None)
        sma200 = features.get("sma200", None)
        prev_sma50 = features.get("prev_sma50", None)
        prev_sma200 = features.get("prev_sma200", None)

        if None in [sma50, sma200, prev_sma50, prev_sma200]:
            return None

        # Check for Golden Cross (SMA50 crosses above SMA200)
        if prev_sma50 <= prev_sma200 and sma50 > sma200:
            return ("buy", 0.75)  # Strong bullish signal

        # Check for Death Cross (SMA50 crosses below SMA200)
        elif prev_sma50 >= prev_sma200 and sma50 < sma200:
            return ("sell", 0.75)  # Strong bearish signal

        return None

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate trend-following signal."""
        # Check for MA crossover first (high priority signal)
        ma_crossover = self.detect_ma_crossover(features)
        if ma_crossover:
            direction, crossover_confidence = ma_crossover
            regime_affinity = 1.0 if current_regime == "trend" else 0.5
            confidence = crossover_confidence * regime_affinity
            
            crossover_type = "Golden Cross" if direction == "buy" else "Death Cross"
            reasoning = f"{crossover_type} detected: SMA50 crossed SMA200"
            
            return AlphaSignal(
                technique=TradingTechnique.TREND,
                direction=direction,
                confidence=min(confidence, 1.0),
                reasoning=reasoning,
                key_features={
                    "sma50": features.get("sma50", 0.0),
                    "sma200": features.get("sma200", 0.0),
                    "ma_crossover": 1.0,
                },
                regime_affinity=regime_affinity,
            )

        # Extract key features
        trend_str = features.get("trend_strength", 0.0)
        ema_slope = features.get("ema_slope", 0.0)
        momentum_slope = features.get("momentum_slope", 0.0)
        htf = features.get("htf_bias", 0.5)
        adx = features.get("adx", 0.0)

        # Regime affinity: Best in TREND regime
        regime_affinity = 1.0 if current_regime == "trend" else 0.3

        # Calculate weighted score
        score = 0.0
        score += self.feature_weights["trend_strength"] * abs(trend_str)
        score += self.feature_weights["ema_slope"] * (1.0 if ema_slope > 0 else 0.0)
        score += self.feature_weights["momentum_slope"] * (
            1.0 if momentum_slope > 0 else 0.0
        )
        score += self.feature_weights["htf_bias"] * htf
        score += self.feature_weights["adx"] * min(adx / 50.0, 1.0)

        # Determine direction
        if trend_str > self.min_trend_strength and adx > self.min_adx:
            direction = "buy" if trend_str > 0 else "sell"
            confidence = score * regime_affinity
            reasoning = f"Strong trend: {trend_str:.2f}, ADX: {adx:.1f}"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"Weak trend: {trend_str:.2f}, ADX: {adx:.1f}"

        return AlphaSignal(
            technique=TradingTechnique.TREND,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={
                "trend_strength": trend_str,
                "adx": adx,
                "ema_slope": ema_slope,
            },
            regime_affinity=regime_affinity,
        )


class RangeEngine:
    """
    Range Engine - Mean reversion in choppy markets.

    Best in: RANGE regime
    Key features: mean_revert_bias, bb_width, compression, volatility_regime
    Strategy: Fade extremes, buy lows sell highs
    """

    def __init__(
        self,
        min_compression: float = 0.6,
        max_adx: float = 25.0,
        min_confidence: float = 0.55,
    ):
        self.min_compression = min_compression
        self.max_adx = max_adx
        self.min_confidence = min_confidence

        self.feature_weights = {
            "mean_revert_bias": 0.30,
            "compression": 0.25,
            "bb_width": 0.20,
            "price_position": 0.25,
        }

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate mean reversion signal."""
        mean_rev = features.get("mean_revert_bias", 0.0)
        compression = features.get("compression", 0.0)
        bb_width = features.get("bb_width", 1.0)
        price_pos = features.get("price_position", 0.5)
        adx = features.get("adx", 50.0)

        # Regime affinity: Best in RANGE regime
        regime_affinity = 1.0 if current_regime == "range" else 0.4

        # Calculate score
        score = 0.0
        score += self.feature_weights["mean_revert_bias"] * abs(mean_rev)
        score += self.feature_weights["compression"] * compression
        score += self.feature_weights["bb_width"] * (1.0 - min(bb_width / 0.05, 1.0))
        score += self.feature_weights["price_position"] * abs(price_pos - 0.5) * 2

        # Determine direction (fade extremes)
        if compression > self.min_compression and adx < self.max_adx:
            if price_pos < 0.3:  # Near bottom of range
                direction = "buy"
                confidence = score * regime_affinity
                reasoning = f"Oversold in range: pos={price_pos:.2f}, comp={compression:.2f}"
            elif price_pos > 0.7:  # Near top of range
                direction = "sell"
                confidence = score * regime_affinity
                reasoning = f"Overbought in range: pos={price_pos:.2f}, comp={compression:.2f}"
            else:
                direction = "hold"
                confidence = 0.0
                reasoning = "Mid-range, no extreme"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"Not ranging: comp={compression:.2f}, adx={adx:.1f}"

        return AlphaSignal(
            technique=TradingTechnique.RANGE,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={
                "mean_revert_bias": mean_rev,
                "compression": compression,
                "price_position": price_pos,
            },
            regime_affinity=regime_affinity,
        )


class BreakoutEngine:
    """
    Breakout Engine - Catches explosive moves from compression.

    Best in: TREND or transition from RANGE to TREND
    Key features: ignition_score, breakout_thrust, breakout_quality, nr7_density
    Strategy: Enter on high-quality breakouts with volume confirmation
    """

    def __init__(
        self,
        min_ignition: float = 60.0,
        min_quality: float = 0.6,
        min_confidence: float = 0.60,
    ):
        self.min_ignition = min_ignition
        self.min_quality = min_quality
        self.min_confidence = min_confidence

        self.feature_weights = {
            "ignition_score": 0.35,
            "breakout_quality": 0.30,
            "breakout_thrust": 0.20,
            "nr7_density": 0.15,
        }

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate breakout signal."""
        ignition = features.get("ignition_score", 0.0)
        quality = features.get("breakout_quality", 0.0)
        thrust = features.get("breakout_thrust", 0.0)
        nr7 = features.get("nr7_density", 0.0)

        # Regime affinity: Good in trend, decent in transition
        regime_affinity = 1.0 if current_regime == "trend" else 0.7

        # Calculate score
        score = 0.0
        score += self.feature_weights["ignition_score"] * min(ignition / 100.0, 1.0)
        score += self.feature_weights["breakout_quality"] * quality
        score += self.feature_weights["breakout_thrust"] * min(abs(thrust), 1.0)
        score += self.feature_weights["nr7_density"] * nr7

        # Determine direction
        if ignition > self.min_ignition and quality > self.min_quality:
            direction = "buy" if thrust > 0 else "sell"
            confidence = score * regime_affinity
            reasoning = f"Breakout: ignition={ignition:.0f}, quality={quality:.2f}"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"No breakout: ignition={ignition:.0f}, quality={quality:.2f}"

        return AlphaSignal(
            technique=TradingTechnique.BREAKOUT,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={
                "ignition_score": ignition,
                "breakout_quality": quality,
                "breakout_thrust": thrust,
            },
            regime_affinity=regime_affinity,
        )


class TapeEngine:
    """
    Tape Engine - Exploits microstructure inefficiencies.

    Best in: All regimes (microstructure always matters)
    Key features: micro_score, uptick_ratio, spread_bps, vol_jump_z
    Strategy: Ride short-term order flow imbalances
    """

    def __init__(self, min_micro_score: float = 60.0, min_confidence: float = 0.50):
        self.min_micro_score = min_micro_score
        self.min_confidence = min_confidence

        self.feature_weights = {
            "micro_score": 0.40,
            "uptick_ratio": 0.30,
            "vol_jump_z": 0.20,
            "spread_bps": 0.10,
        }

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate tape reading signal."""
        micro = features.get("micro_score", 50.0)
        uptick = features.get("uptick_ratio", 0.5)
        vol_jump = features.get("vol_jump_z", 0.0)
        spread = features.get("spread_bps", 50.0)

        # Regime affinity: Works in all regimes
        regime_affinity = 0.8  # Moderate in all conditions

        # Calculate score
        score = 0.0
        score += self.feature_weights["micro_score"] * min(micro / 100.0, 1.0)
        score += self.feature_weights["uptick_ratio"] * uptick
        score += self.feature_weights["vol_jump_z"] * min(abs(vol_jump) / 3.0, 1.0)
        score += self.feature_weights["spread_bps"] * (
            1.0 - min(spread / 100.0, 1.0)
        )  # Lower spread = better

        # Determine direction
        if micro > self.min_micro_score:
            if uptick > 0.6:  # Strong buying
                direction = "buy"
                confidence = score * regime_affinity
                reasoning = f"Strong tape: micro={micro:.0f}, uptick={uptick:.2f}"
            elif uptick < 0.4:  # Strong selling
                direction = "sell"
                confidence = score * regime_affinity
                reasoning = f"Weak tape: micro={micro:.0f}, uptick={uptick:.2f}"
            else:
                direction = "hold"
                confidence = 0.0
                reasoning = "Neutral tape"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"Weak microstructure: {micro:.0f}"

        return AlphaSignal(
            technique=TradingTechnique.TAPE,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={"micro_score": micro, "uptick_ratio": uptick},
            regime_affinity=regime_affinity,
        )


class LeaderEngine:
    """
    Leader Engine - Trades relative strength momentum.

    Best in: TREND regime
    Key features: rs_score, leader_bias, momentum
    Strategy: Buy leaders, avoid laggards
    """

    def __init__(self, min_rs_score: float = 70.0, min_confidence: float = 0.55):
        self.min_rs_score = min_rs_score
        self.min_confidence = min_confidence

        self.feature_weights = {
            "rs_score": 0.40,
            "leader_bias": 0.35,
            "momentum_slope": 0.25,
        }

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate relative strength signal."""
        rs = features.get("rs_score", 50.0)
        leader = features.get("leader_bias", 0.0)
        momentum = features.get("momentum_slope", 0.0)

        # Regime affinity: Best in trends
        regime_affinity = 1.0 if current_regime == "trend" else 0.5

        # Calculate score
        score = 0.0
        score += self.feature_weights["rs_score"] * min(rs / 100.0, 1.0)
        score += self.feature_weights["leader_bias"] * (leader + 1.0) / 2.0  # -1 to 1 -> 0 to 1
        score += self.feature_weights["momentum_slope"] * (
            1.0 if momentum > 0 else 0.0
        )

        # Determine direction
        if rs > self.min_rs_score and leader > 0.3:
            direction = "buy"
            confidence = score * regime_affinity
            reasoning = f"Leading: RS={rs:.0f}, bias={leader:.2f}"
        elif rs < 30 and leader < -0.3:
            direction = "sell"
            confidence = score * regime_affinity * 0.8  # Slightly lower confidence on shorts
            reasoning = f"Lagging: RS={rs:.0f}, bias={leader:.2f}"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"Neutral RS: {rs:.0f}"

        return AlphaSignal(
            technique=TradingTechnique.LEADER,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={"rs_score": rs, "leader_bias": leader},
            regime_affinity=regime_affinity,
        )


class SweepEngine:
    """
    Sweep Engine - Plays liquidity sweeps and stop hunts.

    Best in: All regimes (liquidity events happen always)
    Key features: vol_jump_z, pullback_depth, price_position, kurtosis
    Strategy: Detect fake-outs and trap reversals
    """

    def __init__(
        self, min_vol_jump: float = 2.0, min_pullback: float = 0.3, min_confidence: float = 0.55
    ):
        self.min_vol_jump = min_vol_jump
        self.min_pullback = min_pullback
        self.min_confidence = min_confidence

        self.feature_weights = {
            "vol_jump_z": 0.35,
            "pullback_depth": 0.30,
            "kurtosis": 0.20,
            "price_position": 0.15,
        }

    def generate_signal(
        self, features: Dict[str, float], current_regime: str
    ) -> AlphaSignal:
        """Generate sweep/liquidity signal."""
        vol_jump = features.get("vol_jump_z", 0.0)
        pullback = features.get("pullback_depth", 0.0)
        kurt = features.get("kurtosis", 0.0)
        price_pos = features.get("price_position", 0.5)

        # Regime affinity: Works in all regimes
        regime_affinity = 0.7

        # Calculate score
        score = 0.0
        score += self.feature_weights["vol_jump_z"] * min(abs(vol_jump) / 3.0, 1.0)
        score += self.feature_weights["pullback_depth"] * pullback
        score += self.feature_weights["kurtosis"] * min(abs(kurt) / 5.0, 1.0)
        score += self.feature_weights["price_position"] * abs(price_pos - 0.5) * 2

        # Detect sweep conditions
        if vol_jump > self.min_vol_jump and pullback > self.min_pullback:
            if price_pos < 0.2:  # Sweep of lows
                direction = "buy"
                confidence = score * regime_affinity
                reasoning = f"Low sweep: vol_jump={vol_jump:.1f}, pullback={pullback:.2f}"
            elif price_pos > 0.8:  # Sweep of highs
                direction = "sell"
                confidence = score * regime_affinity
                reasoning = f"High sweep: vol_jump={vol_jump:.1f}, pullback={pullback:.2f}"
            else:
                direction = "hold"
                confidence = 0.0
                reasoning = "Mid-range sweep"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = f"No sweep: vol_jump={vol_jump:.1f}"

        return AlphaSignal(
            technique=TradingTechnique.SWEEP,
            direction=direction,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            key_features={"vol_jump_z": vol_jump, "pullback_depth": pullback},
            regime_affinity=regime_affinity,
        )


class AlphaEngineCoordinator:
    """
    Coordinates all 6 alpha engines and selects best technique.

    Responsibilities:
    1. Run all engines in parallel
    2. Weight by regime affinity
    3. Track engine performance
    4. Select best technique dynamically
    """

    def __init__(self, use_bandit: bool = True):
        """Initialize all engines."""
        self.engines = {
            TradingTechnique.TREND: TrendEngine(),
            TradingTechnique.RANGE: RangeEngine(),
            TradingTechnique.BREAKOUT: BreakoutEngine(),
            TradingTechnique.TAPE: TapeEngine(),
            TradingTechnique.LEADER: LeaderEngine(),
            TradingTechnique.SWEEP: SweepEngine(),
        }

        # Track engine performance
        self.engine_performance: Dict[TradingTechnique, List[float]] = {
            technique: [] for technique in TradingTechnique
        }
        
        # Multi-armed bandit for engine selection
        self.use_bandit = use_bandit
        self.bandit = None
        if use_bandit:
            try:
                from .alpha_engine_bandit import AlphaEngineBandit
                self.bandit = AlphaEngineBandit()
                logger.info("alpha_engine_bandit_enabled")
            except ImportError:
                logger.warning("alpha_engine_bandit_not_available")
                self.use_bandit = False

        logger.info("alpha_engine_coordinator_initialized", num_engines=len(self.engines), use_bandit=use_bandit)

    def generate_all_signals(
        self, features: Dict[str, float], current_regime: str
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """Generate signals from all engines."""
        signals = {}

        for technique, engine in self.engines.items():
            signal = engine.generate_signal(features, current_regime)
            signals[technique] = signal

        return signals

    def generate_all_signals_batch(
        self, symbols_features: Dict[str, Dict[str, float]], current_regimes: Dict[str, str]
    ) -> Dict[str, Dict[TradingTechnique, AlphaSignal]]:
        """
        Generate signals for multiple symbols in batch (more efficient).
        
        Args:
            symbols_features: Dict of {symbol: features}
            current_regimes: Dict of {symbol: regime}
            
        Returns:
            Dict of {symbol: {technique: signal}}
        """
        all_signals = {}
        
        # Process all symbols
        for symbol, features in symbols_features.items():
            regime = current_regimes.get(symbol, 'UNKNOWN')
            signals = self.generate_all_signals(features, regime)
            all_signals[symbol] = signals
        
        return all_signals

    def select_best_technique(
        self, signals: Dict[TradingTechnique, AlphaSignal], current_regime: str = 'unknown'
    ) -> AlphaSignal:
        """
        Select best technique based on:
        1. Signal confidence
        2. Regime affinity
        3. Historical performance
        4. Multi-armed bandit (if enabled)
        """
        # Filter to signals that are not "hold"
        active_signals = {
            tech: sig for tech, sig in signals.items() if sig.direction != "hold"
        }

        if not active_signals:
            # No active signals, return neutral
            return AlphaSignal(
                technique=TradingTechnique.TREND,
                direction="hold",
                confidence=0.0,
                reasoning="No engine has conviction",
                key_features={},
                regime_affinity=0.0,
            )

        # Use bandit if enabled
        if self.use_bandit and self.bandit:
            best_technique, best_signal, bandit_confidence = self.bandit.select_engine(
                current_regime=current_regime,
                all_signals=signals,
            )
            
            # Update signal confidence with bandit confidence
            best_signal.confidence = (best_signal.confidence + bandit_confidence) / 2.0
            
            logger.debug(
                "bandit_technique_selected",
                technique=best_technique.value,
                confidence=best_signal.confidence,
                bandit_confidence=bandit_confidence,
            )
            
            return best_signal

        # Fallback to original scoring method
        # Score each signal: confidence * regime_affinity * historical_performance
        scores = {}
        for technique, signal in active_signals.items():
            # Get historical win rate for this engine
            if self.engine_performance[technique]:
                hist_perf = np.mean(self.engine_performance[technique][-20:])  # Recent 20
            else:
                hist_perf = 0.55  # Assume 55% default

            score = signal.confidence * signal.regime_affinity * hist_perf
            scores[technique] = score

        # Select highest scoring
        best_technique = max(scores, key=scores.get)
        best_signal = active_signals[best_technique]

        logger.debug(
            "best_technique_selected",
            technique=best_technique.value,
            confidence=best_signal.confidence,
            num_active=len(active_signals),
        )

        return best_signal

    def update_engine_performance(
        self, technique: TradingTechnique, performance: float, regime: str = 'unknown', won: bool = False
    ) -> None:
        """
        Update performance tracking for an engine.
        
        Args:
            technique: Trading technique
            performance: Performance metric (0-1)
            regime: Market regime
            won: Whether the trade won (for bandit)
        """
        if technique not in self.engine_performance:
            self.engine_performance[technique] = []
        self.engine_performance[technique].append(performance)

        # Keep recent history
        if len(self.engine_performance[technique]) > 100:
            self.engine_performance[technique] = self.engine_performance[technique][
                -100:
            ]
        
        # Update bandit if enabled
        if self.use_bandit and self.bandit:
            self.bandit.update_performance(technique=technique, regime=regime, won=won)

    def get_engine_stats(self) -> Dict[str, Dict]:
        """Get performance stats for all engines."""
        stats = {}

        for technique in TradingTechnique:
            perf_history = self.engine_performance[technique]

            if perf_history:
                stats[technique.value] = {
                    "total_signals": len(perf_history),
                    "win_rate": np.mean(perf_history),
                    "recent_win_rate": np.mean(perf_history[-20:])
                    if len(perf_history) >= 20
                    else np.mean(perf_history),
                }
            else:
                stats[technique.value] = {
                    "total_signals": 0,
                    "win_rate": 0.55,
                    "recent_win_rate": 0.55,
                }

        return stats

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "engine_performance": {
                tech.value: perf[-50:]  # Keep recent 50
                for tech, perf in self.engine_performance.items()
            }
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        perf_data = state.get("engine_performance", {})

        for tech_str, perf_list in perf_data.items():
            technique = TradingTechnique(tech_str)
            self.engine_performance[technique] = list(perf_list)

        logger.info(
            "alpha_engine_coordinator_state_loaded",
            num_engines=len(self.engine_performance),
        )
