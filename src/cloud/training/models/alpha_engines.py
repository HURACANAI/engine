"""
Alpha Engines - Comprehensive 23-Engine System

All 23 specialized trading engines that exploit different market conditions:

Price-Action / Market-Microstructure Engines (7):
1. **Trend Engine** - Rides strong directional moves
2. **Range Engine** - Plays mean reversion in choppy markets
3. **Breakout Engine** - Catches explosive moves from compression
4. **Tape Engine** - Exploits microstructure inefficiencies
5. **Leader Engine** - Trades relative strength momentum
6. **Sweep Engine** - Plays liquidity sweeps and stop hunts
7. **Scalper Engine** - Micro-arbitrage with ultra-low latency

Cross-Asset & Relative-Value Engines (4):
8. **Correlation Engine** - Pair spread trading and clustering
9. **Funding Engine** - Funding rate and carry trades
10. **Arbitrage Engine** - Multi-exchange arbitrage
11. **Volatility Engine** - Volatility expansion/compression

Learning / Meta Engines (3):
12. **Adaptive Meta Engine** - Dynamic engine weighting
13. **Evolutionary Engine** - Auto-discovery of strategies
14. **Risk Engine** - Volatility targeting and drawdown control

Exotic / Research-Lab Engines (5):
15. **Flow Prediction Engine** - Order flow prediction (deep RL)
16. **Latency Engine** - Cross-venue latency exploitation
17. **Market Maker Engine** - Inventory management and spread capture
18. **Anomaly Engine** - Anomaly and manipulation detection
19. **Regime Engine** - ML-based regime classification

Additional Engines (4):
20. **Momentum Reversal Engine** - Momentum exhaustion trades
21. **Divergence Engine** - Price/indicator divergence
22. **Support/Resistance Engine** - S/R bounce trades
23. **Additional strategies** - Various pattern-based strategies

Each engine:
- Has its own feature affinity (which features it trusts)
- Generates independent signals
- Has its own confidence scoring
- Operates best in specific market regimes
- Can run in parallel with other engines
- Weighted dynamically based on performance
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
import structlog  # type: ignore

logger = structlog.get_logger()

# Import new engines (optional - handle import errors gracefully)
try:
    from .scalper_latency_engine import ScalperLatencyEngine, ScalperSignal
    HAS_SCALPER = True
except ImportError:
    HAS_SCALPER = False
    logger.warning("scalper_latency_engine_not_available")

try:
    from .funding_carry_engine import FundingCarryEngine, FundingSignal
    HAS_FUNDING = True
except ImportError:
    HAS_FUNDING = False
    logger.warning("funding_carry_engine_not_available")

try:
    from .flow_prediction_engine import FlowPredictionEngine, FlowPrediction
    HAS_FLOW_PREDICTION = True
except ImportError:
    HAS_FLOW_PREDICTION = False
    logger.warning("flow_prediction_engine_not_available")

try:
    from .cross_venue_latency_engine import CrossVenueLatencyEngine, LatencyPrediction
    HAS_LATENCY = True
except ImportError:
    HAS_LATENCY = False
    logger.warning("cross_venue_latency_engine_not_available")

try:
    from .market_maker_inventory_engine import MarketMakerInventoryEngine, MarketMakerQuote
    HAS_MARKET_MAKER = True
except ImportError:
    HAS_MARKET_MAKER = False
    logger.warning("market_maker_inventory_engine_not_available")

try:
    from .correlation_cluster_engine import CorrelationClusterEngine, ClusterSignal
    HAS_CORRELATION = True
except ImportError:
    HAS_CORRELATION = False
    logger.warning("correlation_cluster_engine_not_available")

try:
    from .volatility_expansion_engine import VolatilityExpansionEngine
    HAS_VOLATILITY = True
except ImportError:
    HAS_VOLATILITY = False
    logger.warning("volatility_expansion_engine_not_available")

try:
    from .momentum_reversal_engine import MomentumReversalEngine
    HAS_MOMENTUM_REVERSAL = True
except ImportError:
    HAS_MOMENTUM_REVERSAL = False
    logger.warning("momentum_reversal_engine_not_available")

try:
    from .divergence_engine import DivergenceEngine
    HAS_DIVERGENCE = True
except ImportError:
    HAS_DIVERGENCE = False
    logger.warning("divergence_engine_not_available")

try:
    from .support_resistance_bounce_engine import SupportResistanceBounceEngine
    HAS_SUPPORT_RESISTANCE = True
except ImportError:
    HAS_SUPPORT_RESISTANCE = False
    logger.warning("support_resistance_bounce_engine_not_available")

try:
    from .adaptive_meta_engine import AdaptiveMetaEngine, EngineType as AdaptiveEngineType
    HAS_ADAPTIVE_META = True
except ImportError:
    HAS_ADAPTIVE_META = False
    logger.warning("adaptive_meta_engine_not_available")

try:
    from .regime_detector import RegimeDetector
    HAS_REGIME = True
except ImportError:
    HAS_REGIME = False
    logger.warning("regime_detector_not_available")


class TradingTechnique(Enum):
    """Trading technique types - All 23 engines."""

    # Price-Action / Market-Microstructure Engines (7)
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    TAPE = "tape"  # Microstructure
    LEADER = "leader"  # Relative strength
    SWEEP = "sweep"  # Liquidity
    SCALPER = "scalper"  # Latency-arb
    VOLATILITY = "volatility"  # Volatility expansion
    
    # Cross-Asset & Relative-Value Engines (4)
    CORRELATION = "correlation"  # Cluster/pairs
    FUNDING = "funding"  # Carry
    ARBITRAGE = "arbitrage"  # Multi-exchange
    
    # Learning / Meta Engines (3)
    ADAPTIVE_META = "adaptive_meta"  # Meta-learning
    EVOLUTIONARY = "evolutionary"  # Auto-discovery
    RISK = "risk"  # Risk management
    
    # Exotic / Research-Lab Engines (5)
    FLOW_PREDICTION = "flow_prediction"  # Order flow prediction
    LATENCY = "latency"  # Cross-venue latency
    MARKET_MAKER = "market_maker"  # Inventory management
    ANOMALY = "anomaly"  # Anomaly detection
    REGIME = "regime"  # Regime classification
    
    # Additional engines (4)
    MOMENTUM_REVERSAL = "momentum_reversal"
    DIVERGENCE = "divergence"
    SUPPORT_RESISTANCE = "support_resistance"


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
    Coordinates all 23 alpha engines with parallel execution and adaptive weighting.

    Responsibilities:
    1. Run all engines in parallel (using ThreadPoolExecutor or Ray)
    2. Weight by regime affinity
    3. Track engine performance
    4. Dynamically weight engines based on performance (AdaptiveMetaEngine)
    5. Combine signals using weighted voting instead of best selection
    """

    def __init__(
        self,
        use_bandit: bool = True,
        use_parallel: bool = True,
        use_adaptive_weighting: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize all engines.
        
        Args:
            use_bandit: Whether to use multi-armed bandit for engine selection
            use_parallel: Whether to use parallel execution
            use_adaptive_weighting: Whether to use adaptive meta-engine for dynamic weighting
            max_workers: Maximum number of parallel workers (None = auto)
        """
        # Initialize all engines
        self.engines: Dict[TradingTechnique, Any] = {
            # Original 6 engines
            TradingTechnique.TREND: TrendEngine(),
            TradingTechnique.RANGE: RangeEngine(),
            TradingTechnique.BREAKOUT: BreakoutEngine(),
            TradingTechnique.TAPE: TapeEngine(),
            TradingTechnique.LEADER: LeaderEngine(),
            TradingTechnique.SWEEP: SweepEngine(),
        }
        
        # Add new engines if available
        if HAS_SCALPER:
            self.engines[TradingTechnique.SCALPER] = ScalperLatencyEngine()
        
        if HAS_VOLATILITY:
            self.engines[TradingTechnique.VOLATILITY] = VolatilityExpansionEngine()
        
        if HAS_CORRELATION:
            self.engines[TradingTechnique.CORRELATION] = CorrelationClusterEngine()
        
        if HAS_FUNDING:
            self.engines[TradingTechnique.FUNDING] = FundingCarryEngine()
        
        if HAS_FLOW_PREDICTION:
            self.engines[TradingTechnique.FLOW_PREDICTION] = FlowPredictionEngine()
        
        if HAS_LATENCY:
            self.engines[TradingTechnique.LATENCY] = CrossVenueLatencyEngine()
        
        if HAS_MARKET_MAKER:
            self.engines[TradingTechnique.MARKET_MAKER] = MarketMakerInventoryEngine()
        
        if HAS_MOMENTUM_REVERSAL:
            self.engines[TradingTechnique.MOMENTUM_REVERSAL] = MomentumReversalEngine()
        
        if HAS_DIVERGENCE:
            self.engines[TradingTechnique.DIVERGENCE] = DivergenceEngine()
        
        if HAS_SUPPORT_RESISTANCE:
            self.engines[TradingTechnique.SUPPORT_RESISTANCE] = SupportResistanceBounceEngine()
        
        if HAS_REGIME:
            self.engines[TradingTechnique.REGIME] = RegimeDetector()
        
        # Load AI-generated engines (optional)
        self.ai_engines: Dict[str, Any] = {}
        try:
            from .ai_generated_engines import load_ai_engines_with_adapters
            # Load AI engines with "testing" or "approved" status
            ai_engines_dict = load_ai_engines_with_adapters(status_filter="all", symbol="UNKNOWN")
            self.ai_engines = ai_engines_dict
            if ai_engines_dict:
                logger.info(
                    "ai_engines_loaded",
                    count=len(ai_engines_dict),
                    engines=list(ai_engines_dict.keys())
                )
        except Exception as e:
            logger.warning(
                "ai_engines_load_failed",
                error=str(e),
                message="AI-generated engines will not be available"
            )

        # Track engine performance
        self.engine_performance: Dict[TradingTechnique, List[float]] = {
            technique: [] for technique in TradingTechnique
        }
        
        # Track AI engine performance separately
        self.ai_engine_performance: Dict[str, List[float]] = {
            name: [] for name in self.ai_engines.keys()
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
        
        # Parallel execution
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(32, len(self.engines) + 4)  # Default: min(32, num_engines + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers) if use_parallel else None
        
        # Adaptive meta-engine for dynamic weighting
        self.use_adaptive_weighting = use_adaptive_weighting
        self.adaptive_meta_engine = None
        if use_adaptive_weighting and HAS_ADAPTIVE_META:
            self.adaptive_meta_engine = AdaptiveMetaEngine(
                min_win_rate=0.50,
                min_sharpe=0.5,
                lookback_trades=100,
                reweight_frequency=50,
                use_meta_learning=True,
            )
            logger.info("adaptive_meta_engine_enabled")
        else:
            logger.warning("adaptive_meta_engine_not_available_or_disabled")

        logger.info(
            "alpha_engine_coordinator_initialized",
            num_engines=len(self.engines),
            use_bandit=use_bandit,
            use_parallel=use_parallel,
            use_adaptive_weighting=use_adaptive_weighting,
            max_workers=self.max_workers,
        )

    def generate_all_signals(
        self, features: Dict[str, float], current_regime: str, order_book_data: Optional[Dict] = None
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """
        Generate signals from all engines in parallel.
        
        Args:
            features: Market features
            current_regime: Current market regime
            order_book_data: Optional order book data for engines that need it
        
        Returns:
            Dict of {technique: signal}
        """
        # Generate signals from standard engines
        if self.use_parallel and self.executor:
            signals = self._generate_all_signals_parallel(features, current_regime, order_book_data)
        else:
            signals = self._generate_all_signals_sequential(features, current_regime, order_book_data)
        
        # Add AI-generated engine signals (they use adapters so they work with the standard interface)
        # Note: AI engines are integrated via adapters, so they can be called like standard engines
        # but we store them separately for tracking purposes
        ai_signals = self._generate_ai_engine_signals(features, current_regime)
        # AI engines don't map to TradingTechnique, so we'll incorporate them into existing signals
        # by taking the best AI signal and adding it to the appropriate technique
        
        return signals
    
    def _generate_all_signals_parallel(
        self, features: Dict[str, float], current_regime: str, order_book_data: Optional[Dict] = None
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """Generate signals from all engines in parallel using ThreadPoolExecutor."""
        signals = {}
        futures = {}
        
        start_time = time.time()
        
        # Submit all engine tasks
        for technique, engine in self.engines.items():
            future = self.executor.submit(
                self._run_engine_safe,
                engine=engine,
                technique=technique,
                features=features,
                current_regime=current_regime,
                order_book_data=order_book_data,
            )
            futures[future] = technique
        
        # Collect results
        for future in as_completed(futures):
            technique = futures[future]
            try:
                signal = future.result(timeout=5.0)  # 5 second timeout per engine
                if signal:
                    signals[technique] = signal
            except Exception as e:
                logger.warning(
                    "engine_signal_generation_failed",
                    technique=technique.value,
                    error=str(e),
                )
                # Create a hold signal on error
                signals[technique] = AlphaSignal(
                    technique=technique,
                    direction="hold",
                    confidence=0.0,
                    reasoning=f"Engine error: {str(e)}",
                    key_features={},
                    regime_affinity=0.0,
                )
        
        elapsed_time = time.time() - start_time
        logger.debug(
            "parallel_signal_generation_complete",
            num_engines=len(self.engines),
            num_signals=len(signals),
            elapsed_time_ms=elapsed_time * 1000,
        )
        
        return signals
    
    def _generate_all_signals_sequential(
        self, features: Dict[str, float], current_regime: str, order_book_data: Optional[Dict] = None
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """Generate signals from all engines sequentially (fallback)."""
        signals = {}
        
        start_time = time.time()
        
        for technique, engine in self.engines.items():
            try:
                signal = self._run_engine_safe(
                    engine=engine,
                    technique=technique,
                    features=features,
                    current_regime=current_regime,
                    order_book_data=order_book_data,
                )
                if signal:
                    signals[technique] = signal
            except Exception as e:
                logger.warning(
                    "engine_signal_generation_failed",
                    technique=technique.value,
                    error=str(e),
                )
                # Create a hold signal on error
                signals[technique] = AlphaSignal(
                    technique=technique,
                    direction="hold",
                    confidence=0.0,
                    reasoning=f"Engine error: {str(e)}",
                    key_features={},
                    regime_affinity=0.0,
                )
        
        elapsed_time = time.time() - start_time
        logger.debug(
            "sequential_signal_generation_complete",
            num_engines=len(self.engines),
            num_signals=len(signals),
            elapsed_time_ms=elapsed_time * 1000,
        )
        
        return signals
    
    def _run_engine_safe(
        self,
        engine: Any,
        technique: TradingTechnique,
        features: Dict[str, float],
        current_regime: str,
        order_book_data: Optional[Dict] = None,
    ) -> Optional[AlphaSignal]:
        """
        Safely run an engine and convert its signal to AlphaSignal format.
        
        Args:
            engine: Engine instance
            technique: Trading technique
            features: Market features
            current_regime: Current market regime
            order_book_data: Optional order book data
        
        Returns:
            AlphaSignal or None
        """
        try:
            # Different engines have different interfaces
            if technique == TradingTechnique.SCALPER and HAS_SCALPER:
                scalper_signal = engine.generate_signal(features, current_regime, order_book_data)
                return self._convert_scalper_signal(scalper_signal, technique)
            
            elif technique == TradingTechnique.FUNDING and HAS_FUNDING:
                funding_signal = engine.generate_signal(features, current_regime)
                return self._convert_funding_signal(funding_signal, technique)
            
            elif technique == TradingTechnique.FLOW_PREDICTION and HAS_FLOW_PREDICTION:
                flow_prediction = engine.predict_flow(features, current_regime, order_book_data)
                return self._convert_flow_prediction(flow_prediction, technique)
            
            elif technique == TradingTechnique.CORRELATION and HAS_CORRELATION:
                # Correlation engine needs two symbols - try to extract from features
                # If correlation features are available, use them
                if "correlation_pair" in features and "correlation_spread_bps" in features:
                    # Use correlation features if available
                    spread_bps = features.get("correlation_spread_bps", 0.0)
                    spread_zscore = features.get("correlation_spread_zscore", 0.0)
                    correlation = features.get("correlation", 0.0)
                    
                    # Generate signal based on spread
                    if abs(spread_zscore) > 1.0 and abs(correlation) > 0.7:
                        direction = "buy" if spread_zscore < -1.0 else "sell"
                        confidence = min(0.8, abs(spread_zscore) / 2.0)
                        return AlphaSignal(
                            technique=technique,
                            direction=direction,
                            confidence=confidence,
                            reasoning=f"Correlation spread detected (z={spread_zscore:.2f}, corr={correlation:.2f})",
                            key_features={"spread_bps": spread_bps, "spread_zscore": spread_zscore, "correlation": correlation},
                            regime_affinity=1.0 if current_regime == "RANGE" else 0.5,
                        )
                return None
            
            elif technique == TradingTechnique.LATENCY and HAS_LATENCY:
                # Latency engine needs symbol - try to extract from features
                # If latency features are available, use them
                if "latency_diff_ms" in features and "price_diff_bps" in features:
                    latency_diff = features.get("latency_diff_ms", 0.0)
                    price_diff = features.get("price_diff_bps", 0.0)
                    
                    # Generate signal based on latency arbitrage
                    if latency_diff > 10.0 and price_diff > 2.0:
                        direction = "buy" if price_diff > 0 else "sell"
                        confidence = min(0.8, (latency_diff / 100.0) * 0.5 + (price_diff / 50.0) * 0.5)
                        return AlphaSignal(
                            technique=technique,
                            direction=direction,
                            confidence=confidence,
                            reasoning=f"Latency arbitrage detected (latency_diff={latency_diff:.1f}ms, price_diff={price_diff:.1f}bps)",
                            key_features={"latency_diff_ms": latency_diff, "price_diff_bps": price_diff},
                            regime_affinity=1.0,  # Works in all regimes
                        )
                return None
            
            elif technique == TradingTechnique.MARKET_MAKER and HAS_MARKET_MAKER:
                # Market maker engine needs mid_price - try to extract from features
                mid_price = features.get("mid_price") or features.get("close") or features.get("price")
                if mid_price:
                    quote = engine.generate_quotes(
                        symbol="UNKNOWN",  # Symbol not available
                        mid_price=mid_price,
                        features=features,
                        current_regime=current_regime,
                    )
                    if quote:
                        # Convert quote to signal (buy if spread is wide enough)
                        if quote.spread_bps > 5.0:
                            direction = "buy"  # Provide liquidity
                            confidence = min(0.7, quote.confidence)
                            return AlphaSignal(
                                technique=technique,
                                direction=direction,
                                confidence=confidence,
                                reasoning=f"Market maker quote: {quote.reasoning}",
                                key_features=quote.key_features,
                                regime_affinity=1.0,  # Works in all regimes
                            )
                return None
            
            elif technique == TradingTechnique.REGIME and HAS_REGIME:
                # Regime detector returns regime, not signal - convert to signal
                # Use regime confidence as signal confidence
                # This is a meta-signal that indicates regime quality
                regime_confidence = features.get("regime_confidence", 0.5)
                regime_score = features.get("regime_score", 0.5)
                
                # High regime confidence = good market conditions = buy signal
                # Low regime confidence = uncertain = hold
                if regime_confidence > 0.7:
                    direction = "buy" if regime_score > 0.5 else "sell"
                    confidence = regime_confidence * 0.8  # Scale down
                    return AlphaSignal(
                        technique=technique,
                        direction=direction,
                        confidence=confidence,
                        reasoning=f"Strong regime detected (confidence={regime_confidence:.2f}, score={regime_score:.2f})",
                        key_features={"regime_confidence": regime_confidence, "regime_score": regime_score},
                        regime_affinity=1.0,
                    )
                return None
            
            else:
                # Standard engines with generate_signal method
                # Also handle AI-generated engines (they're wrapped in adapters)
                signal = engine.generate_signal(features, current_regime)
                return signal
        
        except Exception as e:
            logger.warning(
                "engine_execution_error",
                technique=technique.value if technique else "unknown",
                error=str(e),
            )
            return None
    
    def _generate_ai_engine_signals(
        self,
        features: Dict[str, float],
        current_regime: str
    ) -> Dict[str, AlphaSignal]:
        """
        Generate signals from AI-generated engines.
        
        Args:
            features: Market features
            current_regime: Current market regime
        
        Returns:
            Dict of {engine_name: signal}
        """
        ai_signals = {}
        
        for engine_name, adapter in self.ai_engines.items():
            try:
                signal = adapter.generate_signal(features, current_regime)
                if signal:
                    ai_signals[engine_name] = signal
            except Exception as e:
                logger.warning(
                    "ai_engine_signal_generation_failed",
                    engine_name=engine_name,
                    error=str(e)
                )
        
        return ai_signals
    
    def _convert_scalper_signal(self, scalper_signal: ScalperSignal, technique: TradingTechnique) -> AlphaSignal:
        """Convert ScalperSignal to AlphaSignal."""
        return AlphaSignal(
            technique=technique,
            direction=scalper_signal.direction,
            confidence=scalper_signal.confidence,
            reasoning=scalper_signal.reasoning,
            key_features=scalper_signal.key_features,
            regime_affinity=1.0,  # Scalper works in all regimes
        )
    
    def _convert_funding_signal(self, funding_signal: FundingSignal, technique: TradingTechnique) -> AlphaSignal:
        """Convert FundingSignal to AlphaSignal."""
        return AlphaSignal(
            technique=technique,
            direction=funding_signal.direction,
            confidence=funding_signal.confidence,
            reasoning=funding_signal.reasoning,
            key_features=funding_signal.key_features,
            regime_affinity=1.0,  # Funding works in all regimes
        )
    
    def _convert_flow_prediction(self, flow_prediction: FlowPrediction, technique: TradingTechnique) -> AlphaSignal:
        """Convert FlowPrediction to AlphaSignal."""
        return AlphaSignal(
            technique=technique,
            direction=flow_prediction.direction,
            confidence=flow_prediction.confidence,
            reasoning=flow_prediction.reasoning,
            key_features=flow_prediction.key_features,
            regime_affinity=1.0,  # Flow prediction works in all regimes
        )

    def generate_all_signals_batch(
        self,
        symbols_features: Dict[str, Dict[str, float]],
        current_regimes: Dict[str, str],
        order_book_data: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[TradingTechnique, AlphaSignal]]:
        """
        Generate signals for multiple symbols in batch (more efficient).
        
        Args:
            symbols_features: Dict of {symbol: features}
            current_regimes: Dict of {symbol: regime}
            order_book_data: Optional dict of {symbol: order_book_data}
            
        Returns:
            Dict of {symbol: {technique: signal}}
        """
        all_signals = {}
        
        # Process all symbols
        for symbol, features in symbols_features.items():
            regime = current_regimes.get(symbol, 'UNKNOWN')
            symbol_order_book = order_book_data.get(symbol) if order_book_data else None
            signals = self.generate_all_signals(features, regime, symbol_order_book)
            all_signals[symbol] = signals
        
        return all_signals

    def combine_signals(
        self, signals: Dict[TradingTechnique, AlphaSignal], current_regime: str = 'unknown'
    ) -> AlphaSignal:
        """
        Combine signals from all engines using weighted voting.
        
        Uses adaptive weighting if enabled, otherwise uses confidence-based weighting.
        
        Args:
            signals: Dict of {technique: signal}
            current_regime: Current market regime
        
        Returns:
            Combined AlphaSignal
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
        
        # Get engine weights (adaptive or default)
        if self.use_adaptive_weighting and self.adaptive_meta_engine:
            # Use adaptive meta-engine weights
            engine_weights = self.adaptive_meta_engine.get_engine_weights(current_regime)
            # Convert AdaptiveEngineType to TradingTechnique
            technique_weights = {}
            for tech, sig in active_signals.items():
                # Map TradingTechnique to AdaptiveEngineType
                adaptive_type = self._map_technique_to_adaptive_type(tech)
                if adaptive_type and adaptive_type in engine_weights:
                    technique_weights[tech] = engine_weights[adaptive_type]
                else:
                    # Default weight if not found
                    technique_weights[tech] = 1.0 / len(active_signals)
        else:
            # Use confidence-based weighting
            technique_weights = self._calculate_confidence_weights(active_signals, current_regime)
        
        # Weighted voting: combine signals by direction
        buy_votes = 0.0
        sell_votes = 0.0
        hold_votes = 0.0
        
        buy_confidence_sum = 0.0
        sell_confidence_sum = 0.0
        
        buy_reasons = []
        sell_reasons = []
        
        for technique, signal in active_signals.items():
            weight = technique_weights.get(technique, 0.0)
            weighted_confidence = signal.confidence * weight
            
            if signal.direction == "buy":
                buy_votes += weight
                buy_confidence_sum += weighted_confidence
                buy_reasons.append(f"{technique.value}: {signal.reasoning}")
            elif signal.direction == "sell":
                sell_votes += weight
                sell_confidence_sum += weighted_confidence
                sell_reasons.append(f"{technique.value}: {signal.reasoning}")
            else:
                hold_votes += weight
        
        # Determine final direction
        if buy_votes > sell_votes and buy_votes > hold_votes:
            direction = "buy"
            confidence = buy_confidence_sum / buy_votes if buy_votes > 0 else 0.0
            reasoning = f"Weighted buy consensus ({buy_votes:.2f} votes): " + "; ".join(buy_reasons[:3])
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            direction = "sell"
            confidence = sell_confidence_sum / sell_votes if sell_votes > 0 else 0.0
            reasoning = f"Weighted sell consensus ({sell_votes:.2f} votes): " + "; ".join(sell_reasons[:3])
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = "No clear consensus among engines"
        
        # Use bandit if enabled (for additional confidence adjustment)
        if self.use_bandit and self.bandit and active_signals:
            try:
                best_technique_bandit, best_signal_bandit, bandit_confidence = self.bandit.select_engine(
                    current_regime=current_regime,
                    all_signals=signals,
                )
                # Blend bandit confidence with weighted confidence
                if direction != "hold":
                    confidence = (confidence + bandit_confidence) / 2.0
            except Exception as e:
                logger.warning("bandit_selection_failed", error=str(e))
        
        # Combine key features from top signals
        combined_features = {}
        avg_regime_affinity = 0.0
        best_technique = TradingTechnique.TREND
        
        if active_signals and technique_weights:
            top_signals = sorted(active_signals.items(), key=lambda x: technique_weights.get(x[0], 0.0), reverse=True)[:3]
            for technique, signal in top_signals:
                combined_features.update(signal.key_features)
            
            # Calculate average regime affinity
            regime_affinities = [sig.regime_affinity for sig in active_signals.values()]
            if regime_affinities:
                avg_regime_affinity = float(np.mean(regime_affinities))
            
            # Select most weighted technique as representative
            if technique_weights:
                best_technique = max(technique_weights.items(), key=lambda x: x[1])[0]
        
        logger.debug(
            "signals_combined",
            direction=direction,
            confidence=confidence,
            num_active=len(active_signals),
            buy_votes=buy_votes,
            sell_votes=sell_votes,
            best_technique=best_technique.value,
        )
        
        return AlphaSignal(
            technique=best_technique,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_features=combined_features,
            regime_affinity=avg_regime_affinity,
        )
    
    def select_best_technique(
        self, signals: Dict[TradingTechnique, AlphaSignal], current_regime: str = 'unknown'
    ) -> AlphaSignal:
        """
        Select best technique (legacy method - now uses combine_signals).
        
        For backward compatibility, this method now calls combine_signals.
        """
        return self.combine_signals(signals, current_regime)
    
    def _calculate_confidence_weights(
        self, active_signals: Dict[TradingTechnique, AlphaSignal], current_regime: str
    ) -> Dict[TradingTechnique, float]:
        """Calculate weights based on confidence, regime affinity, and historical performance."""
        weights = {}
        total_score = 0.0
        
        for technique, signal in active_signals.items():
            # Get historical win rate for this engine
            if self.engine_performance[technique]:
                hist_perf = np.mean(self.engine_performance[technique][-20:])  # Recent 20
            else:
                hist_perf = 0.55  # Assume 55% default
            
            # Score: confidence * regime_affinity * historical_performance
            score = signal.confidence * signal.regime_affinity * hist_perf
            weights[technique] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for technique in weights:
                weights[technique] = weights[technique] / total_score
        else:
            # Equal weights if all scores are zero
            n_engines = len(weights)
            for technique in weights:
                weights[technique] = 1.0 / n_engines if n_engines > 0 else 0.0
        
        return weights
    
    def _map_technique_to_adaptive_type(self, technique: TradingTechnique) -> Optional[Any]:
        """Map TradingTechnique to AdaptiveEngineType."""
        if not HAS_ADAPTIVE_META:
            return None
        
        mapping = {
            TradingTechnique.TREND: AdaptiveEngineType.TREND,
            TradingTechnique.RANGE: AdaptiveEngineType.RANGE,
            TradingTechnique.BREAKOUT: AdaptiveEngineType.BREAKOUT,
            TradingTechnique.TAPE: AdaptiveEngineType.TAPE,
            TradingTechnique.LEADER: AdaptiveEngineType.LEADER,
            TradingTechnique.SWEEP: AdaptiveEngineType.SWEEP,
            TradingTechnique.SCALPER: AdaptiveEngineType.SCALPER,
            TradingTechnique.VOLATILITY: AdaptiveEngineType.VOLATILITY,
            TradingTechnique.FUNDING: AdaptiveEngineType.FUNDING,
            TradingTechnique.ARBITRAGE: AdaptiveEngineType.ARBITRAGE,
            TradingTechnique.CORRELATION: AdaptiveEngineType.CORRELATION,
            TradingTechnique.FLOW_PREDICTION: AdaptiveEngineType.FLOW_PREDICTION,
            TradingTechnique.LATENCY: AdaptiveEngineType.LATENCY,
            TradingTechnique.MARKET_MAKER: AdaptiveEngineType.MARKET_MAKER,
        }
        
        return mapping.get(technique)

    def update_engine_performance(
        self, technique: TradingTechnique, performance: float, regime: str = 'unknown', won: bool = False, profit_bps: float = 0.0
    ) -> None:
        """
        Update performance tracking for an engine.
        
        Args:
            technique: Trading technique
            performance: Performance metric (0-1)
            regime: Market regime
            won: Whether the trade won (for bandit)
            profit_bps: Profit in basis points (for adaptive meta-engine)
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
        
        # Update adaptive meta-engine if enabled
        if self.use_adaptive_weighting and self.adaptive_meta_engine:
            adaptive_type = self._map_technique_to_adaptive_type(technique)
            if adaptive_type:
                self.adaptive_meta_engine.update_engine_performance(
                    engine_type=adaptive_type,
                    trade_result={
                        "won": won,
                        "profit_bps": profit_bps,
                        "regime": regime,
                        "performance": performance,
                    },
                )

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
    
    def shutdown(self) -> None:
        """Shutdown coordinator and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("alpha_engine_coordinator_shutdown")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.shutdown()
        return False
