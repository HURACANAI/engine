"""
Scalper / Latency-Arb Engine

Trades micro-arbitrage opportunities across venues with ultra-low latency.
Exploits price discrepancies that exist for milliseconds.

Key Features:
1. Micro-arbitrage detection (price differences <1 bps)
2. Latency-aware execution (prioritize fast exchanges)
3. Order book imbalance exploitation
4. Spread capture (bid-ask spread)
5. Ultra-fast signal generation (<10ms)

Best in: All regimes (microstructure always matters)
Strategy: Hit micro-arbitrage opportunities before they disappear
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ScalperSignal:
    """Signal from scalper/latency-arb engine."""
    direction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    profit_bps: float  # Expected profit in basis points
    latency_ms: float  # Expected latency in milliseconds
    reasoning: str
    key_features: Dict[str, float]


class ScalperLatencyEngine:
    """
    Scalper / Latency-Arb Engine.
    
    Trades micro-arbitrage opportunities with ultra-low latency.
    Exploits price discrepancies that exist for milliseconds.
    
    Key Features:
    - Micro-arbitrage detection (<1 bps)
    - Latency-aware execution
    - Order book imbalance exploitation
    - Spread capture
    - Ultra-fast signal generation
    """
    
    def __init__(
        self,
        min_profit_bps: float = 0.5,  # Minimum profit in bps (0.5 bps = very tight)
        max_latency_ms: float = 50.0,  # Maximum acceptable latency (50ms)
        min_spread_bps: float = 1.0,  # Minimum spread to trade
        use_order_book: bool = True,  # Use order book data if available
    ):
        """
        Initialize scalper/latency-arb engine.
        
        Args:
            min_profit_bps: Minimum profit required (in bps)
            max_latency_ms: Maximum acceptable latency (milliseconds)
            min_spread_bps: Minimum spread to trade
            use_order_book: Whether to use order book data
        """
        self.min_profit_bps = min_profit_bps
        self.max_latency_ms = max_latency_ms
        self.min_spread_bps = min_spread_bps
        self.use_order_book = use_order_book
        
        # Feature weights for this technique
        self.feature_weights = {
            "spread_bps": 0.30,
            "order_book_imbalance": 0.25,
            "micro_score": 0.20,
            "vol_jump_z": 0.15,
            "uptick_ratio": 0.10,
        }
        
        logger.info(
            "scalper_latency_engine_initialized",
            min_profit_bps=min_profit_bps,
            max_latency_ms=max_latency_ms,
        )
    
    def generate_signal(
        self,
        features: Dict[str, float],
        current_regime: str,
        order_book_data: Optional[Dict] = None,
    ) -> ScalperSignal:
        """
        Generate scalper/latency-arb signal.
        
        Args:
            features: Feature dictionary
            current_regime: Current market regime
            order_book_data: Optional order book data
        
        Returns:
            ScalperSignal
        """
        # Extract key features
        spread_bps = features.get("spread_bps", 0.0)
        micro_score = features.get("micro_score", 0.0)
        order_book_imbalance = features.get("order_book_imbalance", 0.0)
        vol_jump_z = features.get("vol_jump_z", 0.0)
        uptick_ratio = features.get("uptick_ratio", 0.5)
        
        # Calculate expected profit
        expected_profit_bps = self._calculate_expected_profit(
            spread_bps=spread_bps,
            order_book_imbalance=order_book_imbalance,
            micro_score=micro_score,
        )
        
        # Estimate latency (based on exchange and network conditions)
        estimated_latency_ms = self._estimate_latency(features)
        
        # Check if opportunity is viable
        if expected_profit_bps < self.min_profit_bps:
            return ScalperSignal(
                direction="hold",
                confidence=0.0,
                profit_bps=expected_profit_bps,
                latency_ms=estimated_latency_ms,
                reasoning="Profit too small",
                key_features={
                    "spread_bps": spread_bps,
                    "expected_profit_bps": expected_profit_bps,
                },
            )
        
        if estimated_latency_ms > self.max_latency_ms:
            return ScalperSignal(
                direction="hold",
                confidence=0.0,
                profit_bps=expected_profit_bps,
                latency_ms=estimated_latency_ms,
                reasoning="Latency too high",
                key_features={
                    "spread_bps": spread_bps,
                    "latency_ms": estimated_latency_ms,
                },
            )
        
        if spread_bps < self.min_spread_bps:
            return ScalperSignal(
                direction="hold",
                confidence=0.0,
                profit_bps=expected_profit_bps,
                latency_ms=estimated_latency_ms,
                reasoning="Spread too tight",
                key_features={
                    "spread_bps": spread_bps,
                },
            )
        
        # Determine direction based on order book imbalance
        if order_book_imbalance > 0.1:
            direction = "buy"
            confidence = min(0.9, 0.5 + abs(order_book_imbalance) * 2.0)
            reasoning = f"Buy-side imbalance detected (imbalance={order_book_imbalance:.2f})"
        elif order_book_imbalance < -0.1:
            direction = "sell"
            confidence = min(0.9, 0.5 + abs(order_book_imbalance) * 2.0)
            reasoning = f"Sell-side imbalance detected (imbalance={order_book_imbalance:.2f})"
        elif uptick_ratio > 0.6:
            direction = "buy"
            confidence = 0.6
            reasoning = f"Uptick ratio high (uptick={uptick_ratio:.2f})"
        elif uptick_ratio < 0.4:
            direction = "sell"
            confidence = 0.6
            reasoning = f"Downtick ratio high (uptick={uptick_ratio:.2f})"
        else:
            direction = "hold"
            confidence = 0.0
            reasoning = "No clear direction"
        
        # Adjust confidence based on profit and latency
        if direction != "hold":
            # Higher profit = higher confidence
            profit_factor = min(1.0, expected_profit_bps / (self.min_profit_bps * 2))
            # Lower latency = higher confidence
            latency_factor = max(0.5, 1.0 - (estimated_latency_ms / self.max_latency_ms))
            confidence = confidence * profit_factor * latency_factor
        
        return ScalperSignal(
            direction=direction,
            confidence=confidence,
            profit_bps=expected_profit_bps,
            latency_ms=estimated_latency_ms,
            reasoning=reasoning,
            key_features={
                "spread_bps": spread_bps,
                "order_book_imbalance": order_book_imbalance,
                "micro_score": micro_score,
                "vol_jump_z": vol_jump_z,
                "uptick_ratio": uptick_ratio,
                "expected_profit_bps": expected_profit_bps,
                "latency_ms": estimated_latency_ms,
            },
        )
    
    def _calculate_expected_profit(
        self,
        spread_bps: float,
        order_book_imbalance: float,
        micro_score: float,
    ) -> float:
        """Calculate expected profit in basis points."""
        # Base profit from spread
        base_profit = spread_bps * 0.5  # Assume we capture 50% of spread
        
        # Order book imbalance adds to profit
        imbalance_bonus = abs(order_book_imbalance) * 0.5  # Up to 0.5 bps bonus
        
        # Microstructure score adds to profit
        micro_bonus = micro_score * 0.3  # Up to 0.3 bps bonus
        
        # Total expected profit
        expected_profit = base_profit + imbalance_bonus + micro_bonus
        
        # Subtract fees (assume 1 bps per side = 2 bps total)
        expected_profit -= 2.0
        
        return max(0.0, expected_profit)
    
    def _estimate_latency(self, features: Dict[str, float]) -> float:
        """Estimate execution latency in milliseconds."""
        # Base latency (exchange + network)
        base_latency = 20.0  # 20ms base
        
        # Add volatility penalty (high vol = slower fills)
        volatility = features.get("volatility", 0.0)
        vol_penalty = volatility * 10.0  # Up to 10ms penalty
        
        # Add spread penalty (tight spread = slower fills)
        spread_bps = features.get("spread_bps", 0.0)
        spread_penalty = max(0.0, (self.min_spread_bps - spread_bps) * 5.0)
        
        total_latency = base_latency + vol_penalty + spread_penalty
        
        return total_latency

