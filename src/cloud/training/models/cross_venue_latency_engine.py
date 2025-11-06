"""
Cross-Venue Latency Engine

Predicts which exchange will move first and pre-positions liquidity.
Exploits latency differences between exchanges.

Key Features:
1. Exchange latency measurement (ping times, order latency)
2. Price movement prediction (which exchange moves first)
3. Pre-positioning strategy (place orders on fast exchanges)
4. Cross-venue arbitrage detection
5. Latency-aware execution

Best in: All regimes (latency always matters)
Strategy: Predict which exchange moves first and trade ahead
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LatencyPrediction:
    """Latency prediction for cross-venue trading."""
    fastest_exchange: str  # Exchange that will move first
    confidence: float  # 0-1
    expected_latency_ms: float  # Expected latency in milliseconds
    price_movement_bps: float  # Expected price movement in bps
    recommended_action: str  # "buy", "sell", "hold"
    reasoning: str
    key_features: Dict[str, float]


class CrossVenueLatencyEngine:
    """
    Cross-Venue Latency Engine.
    
    Predicts which exchange will move first and pre-positions liquidity.
    Exploits latency differences between exchanges.
    
    Key Features:
    - Exchange latency measurement
    - Price movement prediction
    - Pre-positioning strategy
    - Cross-venue arbitrage detection
    - Latency-aware execution
    """
    
    def __init__(
        self,
        exchanges: List[str] = ["binance", "coinbase", "kraken"],
        min_latency_diff_ms: float = 10.0,  # Minimum latency difference to exploit
        min_price_diff_bps: float = 2.0,  # Minimum price difference to trade
    ):
        """
        Initialize cross-venue latency engine.
        
        Args:
            exchanges: List of exchanges to monitor
            min_latency_diff_ms: Minimum latency difference to exploit
            min_price_diff_bps: Minimum price difference to trade
        """
        self.exchanges = exchanges
        self.min_latency_diff = min_latency_diff_ms
        self.min_price_diff = min_price_diff_bps
        
        # Exchange latency tracking
        self.exchange_latencies: Dict[str, List[float]] = {
            exchange: [] for exchange in exchanges
        }
        
        # Exchange price tracking
        self.exchange_prices: Dict[str, Dict[str, float]] = {
            exchange: {} for exchange in exchanges
        }
        
        logger.info(
            "cross_venue_latency_engine_initialized",
            exchanges=exchanges,
            min_latency_diff_ms=min_latency_diff_ms,
        )
    
    def update_latency(
        self,
        exchange: str,
        latency_ms: float,
    ) -> None:
        """Update latency measurement for an exchange."""
        if exchange not in self.exchange_latencies:
            self.exchange_latencies[exchange] = []
        
        self.exchange_latencies[exchange].append(latency_ms)
        
        # Keep only last 100 measurements
        if len(self.exchange_latencies[exchange]) > 100:
            self.exchange_latencies[exchange] = self.exchange_latencies[exchange][-100:]
    
    def update_price(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
    ) -> None:
        """Update price data for an exchange."""
        if exchange not in self.exchange_prices:
            self.exchange_prices[exchange] = {}
        
        self.exchange_prices[exchange][symbol] = {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "timestamp": datetime.utcnow(),
        }
    
    def predict_fastest_exchange(
        self,
        symbol: str,
    ) -> Optional[LatencyPrediction]:
        """
        Predict which exchange will move first.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            LatencyPrediction if prediction is possible, None otherwise
        """
        # Get average latency for each exchange
        exchange_avg_latencies = {}
        for exchange in self.exchanges:
            if exchange in self.exchange_latencies and len(self.exchange_latencies[exchange]) > 0:
                avg_latency = np.mean(self.exchange_latencies[exchange])
                exchange_avg_latencies[exchange] = avg_latency
        
        if len(exchange_avg_latencies) < 2:
            # Need at least 2 exchanges
            return None
        
        # Find fastest exchange
        fastest_exchange = min(exchange_avg_latencies.items(), key=lambda x: x[1])[0]
        fastest_latency = exchange_avg_latencies[fastest_exchange]
        
        # Find slowest exchange
        slowest_exchange = max(exchange_avg_latencies.items(), key=lambda x: x[1])[0]
        slowest_latency = exchange_avg_latencies[slowest_exchange]
        
        # Calculate latency difference
        latency_diff = slowest_latency - fastest_latency
        
        if latency_diff < self.min_latency_diff:
            # Latency difference too small
            return LatencyPrediction(
                fastest_exchange=fastest_exchange,
                confidence=0.0,
                expected_latency_ms=fastest_latency,
                price_movement_bps=0.0,
                recommended_action="hold",
                reasoning=f"Latency difference too small ({latency_diff:.1f} ms < {self.min_latency_diff:.1f} ms)",
                key_features={
                    "fastest_exchange": fastest_exchange,
                    "fastest_latency_ms": fastest_latency,
                    "latency_diff_ms": latency_diff,
                },
            )
        
        # Get price data for exchanges
        exchange_mids = {}
        for exchange in self.exchanges:
            if exchange in self.exchange_prices and symbol in self.exchange_prices[exchange]:
                mid = self.exchange_prices[exchange][symbol]["mid"]
                exchange_mids[exchange] = mid
        
        if len(exchange_mids) < 2:
            # Need price data from at least 2 exchanges
            return LatencyPrediction(
                fastest_exchange=fastest_exchange,
                confidence=0.0,
                expected_latency_ms=fastest_latency,
                price_movement_bps=0.0,
                recommended_action="hold",
                reasoning="Insufficient price data",
                key_features={
                    "fastest_exchange": fastest_exchange,
                },
            )
        
        # Calculate price differences
        price_diffs = {}
        for exchange, mid in exchange_mids.items():
            if exchange != fastest_exchange:
                price_diff_bps = abs((mid - exchange_mids[fastest_exchange]) / exchange_mids[fastest_exchange]) * 10000.0
                price_diffs[exchange] = price_diff_bps
        
        if not price_diffs:
            return None
        
        max_price_diff = max(price_diffs.values())
        max_diff_exchange = max(price_diffs.items(), key=lambda x: x[1])[0]
        
        if max_price_diff < self.min_price_diff:
            # Price difference too small
            return LatencyPrediction(
                fastest_exchange=fastest_exchange,
                confidence=0.0,
                expected_latency_ms=fastest_latency,
                price_movement_bps=max_price_diff,
                recommended_action="hold",
                reasoning=f"Price difference too small ({max_price_diff:.1f} bps < {self.min_price_diff:.1f} bps)",
                key_features={
                    "fastest_exchange": fastest_exchange,
                    "price_diff_bps": max_price_diff,
                },
            )
        
        # Determine recommended action
        if exchange_mids[fastest_exchange] < exchange_mids[max_diff_exchange]:
            # Fastest exchange has lower price → BUY on fastest, SELL on slowest
            recommended_action = "buy"
            confidence = min(0.9, 0.5 + (latency_diff / 100.0) * 0.4)  # Higher latency diff = higher confidence
            reasoning = f"Fastest exchange ({fastest_exchange}) has lower price: BUY on {fastest_exchange}, SELL on {max_diff_exchange}"
        else:
            # Fastest exchange has higher price → SELL on fastest, BUY on slowest
            recommended_action = "sell"
            confidence = min(0.9, 0.5 + (latency_diff / 100.0) * 0.4)
            reasoning = f"Fastest exchange ({fastest_exchange}) has higher price: SELL on {fastest_exchange}, BUY on {max_diff_exchange}"
        
        return LatencyPrediction(
            fastest_exchange=fastest_exchange,
            confidence=confidence,
            expected_latency_ms=fastest_latency,
            price_movement_bps=max_price_diff,
            recommended_action=recommended_action,
            reasoning=reasoning,
            key_features={
                "fastest_exchange": fastest_exchange,
                "fastest_latency_ms": fastest_latency,
                "latency_diff_ms": latency_diff,
                "price_diff_bps": max_price_diff,
                "max_diff_exchange": max_diff_exchange,
            },
        )

