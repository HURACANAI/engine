"""
Smart Execution Router - Reduces market footprint.

Features:
- Venue routing with real-time spread and queue awareness
- TWAP (Time-Weighted Average Price) execution
- POV (Percentage of Volume) execution
- Post-only maker when spread is wide
- Iceberg orders for larger sizes
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VenueQuote:
    """Venue quote with spread and queue info."""
    venue: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    queue_position_bid: int
    queue_position_ask: int
    latency_ms: float
    reliability: float


@dataclass
class ExecutionPlan:
    """Execution plan for order."""
    venue: str
    order_type: str  # "market", "limit", "post_only", "iceberg", "twap", "pov"
    size_usd: float
    child_orders: List[Dict[str, any]]
    expected_slippage_bps: float
    expected_fill_time_seconds: float


class SmartExecutionRouter:
    """
    Smart execution router to reduce market footprint.
    
    Features:
    - Venue scoring based on spread, queue, latency
    - TWAP execution for persistent signals
    - POV execution for volume participation
    - Post-only maker orders when spread is wide
    - Iceberg orders for large sizes
    """
    
    def __init__(
        self,
        min_spread_for_maker_bps: float = 10.0,  # Use maker if spread > 10 bps
        twap_duration_minutes: int = 5,
        pov_percentage: float = 0.1,  # 10% of volume
        iceberg_visible_pct: float = 0.2,  # Show 20% of order
    ) -> None:
        """
        Initialize smart execution router.
        
        Args:
            min_spread_for_maker_bps: Minimum spread to use maker orders
            twap_duration_minutes: TWAP duration in minutes
            pov_percentage: POV percentage
            iceberg_visible_pct: Visible portion of iceberg order
        """
        self.min_spread_for_maker_bps = min_spread_for_maker_bps
        self.twap_duration_minutes = twap_duration_minutes
        self.pov_percentage = pov_percentage
        self.iceberg_visible_pct = iceberg_visible_pct
        
        logger.info("smart_execution_router_initialized")
    
    def score_venue(
        self,
        quote: VenueQuote,
        side: str,  # "buy" or "sell"
        size_usd: float
    ) -> float:
        """
        Score venue for execution.
        
        Considers:
        - Spread (tighter = better)
        - Queue position (better position = better)
        - Latency (lower = better)
        - Reliability (higher = better)
        - Size availability (enough size = better)
        
        Args:
            quote: Venue quote
            side: Order side
            size_usd: Order size
        
        Returns:
            Venue score (higher = better)
        """
        # Calculate spread
        spread_bps = ((quote.best_ask - quote.best_bid) / quote.best_bid) * 10000
        
        # Get relevant price and size
        if side == "buy":
            price = quote.best_ask
            size_available = quote.ask_size
            queue_position = quote.queue_position_ask
        else:
            price = quote.best_bid
            size_available = quote.bid_size
            queue_position = quote.queue_position_bid
        
        # Calculate size in units
        size_units = size_usd / price
        
        # Score components (normalized to 0-1)
        spread_score = 1.0 / (1.0 + spread_bps / 10.0)  # Lower spread = higher score
        queue_score = 1.0 / (1.0 + queue_position / 10.0)  # Better queue = higher score
        latency_score = 1.0 / (1.0 + quote.latency_ms / 100.0)  # Lower latency = higher score
        reliability_score = quote.reliability  # Already 0-1
        size_score = min(1.0, size_available / size_units)  # Enough size = 1.0
        
        # Weighted combination
        total_score = (
            0.3 * spread_score +
            0.2 * queue_score +
            0.2 * latency_score +
            0.15 * reliability_score +
            0.15 * size_score
        )
        
        return total_score
    
    def select_venue(
        self,
        quotes: List[VenueQuote],
        side: str,
        size_usd: float
    ) -> Optional[VenueQuote]:
        """
        Select best venue for execution.
        
        Args:
            quotes: List of venue quotes
            side: Order side
            size_usd: Order size
        
        Returns:
            Best venue quote or None
        """
        if not quotes:
            return None
        
        # Score all venues
        scored_venues = [
            (quote, self.score_venue(quote, side, size_usd))
            for quote in quotes
        ]
        
        # Sort by score
        scored_venues.sort(key=lambda x: x[1], reverse=True)
        
        best_quote, best_score = scored_venues[0]
        
        logger.debug(
            "venue_selected",
            venue=best_quote.venue,
            score=best_score,
            side=side,
            size_usd=size_usd
        )
        
        return best_quote
    
    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quotes: List[VenueQuote],
        signal_persistence: float = 0.5,  # 0 to 1, how persistent is signal
        avg_volume_usd: Optional[float] = None
    ) -> ExecutionPlan:
        """
        Create execution plan based on conditions.
        
        Args:
            symbol: Trading symbol
            side: Order side
            size_usd: Order size
            quotes: Available venue quotes
            signal_persistence: How persistent is signal (0-1)
            avg_volume_usd: Average volume for POV calculation
        
        Returns:
            ExecutionPlan
        """
        # Select best venue
        best_venue = self.select_venue(quotes, side, size_usd)
        if best_venue is None:
            raise ValueError("No suitable venue found")
        
        # Determine execution strategy
        spread_bps = ((best_venue.best_ask - best_venue.best_bid) / best_venue.best_bid) * 10000
        
        # Strategy selection
        if spread_bps > self.min_spread_for_maker_bps:
            # Wide spread: use post-only maker
            order_type = "post_only"
            child_orders = self._create_maker_order(symbol, side, size_usd, best_venue)
            expected_slippage = 0.0  # Maker orders have no slippage
            expected_fill_time = 60.0  # Assume 1 minute
        
        elif signal_persistence > 0.7 and avg_volume_usd:
            # Persistent signal: use TWAP
            order_type = "twap"
            child_orders = self._create_twap_orders(symbol, side, size_usd, best_venue)
            expected_slippage = spread_bps / 4.0  # Average of spread
            expected_fill_time = self.twap_duration_minutes * 60
        
        elif avg_volume_usd and size_usd > avg_volume_usd * 0.1:
            # Large order: use POV
            order_type = "pov"
            child_orders = self._create_pov_orders(symbol, side, size_usd, best_venue, avg_volume_usd)
            expected_slippage = spread_bps / 2.0
            expected_fill_time = (size_usd / (avg_volume_usd * self.pov_percentage)) * 3600  # Seconds
        
        elif size_usd > 50000:  # Large order
            # Use iceberg
            order_type = "iceberg"
            child_orders = self._create_iceberg_orders(symbol, side, size_usd, best_venue)
            expected_slippage = spread_bps / 3.0
            expected_fill_time = 300.0  # 5 minutes
        
        else:
            # Small order: market or limit
            order_type = "limit"
            child_orders = self._create_limit_order(symbol, side, size_usd, best_venue)
            expected_slippage = spread_bps / 2.0
            expected_fill_time = 30.0  # 30 seconds
        
        return ExecutionPlan(
            venue=best_venue.venue,
            order_type=order_type,
            size_usd=size_usd,
            child_orders=child_orders,
            expected_slippage_bps=expected_slippage,
            expected_fill_time_seconds=expected_fill_time
        )
    
    def _create_maker_order(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quote: VenueQuote
    ) -> List[Dict[str, any]]:
        """Create post-only maker order."""
        price = quote.best_bid if side == "buy" else quote.best_ask
        # Maker price: slightly better than best
        if side == "buy":
            price = quote.best_bid * 1.0001  # Slightly above bid
        else:
            price = quote.best_ask * 0.9999  # Slightly below ask
        
        return [{
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "price": price,
            "size_usd": size_usd,
            "post_only": True,
        }]
    
    def _create_twap_orders(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quote: VenueQuote
    ) -> List[Dict[str, any]]:
        """Create TWAP child orders."""
        num_orders = self.twap_duration_minutes  # One order per minute
        size_per_order = size_usd / num_orders
        interval_seconds = 60
        
        orders = []
        for i in range(num_orders):
            orders.append({
                "symbol": symbol,
                "side": side,
                "type": "limit",
                "size_usd": size_per_order,
                "delay_seconds": i * interval_seconds,
            })
        
        return orders
    
    def _create_pov_orders(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quote: VenueQuote,
        avg_volume_usd: float
    ) -> List[Dict[str, any]]:
        """Create POV (Percentage of Volume) orders."""
        # POV: participate at X% of volume
        volume_per_minute = avg_volume_usd / (24 * 60)  # Per minute
        target_volume_per_minute = volume_per_minute * self.pov_percentage
        
        # Estimate duration
        duration_minutes = int(size_usd / target_volume_per_minute) + 1
        
        orders = []
        for i in range(duration_minutes):
            orders.append({
                "symbol": symbol,
                "side": side,
                "type": "market",
                "size_usd": target_volume_per_minute,
                "delay_seconds": i * 60,
            })
        
        return orders
    
    def _create_iceberg_orders(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quote: VenueQuote
    ) -> List[Dict[str, any]]:
        """Create iceberg orders."""
        visible_size = size_usd * self.iceberg_visible_pct
        hidden_size = size_usd - visible_size
        
        # Create visible order
        orders = [{
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "size_usd": visible_size,
            "iceberg": True,
            "total_size_usd": size_usd,
            "hidden_size_usd": hidden_size,
        }]
        
        return orders
    
    def _create_limit_order(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        quote: VenueQuote
    ) -> List[Dict[str, any]]:
        """Create simple limit order."""
        price = quote.best_ask if side == "buy" else quote.best_bid
        
        return [{
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "price": price,
            "size_usd": size_usd,
        }]

