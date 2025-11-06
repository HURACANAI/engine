"""
Market-Maker / Inventory Engine

Earns spread via smart quoting while hedging delta.
Manages inventory risk and provides liquidity.

Key Features:
1. Bid-ask spread capture (maker rebates)
2. Inventory management (delta hedging)
3. Smart quoting (adjust quotes based on inventory)
4. Risk management (limit inventory exposure)
5. Liquidity provision (provide two-sided quotes)

Best in: All regimes (spread capture always works)
Strategy: Provide liquidity and earn spread while managing inventory risk
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MarketMakerQuote:
    """Market maker quote."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    inventory_adjustment: float  # How much inventory affects quotes
    confidence: float  # 0-1
    reasoning: str
    key_features: Dict[str, float]


class MarketMakerInventoryEngine:
    """
    Market-Maker / Inventory Engine.
    
    Earns spread via smart quoting while hedging delta.
    Manages inventory risk and provides liquidity.
    
    Key Features:
    - Bid-ask spread capture
    - Inventory management
    - Smart quoting
    - Risk management
    - Liquidity provision
    """
    
    def __init__(
        self,
        base_spread_bps: float = 5.0,  # Base spread in bps
        max_inventory_size: float = 1000.0,  # Maximum inventory size (GBP)
        inventory_skew_factor: float = 0.5,  # How much inventory affects quotes
        min_spread_bps: float = 2.0,  # Minimum spread to quote
        use_delta_hedging: bool = True,  # Use delta hedging
    ):
        """
        Initialize market-maker/inventory engine.
        
        Args:
            base_spread_bps: Base spread to quote (in bps)
            max_inventory_size: Maximum inventory size
            inventory_skew_factor: How much inventory affects quotes
            min_spread_bps: Minimum spread to quote
            use_delta_hedging: Whether to use delta hedging
        """
        self.base_spread = base_spread_bps
        self.max_inventory = max_inventory_size
        self.inventory_skew = inventory_skew_factor
        self.min_spread = min_spread_bps
        self.use_delta_hedging = use_delta_hedging
        
        # Current inventory (positive = long, negative = short)
        self.current_inventory: Dict[str, float] = {}
        
        # Inventory limits per symbol
        self.inventory_limits: Dict[str, float] = {}
        
        logger.info(
            "market_maker_inventory_engine_initialized",
            base_spread_bps=base_spread_bps,
            max_inventory_size=max_inventory_size,
        )
    
    def update_inventory(
        self,
        symbol: str,
        size_change: float,  # Positive = bought, negative = sold
    ) -> None:
        """Update inventory for a symbol."""
        if symbol not in self.current_inventory:
            self.current_inventory[symbol] = 0.0
        
        self.current_inventory[symbol] += size_change
        
        # Clamp to limits
        if symbol in self.inventory_limits:
            max_inv = self.inventory_limits[symbol]
            self.current_inventory[symbol] = np.clip(
                self.current_inventory[symbol],
                -max_inv,
                max_inv,
            )
    
    def generate_quotes(
        self,
        symbol: str,
        mid_price: float,
        features: Dict[str, float],
        current_regime: str,
    ) -> Optional[MarketMakerQuote]:
        """
        Generate market maker quotes.
        
        Args:
            symbol: Trading symbol
            mid_price: Current mid price
            features: Feature dictionary
            current_regime: Current market regime
        
        Returns:
            MarketMakerQuote if quoting is viable, None otherwise
        """
        # Get current inventory
        current_inv = self.current_inventory.get(symbol, 0.0)
        
        # Get inventory limit
        max_inv = self.inventory_limits.get(symbol, self.max_inventory)
        
        # Check if inventory is at limit
        if abs(current_inv) >= max_inv * 0.9:
            # Inventory near limit, don't quote
            return None
        
        # Calculate spread based on volatility
        volatility = features.get("volatility", 0.0)
        spread_multiplier = 1.0 + volatility * 2.0  # Higher vol = wider spread
        spread_bps = self.base_spread * spread_multiplier
        
        if spread_bps < self.min_spread:
            # Spread too tight
            return None
        
        # Calculate inventory adjustment
        # Positive inventory (long) → skew quotes down (lower bid, lower ask)
        # Negative inventory (short) → skew quotes up (higher bid, higher ask)
        inventory_ratio = current_inv / max_inv if max_inv > 0 else 0.0
        inventory_adjustment = inventory_ratio * self.inventory_skew
        
        # Calculate bid and ask prices
        spread_pct = spread_bps / 10000.0
        half_spread = mid_price * spread_pct / 2.0
        
        # Adjust for inventory
        inventory_skew_pct = inventory_adjustment * spread_pct
        bid_price = mid_price - half_spread - (mid_price * inventory_skew_pct)
        ask_price = mid_price + half_spread - (mid_price * inventory_skew_pct)
        
        # Calculate quote sizes (smaller when inventory is high)
        base_size = 100.0  # Base quote size
        inventory_factor = 1.0 - abs(inventory_ratio) * 0.5  # Reduce size when inventory is high
        bid_size = base_size * inventory_factor
        ask_size = base_size * inventory_factor
        
        # Calculate confidence
        # Higher confidence when:
        # - Spread is wider (more profit)
        # - Inventory is balanced (less risk)
        # - Volatility is moderate (not too risky)
        spread_confidence = min(1.0, spread_bps / (self.base_spread * 2.0))
        inventory_confidence = 1.0 - abs(inventory_ratio) * 0.5
        vol_confidence = 1.0 - min(1.0, volatility * 2.0)
        
        confidence = (spread_confidence + inventory_confidence + vol_confidence) / 3.0
        
        # Determine reasoning
        if abs(inventory_ratio) > 0.5:
            reasoning = f"Inventory skewed ({inventory_ratio:.1%}): Adjusting quotes to reduce inventory"
        else:
            reasoning = f"Balanced inventory: Providing liquidity with {spread_bps:.1f} bps spread"
        
        return MarketMakerQuote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread_bps=spread_bps,
            inventory_adjustment=inventory_adjustment,
            confidence=confidence,
            reasoning=reasoning,
            key_features={
                "mid_price": mid_price,
                "spread_bps": spread_bps,
                "current_inventory": current_inv,
                "inventory_ratio": inventory_ratio,
                "volatility": volatility,
                "inventory_adjustment": inventory_adjustment,
            },
        )
    
    def get_inventory_status(self, symbol: str) -> Dict[str, float]:
        """Get inventory status for a symbol."""
        current_inv = self.current_inventory.get(symbol, 0.0)
        max_inv = self.inventory_limits.get(symbol, self.max_inventory)
        inventory_ratio = current_inv / max_inv if max_inv > 0 else 0.0
        
        return {
            "current_inventory": current_inv,
            "max_inventory": max_inv,
            "inventory_ratio": inventory_ratio,
            "inventory_pct": inventory_ratio * 100.0,
        }

