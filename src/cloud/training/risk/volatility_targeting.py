"""
Volatility Targeting Position Sizing.

Implements:
- Target per trade risk using ATR or realized vol
- Cap single trade risk (0.5-1.5% of equity)
- Cap total exposure by coin and by factor
- Kelly fraction (divided by 4 at most)
- Max leverage and max inventory enforcement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizeResult:
    """Position sizing result."""
    size_usd: float
    size_units: float
    risk_usd: float
    risk_pct: float
    kelly_fraction: float
    volatility_adjusted: bool
    caps_applied: Dict[str, bool]
    metadata: Dict[str, any]


class VolatilityTargetingSizer:
    """
    Volatility targeting position sizer.
    
    Features:
    - Target risk using ATR or realized volatility
    - Single trade risk cap (0.5-1.5% of equity)
    - Total exposure caps (by coin, by factor)
    - Kelly fraction with conservative cap (divide by 4)
    - Max leverage and inventory limits
    """
    
    def __init__(
        self,
        equity_usd: float = 100000.0,
        target_risk_pct: float = 1.0,  # 1% of equity per trade
        max_single_trade_risk_pct: float = 1.5,  # Max 1.5% per trade
        min_single_trade_risk_pct: float = 0.5,  # Min 0.5% per trade
        max_single_coin_exposure_pct: float = 25.0,  # Max 25% in one coin
        max_leverage: float = 3.0,
        kelly_divisor: int = 4,  # Divide Kelly by 4 for safety
    ) -> None:
        """
        Initialize volatility targeting sizer.
        
        Args:
            equity_usd: Total equity in USD
            target_risk_pct: Target risk per trade as % of equity
            max_single_trade_risk_pct: Maximum risk per trade
            min_single_trade_risk_pct: Minimum risk per trade
            max_single_coin_exposure_pct: Maximum exposure to single coin
            max_leverage: Maximum leverage allowed
            kelly_divisor: Divisor for Kelly fraction (default: 4)
        """
        self.equity_usd = equity_usd
        self.target_risk_pct = target_risk_pct
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.min_single_trade_risk_pct = min_single_trade_risk_pct
        self.max_single_coin_exposure_pct = max_single_coin_exposure_pct
        self.max_leverage = max_leverage
        self.kelly_divisor = kelly_divisor
        
        # Track current positions
        self.current_positions: Dict[str, float] = {}  # symbol -> size_usd
        self.current_exposure_by_factor: Dict[str, float] = {}  # factor -> exposure_usd
        
        logger.info(
            "volatility_targeting_sizer_initialized",
            equity_usd=equity_usd,
            target_risk_pct=target_risk_pct,
            max_leverage=max_leverage
        )
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        atr: Optional[float] = None,
        realized_vol: Optional[float] = None,
        win_rate: float = 0.5,
        avg_win_pct: float = 0.02,
        avg_loss_pct: float = 0.01,
        confidence: float = 0.5,
    ) -> PositionSizeResult:
        """
        Calculate position size using volatility targeting.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            atr: Average True Range (optional)
            realized_vol: Realized volatility (optional)
            win_rate: Historical win rate
            avg_win_pct: Average win as % of entry
            avg_loss_pct: Average loss as % of entry
            confidence: Signal confidence (0 to 1)
        
        Returns:
            PositionSizeResult with size and metadata
        """
        # Step 1: Calculate volatility-based risk
        if atr is not None:
            # Use ATR for risk calculation
            volatility_risk = atr / entry_price  # As fraction
        elif realized_vol is not None:
            # Use realized volatility
            volatility_risk = realized_vol
        else:
            # Fallback: use stop loss distance
            volatility_risk = abs(entry_price - stop_loss_price) / entry_price
        
        # Step 2: Calculate target risk in USD
        target_risk_usd = self.equity_usd * (self.target_risk_pct / 100.0)
        
        # Step 3: Calculate base position size
        # Size = Risk / (Price * Volatility_Risk)
        base_size_units = target_risk_usd / (entry_price * volatility_risk)
        base_size_usd = base_size_units * entry_price
        
        # Step 4: Apply Kelly fraction (conservative)
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win_pct, avg_loss_pct)
        kelly_adjusted_size = base_size_usd * (kelly_fraction / self.kelly_divisor)
        
        # Step 5: Apply confidence adjustment
        confidence_adjusted_size = kelly_adjusted_size * confidence
        
        # Step 6: Apply caps
        final_size_usd, caps_applied = self._apply_caps(
            symbol,
            confidence_adjusted_size,
            entry_price
        )
        
        # Calculate final risk
        final_size_units = final_size_usd / entry_price
        final_risk_usd = final_size_units * abs(entry_price - stop_loss_price)
        final_risk_pct = (final_risk_usd / self.equity_usd) * 100.0
        
        return PositionSizeResult(
            size_usd=final_size_usd,
            size_units=final_size_units,
            risk_usd=final_risk_usd,
            risk_pct=final_risk_pct,
            kelly_fraction=kelly_fraction,
            volatility_adjusted=True,
            caps_applied=caps_applied,
            metadata={
                "base_size_usd": base_size_usd,
                "kelly_adjusted_size": kelly_adjusted_size,
                "confidence_adjusted_size": confidence_adjusted_size,
                "volatility_risk": volatility_risk,
                "target_risk_usd": target_risk_usd,
            }
        )
    
    def _calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly = (p * b - q) / b
        Where:
        - p = win rate
        - q = 1 - p (loss rate)
        - b = avg_win / avg_loss (odds)
        
        Args:
            win_rate: Win rate (0 to 1)
            avg_win_pct: Average win as % of entry
            avg_loss_pct: Average loss as % of entry
        
        Returns:
            Kelly fraction (0 to 1)
        """
        if avg_loss_pct <= 0:
            return 0.0
        
        p = win_rate
        q = 1.0 - p
        b = avg_win_pct / avg_loss_pct
        
        kelly = (p * b - q) / b
        
        # Clip to [0, 1]
        kelly = np.clip(kelly, 0.0, 1.0)
        
        return kelly
    
    def _apply_caps(
        self,
        symbol: str,
        size_usd: float,
        entry_price: float
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Apply all position size caps.
        
        Args:
            symbol: Trading symbol
            size_usd: Proposed size in USD
            entry_price: Entry price
        
        Returns:
            (final_size_usd, caps_applied_dict)
        """
        caps_applied = {}
        final_size = size_usd
        
        # Cap 1: Single trade risk cap
        max_risk_usd = self.equity_usd * (self.max_single_trade_risk_pct / 100.0)
        # Assuming 2% stop loss
        max_size_from_risk = max_risk_usd / (entry_price * 0.02)
        if final_size > max_size_from_risk:
            final_size = max_size_from_risk
            caps_applied["max_single_trade_risk"] = True
        
        # Cap 2: Single coin exposure cap
        current_coin_exposure = self.current_positions.get(symbol, 0.0)
        max_coin_exposure = self.equity_usd * (self.max_single_coin_exposure_pct / 100.0)
        max_additional_exposure = max_coin_exposure - current_coin_exposure
        
        if final_size > max_additional_exposure:
            final_size = max_additional_exposure
            caps_applied["max_single_coin_exposure"] = True
        
        # Cap 3: Max leverage
        total_exposure = sum(self.current_positions.values()) + final_size
        max_total_exposure = self.equity_usd * self.max_leverage
        
        if total_exposure > max_total_exposure:
            max_additional = max_total_exposure - sum(self.current_positions.values())
            if max_additional > 0:
                final_size = min(final_size, max_additional)
                caps_applied["max_leverage"] = True
            else:
                final_size = 0.0
                caps_applied["max_leverage"] = True
        
        # Cap 4: Minimum trade size (if too small, reject)
        min_risk_usd = self.equity_usd * (self.min_single_trade_risk_pct / 100.0)
        min_size_from_risk = min_risk_usd / (entry_price * 0.02)
        if final_size < min_size_from_risk:
            final_size = 0.0  # Reject trade
            caps_applied["min_trade_size"] = True
        
        # Ensure non-negative
        final_size = max(0.0, final_size)
        
        return final_size, caps_applied
    
    def update_position(self, symbol: str, size_usd: float) -> None:
        """
        Update current position tracking.
        
        Args:
            symbol: Trading symbol
            size_usd: Position size in USD (0 to close)
        """
        if abs(size_usd) < 0.01:
            # Close position
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = size_usd
        
        logger.debug("position_updated", symbol=symbol, size_usd=size_usd)
    
    def get_total_exposure(self) -> float:
        """Get total current exposure in USD."""
        return sum(abs(v) for v in self.current_positions.values())
    
    def get_exposure_pct(self) -> float:
        """Get total exposure as % of equity."""
        return (self.get_total_exposure() / self.equity_usd) * 100.0

