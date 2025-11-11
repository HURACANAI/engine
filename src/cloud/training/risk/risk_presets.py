"""
Risk Preset System

Defines and enforces risk presets (conservative, balanced, aggressive) across
all simulations and live trading. Ensures consistent risk management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class RiskPreset(str, Enum):
    """Risk preset levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskLimits:
    """Risk limits for a preset."""
    per_trade_risk_pct: float  # Risk per trade as % of equity
    daily_loss_limit_pct: float  # Daily loss limit as % of equity
    max_leverage: float  # Maximum leverage
    min_confidence: float  # Minimum confidence to trade
    max_position_size_pct: float  # Maximum position size as % of equity
    max_daily_trades: int  # Maximum trades per day


class RiskPresetManager:
    """
    Risk preset manager.
    
    Features:
    - Three preset levels (conservative, balanced, aggressive)
    - Configurable limits per preset
    - Enforcement in all simulations
    - Real-time risk monitoring
    """
    
    def __init__(
        self,
        presets: Optional[Dict[RiskPreset, RiskLimits]] = None,
        default_preset: RiskPreset = RiskPreset.BALANCED,
    ) -> None:
        """
        Initialize risk preset manager.
        
        Args:
            presets: Custom preset configurations (uses defaults if None)
            default_preset: Default preset to use
        """
        self.presets = presets or self._default_presets()
        self.current_preset = default_preset
        
        logger.info(
            "risk_preset_manager_initialized",
            current_preset=default_preset.value,
            num_presets=len(self.presets),
        )
    
    def _default_presets(self) -> Dict[RiskPreset, RiskLimits]:
        """Get default risk presets."""
        return {
            RiskPreset.CONSERVATIVE: RiskLimits(
                per_trade_risk_pct=0.5,
                daily_loss_limit_pct=1.5,
                max_leverage=1.5,
                min_confidence=0.65,
                max_position_size_pct=10.0,
                max_daily_trades=50,
            ),
            RiskPreset.BALANCED: RiskLimits(
                per_trade_risk_pct=1.0,
                daily_loss_limit_pct=2.5,
                max_leverage=2.0,
                min_confidence=0.55,
                max_position_size_pct=20.0,
                max_daily_trades=100,
            ),
            RiskPreset.AGGRESSIVE: RiskLimits(
                per_trade_risk_pct=1.5,
                daily_loss_limit_pct=4.0,
                max_leverage=3.0,
                min_confidence=0.50,
                max_position_size_pct=30.0,
                max_daily_trades=200,
            ),
        }
    
    def get_limits(self, preset: Optional[RiskPreset] = None) -> RiskLimits:
        """
        Get risk limits for a preset.
        
        Args:
            preset: Preset to get limits for (uses current if None)
        
        Returns:
            Risk limits
        """
        preset = preset or self.current_preset
        return self.presets.get(preset, self.presets[RiskPreset.BALANCED])
    
    def set_preset(self, preset: RiskPreset) -> None:
        """
        Set current risk preset.
        
        Args:
            preset: Preset to use
        """
        if preset not in self.presets:
            logger.warning("unknown_preset", preset=preset.value)
            return
        
        self.current_preset = preset
        logger.info("risk_preset_changed", preset=preset.value)
    
    def check_trade_allowed(
        self,
        confidence: float,
        position_size_pct: float,
        daily_loss_pct: float,
        daily_trade_count: int,
        current_leverage: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed under current risk limits.
        
        Args:
            confidence: Trade confidence (0.0 to 1.0)
            position_size_pct: Position size as % of equity
            daily_loss_pct: Daily loss so far as % of equity
            daily_trade_count: Number of trades today
            current_leverage: Current leverage
        
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        limits = self.get_limits()
        
        # Check confidence
        if confidence < limits.min_confidence:
            return False, f"Confidence {confidence:.2f} below minimum {limits.min_confidence:.2f}"
        
        # Check position size
        if position_size_pct > limits.max_position_size_pct:
            return False, f"Position size {position_size_pct:.2f}% exceeds maximum {limits.max_position_size_pct:.2f}%"
        
        # Check daily loss limit
        if daily_loss_pct >= limits.daily_loss_limit_pct:
            return False, f"Daily loss {daily_loss_pct:.2f}% exceeds limit {limits.daily_loss_limit_pct:.2f}%"
        
        # Check daily trade count
        if daily_trade_count >= limits.max_daily_trades:
            return False, f"Daily trade count {daily_trade_count} exceeds maximum {limits.max_daily_trades}"
        
        # Check leverage
        if current_leverage >= limits.max_leverage:
            return False, f"Leverage {current_leverage:.2f}x exceeds maximum {limits.max_leverage:.2f}x"
        
        return True, None
    
    def calculate_position_size(
        self,
        equity: float,
        stop_loss_pct: float,
    ) -> float:
        """
        Calculate position size based on risk limits.
        
        Args:
            equity: Account equity
            stop_loss_pct: Stop loss as % of entry price
        
        Returns:
            Position size in USD
        """
        limits = self.get_limits()
        
        # Risk per trade
        risk_amount = equity * (limits.per_trade_risk_pct / 100.0)
        
        # Position size = risk_amount / stop_loss_pct
        if stop_loss_pct > 0:
            position_size = risk_amount / (stop_loss_pct / 100.0)
        else:
            position_size = 0.0
        
        # Cap by max position size
        max_position_size = equity * (limits.max_position_size_pct / 100.0)
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def check_daily_limit_exceeded(self, daily_loss_pct: float) -> bool:
        """
        Check if daily loss limit is exceeded.
        
        Args:
            daily_loss_pct: Daily loss as % of equity
        
        Returns:
            True if limit exceeded
        """
        limits = self.get_limits()
        return daily_loss_pct >= limits.daily_loss_limit_pct

