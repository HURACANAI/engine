"""
Risk Engine Presets - Pre-configured risk profiles.

Provides Conservative, Balanced, and Aggressive presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict
import structlog

logger = structlog.get_logger(__name__)


class RiskProfile(Enum):
    """Risk profile types."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskPreset:
    """Risk preset configuration."""
    profile: RiskProfile
    per_trade_risk_pct: float  # % of equity per trade
    daily_loss_stop_pct: float  # % of equity daily loss limit
    max_leverage: float
    single_coin_exposure_pct: float  # Max % in one coin
    sector_exposure_pct: float  # Max % in one sector
    max_total_open_risk_pct: float  # Sum of stop losses ≤ this


class RiskPresetManager:
    """
    Risk preset manager.
    
    Provides pre-configured risk profiles:
    - Conservative: Low risk, low leverage
    - Balanced: Moderate risk, moderate leverage
    - Aggressive: Higher risk, higher leverage
    """
    
    def __init__(self) -> None:
        """Initialize risk preset manager."""
        self.presets: Dict[RiskProfile, RiskPreset] = {
            RiskProfile.CONSERVATIVE: RiskPreset(
                profile=RiskProfile.CONSERVATIVE,
                per_trade_risk_pct=0.25,  # 0.25% per trade
                daily_loss_stop_pct=1.5,  # 1.5% daily loss stop
                max_leverage=1.5,  # 1.5x max leverage
                single_coin_exposure_pct=25.0,  # 25% max in one coin
                sector_exposure_pct=50.0,  # 50% max in one sector
                max_total_open_risk_pct=1.5,  # Total open risk ≤ daily loss stop
            ),
            RiskProfile.BALANCED: RiskPreset(
                profile=RiskProfile.BALANCED,
                per_trade_risk_pct=0.5,  # 0.5% per trade
                daily_loss_stop_pct=2.5,  # 2.5% daily loss stop
                max_leverage=2.0,  # 2x max leverage
                single_coin_exposure_pct=25.0,  # 25% max in one coin
                sector_exposure_pct=50.0,  # 50% max in one sector
                max_total_open_risk_pct=2.5,  # Total open risk ≤ daily loss stop
            ),
            RiskProfile.AGGRESSIVE: RiskPreset(
                profile=RiskProfile.AGGRESSIVE,
                per_trade_risk_pct=1.0,  # 1.0% per trade
                daily_loss_stop_pct=3.5,  # 3.5% daily loss stop
                max_leverage=3.0,  # 3x max leverage
                single_coin_exposure_pct=25.0,  # 25% max in one coin
                sector_exposure_pct=50.0,  # 50% max in one sector
                max_total_open_risk_pct=3.5,  # Total open risk ≤ daily loss stop
            ),
        }
        
        logger.info("risk_preset_manager_initialized", presets=list(self.presets.keys()))
    
    def get_preset(self, profile: RiskProfile) -> RiskPreset:
        """
        Get risk preset configuration.
        
        Args:
            profile: Risk profile type
        
        Returns:
            RiskPreset configuration
        """
        if profile not in self.presets:
            raise ValueError(f"Unknown risk profile: {profile}")
        
        return self.presets[profile]
    
    def apply_preset(
        self,
        profile: RiskProfile,
        equity_usd: float
    ) -> Dict[str, float]:
        """
        Apply risk preset and return configuration dictionary.
        
        Args:
            profile: Risk profile type
            equity_usd: Total equity in USD
        
        Returns:
            Configuration dictionary for risk components
        """
        preset = self.get_preset(profile)
        
        config = {
            "equity_usd": equity_usd,
            "per_trade_risk_pct": preset.per_trade_risk_pct,
            "daily_loss_stop_pct": preset.daily_loss_stop_pct,
            "max_leverage": preset.max_leverage,
            "single_coin_exposure_pct": preset.single_coin_exposure_pct,
            "sector_exposure_pct": preset.sector_exposure_pct,
            "max_total_open_risk_pct": preset.max_total_open_risk_pct,
            # Calculated values
            "per_trade_risk_usd": equity_usd * (preset.per_trade_risk_pct / 100.0),
            "daily_loss_stop_usd": equity_usd * (preset.daily_loss_stop_pct / 100.0),
            "single_coin_exposure_usd": equity_usd * (preset.single_coin_exposure_pct / 100.0),
            "sector_exposure_usd": equity_usd * (preset.sector_exposure_pct / 100.0),
            "max_total_open_risk_usd": equity_usd * (preset.max_total_open_risk_pct / 100.0),
        }
        
        logger.info(
            "risk_preset_applied",
            profile=profile.value,
            equity_usd=equity_usd,
            per_trade_risk_pct=preset.per_trade_risk_pct,
            daily_loss_stop_pct=preset.daily_loss_stop_pct,
            max_leverage=preset.max_leverage
        )
        
        return config
    
    def validate_risk_limits(
        self,
        profile: RiskProfile,
        current_positions: Dict[str, float],
        proposed_trade_risk: float,
        equity_usd: float
    ) -> tuple[bool, str]:
        """
        Validate that proposed trade doesn't violate risk limits.
        
        Args:
            profile: Risk profile
            current_positions: Current positions (symbol -> size_usd)
            proposed_trade_risk: Proposed trade risk in USD
            equity_usd: Total equity
        
        Returns:
            (is_valid, error_message)
        """
        preset = self.get_preset(profile)
        
        # Check per-trade risk
        max_per_trade_risk = equity_usd * (preset.per_trade_risk_pct / 100.0)
        if proposed_trade_risk > max_per_trade_risk:
            return False, f"Trade risk {proposed_trade_risk:.2f} exceeds per-trade limit {max_per_trade_risk:.2f}"
        
        # Check total open risk (sum of stop losses)
        total_open_risk = sum(
            abs(size) * 0.02 for size in current_positions.values()  # Assume 2% stop loss
        ) + proposed_trade_risk
        
        max_total_risk = equity_usd * (preset.max_total_open_risk_pct / 100.0)
        if total_open_risk > max_total_risk:
            return False, f"Total open risk {total_open_risk:.2f} exceeds limit {max_total_risk:.2f}"
        
        # Check single coin exposure
        # This would need coin information - simplified here
        max_single_coin = equity_usd * (preset.single_coin_exposure_pct / 100.0)
        for symbol, size in current_positions.items():
            if abs(size) > max_single_coin:
                return False, f"Position {symbol} size {size:.2f} exceeds single coin limit {max_single_coin:.2f}"
        
        # Check leverage
        total_exposure = sum(abs(size) for size in current_positions.values())
        max_exposure = equity_usd * preset.max_leverage
        if total_exposure > max_exposure:
            return False, f"Total exposure {total_exposure:.2f} exceeds max leverage {max_exposure:.2f}"
        
        return True, "OK"

