"""
Dynamic Position Sizing

Determines optimal position size based on:
- Agent confidence
- Market volatility
- Current portfolio risk
- Win rate history
- Kelly Criterion

Traditional: Fixed position size (e.g., always $100)
Dynamic: Scale size based on edge and risk

Example:
- High confidence (0.9) + Low volatility → Large position
- Low confidence (0.6) + High volatility → Small position
- Near risk limit → Reduce all positions
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""

    base_position_size_gbp: float = 100.0  # Base position size
    max_position_size_gbp: float = 500.0  # Maximum position size
    min_position_size_gbp: float = 20.0  # Minimum position size
    kelly_fraction: float = 0.25  # Fractional Kelly (conservative)
    volatility_scaling: bool = True  # Scale by volatility
    confidence_scaling: bool = True  # Scale by agent confidence
    risk_budget_gbp: float = 1000.0  # Total risk budget
    max_portfolio_heat: float = 0.15  # Max 15% portfolio at risk


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation."""

    size_gbp: float  # Recommended position size
    leverage: float  # Recommended leverage (1.0 = no leverage)
    stop_loss_bps: float  # Recommended stop loss
    risk_gbp: float  # Amount at risk (size * stop_loss)
    confidence_factor: float  # Confidence scaling factor
    volatility_factor: float  # Volatility scaling factor
    kelly_factor: float  # Kelly criterion factor
    metadata: Dict  # Additional info


class DynamicPositionSizer:
    """
    Dynamic position sizing based on multiple factors.

    Adjusts position size dynamically based on:
    1. Agent confidence
    2. Market volatility
    3. Current portfolio heat
    4. Historical win rate and edge
    5. Kelly Criterion
    """

    def __init__(self, config: PositionSizingConfig):
        """
        Initialize position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config

        # Track portfolio heat
        self.current_heat_gbp = 0.0  # Current risk exposure
        self.open_positions: Dict[str, float] = {}  # Symbol → risk_gbp

        # Track performance for Kelly
        self.win_rate = 0.5  # Default 50%
        self.avg_win_bps = 100.0  # Default +100 bps
        self.avg_loss_bps = 50.0  # Default -50 bps
        self.total_trades = 0

        logger.info(
            "dynamic_position_sizer_initialized",
            base_size=config.base_position_size_gbp,
            kelly_fraction=config.kelly_fraction,
            risk_budget=config.risk_budget_gbp,
        )

    def calculate_position_size(
        self,
        symbol: str,
        confidence: float,
        volatility: float,
        stop_loss_bps: float,
        expected_return_bps: float,
        current_price: float,
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size.

        Args:
            symbol: Asset symbol
            confidence: Agent confidence (0-1)
            volatility: Asset volatility (annualized)
            stop_loss_bps: Stop loss in basis points
            expected_return_bps: Expected return in basis points
            current_price: Current asset price

        Returns:
            Position size recommendation
        """
        # Start with base size
        size = self.config.base_position_size_gbp

        # Factor 1: Confidence scaling
        if self.config.confidence_scaling:
            confidence_factor = self._confidence_scaling_factor(confidence)
            size *= confidence_factor
        else:
            confidence_factor = 1.0

        # Factor 2: Volatility scaling
        if self.config.volatility_scaling:
            volatility_factor = self._volatility_scaling_factor(volatility)
            size *= volatility_factor
        else:
            volatility_factor = 1.0

        # Factor 3: Kelly criterion
        kelly_factor = self._kelly_criterion(expected_return_bps, stop_loss_bps)
        size *= kelly_factor

        # Factor 4: Portfolio heat constraint
        heat_factor = self._heat_scaling_factor(symbol, size, stop_loss_bps)
        size *= heat_factor

        # Factor 5: Fear & Greed Index adjustment
        try:
            from src.cloud.training.analysis.fear_greed_index import FearGreedIndex
            fg_index = FearGreedIndex()
            fear_greed_data = fg_index.get_current_index()
            fg_multiplier = fg_index.get_position_size_multiplier(fear_greed_data)
            size *= fg_multiplier
            logger.debug("fear_greed_position_adjustment", multiplier=fg_multiplier, fear_greed_value=fear_greed_data.value)
        except Exception as e:
            logger.debug("fear_greed_index_not_available", error=str(e))

        # Enforce limits
        size = np.clip(size, self.config.min_position_size_gbp, self.config.max_position_size_gbp)

        # Calculate risk
        risk_gbp = size * (stop_loss_bps / 10000.0)  # bps to fraction

        # Determine leverage (default 1.0 = no leverage)
        leverage = 1.0
        if volatility < 0.3:  # Low volatility → can use leverage
            leverage = min(2.0, 1.0 / (volatility + 0.1))

        recommendation = PositionSizeRecommendation(
            size_gbp=size,
            leverage=leverage,
            stop_loss_bps=stop_loss_bps,
            risk_gbp=risk_gbp,
            confidence_factor=confidence_factor,
            volatility_factor=volatility_factor,
            kelly_factor=kelly_factor,
            metadata={
                "symbol": symbol,
                "heat_factor": heat_factor,
                "current_portfolio_heat": self.current_heat_gbp,
                "available_heat": self.config.risk_budget_gbp - self.current_heat_gbp,
            },
        )

        logger.debug(
            "position_size_calculated",
            symbol=symbol,
            size=size,
            risk=risk_gbp,
            confidence_factor=confidence_factor,
            volatility_factor=volatility_factor,
            kelly_factor=kelly_factor,
        )

        return recommendation

    def _confidence_scaling_factor(self, confidence: float) -> float:
        """
        Scale position by agent confidence.

        Confidence 0.5 → 0.5x size (reduce to half)
        Confidence 0.7 → 1.0x size (base)
        Confidence 0.9 → 1.5x size (increase)
        """
        # Linear scaling: 0.5 → 0.5, 0.7 → 1.0, 0.9 → 1.5
        # Formula: factor = (confidence - 0.5) * 2.5 + 0.5
        factor = (confidence - 0.5) * 2.5 + 0.5
        return np.clip(factor, 0.3, 2.0)  # Limit to [0.3, 2.0]

    def _volatility_scaling_factor(self, volatility: float) -> float:
        """
        Scale position inversely by volatility.

        Low vol (0.2) → 1.5x size
        Med vol (0.4) → 1.0x size
        High vol (0.8) → 0.5x size

        Target: Keep risk constant across different volatilities.
        """
        # Inverse volatility scaling
        # Target volatility = 0.4 (baseline)
        target_vol = 0.4
        factor = target_vol / (volatility + 0.01)  # Add small epsilon

        return np.clip(factor, 0.3, 2.0)

    def _kelly_criterion(
        self,
        expected_return_bps: float,
        stop_loss_bps: float,
    ) -> float:
        """
        Kelly Criterion position sizing.

        Kelly fraction = (p * b - q) / b
        where:
        - p = win probability
        - q = 1 - p = loss probability
        - b = win/loss ratio

        Then apply fractional Kelly for safety (default 0.25).
        """
        if self.total_trades < 10:
            # Not enough data, use conservative sizing
            return 0.5

        # Win/loss ratio
        b = self.avg_win_bps / (self.avg_loss_bps + 1e-6)

        # Kelly fraction
        p = self.win_rate
        q = 1.0 - p

        kelly_f = (p * b - q) / b

        # Fractional Kelly (conservative)
        kelly_f *= self.config.kelly_fraction

        # Ensure non-negative
        kelly_f = max(0.0, kelly_f)

        # Adjust by expected return confidence
        # If expected return matches historical, use full Kelly
        # If expected return differs, adjust
        expected_b = abs(expected_return_bps) / (stop_loss_bps + 1e-6)
        adjustment = expected_b / (b + 1e-6)
        kelly_f *= np.clip(adjustment, 0.5, 1.5)

        return np.clip(kelly_f, 0.1, 2.0)

    def _heat_scaling_factor(
        self,
        symbol: str,
        proposed_size: float,
        stop_loss_bps: float,
    ) -> float:
        """
        Scale down if approaching portfolio heat limit.

        Portfolio heat = sum of all open position risks
        If heat is near limit, reduce new positions.
        """
        # Calculate proposed risk
        proposed_risk = proposed_size * (stop_loss_bps / 10000.0)

        # Total heat if we take this position
        total_heat = self.current_heat_gbp + proposed_risk

        # Available budget
        max_heat = self.config.risk_budget_gbp * self.config.max_portfolio_heat

        # If within budget, no scaling
        if total_heat <= max_heat:
            return 1.0

        # If exceeds budget, scale down
        available = max_heat - self.current_heat_gbp

        if available <= 0:
            logger.warning(
                "portfolio_heat_limit_reached",
                current_heat=self.current_heat_gbp,
                max_heat=max_heat,
            )
            return 0.0  # No capacity

        # Scale to fit available budget
        scale_factor = available / proposed_risk
        return np.clip(scale_factor, 0.0, 1.0)

    def add_position(self, symbol: str, size_gbp: float, stop_loss_bps: float) -> None:
        """
        Track new open position.

        Args:
            symbol: Asset symbol
            size_gbp: Position size
            stop_loss_bps: Stop loss
        """
        risk_gbp = size_gbp * (stop_loss_bps / 10000.0)
        self.open_positions[symbol] = risk_gbp
        self.current_heat_gbp += risk_gbp

        logger.debug(
            "position_added",
            symbol=symbol,
            size=size_gbp,
            risk=risk_gbp,
            total_heat=self.current_heat_gbp,
        )

    def close_position(self, symbol: str, pnl_bps: float) -> None:
        """
        Remove closed position and update statistics.

        Args:
            symbol: Asset symbol
            pnl_bps: Realized PnL in basis points
        """
        if symbol in self.open_positions:
            risk_gbp = self.open_positions[symbol]
            self.current_heat_gbp -= risk_gbp
            del self.open_positions[symbol]

            # Update win/loss statistics
            self._update_statistics(pnl_bps)

            logger.debug(
                "position_closed",
                symbol=symbol,
                pnl_bps=pnl_bps,
                remaining_heat=self.current_heat_gbp,
            )

    def _update_statistics(self, pnl_bps: float) -> None:
        """Update win rate and average win/loss."""
        self.total_trades += 1

        # Running average update
        alpha = 0.1  # Exponential smoothing factor

        if pnl_bps > 0:
            # Win
            self.win_rate = self.win_rate * (1 - alpha) + alpha
            self.avg_win_bps = self.avg_win_bps * (1 - alpha) + pnl_bps * alpha
        else:
            # Loss
            self.win_rate = self.win_rate * (1 - alpha)
            self.avg_loss_bps = self.avg_loss_bps * (1 - alpha) + abs(pnl_bps) * alpha

    def get_portfolio_heat(self) -> Dict[str, float]:
        """Get current portfolio heat statistics."""
        max_heat = self.config.risk_budget_gbp * self.config.max_portfolio_heat

        return {
            "current_heat_gbp": self.current_heat_gbp,
            "max_heat_gbp": max_heat,
            "heat_utilization": self.current_heat_gbp / max_heat if max_heat > 0 else 0.0,
            "available_heat_gbp": max_heat - self.current_heat_gbp,
            "num_open_positions": len(self.open_positions),
            "win_rate": self.win_rate,
            "avg_win_bps": self.avg_win_bps,
            "avg_loss_bps": self.avg_loss_bps,
            "total_trades": self.total_trades,
        }

    def reset_heat(self) -> None:
        """Reset portfolio heat (e.g., start of new trading day)."""
        self.open_positions.clear()
        self.current_heat_gbp = 0.0
        logger.info("portfolio_heat_reset")


def calculate_optimal_leverage(
    volatility: float,
    sharpe_ratio: float,
    max_leverage: float = 3.0,
    target_volatility: float = 0.15,
) -> float:
    """
    Calculate optimal leverage based on volatility and Sharpe ratio.

    Uses formula: leverage = target_vol / asset_vol * sqrt(sharpe)

    Args:
        volatility: Asset volatility
        sharpe_ratio: Expected Sharpe ratio
        max_leverage: Maximum allowed leverage
        target_volatility: Target portfolio volatility

    Returns:
        Optimal leverage
    """
    if volatility <= 0 or sharpe_ratio <= 0:
        return 1.0

    # Base leverage to hit target volatility
    base_leverage = target_volatility / volatility

    # Adjust by Sharpe (higher Sharpe → can use more leverage)
    sharpe_adjustment = np.sqrt(max(sharpe_ratio, 0.1))
    optimal_leverage = base_leverage * sharpe_adjustment

    # Enforce limits
    optimal_leverage = np.clip(optimal_leverage, 1.0, max_leverage)

    return optimal_leverage
