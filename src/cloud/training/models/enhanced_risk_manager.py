"""
Enhanced Risk Management Module

Advanced position sizing and risk management based on:
1. Confidence scores (higher confidence = larger size)
2. Volatility (higher volatility = smaller size)
3. Portfolio drawdown (reduce size in drawdowns)
4. Kelly Criterion for optimal sizing

Based on modern portfolio theory and Kelly betting.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""

    recommended_size_gbp: float
    size_multiplier: float  # Relative to base size (1.0 = base, 0.5 = half, etc.)
    confidence_factor: float  # Contribution from confidence (0-1)
    volatility_factor: float  # Contribution from volatility (0-1)
    drawdown_factor: float  # Contribution from drawdown (0-1)
    kelly_factor: float  # Contribution from Kelly Criterion (0-1)
    reasoning: str


class EnhancedRiskManager:
    """
    Advanced risk management with dynamic position sizing.

    Key features:
    1. Confidence-based sizing (more confident = larger position)
    2. Volatility-adjusted sizing (more volatile = smaller position)
    3. Drawdown-aware sizing (in drawdown = smaller positions)
    4. Kelly Criterion optimization
    """

    def __init__(
        self,
        base_position_size_gbp: float = 100.0,
        max_position_multiplier: float = 2.0,  # Max 2x base size
        min_position_multiplier: float = 0.25,  # Min 0.25x base size
        kelly_fraction: float = 0.25,  # Use 25% of full Kelly (Kelly / 4)
        volatility_target_bps: float = 200.0,  # Target 200 bps daily volatility
        max_portfolio_volatility: float = 0.03,  # Max 3% daily portfolio vol
    ):
        """
        Initialize enhanced risk manager.

        Args:
            base_position_size_gbp: Base position size
            max_position_multiplier: Maximum size multiplier
            min_position_multiplier: Minimum size multiplier
            kelly_fraction: Fraction of Kelly to use (for safety)
            volatility_target_bps: Target position volatility in basis points
            max_portfolio_volatility: Maximum portfolio volatility (daily)
        """
        self.base_size = base_position_size_gbp
        self.max_multiplier = max_position_multiplier
        self.min_multiplier = min_position_multiplier
        self.kelly_fraction = kelly_fraction
        self.volatility_target_bps = volatility_target_bps
        self.max_portfolio_vol = max_portfolio_volatility

        logger.info(
            "enhanced_risk_manager_initialized",
            base_size_gbp=base_position_size_gbp,
            max_multiplier=max_position_multiplier,
            kelly_fraction=kelly_fraction,
        )

    def calculate_position_size(
        self,
        confidence: float,
        asset_volatility_bps: float,
        current_drawdown_pct: float = 0.0,
        win_rate: Optional[float] = None,
        avg_win_pct: Optional[float] = None,
        avg_loss_pct: Optional[float] = None,
    ) -> PositionSizingResult:
        """
        Calculate optimal position size using multiple factors.

        Args:
            confidence: Trade confidence score (0-1)
            asset_volatility_bps: Asset volatility in basis points (daily)
            current_drawdown_pct: Current portfolio drawdown percentage
            win_rate: Historical win rate (0-1), for Kelly calculation
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage (positive number)

        Returns:
            PositionSizingResult with recommended size and reasoning
        """
        # 1. Confidence factor (linear scaling)
        confidence_factor = np.clip(confidence, 0.0, 1.0)

        # 2. Volatility factor (inverse relationship)
        # More volatile = smaller position
        if asset_volatility_bps > 0:
            volatility_factor = min(1.0, self.volatility_target_bps / asset_volatility_bps)
        else:
            volatility_factor = 1.0

        # 3. Drawdown factor (reduce size in drawdowns)
        # Drawdown > 10% starts reducing size
        if current_drawdown_pct <= 10.0:
            drawdown_factor = 1.0
        elif current_drawdown_pct >= 30.0:
            drawdown_factor = 0.5  # Half size at 30% drawdown
        else:
            # Linear interpolation between 10% and 30%
            drawdown_factor = 1.0 - (current_drawdown_pct - 10.0) / 40.0

        # 4. Kelly factor (if we have win rate data)
        kelly_factor = 1.0
        if win_rate is not None and avg_win_pct is not None and avg_loss_pct is not None:
            kelly_factor = self._calculate_kelly_fraction(win_rate, avg_win_pct, avg_loss_pct)

        # Combine all factors (multiplicative)
        combined_multiplier = confidence_factor * volatility_factor * drawdown_factor * kelly_factor

        # Clip to min/max bounds
        final_multiplier = np.clip(
            combined_multiplier,
            self.min_multiplier,
            self.max_multiplier,
        )

        recommended_size = self.base_size * final_multiplier

        # Generate reasoning
        reasoning_parts = []

        if confidence_factor < 0.6:
            reasoning_parts.append(f"Low confidence ({confidence_factor:.2f}) → reduced size")
        elif confidence_factor > 0.8:
            reasoning_parts.append(f"High confidence ({confidence_factor:.2f}) → increased size")

        if volatility_factor < 0.7:
            reasoning_parts.append(
                f"High volatility ({asset_volatility_bps:.0f}bps) → reduced size"
            )

        if drawdown_factor < 1.0:
            reasoning_parts.append(
                f"Portfolio drawdown ({current_drawdown_pct:.1f}%) → reduced size"
            )

        if kelly_factor < 0.8:
            reasoning_parts.append(f"Kelly criterion suggests smaller size")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Normal sizing"

        return PositionSizingResult(
            recommended_size_gbp=recommended_size,
            size_multiplier=final_multiplier,
            confidence_factor=confidence_factor,
            volatility_factor=volatility_factor,
            drawdown_factor=drawdown_factor,
            kelly_factor=kelly_factor,
            reasoning=reasoning,
        )

    def _calculate_kelly_fraction(
        self, win_rate: float, avg_win_pct: float, avg_loss_pct: float
    ) -> float:
        """
        Calculate Kelly fraction for position sizing.

        Kelly formula: f = (p * b - q) / b
        where:
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio

        Args:
            win_rate: Probability of winning (0-1)
            avg_win_pct: Average win size as percentage
            avg_loss_pct: Average loss size as percentage (positive number)

        Returns:
            Kelly fraction (0-1), adjusted by kelly_fraction parameter
        """
        if avg_loss_pct <= 0 or win_rate <= 0 or win_rate >= 1:
            return 1.0  # Invalid inputs, use full size

        # Win/loss ratio
        b = avg_win_pct / avg_loss_pct

        # Kelly formula
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b

        # Clip to reasonable range
        kelly = np.clip(kelly, 0.0, 1.0)

        # Apply fractional Kelly (for safety)
        fractional_kelly = kelly * self.kelly_fraction

        return float(fractional_kelly)

    def calculate_stop_loss(
        self,
        entry_price: float,
        asset_volatility_bps: float,
        confidence: float,
        min_stop_bps: float = 50.0,
        max_stop_bps: float = 300.0,
    ) -> float:
        """
        Calculate dynamic stop loss based on volatility and confidence.

        Args:
            entry_price: Entry price
            asset_volatility_bps: Asset volatility in basis points
            confidence: Trade confidence (0-1)
            min_stop_bps: Minimum stop loss in basis points
            max_stop_bps: Maximum stop loss in basis points

        Returns:
            Stop loss price
        """
        # Base stop at 2x volatility (2 standard deviations)
        base_stop_bps = 2.0 * asset_volatility_bps

        # Adjust based on confidence:
        # High confidence = wider stop (give it room)
        # Low confidence = tighter stop (cut losses quickly)
        confidence_adjustment = 0.5 + confidence  # Range: 0.5 to 1.5

        adjusted_stop_bps = base_stop_bps * confidence_adjustment

        # Clip to min/max
        final_stop_bps = np.clip(adjusted_stop_bps, min_stop_bps, max_stop_bps)

        # Calculate stop price
        stop_price = entry_price * (1 - final_stop_bps / 10000)

        return float(stop_price)

    def calculate_take_profit(
        self,
        entry_price: float,
        asset_volatility_bps: float,
        confidence: float,
        risk_reward_ratio: float = 2.0,
    ) -> float:
        """
        Calculate dynamic take profit target.

        Args:
            entry_price: Entry price
            asset_volatility_bps: Asset volatility in basis points
            confidence: Trade confidence (0-1)
            risk_reward_ratio: Target risk/reward ratio

        Returns:
            Take profit price
        """
        # Calculate stop loss size
        base_stop_bps = 2.0 * asset_volatility_bps
        confidence_adjustment = 0.5 + confidence
        stop_bps = base_stop_bps * confidence_adjustment

        # Take profit at risk_reward_ratio * stop
        take_profit_bps = stop_bps * risk_reward_ratio

        # Calculate take profit price
        tp_price = entry_price * (1 + take_profit_bps / 10000)

        return float(tp_price)

    def should_scale_out(
        self,
        entry_price: float,
        current_price: float,
        unrealized_profit_pct: float,
        target_profit_pct: float,
    ) -> tuple[bool, float]:
        """
        Determine if position should be partially closed (scaled out).

        Args:
            entry_price: Entry price
            current_price: Current market price
            unrealized_profit_pct: Current unrealized profit percentage
            target_profit_pct: Target profit percentage

        Returns:
            (should_scale, scale_fraction) where scale_fraction is amount to close (0-1)
        """
        # Scale out 50% at 50% of target
        if unrealized_profit_pct >= target_profit_pct * 0.5:
            return (True, 0.5)

        # Scale out additional 25% at 75% of target
        if unrealized_profit_pct >= target_profit_pct * 0.75:
            return (True, 0.25)

        return (False, 0.0)

    def calculate_portfolio_volatility(
        self, position_volatilities: list[float], position_sizes: list[float]
    ) -> float:
        """
        Calculate portfolio-level volatility (simplified, assumes uncorrelated).

        Args:
            position_volatilities: List of asset volatilities (daily %)
            position_sizes: List of position sizes in same units

        Returns:
            Portfolio volatility (daily %)
        """
        if not position_volatilities or not position_sizes:
            return 0.0

        # Convert to numpy arrays
        vols = np.array(position_volatilities)
        sizes = np.array(position_sizes)

        # Total size
        total_size = np.sum(sizes)

        if total_size == 0:
            return 0.0

        # Weight by position size
        weights = sizes / total_size

        # Portfolio variance (assuming uncorrelated for simplicity)
        # For correlated assets, would use full covariance matrix
        portfolio_var = np.sum((weights * vols) ** 2)
        portfolio_vol = np.sqrt(portfolio_var)

        return float(portfolio_vol)
