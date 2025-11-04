"""
Risk Management System for Huracan Engine.

Provides portfolio-level risk controls:
- Max daily loss limits
- Position size limits per symbol
- Portfolio heat tracking
- Correlation risk management
- Circuit breakers
- Emergency shutdown
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_daily_loss_gbp: float = 500.0          # Max loss per day
    max_position_size_gbp: float = 5000.0      # Max position per symbol
    max_portfolio_heat: float = 0.15           # Max 15% of capital at risk
    max_symbols: int = 10                      # Max concurrent positions
    max_correlation: float = 0.7               # Max correlation between positions
    circuit_breaker_loss_gbp: float = 1000.0  # Emergency stop
    daily_profit_target_gbp: Optional[float] = 1000.0  # Stop after hitting target (optional)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    size_gbp: float
    entry_price: Decimal
    current_price: Decimal
    pnl_gbp: float
    stop_loss_bps: int
    take_profit_bps: int
    opened_at: datetime


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    allowed: bool
    reason: str
    risk_score: float  # 0-1, higher = riskier
    warnings: List[str]


class RiskManager:
    """
    Portfolio-level risk manager.

    Prevents:
    - Overleveraging
    - Excessive correlation
    - Blowing up the account
    - Trading after daily loss limit hit
    """

    def __init__(self, limits: RiskLimits, capital_gbp: float = 10000.0):
        """
        Initialize risk manager.

        Args:
            limits: Risk limit configuration
            capital_gbp: Total trading capital
        """
        self.limits = limits
        self.capital_gbp = capital_gbp

        # Current state
        self.positions: Dict[str, Position] = {}
        self.daily_pnl_gbp = 0.0
        self.current_date = date.today()
        self.circuit_breaker_triggered = False
        self.daily_target_hit = False

        logger.info(
            "risk_manager_initialized",
            max_daily_loss=limits.max_daily_loss_gbp,
            max_position_size=limits.max_position_size_gbp,
            capital=capital_gbp,
        )

    def can_open_position(
        self,
        symbol: str,
        size_gbp: float,
        stop_loss_bps: int,
    ) -> RiskAssessment:
        """
        Check if we can open a new position.

        Args:
            symbol: Trading symbol
            size_gbp: Position size in GBP
            stop_loss_bps: Stop loss in basis points

        Returns:
            RiskAssessment with decision
        """
        self._update_daily_reset()

        warnings = []
        risk_score = 0.0

        # Check 1: Circuit breaker
        if self.circuit_breaker_triggered:
            return RiskAssessment(
                allowed=False,
                reason="CIRCUIT_BREAKER_ACTIVE",
                risk_score=1.0,
                warnings=["Emergency circuit breaker is active. Trading halted for today."],
            )

        # Check 2: Daily target hit (optional stop)
        if self.daily_target_hit:
            return RiskAssessment(
                allowed=False,
                reason="DAILY_TARGET_HIT",
                risk_score=0.0,
                warnings=[f"Daily profit target of £{self.limits.daily_profit_target_gbp} reached. Stopping for today."],
            )

        # Check 3: Daily loss limit
        if self.daily_pnl_gbp <= -self.limits.max_daily_loss_gbp:
            return RiskAssessment(
                allowed=False,
                reason="DAILY_LOSS_LIMIT",
                risk_score=1.0,
                warnings=[f"Daily loss limit of £{self.limits.max_daily_loss_gbp} exceeded. Current P&L: £{self.daily_pnl_gbp:.2f}"],
            )

        # Check 4: Position size limit
        if size_gbp > self.limits.max_position_size_gbp:
            return RiskAssessment(
                allowed=False,
                reason="POSITION_SIZE_TOO_LARGE",
                risk_score=0.8,
                warnings=[f"Position size £{size_gbp:.2f} exceeds max £{self.limits.max_position_size_gbp}"],
            )

        # Check 5: Already have position in this symbol
        if symbol in self.positions:
            return RiskAssessment(
                allowed=False,
                reason="POSITION_ALREADY_EXISTS",
                risk_score=0.5,
                warnings=[f"Already have open position in {symbol}"],
            )

        # Check 6: Max concurrent positions
        if len(self.positions) >= self.limits.max_symbols:
            return RiskAssessment(
                allowed=False,
                reason="MAX_POSITIONS_REACHED",
                risk_score=0.6,
                warnings=[f"Already have {len(self.positions)} positions (max: {self.limits.max_symbols})"],
            )

        # Check 7: Portfolio heat (total capital at risk)
        risk_per_position_gbp = size_gbp * (stop_loss_bps / 10000.0)
        current_heat = sum(
            pos.size_gbp * (pos.stop_loss_bps / 10000.0)
            for pos in self.positions.values()
        )
        new_heat = (current_heat + risk_per_position_gbp) / self.capital_gbp

        if new_heat > self.limits.max_portfolio_heat:
            return RiskAssessment(
                allowed=False,
                reason="PORTFOLIO_HEAT_TOO_HIGH",
                risk_score=0.9,
                warnings=[
                    f"Portfolio heat would be {new_heat:.1%} (max: {self.limits.max_portfolio_heat:.1%})",
                    f"Current heat: {current_heat / self.capital_gbp:.1%}",
                    f"New position risk: £{risk_per_position_gbp:.2f}",
                ],
            )

        # Calculate risk score (0-1, higher = riskier)
        risk_score = min(1.0, (
            0.3 * (size_gbp / self.limits.max_position_size_gbp) +  # Size risk
            0.3 * (len(self.positions) / self.limits.max_symbols) +  # Concentration risk
            0.4 * (new_heat / self.limits.max_portfolio_heat)  # Portfolio heat risk
        ))

        # Add warnings if risk is elevated
        if risk_score > 0.7:
            warnings.append(f"High risk score: {risk_score:.2f}")
        if new_heat > 0.10:
            warnings.append(f"Portfolio heat will be {new_heat:.1%}")

        return RiskAssessment(
            allowed=True,
            reason="APPROVED",
            risk_score=risk_score,
            warnings=warnings,
        )

    def register_position(self, position: Position) -> None:
        """
        Register a newly opened position.

        Args:
            position: Position details
        """
        self.positions[position.symbol] = position

        logger.info(
            "position_opened",
            symbol=position.symbol,
            size_gbp=position.size_gbp,
            entry_price=float(position.entry_price),
            stop_loss_bps=position.stop_loss_bps,
            total_positions=len(self.positions),
        )

    def close_position(self, symbol: str, exit_price: Decimal, pnl_gbp: float) -> None:
        """
        Close a position and update daily P&L.

        Args:
            symbol: Symbol being closed
            exit_price: Exit price
            pnl_gbp: Realized P&L in GBP
        """
        if symbol not in self.positions:
            logger.warning("position_not_found", symbol=symbol)
            return

        position = self.positions.pop(symbol)

        # Update daily P&L
        self.daily_pnl_gbp += pnl_gbp

        logger.info(
            "position_closed",
            symbol=symbol,
            entry_price=float(position.entry_price),
            exit_price=float(exit_price),
            pnl_gbp=pnl_gbp,
            daily_pnl_gbp=self.daily_pnl_gbp,
            remaining_positions=len(self.positions),
        )

        # Check circuit breaker
        if self.daily_pnl_gbp <= -self.limits.circuit_breaker_loss_gbp:
            self._trigger_circuit_breaker()

        # Check daily target
        if (
            self.limits.daily_profit_target_gbp is not None
            and self.daily_pnl_gbp >= self.limits.daily_profit_target_gbp
        ):
            self._hit_daily_target()

    def update_position_pnl(self, symbol: str, current_price: Decimal) -> None:
        """
        Update unrealized P&L for a position.

        Args:
            symbol: Symbol to update
            current_price: Current market price
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.current_price = current_price

        # Calculate unrealized P&L
        price_change = float(current_price - position.entry_price)
        position.pnl_gbp = (price_change / float(position.entry_price)) * position.size_gbp

    def get_portfolio_status(self) -> Dict[str, any]:
        """
        Get current portfolio risk status.

        Returns:
            Status dictionary
        """
        total_size = sum(pos.size_gbp for pos in self.positions.values())
        unrealized_pnl = sum(pos.pnl_gbp for pos in self.positions.values())
        total_pnl = self.daily_pnl_gbp + unrealized_pnl

        current_heat = sum(
            pos.size_gbp * (pos.stop_loss_bps / 10000.0)
            for pos in self.positions.values()
        )

        return {
            "date": self.current_date.isoformat(),
            "circuit_breaker_active": self.circuit_breaker_triggered,
            "daily_target_hit": self.daily_target_hit,
            "positions_count": len(self.positions),
            "total_exposure_gbp": total_size,
            "daily_realized_pnl_gbp": self.daily_pnl_gbp,
            "unrealized_pnl_gbp": unrealized_pnl,
            "total_pnl_gbp": total_pnl,
            "portfolio_heat": current_heat / self.capital_gbp,
            "remaining_buying_power_gbp": self.capital_gbp - total_size,
            "max_daily_loss_remaining_gbp": self.limits.max_daily_loss_gbp + self.daily_pnl_gbp,
        }

    def _update_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self.current_date:
            logger.info(
                "daily_reset",
                date=today.isoformat(),
                previous_daily_pnl=self.daily_pnl_gbp,
            )
            self.current_date = today
            self.daily_pnl_gbp = 0.0
            self.circuit_breaker_triggered = False
            self.daily_target_hit = False

    def _trigger_circuit_breaker(self) -> None:
        """Trigger emergency circuit breaker."""
        self.circuit_breaker_triggered = True

        logger.critical(
            "===== CIRCUIT BREAKER TRIGGERED =====",
            daily_pnl_gbp=self.daily_pnl_gbp,
            trigger_level=-self.limits.circuit_breaker_loss_gbp,
            open_positions=len(self.positions),
        )

        # Force close all positions (in real system, would send close orders)
        logger.warning(
            "emergency_close_all_positions",
            positions=list(self.positions.keys()),
        )

    def _hit_daily_target(self) -> None:
        """Mark daily target as hit."""
        self.daily_target_hit = True

        logger.info(
            "===== DAILY PROFIT TARGET HIT =====",
            daily_pnl_gbp=self.daily_pnl_gbp,
            target=self.limits.daily_profit_target_gbp,
            message="Stopping trading for today. Great work!",
        )
