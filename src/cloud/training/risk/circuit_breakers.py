"""
Circuit Breakers - Daily and Intraday Risk Limits.

Implements:
- Hard daily max drawdown (e.g., 3% equity)
- Soft streak breaker (halve size after 3 consecutive losses)
- Volatility breaker (defense mode when vol exceeds percentile)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status."""
    is_triggered: bool
    breaker_type: str  # "daily_drawdown", "streak", "volatility"
    message: str
    action: str  # "stop_trading", "reduce_size", "defense_mode"
    reset_time: Optional[datetime] = None


class CircuitBreakerSystem:
    """
    Circuit breaker system for risk management.
    
    Features:
    - Hard daily max drawdown (stop trading for the day)
    - Soft streak breaker (reduce size after losses)
    - Volatility breaker (switch to defense mode)
    """
    
    def __init__(
        self,
        daily_max_drawdown_pct: float = 3.0,  # 3% of equity
        streak_loss_limit: int = 3,  # 3 consecutive losses
        streak_size_reduction: float = 0.5,  # Halve size
        volatility_percentile: float = 95.0,  # 95th percentile
        volatility_lookback_days: int = 30,
    ) -> None:
        """
        Initialize circuit breaker system.
        
        Args:
            daily_max_drawdown_pct: Maximum daily drawdown % (default: 3.0)
            streak_loss_limit: Consecutive losses before breaker (default: 3)
            streak_size_reduction: Size reduction factor (default: 0.5)
            volatility_percentile: Volatility percentile threshold (default: 95.0)
            volatility_lookback_days: Days to look back for volatility (default: 30)
        """
        self.daily_max_drawdown_pct = daily_max_drawdown_pct
        self.streak_loss_limit = streak_loss_limit
        self.streak_size_reduction = streak_size_reduction
        self.volatility_percentile = volatility_percentile
        self.volatility_lookback_days = volatility_lookback_days
        
        # Daily tracking
        self.daily_start_equity: Optional[float] = None
        self.daily_peak_equity: Optional[float] = None
        self.daily_low_equity: Optional[float] = None
        self.current_date: Optional[datetime] = None
        
        # Streak tracking
        self.consecutive_losses: int = 0
        self.recent_trades: List[Dict[str, any]] = []
        
        # Volatility tracking
        self.volatility_history: List[float] = []
        
        logger.info(
            "circuit_breaker_system_initialized",
            daily_max_drawdown_pct=daily_max_drawdown_pct,
            streak_loss_limit=streak_loss_limit,
            volatility_percentile=volatility_percentile
        )
    
    def check_daily_drawdown(
        self,
        current_equity: float,
        current_date: datetime
    ) -> CircuitBreakerStatus:
        """
        Check daily drawdown circuit breaker.
        
        Args:
            current_equity: Current equity
            current_date: Current date
        
        Returns:
            CircuitBreakerStatus
        """
        # Reset if new day
        if self.current_date is None or current_date.date() != self.current_date.date():
            self.daily_start_equity = current_equity
            self.daily_peak_equity = current_equity
            self.daily_low_equity = current_equity
            self.current_date = current_date
            self.consecutive_losses = 0  # Reset streak on new day
        
        # Update peak and low
        if self.daily_peak_equity is not None:
            self.daily_peak_equity = max(self.daily_peak_equity, current_equity)
        else:
            self.daily_peak_equity = current_equity
        
        if self.daily_low_equity is not None:
            self.daily_low_equity = min(self.daily_low_equity, current_equity)
        else:
            self.daily_low_equity = current_equity
        
        # Calculate drawdown from peak
        if self.daily_peak_equity is not None and self.daily_peak_equity > 0:
            drawdown_pct = ((self.daily_peak_equity - current_equity) / self.daily_peak_equity) * 100.0
            
            if drawdown_pct >= self.daily_max_drawdown_pct:
                # Calculate reset time (next day at 00:00)
                reset_time = (current_date + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                
                logger.warning(
                    "daily_drawdown_breaker_triggered",
                    drawdown_pct=drawdown_pct,
                    max_allowed=self.daily_max_drawdown_pct,
                    reset_time=reset_time.isoformat()
                )
                
                return CircuitBreakerStatus(
                    is_triggered=True,
                    breaker_type="daily_drawdown",
                    message=f"Daily drawdown {drawdown_pct:.2f}% exceeds limit {self.daily_max_drawdown_pct}%",
                    action="stop_trading",
                    reset_time=reset_time
                )
        
        return CircuitBreakerStatus(
            is_triggered=False,
            breaker_type="daily_drawdown",
            message="Daily drawdown within limits",
            action="continue"
        )
    
    def check_streak_breaker(
        self,
        trade_result: Dict[str, any]  # Should have 'won' (bool) and 'pnl' (float)
    ) -> CircuitBreakerStatus:
        """
        Check streak breaker (soft breaker - reduces size).
        
        Args:
            trade_result: Trade result with 'won' and 'pnl'
        
        Returns:
            CircuitBreakerStatus
        """
        won = trade_result.get('won', False)
        pnl = trade_result.get('pnl', 0.0)
        
        # Update streak
        if not won and pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Add to recent trades
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)
        
        if self.consecutive_losses >= self.streak_loss_limit:
            logger.warning(
                "streak_breaker_triggered",
                consecutive_losses=self.consecutive_losses,
                size_reduction=self.streak_size_reduction
            )
            
            return CircuitBreakerStatus(
                is_triggered=True,
                breaker_type="streak",
                message=f"{self.consecutive_losses} consecutive losses",
                action="reduce_size",
                reset_time=None  # Resets on next win
            )
        
        return CircuitBreakerStatus(
            is_triggered=False,
            breaker_type="streak",
            message=f"Streak: {self.consecutive_losses} losses",
            action="continue"
        )
    
    def check_volatility_breaker(
        self,
        current_volatility: float
    ) -> CircuitBreakerStatus:
        """
        Check volatility breaker.
        
        Args:
            current_volatility: Current realized volatility
        
        Returns:
            CircuitBreakerStatus
        """
        # Add to history
        self.volatility_history.append(current_volatility)
        if len(self.volatility_history) > self.volatility_lookback_days:
            self.volatility_history.pop(0)
        
        if len(self.volatility_history) < 10:
            # Not enough data
            return CircuitBreakerStatus(
                is_triggered=False,
                breaker_type="volatility",
                message="Insufficient volatility history",
                action="continue"
            )
        
        # Calculate percentile threshold
        threshold = np.percentile(self.volatility_history, self.volatility_percentile)
        
        if current_volatility > threshold:
            logger.warning(
                "volatility_breaker_triggered",
                current_volatility=current_volatility,
                threshold=threshold,
                percentile=self.volatility_percentile
            )
            
            return CircuitBreakerStatus(
                is_triggered=True,
                breaker_type="volatility",
                message=f"Volatility {current_volatility:.4f} exceeds {self.volatility_percentile}th percentile {threshold:.4f}",
                action="defense_mode",
                reset_time=None  # Resets when volatility drops
            )
        
        return CircuitBreakerStatus(
            is_triggered=False,
            breaker_type="volatility",
            message=f"Volatility {current_volatility:.4f} within normal range",
            action="continue"
        )
    
    def check_all(
        self,
        current_equity: float,
        current_date: datetime,
        current_volatility: float,
        recent_trade_result: Optional[Dict[str, any]] = None
    ) -> List[CircuitBreakerStatus]:
        """
        Check all circuit breakers.
        
        Args:
            current_equity: Current equity
            current_date: Current date
            current_volatility: Current volatility
            recent_trade_result: Recent trade result (optional)
        
        Returns:
            List of circuit breaker statuses
        """
        statuses = []
        
        # Daily drawdown
        daily_status = self.check_daily_drawdown(current_equity, current_date)
        statuses.append(daily_status)
        
        # Streak breaker
        if recent_trade_result:
            streak_status = self.check_streak_breaker(recent_trade_result)
            statuses.append(streak_status)
        
        # Volatility breaker
        vol_status = self.check_volatility_breaker(current_volatility)
        statuses.append(vol_status)
        
        return statuses
    
    def get_size_multiplier(self) -> float:
        """
        Get current size multiplier based on circuit breakers.
        
        Returns:
            Size multiplier (1.0 = normal, 0.5 = halved, 0.0 = stopped)
        """
        # Check if any breaker is triggered
        # This is a simplified version - in practice, check all breakers
        
        if self.consecutive_losses >= self.streak_loss_limit:
            return self.streak_size_reduction
        
        # Daily drawdown would return 0.0 if triggered
        # Volatility breaker would return 0.5 (defense mode)
        
        return 1.0

