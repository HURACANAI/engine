"""
Enhanced Circuit Breaker System for £1000 Capital

Multi-level protection system:
- Level 1: Single Trade Protection (1% = £10 max loss per trade)
- Level 2: Hourly Protection (3% = £30 max loss per hour)
- Level 3: Daily Protection (5% = £50 max loss per day)
- Level 4: Drawdown Protection (10% = £100 max drawdown from peak)

All levels auto-pause trading when triggered.
Manual review required to resume after Level 3/4 triggers.

Usage:
    circuit_breaker = EnhancedCircuitBreaker(capital_gbp=1000.0)
    
    # Check before each trade
    if not circuit_breaker.can_trade():
        logger.warning("Trading paused", reason=circuit_breaker.get_status())
        return
    
    # Record trade outcome
    circuit_breaker.record_trade(pnl_gbp=-5.0)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class BreakerLevel(Enum):
    """Circuit breaker levels"""
    NONE = "none"
    LEVEL1 = "level1"  # Single trade
    LEVEL2 = "level2"  # Hourly
    LEVEL3 = "level3"  # Daily
    LEVEL4 = "level4"  # Drawdown


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status"""
    unlimited_mode: bool  # True = shadow trading (unlimited capital)
    level1_active: bool
    level1_limit: float
    level1_current: float
    
    level2_active: bool
    level2_limit: float
    level2_current: float
    level2_window_start: datetime
    
    level3_active: bool
    level3_limit: float
    level3_current: float
    level3_date: str
    
    level4_active: bool
    level4_limit: float
    level4_peak: float
    level4_current: float
    
    trading_paused: bool
    pause_reason: Optional[str]
    manual_review_required: bool


class EnhancedCircuitBreaker:
    """
    Enhanced multi-level circuit breaker system.
    
    Protects capital with cascading protection levels.
    """

    def __init__(
        self,
        capital_gbp: Optional[float] = None,  # None = unlimited (shadow trading)
        level1_pct: float = 0.01,  # 1% per trade
        level2_pct: float = 0.03,  # 3% per hour
        level3_pct: float = 0.05,  # 5% per day
        level4_pct: float = 0.10,  # 10% drawdown
        shadow_trading_mode: bool = True,  # True = unlimited capital for shadow trading
    ):
        """
        Initialize circuit breaker.
        
        Args:
            capital_gbp: Starting capital (None = unlimited for shadow trading)
            level1_pct: Max loss per trade (as % of capital)
            level2_pct: Max loss per hour (as % of capital)
            level3_pct: Max loss per day (as % of capital)
            level4_pct: Max drawdown from peak (as % of capital)
            shadow_trading_mode: If True, capital is unlimited (shadow trading only)
        """
        self.shadow_trading_mode = shadow_trading_mode
        
        if shadow_trading_mode or capital_gbp is None:
            # Shadow trading: unlimited capital, but still track limits for learning
            self.capital_gbp = None  # Unlimited
            self.unlimited_mode = True
            # Use a virtual capital for limit calculations (for reporting only)
            self.virtual_capital_gbp = 1000.0
        else:
            # Real trading: use actual capital
            self.capital_gbp = capital_gbp
            self.unlimited_mode = False
            self.virtual_capital_gbp = capital_gbp
        
        # Level 1: Single Trade
        self.level1_limit = (self.virtual_capital_gbp if self.unlimited_mode else self.capital_gbp) * level1_pct
        self.level1_current = 0.0
        
        # Level 2: Hourly
        self.level2_limit = (self.virtual_capital_gbp if self.unlimited_mode else self.capital_gbp) * level2_pct
        self.level2_current = 0.0
        self.level2_window_start = datetime.utcnow()
        self.level2_trades: List[Dict] = []  # Track trades in current hour
        
        # Level 3: Daily
        self.level3_limit = (self.virtual_capital_gbp if self.unlimited_mode else self.capital_gbp) * level3_pct
        self.level3_current = 0.0
        self.level3_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Level 4: Drawdown
        self.level4_limit = (self.virtual_capital_gbp if self.unlimited_mode else self.capital_gbp) * level4_pct
        self.level4_peak = self.virtual_capital_gbp if self.unlimited_mode else self.capital_gbp
        self.level4_current = self.level4_peak
        
        # State
        self.trading_paused = False
        self.pause_reason: Optional[str] = None
        self.manual_review_required = False
        
        logger.info(
            "enhanced_circuit_breaker_initialized",
            capital=capital_gbp if not self.unlimited_mode else "UNLIMITED (shadow trading)",
            shadow_trading_mode=shadow_trading_mode,
            unlimited_mode=self.unlimited_mode,
            level1_limit=self.level1_limit,
            level2_limit=self.level2_limit,
            level3_limit=self.level3_limit,
            level4_limit=self.level4_limit,
        )

    def can_trade(self, trade_size_gbp: float = 0.0) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed.
        
        Args:
            trade_size_gbp: Size of proposed trade (for Level 1 check)
            
        Returns:
            (allowed, reason) tuple
        """
        # Shadow trading mode: Always allow (unlimited capital)
        # But still track limits for learning and reporting
        if self.unlimited_mode:
            # Still track for learning, but don't block
            self._reset_if_needed()
            return True, None
        
        # Real trading mode: Enforce limits
        # Reset daily/hourly if needed
        self._reset_if_needed()
        
        # Check Level 4: Drawdown (highest priority)
        if self.level4_current < (self.level4_peak - self.level4_limit):
            self.trading_paused = True
            self.pause_reason = f"LEVEL4_DRAWDOWN: Current {self.level4_current:.2f} below peak {self.level4_peak:.2f} by {self.level4_limit:.2f}"
            self.manual_review_required = True
            logger.critical("circuit_breaker_level4_triggered", reason=self.pause_reason)
            return False, self.pause_reason
        
        # Check Level 3: Daily
        if abs(self.level3_current) >= self.level3_limit:
            self.trading_paused = True
            self.pause_reason = f"LEVEL3_DAILY: Daily loss {abs(self.level3_current):.2f} exceeds limit {self.level3_limit:.2f}"
            self.manual_review_required = True
            logger.critical("circuit_breaker_level3_triggered", reason=self.pause_reason)
            return False, self.pause_reason
        
        # Check Level 2: Hourly
        if abs(self.level2_current) >= self.level2_limit:
            self.trading_paused = True
            self.pause_reason = f"LEVEL2_HOURLY: Hourly loss {abs(self.level2_current):.2f} exceeds limit {self.level2_limit:.2f}"
            logger.warning("circuit_breaker_level2_triggered", reason=self.pause_reason)
            return False, self.pause_reason
        
        # Check Level 1: Single Trade
        if trade_size_gbp > 0 and trade_size_gbp > self.level1_limit:
            self.trading_paused = True
            self.pause_reason = f"LEVEL1_TRADE: Trade size {trade_size_gbp:.2f} exceeds limit {self.level1_limit:.2f}"
            logger.warning("circuit_breaker_level1_triggered", reason=self.pause_reason)
            return False, self.pause_reason
        
        # All checks passed
        if self.trading_paused:
            # Was paused but conditions cleared
            logger.info("circuit_breaker_cleared", previous_reason=self.pause_reason)
            self.trading_paused = False
            self.pause_reason = None
        
        return True, None

    def record_trade(self, pnl_gbp: float, trade_size_gbp: float = 0.0):
        """
        Record trade outcome and update circuit breaker state.
        
        Args:
            pnl_gbp: Profit/loss from trade
            trade_size_gbp: Size of trade (for Level 1 tracking)
        """
        # Update Level 1: Single Trade
        if trade_size_gbp > 0:
            # Check if this trade alone would exceed limit
            if abs(pnl_gbp) > self.level1_limit:
                logger.warning(
                    "level1_trade_exceeded",
                    pnl=pnl_gbp,
                    limit=self.level1_limit,
                )
        
        # Update Level 2: Hourly
        self.level2_current += pnl_gbp
        self.level2_trades.append({
            'timestamp': datetime.utcnow(),
            'pnl': pnl_gbp,
        })
        
        # Update Level 3: Daily
        self.level3_current += pnl_gbp
        
        # Update Level 4: Drawdown
        self.level4_current += pnl_gbp
        if self.level4_current > self.level4_peak:
            self.level4_peak = self.level4_current
        
        # Check if any levels triggered
        self.can_trade()  # This will check and update pause status
        
        logger.debug(
            "circuit_breaker_updated",
            pnl=pnl_gbp,
            level2_current=self.level2_current,
            level3_current=self.level3_current,
            level4_current=self.level4_current,
            level4_peak=self.level4_peak,
        )

    def _reset_if_needed(self):
        """Reset hourly/daily windows if needed"""
        now = datetime.utcnow()
        
        # Reset Level 2: Hourly
        if (now - self.level2_window_start).total_seconds() >= 3600:
            logger.info(
                "level2_hourly_reset",
                previous_hourly_pnl=self.level2_current,
            )
            self.level2_current = 0.0
            self.level2_window_start = now
            self.level2_trades = []
        
        # Reset Level 3: Daily
        current_date = now.strftime("%Y-%m-%d")
        if current_date != self.level3_date:
            logger.info(
                "level3_daily_reset",
                previous_daily_pnl=self.level3_current,
            )
            self.level3_current = 0.0
            self.level3_date = current_date

    def get_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status"""
        return CircuitBreakerStatus(
            unlimited_mode=self.unlimited_mode,
            level1_active=abs(self.level1_current) >= self.level1_limit if not self.unlimited_mode else False,
            level1_limit=self.level1_limit,
            level1_current=self.level1_current,
            level2_active=abs(self.level2_current) >= self.level2_limit if not self.unlimited_mode else False,
            level2_limit=self.level2_limit,
            level2_current=self.level2_current,
            level2_window_start=self.level2_window_start,
            level3_active=abs(self.level3_current) >= self.level3_limit if not self.unlimited_mode else False,
            level3_limit=self.level3_limit,
            level3_current=self.level3_current,
            level3_date=self.level3_date,
            level4_active=self.level4_current < (self.level4_peak - self.level4_limit) if not self.unlimited_mode else False,
            level4_limit=self.level4_limit,
            level4_peak=self.level4_peak,
            level4_current=self.level4_current,
            trading_paused=self.trading_paused if not self.unlimited_mode else False,
            pause_reason=self.pause_reason if not self.unlimited_mode else None,
            manual_review_required=self.manual_review_required if not self.unlimited_mode else False,
        )

    def resume_trading(self, reason: str = "Manual resume"):
        """
        Resume trading after manual review.
        
        Args:
            reason: Reason for resuming
        """
        if not self.manual_review_required:
            logger.warning("resume_called_without_manual_review", reason=reason)
            return
        
        self.trading_paused = False
        self.pause_reason = None
        self.manual_review_required = False
        
        # Reset levels that triggered
        if self.level4_current < (self.level4_peak - self.level4_limit):
            # Reset drawdown peak
            self.level4_peak = self.level4_current
        
        logger.info("trading_resumed", reason=reason)

    def update_capital(self, new_capital_gbp: Optional[float]):
        """
        Update capital (e.g., after profitable day).
        
        Args:
            new_capital_gbp: New capital amount (None = unlimited for shadow trading)
        """
        if self.unlimited_mode:
            # Shadow trading: update virtual capital for reporting
            old_capital = self.virtual_capital_gbp
            self.virtual_capital_gbp = new_capital_gbp if new_capital_gbp else 1000.0
            
            # Recalculate limits (for reporting only)
            ratio = self.virtual_capital_gbp / old_capital if old_capital > 0 else 1.0
            self.level1_limit = self.virtual_capital_gbp * 0.01
            self.level2_limit = self.virtual_capital_gbp * 0.03
            self.level3_limit = self.virtual_capital_gbp * 0.05
            self.level4_limit = self.virtual_capital_gbp * 0.10
            self.level4_peak *= ratio
            self.level4_current *= ratio
            
            logger.info(
                "virtual_capital_updated",
                old_capital=old_capital,
                new_capital=self.virtual_capital_gbp,
                ratio=ratio,
                note="Shadow trading mode - unlimited capital, limits for reporting only",
            )
        else:
            # Real trading: update actual capital
            old_capital = self.capital_gbp
            if new_capital_gbp is None:
                raise ValueError("Cannot set capital to None in real trading mode")
            
            self.capital_gbp = new_capital_gbp
            
            # Recalculate limits
            self.level1_limit = new_capital_gbp * 0.01
            self.level2_limit = new_capital_gbp * 0.03
            self.level3_limit = new_capital_gbp * 0.05
            self.level4_limit = new_capital_gbp * 0.10
            
            # Adjust current values proportionally
            ratio = new_capital_gbp / old_capital if old_capital > 0 else 1.0
            self.level4_peak *= ratio
            self.level4_current *= ratio
            
            logger.info(
                "capital_updated",
                old_capital=old_capital,
                new_capital=new_capital_gbp,
                ratio=ratio,
            )

