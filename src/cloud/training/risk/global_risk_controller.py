"""
Global Risk Controller for Scalable Architecture

Monitors all trades across coins, sectors, and exchanges.
Enforces limits and provides soft throttling for 400 coins and 500 trades.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class RiskDecision(Enum):
    """Risk decision result."""
    APPROVE = "approve"
    BLOCK = "block"
    THROTTLE = "throttle"
    REDUCE_SIZE = "reduce_size"


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    global_max_exposure_pct: float = 200.0  # 2x leverage max
    per_coin_max_exposure_pct: float = 40.0
    per_sector_max_exposure_pct: float = 60.0
    per_exchange_max_exposure_pct: float = 150.0
    soft_throttle_threshold_pct: float = 80.0  # Throttle at 80% of limits
    max_concurrent_trades: int = 500
    max_active_trades: int = 100  # Soft throttle


@dataclass
class Exposure:
    """Exposure tracking for a category."""
    total_exposure_pct: float = 0.0
    long_exposure_pct: float = 0.0
    short_exposure_pct: float = 0.0
    net_exposure_pct: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class TradeRequest:
    """Trade request for risk checking."""
    coin: str
    direction: str  # 'long' or 'short'
    size_usd: float
    exchange: str
    sector: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskCheckResult:
    """Result of risk check."""
    decision: RiskDecision
    reason: str
    current_exposure_pct: float
    limit_exposure_pct: float
    recommended_size_usd: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker for risk management."""
    
    def __init__(
        self,
        max_drawdown_pct: float = 15.0,
        max_loss_per_hour: float = 1000.0,
        cooldown_seconds: int = 3600,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            max_drawdown_pct: Maximum drawdown percentage
            max_loss_per_hour: Maximum loss per hour in USD
            cooldown_seconds: Cooldown period after trigger
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.max_loss_per_hour = max_loss_per_hour
        self.cooldown_seconds = cooldown_seconds
        
        self.triggered = False
        self.triggered_at: Optional[float] = None
        self.portfolio_peak = 0.0
        self.current_portfolio_value = 0.0
        self.hourly_losses: List[Tuple[float, float]] = []  # (timestamp, loss)
        
        logger.info(
            "circuit_breaker_initialized",
            max_drawdown_pct=max_drawdown_pct,
            max_loss_per_hour=max_loss_per_hour,
        )
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value and check for drawdown."""
        self.current_portfolio_value = value
        
        if value > self.portfolio_peak:
            self.portfolio_peak = value
        
        # Check drawdown
        if self.portfolio_peak > 0:
            drawdown_pct = (
                (self.portfolio_peak - value) / self.portfolio_peak * 100.0
            )
            
            if drawdown_pct >= self.max_drawdown_pct:
                self.trigger()
                logger.warning(
                    "circuit_breaker_triggered_drawdown",
                    drawdown_pct=drawdown_pct,
                    max_drawdown_pct=self.max_drawdown_pct,
                )
    
    def record_loss(self, loss_usd: float) -> None:
        """Record a loss and check hourly limit."""
        current_time = time.time()
        self.hourly_losses.append((current_time, loss_usd))
        
        # Remove losses older than 1 hour
        cutoff_time = current_time - 3600
        self.hourly_losses = [
            (ts, loss) for ts, loss in self.hourly_losses
            if ts > cutoff_time
        ]
        
        # Calculate total loss in last hour
        total_loss = sum(loss for _, loss in self.hourly_losses)
        
        if total_loss >= self.max_loss_per_hour:
            self.trigger()
            logger.warning(
                "circuit_breaker_triggered_loss",
                total_loss=total_loss,
                max_loss_per_hour=self.max_loss_per_hour,
            )
    
    def trigger(self) -> None:
        """Trigger circuit breaker."""
        if not self.triggered:
            self.triggered = True
            self.triggered_at = time.time()
            logger.critical("circuit_breaker_triggered")
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        self.triggered = False
        self.triggered_at = None
        logger.info("circuit_breaker_reset")
    
    def is_triggered(self) -> bool:
        """Check if circuit breaker is triggered."""
        if not self.triggered:
            return False
        
        # Check cooldown
        if self.triggered_at:
            elapsed = time.time() - self.triggered_at
            if elapsed >= self.cooldown_seconds:
                self.reset()
                return False
        
        return True


class GlobalRiskController:
    """
    Global risk controller for multi-coin, multi-exchange monitoring.
    
    Features:
    - Per-coin exposure limits
    - Per-sector exposure limits
    - Per-exchange exposure limits
    - Global exposure limits
    - Soft throttling at 80% of limits
    - Circuit breakers for drawdowns
    - Dynamic position sizing
    """
    
    def __init__(self, limits: RiskLimits):
        """
        Initialize global risk controller.
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits
        
        # Exposure tracking
        self.coin_exposure: Dict[str, Exposure] = {}
        self.sector_exposure: Dict[str, Exposure] = {}
        self.exchange_exposure: Dict[str, Exposure] = {}
        self.global_exposure = Exposure()
        
        # Active trades tracking
        self.active_trades: Dict[str, TradeRequest] = {}  # trade_id -> TradeRequest
        self.coin_trades: Dict[str, Set[str]] = defaultdict(set)  # coin -> trade_ids
        self.exchange_trades: Dict[str, Set[str]] = defaultdict(set)  # exchange -> trade_ids
        
        # Circuit breakers
        self.circuit_breaker = CircuitBreaker()
        
        # Coin to sector mapping (would come from config/data)
        self.coin_sectors: Dict[str, str] = {}
        
        logger.info(
            "global_risk_controller_initialized",
            limits=limits,
        )
    
    def set_coin_sector(self, coin: str, sector: str) -> None:
        """Set sector for a coin."""
        self.coin_sectors[coin] = sector
        if sector not in self.sector_exposure:
            self.sector_exposure[sector] = Exposure()
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value for circuit breaker."""
        self.circuit_breaker.update_portfolio_value(value)
    
    def record_loss(self, loss_usd: float) -> None:
        """Record a loss for circuit breaker."""
        self.circuit_breaker.record_loss(loss_usd)
    
    def check_trade(self, trade: TradeRequest) -> RiskCheckResult:
        """
        Check if a trade should be approved.
        
        Args:
            trade: Trade request to check
        
        Returns:
            RiskCheckResult with decision and reasoning
        """
        # Check circuit breaker
        if self.circuit_breaker.is_triggered():
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason="Circuit breaker triggered",
                current_exposure_pct=0.0,
                limit_exposure_pct=0.0,
            )
        
        # Check active trades limit
        if len(self.active_trades) >= self.limits.max_concurrent_trades:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=f"Max concurrent trades reached: {self.limits.max_concurrent_trades}",
                current_exposure_pct=len(self.active_trades),
                limit_exposure_pct=self.limits.max_concurrent_trades,
            )
        
        # Check soft throttle
        if len(self.active_trades) >= self.limits.max_active_trades:
            return RiskCheckResult(
                decision=RiskDecision.THROTTLE,
                reason=f"Soft throttle: {len(self.active_trades)} active trades",
                current_exposure_pct=len(self.active_trades),
                limit_exposure_pct=self.limits.max_active_trades,
            )
        
        # Check per-coin exposure
        coin_exposure = self.coin_exposure.get(trade.coin, Exposure())
        coin_limit = self.limits.per_coin_max_exposure_pct
        coin_throttle = coin_limit * self.limits.soft_throttle_threshold_pct / 100.0
        
        if trade.direction == "long":
            new_exposure = coin_exposure.long_exposure_pct + (trade.size_usd / 10000.0 * 100.0)  # Simplified
        else:
            new_exposure = coin_exposure.short_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
        
        if new_exposure > coin_limit:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=f"Per-coin exposure limit exceeded: {trade.coin}",
                current_exposure_pct=new_exposure,
                limit_exposure_pct=coin_limit,
            )
        elif new_exposure > coin_throttle:
            # Reduce size to stay under throttle
            recommended_size = (
                (coin_throttle - coin_exposure.long_exposure_pct if trade.direction == "long" 
                 else coin_throttle - coin_exposure.short_exposure_pct) * 100.0
            )
            return RiskCheckResult(
                decision=RiskDecision.REDUCE_SIZE,
                reason=f"Per-coin exposure throttle: {trade.coin}",
                current_exposure_pct=new_exposure,
                limit_exposure_pct=coin_throttle,
                recommended_size_usd=max(0, recommended_size),
            )
        
        # Check per-sector exposure
        if trade.sector:
            sector_exposure = self.sector_exposure.get(trade.sector, Exposure())
            sector_limit = self.limits.per_sector_max_exposure_pct
            sector_throttle = sector_limit * self.limits.soft_throttle_threshold_pct / 100.0
            
            if trade.direction == "long":
                new_sector_exposure = sector_exposure.long_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
            else:
                new_sector_exposure = sector_exposure.short_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
            
            if new_sector_exposure > sector_limit:
                return RiskCheckResult(
                    decision=RiskDecision.BLOCK,
                    reason=f"Per-sector exposure limit exceeded: {trade.sector}",
                    current_exposure_pct=new_sector_exposure,
                    limit_exposure_pct=sector_limit,
                )
        
        # Check per-exchange exposure
        exchange_exposure = self.exchange_exposure.get(trade.exchange, Exposure())
        exchange_limit = self.limits.per_exchange_max_exposure_pct
        exchange_throttle = exchange_limit * self.limits.soft_throttle_threshold_pct / 100.0
        
        if trade.direction == "long":
            new_exchange_exposure = exchange_exposure.long_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
        else:
            new_exchange_exposure = exchange_exposure.short_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
        
        if new_exchange_exposure > exchange_limit:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=f"Per-exchange exposure limit exceeded: {trade.exchange}",
                current_exposure_pct=new_exchange_exposure,
                limit_exposure_pct=exchange_limit,
            )
        
        # Check global exposure
        global_limit = self.limits.global_max_exposure_pct
        global_throttle = global_limit * self.limits.soft_throttle_threshold_pct / 100.0
        
        if trade.direction == "long":
            new_global_exposure = self.global_exposure.long_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
        else:
            new_global_exposure = self.global_exposure.short_exposure_pct + (trade.size_usd / 10000.0 * 100.0)
        
        if new_global_exposure > global_limit:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason="Global exposure limit exceeded",
                current_exposure_pct=new_global_exposure,
                limit_exposure_pct=global_limit,
            )
        
        # All checks passed
        return RiskCheckResult(
            decision=RiskDecision.APPROVE,
            reason="All risk checks passed",
            current_exposure_pct=new_global_exposure,
            limit_exposure_pct=global_limit,
        )
    
    def register_trade(self, trade_id: str, trade: TradeRequest) -> None:
        """Register an active trade."""
        self.active_trades[trade_id] = trade
        self.coin_trades[trade.coin].add(trade_id)
        self.exchange_trades[trade.exchange].add(trade_id)
        
        # Update exposure
        self._update_exposure(trade, trade.size_usd)
        
        logger.debug(
            "trade_registered",
            trade_id=trade_id,
            coin=trade.coin,
            size=trade.size_usd,
        )
    
    def unregister_trade(self, trade_id: str) -> None:
        """Unregister a completed trade."""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        del self.active_trades[trade_id]
        self.coin_trades[trade.coin].discard(trade_id)
        self.exchange_trades[trade.exchange].discard(trade_id)
        
        # Update exposure (subtract)
        self._update_exposure(trade, -trade.size_usd)
        
        logger.debug("trade_unregistered", trade_id=trade_id)
    
    def _update_exposure(self, trade: TradeRequest, size_delta: float) -> None:
        """Update exposure tracking."""
        exposure_delta = size_delta / 10000.0 * 100.0  # Simplified: assume $10k portfolio
        
        # Update coin exposure
        if trade.coin not in self.coin_exposure:
            self.coin_exposure[trade.coin] = Exposure()
        
        coin_exposure = self.coin_exposure[trade.coin]
        if trade.direction == "long":
            coin_exposure.long_exposure_pct += exposure_delta
        else:
            coin_exposure.short_exposure_pct += exposure_delta
        coin_exposure.net_exposure_pct = coin_exposure.long_exposure_pct - coin_exposure.short_exposure_pct
        coin_exposure.total_exposure_pct = coin_exposure.long_exposure_pct + coin_exposure.short_exposure_pct
        coin_exposure.last_update = time.time()
        
        # Update sector exposure
        if trade.sector:
            if trade.sector not in self.sector_exposure:
                self.sector_exposure[trade.sector] = Exposure()
            
            sector_exposure = self.sector_exposure[trade.sector]
            if trade.direction == "long":
                sector_exposure.long_exposure_pct += exposure_delta
            else:
                sector_exposure.short_exposure_pct += exposure_delta
            sector_exposure.net_exposure_pct = sector_exposure.long_exposure_pct - sector_exposure.short_exposure_pct
            sector_exposure.total_exposure_pct = sector_exposure.long_exposure_pct + sector_exposure.short_exposure_pct
            sector_exposure.last_update = time.time()
        
        # Update exchange exposure
        if trade.exchange not in self.exchange_exposure:
            self.exchange_exposure[trade.exchange] = Exposure()
        
        exchange_exposure = self.exchange_exposure[trade.exchange]
        if trade.direction == "long":
            exchange_exposure.long_exposure_pct += exposure_delta
        else:
            exchange_exposure.short_exposure_pct += exposure_delta
        exchange_exposure.net_exposure_pct = exchange_exposure.long_exposure_pct - exchange_exposure.short_exposure_pct
        exchange_exposure.total_exposure_pct = exchange_exposure.long_exposure_pct + exchange_exposure.short_exposure_pct
        exchange_exposure.last_update = time.time()
        
        # Update global exposure
        if trade.direction == "long":
            self.global_exposure.long_exposure_pct += exposure_delta
        else:
            self.global_exposure.short_exposure_pct += exposure_delta
        self.global_exposure.net_exposure_pct = self.global_exposure.long_exposure_pct - self.global_exposure.short_exposure_pct
        self.global_exposure.total_exposure_pct = self.global_exposure.long_exposure_pct + self.global_exposure.short_exposure_pct
        self.global_exposure.last_update = time.time()
    
    def get_exposure_summary(self) -> Dict[str, Any]:
        """Get exposure summary for monitoring."""
        return {
            "global": {
                "total_exposure_pct": self.global_exposure.total_exposure_pct,
                "long_exposure_pct": self.global_exposure.long_exposure_pct,
                "short_exposure_pct": self.global_exposure.short_exposure_pct,
                "net_exposure_pct": self.global_exposure.net_exposure_pct,
            },
            "per_coin": {
                coin: {
                    "total_exposure_pct": exp.total_exposure_pct,
                    "long_exposure_pct": exp.long_exposure_pct,
                    "short_exposure_pct": exp.short_exposure_pct,
                }
                for coin, exp in self.coin_exposure.items()
            },
            "per_sector": {
                sector: {
                    "total_exposure_pct": exp.total_exposure_pct,
                    "long_exposure_pct": exp.long_exposure_pct,
                    "short_exposure_pct": exp.short_exposure_pct,
                }
                for sector, exp in self.sector_exposure.items()
            },
            "per_exchange": {
                exchange: {
                    "total_exposure_pct": exp.total_exposure_pct,
                    "long_exposure_pct": exp.long_exposure_pct,
                    "short_exposure_pct": exp.short_exposure_pct,
                }
                for exchange, exp in self.exchange_exposure.items()
            },
            "active_trades": len(self.active_trades),
            "circuit_breaker": {
                "triggered": self.circuit_breaker.is_triggered(),
                "triggered_at": self.circuit_breaker.triggered_at,
            },
        }

