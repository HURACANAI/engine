"""
Pre-Trade Risk Engine

Fast risk validation layer between Hamilton's trade signal and final API send.
Ensures no strategy can over-expose capital.

Key Features:
- Exposure limit checks
- Position size validation
- Daily loss limits
- Concentration limits
- Leverage limits
- Fast validation (< 1ms)

Author: Huracan Engine Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class RiskCheckResult(Enum):
    """Risk check result"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REDUCED = "reduced"  # Size reduced but approved
    DELAYED = "delayed"  # Temporary delay


@dataclass
class RiskCheck:
    """Individual risk check"""
    check_name: str
    passed: bool
    message: str
    limit_value: Optional[float] = None
    current_value: Optional[float] = None
    duration_ns: int = 0


@dataclass
class PreTradeRiskResult:
    """Result of pre-trade risk checks"""
    approved: bool
    result: RiskCheckResult
    checks: List[RiskCheck]
    recommended_size: float  # May be reduced from requested size
    rejection_reason: Optional[str] = None
    total_duration_ns: int = 0


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size_usd: float = 10000.0
    max_daily_loss_usd: float = 1000.0
    max_leverage: float = 1.0  # 1.0 = no leverage
    max_concentration_pct: float = 0.20  # 20% max per symbol
    max_correlation_exposure: float = 0.50  # 50% max correlated positions
    min_liquidity_score: float = 0.5  # Minimum liquidity for trading
    max_spread_bps: float = 50.0  # Maximum spread in bps
    max_single_trade_usd: float = 5000.0
    max_exposure_per_symbol: Dict[str, float] = field(default_factory=dict)


class PreTradeRiskEngine:
    """
    Fast pre-trade risk validation engine.
    
    Validates trades before execution to prevent over-exposure.
    Designed for low latency (< 1ms typical).
    
    Usage:
        engine = PreTradeRiskEngine(limits=RiskLimits())
        result = engine.validate_trade(
            symbol="BTCUSDT",
            direction="buy",
            size_usd=1000.0,
            current_positions={...},
            daily_pnl=-100.0
        )
        
        if result.approved:
            execute_trade(size=result.recommended_size)
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        enable_all_checks: bool = True
    ):
        """
        Initialize pre-trade risk engine.
        
        Args:
            limits: Risk limits configuration
            enable_all_checks: Enable all risk checks
        """
        self.limits = limits or RiskLimits()
        self.enable_all_checks = enable_all_checks
        
        # Track daily metrics
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.current_positions: Dict[str, float] = {}  # symbol -> size_usd
        self.daily_exposure: Dict[str, float] = {}  # symbol -> exposure_usd
        
        # Reset daily at start
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        logger.info(
            "pre_trade_risk_engine_initialized",
            max_position_size=self.limits.max_position_size_usd,
            max_daily_loss=self.limits.max_daily_loss_usd,
            max_leverage=self.limits.max_leverage
        )
    
    def validate_trade(
        self,
        symbol: str,
        direction: str,  # "buy" or "sell"
        size_usd: float,
        current_price: float,
        spread_bps: Optional[float] = None,
        liquidity_score: Optional[float] = None,
        correlation_exposure: Optional[float] = None,
        leverage: float = 1.0
    ) -> PreTradeRiskResult:
        """
        Validate trade before execution.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ("buy" or "sell")
            size_usd: Trade size in USD
            current_price: Current market price
            spread_bps: Current spread in bps (optional)
            liquidity_score: Liquidity score (optional)
            correlation_exposure: Correlation exposure (optional)
            leverage: Leverage factor
        
        Returns:
            PreTradeRiskResult with approval status and checks
        """
        start_time_ns = time.perf_counter_ns()
        
        # Reset daily metrics if new day
        self._reset_daily_if_needed()
        
        checks: List[RiskCheck] = []
        recommended_size = size_usd
        all_passed = True
        rejection_reason = None
        
        # 1. Check daily loss limit
        if self.enable_all_checks:
            loss_check = self._check_daily_loss_limit()
            checks.append(loss_check)
            if not loss_check.passed:
                all_passed = False
                rejection_reason = loss_check.message
        
        # 2. Check position size limits
        if self.enable_all_checks and all_passed:
            size_check, recommended_size = self._check_position_size(
                symbol, size_usd, recommended_size
            )
            checks.append(size_check)
            if not size_check.passed:
                all_passed = False
                rejection_reason = size_check.message
        
        # 3. Check leverage limits
        if self.enable_all_checks and all_passed:
            leverage_check = self._check_leverage(leverage)
            checks.append(leverage_check)
            if not leverage_check.passed:
                all_passed = False
                rejection_reason = leverage_check.message
        
        # 4. Check concentration limits
        if self.enable_all_checks and all_passed:
            concentration_check, recommended_size = self._check_concentration(
                symbol, recommended_size, size_usd
            )
            checks.append(concentration_check)
            if not concentration_check.passed:
                all_passed = False
                rejection_reason = concentration_check.message
        
        # 5. Check liquidity
        if self.enable_all_checks and all_passed and liquidity_score is not None:
            liquidity_check = self._check_liquidity(liquidity_score)
            checks.append(liquidity_check)
            if not liquidity_check.passed:
                all_passed = False
                rejection_reason = liquidity_check.message
        
        # 6. Check spread
        if self.enable_all_checks and all_passed and spread_bps is not None:
            spread_check = self._check_spread(spread_bps)
            checks.append(spread_check)
            if not spread_check.passed:
                all_passed = False
                rejection_reason = spread_check.message
        
        # 7. Check correlation exposure
        if self.enable_all_checks and all_passed and correlation_exposure is not None:
            correlation_check = self._check_correlation_exposure(correlation_exposure)
            checks.append(correlation_check)
            if not correlation_check.passed:
                all_passed = False
                rejection_reason = correlation_check.message
        
        # 8. Check single trade limit
        if self.enable_all_checks and all_passed:
            single_trade_check, recommended_size = self._check_single_trade_limit(
                recommended_size, size_usd
            )
            checks.append(single_trade_check)
            if not single_trade_check.passed:
                all_passed = False
                rejection_reason = single_trade_check.message
        
        # Determine result
        if all_passed:
            if recommended_size < size_usd:
                result = RiskCheckResult.REDUCED
            else:
                result = RiskCheckResult.APPROVED
        else:
            result = RiskCheckResult.REJECTED
        
        duration_ns = time.perf_counter_ns() - start_time_ns
        
        risk_result = PreTradeRiskResult(
            approved=all_passed,
            result=result,
            checks=checks,
            recommended_size=recommended_size,
            rejection_reason=rejection_reason,
            total_duration_ns=duration_ns
        )
        
        logger.debug(
            "pre_trade_risk_check",
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            approved=all_passed,
            recommended_size=recommended_size,
            duration_ns=duration_ns,
            checks_passed=sum(1 for c in checks if c.passed),
            checks_total=len(checks)
        )
        
        return risk_result
    
    def _check_daily_loss_limit(self) -> RiskCheck:
        """Check if daily loss limit would be exceeded"""
        start_ns = time.perf_counter_ns()
        
        passed = self.daily_pnl >= -self.limits.max_daily_loss_usd
        message = (
            f"Daily loss limit exceeded: ${self.daily_pnl:.2f} / ${-self.limits.max_daily_loss_usd:.2f}"
            if not passed
            else "Daily loss limit OK"
        )
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="daily_loss_limit",
            passed=passed,
            message=message,
            limit_value=-self.limits.max_daily_loss_usd,
            current_value=self.daily_pnl,
            duration_ns=duration_ns
        )
    
    def _check_position_size(
        self,
        symbol: str,
        size_usd: float,
        current_recommended: float
    ) -> Tuple[RiskCheck, float]:
        """Check position size limits"""
        start_ns = time.perf_counter_ns()
        
        # Check max position size
        current_position = abs(self.current_positions.get(symbol, 0.0))
        new_position = current_position + size_usd
        
        # Check symbol-specific limit
        symbol_limit = self.limits.max_exposure_per_symbol.get(symbol, self.limits.max_position_size_usd)
        
        if new_position > symbol_limit:
            # Reduce size to fit within limit
            recommended_size = max(0.0, symbol_limit - current_position)
            passed = recommended_size > 0
            message = (
                f"Position size limit: ${new_position:.2f} > ${symbol_limit:.2f}, reduced to ${recommended_size:.2f}"
                if passed
                else f"Position size limit exceeded: ${new_position:.2f} > ${symbol_limit:.2f}"
            )
        else:
            recommended_size = current_recommended
            passed = True
            message = "Position size limit OK"
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="position_size",
            passed=passed,
            message=message,
            limit_value=symbol_limit,
            current_value=new_position,
            duration_ns=duration_ns
        ), recommended_size
    
    def _check_leverage(self, leverage: float) -> RiskCheck:
        """Check leverage limits"""
        start_ns = time.perf_counter_ns()
        
        passed = leverage <= self.limits.max_leverage
        message = (
            f"Leverage limit exceeded: {leverage:.2f}x > {self.limits.max_leverage:.2f}x"
            if not passed
            else f"Leverage OK: {leverage:.2f}x"
        )
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="leverage",
            passed=passed,
            message=message,
            limit_value=self.limits.max_leverage,
            current_value=leverage,
            duration_ns=duration_ns
        )
    
    def _check_concentration(
        self,
        symbol: str,
        recommended_size: float,
        original_size: float
    ) -> Tuple[RiskCheck, float]:
        """Check concentration limits"""
        start_ns = time.perf_counter_ns()
        
        # Calculate total exposure
        total_exposure = sum(abs(v) for v in self.current_positions.values())
        symbol_exposure = abs(self.current_positions.get(symbol, 0.0)) + recommended_size
        
        # Calculate concentration
        if total_exposure > 0:
            concentration = symbol_exposure / total_exposure
        else:
            concentration = 1.0 if recommended_size > 0 else 0.0
        
        if concentration > self.limits.max_concentration_pct:
            # Reduce size to fit within concentration limit
            max_symbol_exposure = total_exposure * self.limits.max_concentration_pct
            recommended_size = max(0.0, max_symbol_exposure - abs(self.current_positions.get(symbol, 0.0)))
            passed = recommended_size > 0
            message = (
                f"Concentration limit: {concentration*100:.1f}% > {self.limits.max_concentration_pct*100:.1f}%, reduced size"
                if passed
                else f"Concentration limit exceeded: {concentration*100:.1f}% > {self.limits.max_concentration_pct*100:.1f}%"
            )
        else:
            passed = True
            message = f"Concentration OK: {concentration*100:.1f}%"
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="concentration",
            passed=passed,
            message=message,
            limit_value=self.limits.max_concentration_pct,
            current_value=concentration,
            duration_ns=duration_ns
        ), recommended_size
    
    def _check_liquidity(self, liquidity_score: float) -> RiskCheck:
        """Check liquidity requirements"""
        start_ns = time.perf_counter_ns()
        
        passed = liquidity_score >= self.limits.min_liquidity_score
        message = (
            f"Liquidity too low: {liquidity_score:.2f} < {self.limits.min_liquidity_score:.2f}"
            if not passed
            else f"Liquidity OK: {liquidity_score:.2f}"
        )
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="liquidity",
            passed=passed,
            message=message,
            limit_value=self.limits.min_liquidity_score,
            current_value=liquidity_score,
            duration_ns=duration_ns
        )
    
    def _check_spread(self, spread_bps: float) -> RiskCheck:
        """Check spread limits"""
        start_ns = time.perf_counter_ns()
        
        passed = spread_bps <= self.limits.max_spread_bps
        message = (
            f"Spread too wide: {spread_bps:.2f} bps > {self.limits.max_spread_bps:.2f} bps"
            if not passed
            else f"Spread OK: {spread_bps:.2f} bps"
        )
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="spread",
            passed=passed,
            message=message,
            limit_value=self.limits.max_spread_bps,
            current_value=spread_bps,
            duration_ns=duration_ns
        )
    
    def _check_correlation_exposure(self, correlation_exposure: float) -> RiskCheck:
        """Check correlation exposure limits"""
        start_ns = time.perf_counter_ns()
        
        passed = correlation_exposure <= self.limits.max_correlation_exposure
        message = (
            f"Correlation exposure too high: {correlation_exposure:.2f} > {self.limits.max_correlation_exposure:.2f}"
            if not passed
            else f"Correlation exposure OK: {correlation_exposure:.2f}"
        )
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="correlation_exposure",
            passed=passed,
            message=message,
            limit_value=self.limits.max_correlation_exposure,
            current_value=correlation_exposure,
            duration_ns=duration_ns
        )
    
    def _check_single_trade_limit(
        self,
        recommended_size: float,
        original_size: float
    ) -> Tuple[RiskCheck, float]:
        """Check single trade size limit"""
        start_ns = time.perf_counter_ns()
        
        if recommended_size > self.limits.max_single_trade_usd:
            recommended_size = self.limits.max_single_trade_usd
            passed = True
            message = f"Single trade limit: reduced to ${recommended_size:.2f}"
        else:
            passed = True
            message = "Single trade limit OK"
        
        duration_ns = time.perf_counter_ns() - start_ns
        
        return RiskCheck(
            check_name="single_trade_limit",
            passed=passed,
            message=message,
            limit_value=self.limits.max_single_trade_usd,
            current_value=recommended_size,
            duration_ns=duration_ns
        ), recommended_size
    
    def update_daily_pnl(self, pnl_usd: float) -> None:
        """Update daily P&L"""
        self.daily_pnl += pnl_usd
    
    def update_position(self, symbol: str, size_usd: float) -> None:
        """Update current position"""
        if size_usd == 0:
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = size_usd
    
    def _reset_daily_if_needed(self) -> None:
        """Reset daily metrics if new day"""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_exposure = {}
            self.last_reset_date = current_date
            logger.info("daily_risk_metrics_reset", date=current_date)

