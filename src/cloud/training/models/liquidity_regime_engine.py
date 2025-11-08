"""
Liquidity Regime Engine

Hard gate for all signals. Only allows entries when liquidity quality is high.
Uses spread in bps, top of book depth, and order book imbalance.

Key Features:
- Liquidity quality scoring
- Hard gate for all signals
- Spread-based filtering
- Order book depth analysis
- Imbalance detection
- Integration with all engines

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class LiquidityRegime(Enum):
    """Liquidity regime"""
    EXCELLENT = "excellent"  # High liquidity, tight spreads
    GOOD = "good"  # Adequate liquidity
    FAIR = "fair"  # Moderate liquidity
    POOR = "poor"  # Low liquidity, wide spreads
    UNACCEPTABLE = "unacceptable"  # Too risky to trade


@dataclass
class LiquidityMetrics:
    """Liquidity metrics"""
    spread_bps: float
    top_of_book_depth_usd: float
    order_book_imbalance: float  # -1 to 1, negative = ask heavy, positive = bid heavy
    liquidity_score: float  # 0-1, higher is better
    regime: LiquidityRegime
    can_trade: bool


@dataclass
class LiquidityGateResult:
    """Result of liquidity gate check"""
    passed: bool
    regime: LiquidityRegime
    liquidity_score: float
    metrics: LiquidityMetrics
    rejection_reason: Optional[str] = None


class LiquidityRegimeEngine:
    """
    Liquidity Regime Engine - Hard gate for all signals.
    
    Only allows entries when liquidity quality is high.
    Gates all other engines through this filter.
    
    Usage:
        engine = LiquidityRegimeEngine()
        
        result = engine.check_liquidity(
            symbol="BTCUSDT",
            spread_bps=5.0,
            bid_depth_usd=10000.0,
            ask_depth_usd=9500.0,
            bid_volume=100.0,
            ask_volume=95.0
        )
        
        if result.passed:
            # Proceed with trade
            pass
        else:
            # Reject trade
            logger.warning(f"Trade rejected: {result.rejection_reason}")
    """
    
    def __init__(
        self,
        min_liquidity_score: float = 0.6,  # Minimum liquidity score to trade
        max_spread_bps: float = 50.0,  # Maximum spread in bps
        min_depth_usd: float = 1000.0,  # Minimum top of book depth
        max_imbalance: float = 0.3,  # Maximum order book imbalance (absolute)
        regime_thresholds: Optional[Dict[LiquidityRegime, float]] = None
    ):
        """
        Initialize liquidity regime engine.
        
        Args:
            min_liquidity_score: Minimum liquidity score to allow trading
            max_spread_bps: Maximum spread in bps
            min_depth_usd: Minimum top of book depth in USD
            max_imbalance: Maximum order book imbalance (absolute value)
            regime_thresholds: Custom thresholds for each regime
        """
        self.min_liquidity_score = min_liquidity_score
        self.max_spread_bps = max_spread_bps
        self.min_depth_usd = min_depth_usd
        self.max_imbalance = max_imbalance
        
        # Default regime thresholds (liquidity score ranges)
        self.regime_thresholds = regime_thresholds or {
            LiquidityRegime.EXCELLENT: 0.9,
            LiquidityRegime.GOOD: 0.7,
            LiquidityRegime.FAIR: 0.5,
            LiquidityRegime.POOR: 0.3,
            LiquidityRegime.UNACCEPTABLE: 0.0
        }
        
        logger.info(
            "liquidity_regime_engine_initialized",
            min_liquidity_score=min_liquidity_score,
            max_spread_bps=max_spread_bps,
            min_depth_usd=min_depth_usd,
            max_imbalance=max_imbalance
        )
    
    def check_liquidity(
        self,
        symbol: str,
        spread_bps: float,
        bid_depth_usd: Optional[float] = None,
        ask_depth_usd: Optional[float] = None,
        bid_volume: Optional[float] = None,
        ask_volume: Optional[float] = None,
        top_of_book_depth_usd: Optional[float] = None,
        order_book_imbalance: Optional[float] = None
    ) -> LiquidityGateResult:
        """
        Check liquidity and determine if trading is allowed.
        
        Args:
            symbol: Trading symbol
            spread_bps: Current spread in bps
            bid_depth_usd: Bid depth in USD (optional)
            ask_depth_usd: Ask depth in USD (optional)
            bid_volume: Bid volume (optional)
            ask_volume: Ask volume (optional)
            top_of_book_depth_usd: Top of book depth in USD (optional)
            order_book_imbalance: Order book imbalance -1 to 1 (optional)
        
        Returns:
            LiquidityGateResult with pass/fail status
        """
        # Calculate top of book depth if not provided
        if top_of_book_depth_usd is None:
            if bid_depth_usd is not None and ask_depth_usd is not None:
                top_of_book_depth_usd = min(bid_depth_usd, ask_depth_usd)
            else:
                top_of_book_depth_usd = 0.0
        
        # Calculate order book imbalance if not provided
        if order_book_imbalance is None:
            if bid_volume is not None and ask_volume is not None:
                total_volume = bid_volume + ask_volume
                if total_volume > 0:
                    order_book_imbalance = (bid_volume - ask_volume) / total_volume
                else:
                    order_book_imbalance = 0.0
            else:
                order_book_imbalance = 0.0
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(
            spread_bps=spread_bps,
            top_of_book_depth_usd=top_of_book_depth_usd,
            order_book_imbalance=order_book_imbalance
        )
        
        # Determine regime
        regime = self._determine_regime(liquidity_score)
        
        # Check if trading is allowed
        can_trade = self._can_trade(
            spread_bps=spread_bps,
            top_of_book_depth_usd=top_of_book_depth_usd,
            order_book_imbalance=order_book_imbalance,
            liquidity_score=liquidity_score,
            regime=regime
        )
        
        # Create metrics
        metrics = LiquidityMetrics(
            spread_bps=spread_bps,
            top_of_book_depth_usd=top_of_book_depth_usd,
            order_book_imbalance=order_book_imbalance,
            liquidity_score=liquidity_score,
            regime=regime,
            can_trade=can_trade
        )
        
        # Create result
        result = LiquidityGateResult(
            passed=can_trade,
            regime=regime,
            liquidity_score=liquidity_score,
            metrics=metrics,
            rejection_reason=None if can_trade else self._get_rejection_reason(
                spread_bps, top_of_book_depth_usd, order_book_imbalance, liquidity_score, regime
            )
        )
        
        logger.debug(
            "liquidity_check",
            symbol=symbol,
            liquidity_score=liquidity_score,
            regime=regime.value,
            can_trade=can_trade,
            spread_bps=spread_bps,
            depth_usd=top_of_book_depth_usd
        )
        
        return result
    
    def _calculate_liquidity_score(
        self,
        spread_bps: float,
        top_of_book_depth_usd: float,
        order_book_imbalance: float
    ) -> float:
        """Calculate liquidity score (0-1, higher is better)"""
        # Spread component (lower spread = higher score)
        # Normalize spread: 0 bps = 1.0, max_spread_bps = 0.0
        spread_score = max(0.0, 1.0 - (spread_bps / self.max_spread_bps))
        
        # Depth component (higher depth = higher score)
        # Normalize depth: min_depth = 0.5, higher = 1.0
        if top_of_book_depth_usd >= self.min_depth_usd:
            depth_score = min(1.0, 0.5 + (top_of_book_depth_usd / self.min_depth_usd) * 0.5)
        else:
            depth_score = (top_of_book_depth_usd / self.min_depth_usd) * 0.5
        
        # Imbalance component (lower imbalance = higher score)
        # Normalize imbalance: 0 = 1.0, max_imbalance = 0.0
        imbalance_abs = abs(order_book_imbalance)
        imbalance_score = max(0.0, 1.0 - (imbalance_abs / self.max_imbalance))
        
        # Weighted combination
        liquidity_score = (
            spread_score * 0.4 +  # Spread is most important
            depth_score * 0.4 +   # Depth is also important
            imbalance_score * 0.2  # Imbalance is less critical
        )
        
        return float(liquidity_score)
    
    def _determine_regime(self, liquidity_score: float) -> LiquidityRegime:
        """Determine liquidity regime from score"""
        if liquidity_score >= self.regime_thresholds[LiquidityRegime.EXCELLENT]:
            return LiquidityRegime.EXCELLENT
        elif liquidity_score >= self.regime_thresholds[LiquidityRegime.GOOD]:
            return LiquidityRegime.GOOD
        elif liquidity_score >= self.regime_thresholds[LiquidityRegime.FAIR]:
            return LiquidityRegime.FAIR
        elif liquidity_score >= self.regime_thresholds[LiquidityRegime.POOR]:
            return LiquidityRegime.POOR
        else:
            return LiquidityRegime.UNACCEPTABLE
    
    def _can_trade(
        self,
        spread_bps: float,
        top_of_book_depth_usd: float,
        order_book_imbalance: float,
        liquidity_score: float,
        regime: LiquidityRegime
    ) -> bool:
        """Determine if trading is allowed"""
        # Must meet minimum liquidity score
        if liquidity_score < self.min_liquidity_score:
            return False
        
        # Must meet spread requirement
        if spread_bps > self.max_spread_bps:
            return False
        
        # Must meet depth requirement
        if top_of_book_depth_usd < self.min_depth_usd:
            return False
        
        # Must meet imbalance requirement
        if abs(order_book_imbalance) > self.max_imbalance:
            return False
        
        # Must not be in unacceptable regime
        if regime == LiquidityRegime.UNACCEPTABLE:
            return False
        
        return True
    
    def _get_rejection_reason(
        self,
        spread_bps: float,
        top_of_book_depth_usd: float,
        order_book_imbalance: float,
        liquidity_score: float,
        regime: LiquidityRegime
    ) -> str:
        """Get rejection reason"""
        reasons = []
        
        if liquidity_score < self.min_liquidity_score:
            reasons.append(f"Liquidity score too low: {liquidity_score:.2f} < {self.min_liquidity_score:.2f}")
        
        if spread_bps > self.max_spread_bps:
            reasons.append(f"Spread too wide: {spread_bps:.2f} bps > {self.max_spread_bps:.2f} bps")
        
        if top_of_book_depth_usd < self.min_depth_usd:
            reasons.append(f"Depth too low: ${top_of_book_depth_usd:.2f} < ${self.min_depth_usd:.2f}")
        
        if abs(order_book_imbalance) > self.max_imbalance:
            reasons.append(f"Imbalance too high: {abs(order_book_imbalance):.2f} > {self.max_imbalance:.2f}")
        
        if regime == LiquidityRegime.UNACCEPTABLE:
            reasons.append(f"Regime unacceptable: {regime.value}")
        
        return "; ".join(reasons) if reasons else "Unknown reason"
    
    def gate_signal(
        self,
        signal: any,  # AlphaSignal or similar
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        spread_bps: Optional[float] = None,
        bid_depth_usd: Optional[float] = None,
        ask_depth_usd: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Gate a signal through liquidity check.
        
        Args:
            signal: Trading signal
            liquidity_metrics: Pre-calculated liquidity metrics (optional)
            spread_bps: Spread in bps (optional)
            bid_depth_usd: Bid depth in USD (optional)
            ask_depth_usd: Ask depth in USD (optional)
        
        Returns:
            (passed, rejection_reason)
        """
        # If metrics provided, use them
        if liquidity_metrics:
            if liquidity_metrics.can_trade:
                return True, None
            else:
                return False, self._get_rejection_reason(
                    liquidity_metrics.spread_bps,
                    liquidity_metrics.top_of_book_depth_usd,
                    liquidity_metrics.order_book_imbalance,
                    liquidity_metrics.liquidity_score,
                    liquidity_metrics.regime
                )
        
        # Otherwise, check liquidity
        if spread_bps is None:
            # Try to get from signal metadata
            spread_bps = getattr(signal, 'spread_bps', 10.0)
        
        symbol = getattr(signal, 'symbol', 'UNKNOWN')
        
        result = self.check_liquidity(
            symbol=symbol,
            spread_bps=spread_bps,
            bid_depth_usd=bid_depth_usd,
            ask_depth_usd=ask_depth_usd
        )
        
        if result.passed:
            return True, None
        else:
            return False, result.rejection_reason

