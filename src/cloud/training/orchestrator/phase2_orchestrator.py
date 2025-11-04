"""
Phase 2 Orchestrator

Integrates all Phase 2 modules into a cohesive trading system:
- Multi-Symbol Coordination
- Enhanced Risk Management
- Advanced Pattern Recognition
- Portfolio-Level Learning

This is the glue that makes all Phase 2 features work together.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

import polars as pl
import structlog

from ..models.advanced_patterns import AdvancedPatternDetector, PatternDetection
from ..models.enhanced_risk_manager import EnhancedRiskManager, PositionSizingResult
from ..models.multi_symbol_coordinator import MultiSymbolCoordinator, SymbolPosition
from ..models.portfolio_learner import PortfolioLearner

logger = structlog.get_logger()


@dataclass
class TradingDecision:
    """Complete trading decision with all Phase 2 intelligence."""

    symbol: str
    action: str  # "enter", "exit", "hold", "skip"
    recommended_size_gbp: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    confidence: float
    detected_patterns: List[PatternDetection]
    risk_reasoning: str
    can_trade: bool
    blocking_reason: Optional[str]


class Phase2Orchestrator:
    """
    Orchestrates all Phase 2 features for intelligent portfolio trading.

    This module coordinates:
    1. Pattern detection across all symbols
    2. Risk management and position sizing
    3. Multi-symbol correlation checks
    4. Portfolio-level learning
    """

    def __init__(
        self,
        total_capital_gbp: float = 10000.0,
        base_position_size_gbp: float = 100.0,
    ):
        """
        Initialize Phase 2 orchestrator.

        Args:
            total_capital_gbp: Total available capital
            base_position_size_gbp: Base position size
        """
        self.total_capital = total_capital_gbp

        # Initialize all Phase 2 modules
        self.multi_symbol_coordinator = MultiSymbolCoordinator(
            max_portfolio_heat=0.7,
            max_correlated_exposure=0.4,
            correlation_threshold=0.7,
        )

        self.risk_manager = EnhancedRiskManager(
            base_position_size_gbp=base_position_size_gbp,
            max_position_multiplier=2.0,
            min_position_multiplier=0.25,
            kelly_fraction=0.25,
        )

        self.pattern_detector = AdvancedPatternDetector(
            min_pattern_quality=0.6,
            lookback_periods=50,
        )

        self.portfolio_learner = PortfolioLearner(
            ema_alpha=0.05,
            cross_symbol_threshold=3,
        )

        # Track active symbols
        self.active_symbols: Set[str] = set()

        logger.info(
            "phase2_orchestrator_initialized",
            total_capital_gbp=total_capital_gbp,
            base_size_gbp=base_position_size_gbp,
        )

    def evaluate_entry(
        self,
        symbol: str,
        df: pl.DataFrame,
        current_idx: int,
        confidence: float,
        asset_volatility_bps: float,
        current_drawdown_pct: float = 0.0,
        win_rate: Optional[float] = None,
        avg_win_pct: Optional[float] = None,
        avg_loss_pct: Optional[float] = None,
    ) -> TradingDecision:
        """
        Evaluate whether to enter a position using all Phase 2 intelligence.

        Args:
            symbol: Symbol to trade
            df: Price data
            current_idx: Current index
            confidence: Base confidence score
            asset_volatility_bps: Asset volatility
            current_drawdown_pct: Portfolio drawdown
            win_rate: Historical win rate
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage

        Returns:
            TradingDecision with all analysis
        """
        # 1. Detect patterns
        patterns = self.pattern_detector.detect_patterns(df, current_idx)

        # Boost confidence if high-quality patterns detected
        pattern_boost = 0.0
        if patterns:
            best_pattern_quality = max(p.quality_score for p in patterns)
            pattern_boost = best_pattern_quality * 0.1  # Up to +10% confidence

        adjusted_confidence = min(1.0, confidence + pattern_boost)

        # 2. Calculate position size using enhanced risk management
        position_sizing = self.risk_manager.calculate_position_size(
            confidence=adjusted_confidence,
            asset_volatility_bps=asset_volatility_bps,
            current_drawdown_pct=current_drawdown_pct,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
        )

        # 3. Check multi-symbol coordination
        can_enter, blocking_reason = self.multi_symbol_coordinator.can_enter_position(
            symbol=symbol,
            position_size_gbp=position_sizing.recommended_size_gbp,
            total_capital_gbp=self.total_capital,
        )

        # 4. Calculate stop loss and take profit
        current_price = float(df["close"][current_idx])

        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price=current_price,
            asset_volatility_bps=asset_volatility_bps,
            confidence=adjusted_confidence,
        )

        take_profit = self.risk_manager.calculate_take_profit(
            entry_price=current_price,
            asset_volatility_bps=asset_volatility_bps,
            confidence=adjusted_confidence,
            risk_reward_ratio=2.0,
        )

        # 5. Make final decision
        action = "enter" if can_enter else "skip"

        return TradingDecision(
            symbol=symbol,
            action=action,
            recommended_size_gbp=position_sizing.recommended_size_gbp,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            confidence=adjusted_confidence,
            detected_patterns=patterns,
            risk_reasoning=position_sizing.reasoning,
            can_trade=can_enter,
            blocking_reason=blocking_reason if not can_enter else None,
        )

    def record_position_entry(
        self,
        symbol: str,
        entry_price: float,
        entry_timestamp: datetime,
        position_size_gbp: float,
        regime: str,
    ) -> None:
        """
        Record a new position entry.

        Args:
            symbol: Symbol
            entry_price: Entry price
            entry_timestamp: Entry timestamp
            position_size_gbp: Position size
            regime: Market regime
        """
        position = SymbolPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            position_size_gbp=position_size_gbp,
            regime=regime,
        )

        self.multi_symbol_coordinator.add_position(position)
        self.active_symbols.add(symbol)

        logger.info(
            "position_recorded",
            symbol=symbol,
            size_gbp=position_size_gbp,
            total_positions=len(self.multi_symbol_coordinator.positions),
        )

    def record_position_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_timestamp: datetime,
        is_winner: bool,
        pnl_gbp: float,
    ) -> None:
        """
        Record a position exit.

        Args:
            symbol: Symbol
            exit_price: Exit price
            exit_timestamp: Exit timestamp
            is_winner: Whether trade was profitable
            pnl_gbp: Profit/loss in GBP
        """
        position = self.multi_symbol_coordinator.remove_position(symbol)
        self.active_symbols.discard(symbol)

        if position:
            logger.info(
                "position_exited",
                symbol=symbol,
                pnl_gbp=pnl_gbp,
                is_winner=is_winner,
                total_positions=len(self.multi_symbol_coordinator.positions),
            )

    def update_price(
        self, symbol: str, timestamp: datetime, price: float
    ) -> None:
        """
        Update price for correlation tracking.

        Args:
            symbol: Symbol
            timestamp: Price timestamp
            price: Current price
        """
        self.multi_symbol_coordinator.update_price(symbol, timestamp, price)

    def update_correlations(self, timestamp: datetime) -> None:
        """
        Update all symbol correlations.

        Args:
            timestamp: Current timestamp
        """
        self.multi_symbol_coordinator.update_correlations(timestamp)

        correlation_groups = self.multi_symbol_coordinator.correlation_groups
        if correlation_groups:
            logger.debug(
                "correlations_updated",
                num_groups=len(correlation_groups),
                timestamp=timestamp.isoformat(),
            )

    def record_portfolio_outcome(
        self,
        timestamp: datetime,
        total_pnl_gbp: float,
        win_rate: float,
        portfolio_regime: str,
        features: Dict[str, float],
    ) -> None:
        """
        Record portfolio-level outcome for learning.

        Args:
            timestamp: Outcome timestamp
            total_pnl_gbp: Total portfolio P&L
            win_rate: Win rate across positions
            portfolio_regime: Portfolio regime
            features: Portfolio features
        """
        symbols = list(self.active_symbols)

        self.portfolio_learner.update_portfolio_outcome(
            timestamp=timestamp,
            symbols=symbols,
            total_pnl_gbp=total_pnl_gbp,
            win_rate=win_rate,
            num_positions=len(symbols),
            portfolio_regime=portfolio_regime,
            features=features,
        )

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary."""
        portfolio_state = self.multi_symbol_coordinator.get_portfolio_state(
            self.total_capital
        )

        return {
            "total_exposure_gbp": portfolio_state.total_exposure_gbp,
            "total_pnl_gbp": portfolio_state.total_unrealized_pnl_gbp,
            "num_positions": portfolio_state.num_positions,
            "heat_score": portfolio_state.heat_score,
            "correlation_groups": len(self.multi_symbol_coordinator.correlation_groups),
            "portfolio_success_rate": self.portfolio_learner.get_portfolio_success_rate(),
        }

    def should_reduce_exposure(
        self, current_drawdown_pct: float
    ) -> tuple[bool, str]:
        """
        Check if portfolio exposure should be reduced.

        Args:
            current_drawdown_pct: Current portfolio drawdown

        Returns:
            (should_reduce, reason) tuple
        """
        return self.multi_symbol_coordinator.should_reduce_exposure(
            self.total_capital, current_drawdown_pct
        )

    def get_state(self) -> Dict:
        """Get complete state for persistence."""
        return {
            "portfolio_learner": self.portfolio_learner.get_state(),
            "total_capital": self.total_capital,
            "active_symbols": list(self.active_symbols),
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        self.portfolio_learner.load_state(state.get("portfolio_learner", {}))
        self.total_capital = state.get("total_capital", self.total_capital)
        self.active_symbols = set(state.get("active_symbols", []))

        logger.info(
            "phase2_orchestrator_state_loaded",
            portfolio_sessions=self.portfolio_learner.total_portfolio_sessions,
        )
