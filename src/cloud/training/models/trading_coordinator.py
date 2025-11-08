"""
[FUTURE/PILOT - NOT USED IN ENGINE]

Trading Coordinator - Integrates Dual-Mode System with Alpha Engines

This module is for Pilot (Local Trader) live trading execution.
The Engine does NOT use this directly - Engine uses shadow_trader.py for LEARNING.

DO NOT USE in Engine daily training pipeline.
This will be used when building Pilot component.

NOTE: This is used by backtesting_framework.py for validation/testing,
but the primary use case is live trading in Pilot.

This is the main orchestration layer that ties everything together:
1. Alpha engines generate signals
2. Engine consensus validates signals
3. Mode selector determines scalp vs runner
4. Gate profiles filter by quality
5. Dual-book manager executes and tracks

Architecture:
    Market Data
        ↓
    Alpha Engines (6) → Signals
        ↓
    Engine Consensus → Validated signals
        ↓
    Mode Selector → Scalp vs Runner routing
        ↓
    Gate Profiles → Quality filtering
        ↓
    Dual-Book Manager → Position execution
        ↓
    P&L Tracking + Metrics

Key Features:
- End-to-end signal-to-execution pipeline
- Automatic mode selection per signal
- Independent gate profiles per mode
- Heat management across both books
- Comprehensive metrics and monitoring
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time

import numpy as np
import structlog

from ..validation.validators import (
    validate_symbol,
    validate_price,
    validate_confidence,
    validate_features,
    validate_regime,
    validate_size,
)

# Import all our components
from .alpha_engines import (
    TrendEngine,
    RangeEngine,
    BreakoutEngine,
    TapeEngine,
    LeaderEngine,
    SweepEngine,
    TradingTechnique,
    AlphaSignal,
)
from .engine_consensus import EngineConsensus, EngineOpinion, ConsensusResult
from .mode_selector import ModeSelector, PreferredMode
from .gate_profiles import ScalpGateProfile, RunnerGateProfile, HybridGateRouter
from .dual_book_manager import DualBookManager, AssetProfile, BookType
from .cost_gate import OrderType
from .gate_counterfactuals import GateCounterfactualTracker

logger = structlog.get_logger(__name__)


@dataclass
class TradeDecision:
    """Final trade decision with all context."""

    # Decision
    approved: bool
    symbol: str
    direction: str  # 'long' or 'short'
    book: Optional[BookType]
    size_usd: float

    # Signal info
    technique: str
    confidence: float
    regime: str
    edge_hat_bps: float

    # Consensus
    consensus_level: str
    agreement_score: float

    # Mode selection
    preferred_mode: str
    mode_reason: str

    # Gate analysis
    gates_passed: int
    gates_blocked: int
    edge_net_bps: float
    win_probability: float

    # Metadata
    timestamp: float
    reasoning: str


class TradingCoordinator:
    """
    Main coordinator for dual-mode trading system.

    Orchestrates the entire signal-to-execution pipeline:
    1. Alpha engines analyze market and generate signals
    2. Engine consensus validates and adjusts confidence
    3. Mode selector routes to scalp vs runner
    4. Gate profiles filter by quality
    5. Dual-book manager executes trades
    6. Counterfactual tracker monitors gate performance
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        asset_symbols: List[str] = None,
        max_short_heat: float = 0.40,
        max_long_heat: float = 0.50,
        reserve_heat: float = 0.10,
    ):
        """
        Initialize trading coordinator.

        Args:
            total_capital: Total trading capital
            asset_symbols: List of symbols to trade
            max_short_heat: Max heat for scalp book
            max_long_heat: Max heat for runner book
            reserve_heat: Reserve allocation
        """
        self.total_capital = total_capital
        self.asset_symbols = asset_symbols or ['ETH-USD', 'SOL-USD', 'BTC-USD']

        # 1. Alpha Engines
        self.trend_engine = TrendEngine()
        self.range_engine = RangeEngine()
        self.breakout_engine = BreakoutEngine()
        self.tape_engine = TapeEngine()
        self.leader_engine = LeaderEngine()
        self.sweep_engine = SweepEngine()

        # 2. Engine Consensus
        self.consensus = EngineConsensus()

        # 3. Dual-Book Manager
        self.book_manager = DualBookManager(
            total_capital=total_capital,
            max_short_heat=max_short_heat,
            max_long_heat=max_long_heat,
            reserve_heat=reserve_heat,
        )

        # Set asset profiles
        for symbol in self.asset_symbols:
            self.book_manager.set_asset_profile(
                symbol,
                AssetProfile(
                    allowed_books=[BookType.SHORT_HOLD, BookType.LONG_HOLD],
                    scalp_target_bps=100.0,  # £1 on £100
                    runner_target_bps=800.0,  # £8 on £100
                    scalp_max_size=200.0,
                    runner_max_size=1000.0,
                ),
            )

        # 4. Gate Profiles
        self.scalp_profile = ScalpGateProfile()
        self.runner_profile = RunnerGateProfile()
        self.hybrid_router = HybridGateRouter()

        # 5. Mode Selector
        self.mode_selector = ModeSelector(
            dual_book_manager=self.book_manager,
            scalp_profile=self.scalp_profile,
            runner_profile=self.runner_profile,
            hybrid_router=self.hybrid_router,
        )

        # 6. Counterfactual Tracker
        self.counterfactual_tracker = GateCounterfactualTracker(
            hold_duration_sec=300.0,  # 5 minutes
            min_trades_for_tuning=50,
            false_negative_threshold=0.30,
        )

        # Statistics
        self.total_signals_processed = 0
        self.trades_executed = 0
        self.trades_blocked_by_consensus = 0
        self.trades_blocked_by_gates = 0
        self.trades_blocked_by_heat = 0

        # Current prices (for counterfactual simulation)
        self.current_prices: Dict[str, float] = {}

        logger.info(
            "trading_coordinator_initialized",
            capital=total_capital,
            assets=self.asset_symbols,
            scalp_heat=max_short_heat,
            runner_heat=max_long_heat,
        )

    def process_signal(
        self,
        symbol: str,
        price: float,
        features: Dict[str, float],
        regime: str,
        spread_bps: float = 8.0,
        liquidity_score: float = 0.75,
    ) -> Optional[TradeDecision]:
        """
        Process market data and potentially generate trade.

        Args:
            symbol: Asset symbol
            price: Current price
            features: Market features
            regime: Current market regime
            spread_bps: Current spread
            liquidity_score: Liquidity score

        Returns:
            TradeDecision if trade generated, None otherwise
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate all inputs
        try:
            symbol = validate_symbol(symbol)
            price = validate_price(price)
            features = validate_features(features)
            regime = validate_regime(regime)
            
            if not isinstance(spread_bps, (int, float)) or spread_bps < 0:
                raise ValueError(f"Invalid spread_bps: must be non-negative, got {spread_bps}")
            if not isinstance(liquidity_score, (int, float)) or not 0 <= liquidity_score <= 1:
                raise ValueError(f"Invalid liquidity_score: must be between 0 and 1, got {liquidity_score}")
        except ValueError as e:
            logger.error("invalid_input", error=str(e), symbol=symbol, price=price)
            raise
        
        self.total_signals_processed += 1
        self.current_prices[symbol] = price

        # 1. Get signals from all 6 alpha engines
        trend_signal = self.trend_engine.generate_signal(features, regime)
        range_signal = self.range_engine.generate_signal(features, regime)
        breakout_signal = self.breakout_engine.generate_signal(features, regime)
        tape_signal = self.tape_engine.generate_signal(features, regime)
        leader_signal = self.leader_engine.generate_signal(features, regime)
        sweep_signal = self.sweep_engine.generate_signal(features, regime)

        all_signals = [
            trend_signal,
            range_signal,
            breakout_signal,
            tape_signal,
            leader_signal,
            sweep_signal,
        ]

        # Find primary signal (highest confidence)
        primary_signal = max(all_signals, key=lambda s: s.confidence)

        if primary_signal.direction == 'hold' or primary_signal.confidence < 0.50:
            return None  # No tradeable signal

        # 2. Get consensus from all engines
        opinions = [
            EngineOpinion(
                technique=s.technique,
                direction=s.direction,
                confidence=s.confidence,
                reasoning=s.reasoning,
                supporting_factors=list(s.key_features.keys()),
            )
            for s in all_signals
        ]

        consensus_result = self.consensus.analyze_consensus(
            primary_engine=primary_signal.technique,
            primary_confidence=primary_signal.confidence,
            all_opinions=opinions,
            current_regime=regime,
        )

        # Block if consensus says skip
        if consensus_result.recommendation == 'SKIP_TRADE':
            self.trades_blocked_by_consensus += 1
            logger.info(
                "trade_blocked_by_consensus",
                symbol=symbol,
                technique=primary_signal.technique.value,
                consensus=consensus_result.consensus_level.value,
                reason=consensus_result.reasoning,
            )
            return None

        # Adjust confidence by consensus
        adjusted_confidence = consensus_result.adjusted_confidence

        # 3. Estimate edge
        edge_hat_bps = self._estimate_edge(
            confidence=adjusted_confidence,
            regime=regime,
            technique=primary_signal.technique.value,
        )

        # Add engine confidence to features for gates
        gate_features = {
            **features,
            'engine_conf': adjusted_confidence,
            'regime': regime,
            'technique': primary_signal.technique.value,
        }

        # 4. Route through mode selector
        routing_decision = self.mode_selector.route_signal(
            technique=primary_signal.technique.value,
            confidence=adjusted_confidence,
            regime=regime,
            edge_hat_bps=edge_hat_bps,
            features=gate_features,
            symbol=symbol,
            order_type=OrderType.MAKER,
            position_size_usd=200.0,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            urgency='moderate',
        )

        # Check if approved
        if not routing_decision.approved:
            # Safely check heat_limit_hit attribute
            if hasattr(routing_decision, 'heat_limit_hit') and routing_decision.heat_limit_hit:
                self.trades_blocked_by_heat += 1
            else:
                self.trades_blocked_by_gates += 1

                # Record counterfactual for blocked trade
                if hasattr(routing_decision, 'gate_decision') and routing_decision.gate_decision:
                    blocked_reason = "Gates blocked"
                    if hasattr(routing_decision.gate_decision, 'gates_blocked'):
                        blocked_reason = f"Gates blocked: {routing_decision.gate_decision.gates_blocked}"
                    
                    self.counterfactual_tracker.record_blocked_trade(
                        gate_name='combined_gates',
                        symbol=symbol,
                        direction='long' if primary_signal.direction == 'buy' else 'short',
                        entry_price=price,
                        size=200.0 / price if price > 0 else 0,  # Convert USD to asset units
                        blocked_reason=blocked_reason,
                        gate_features=gate_features,
                        technique=primary_signal.technique.value,
                        regime=regime,
                        confidence=adjusted_confidence,
                    )

            return None

        # 5. Execute trade in dual-book manager
        direction_map = {'buy': 'long', 'sell': 'short'}
        direction = direction_map.get(primary_signal.direction, 'long')

        # Safely calculate position size (avoid division by zero)
        if price <= 0:
            logger.warning("invalid_price_for_trade", symbol=symbol, price=price)
            return None
        
        recommended_size = routing_decision.recommended_size if hasattr(routing_decision, 'recommended_size') else 200.0
        position_size = recommended_size / price  # Convert to asset units

        position = self.book_manager.add_position(
            symbol=symbol,
            book=routing_decision.recommended_book if hasattr(routing_decision, 'recommended_book') else None,
            entry_price=price,
            size=position_size,
            direction=direction,
            technique=primary_signal.technique.value,
            regime=regime,
            confidence=adjusted_confidence,
        )

        if position:
            self.trades_executed += 1

            edge_net = 0
            if hasattr(routing_decision, 'gate_decision') and routing_decision.gate_decision:
                if hasattr(routing_decision.gate_decision, 'edge_net_bps'):
                    edge_net = routing_decision.gate_decision.edge_net_bps

            logger.info(
                "trade_executed",
                symbol=symbol,
                book=routing_decision.recommended_book.value if hasattr(routing_decision.recommended_book, 'value') else str(routing_decision.recommended_book),
                technique=primary_signal.technique.value,
                confidence=adjusted_confidence,
                edge_net=edge_net,
                size_usd=routing_decision.recommended_size if hasattr(routing_decision, 'recommended_size') else 0,
            )

            # Safely extract gate decision attributes
            gates_passed = 0
            gates_blocked = 0
            edge_net_bps = 0
            win_probability = 0
            
            if hasattr(routing_decision, 'gate_decision') and routing_decision.gate_decision:
                if hasattr(routing_decision.gate_decision, 'gates_passed'):
                    gates_passed = routing_decision.gate_decision.gates_passed
                if hasattr(routing_decision.gate_decision, 'gates_blocked'):
                    gates_blocked = routing_decision.gate_decision.gates_blocked
                if hasattr(routing_decision.gate_decision, 'edge_net_bps'):
                    edge_net_bps = routing_decision.gate_decision.edge_net_bps
                if hasattr(routing_decision.gate_decision, 'win_probability'):
                    win_probability = routing_decision.gate_decision.win_probability

            # Create trade decision record
            return TradeDecision(
                approved=True,
                symbol=symbol,
                direction=direction,
                book=routing_decision.recommended_book if hasattr(routing_decision, 'recommended_book') else None,
                size_usd=routing_decision.recommended_size if hasattr(routing_decision, 'recommended_size') else 0,
                technique=primary_signal.technique.value,
                confidence=adjusted_confidence,
                regime=regime,
                edge_hat_bps=edge_hat_bps,
                consensus_level=consensus_result.consensus_level.value if hasattr(consensus_result.consensus_level, 'value') else str(consensus_result.consensus_level),
                agreement_score=consensus_result.agreement_score,
                preferred_mode=routing_decision.preferred_mode.value if hasattr(routing_decision, 'preferred_mode') and hasattr(routing_decision.preferred_mode, 'value') else 'unknown',
                mode_reason=routing_decision.mode_reason if hasattr(routing_decision, 'mode_reason') else '',
                gates_passed=gates_passed,
                gates_blocked=gates_blocked,
                edge_net_bps=edge_net_bps,
                win_probability=win_probability,
                timestamp=time.time(),
                reasoning=primary_signal.reasoning,
            )

        return None

    def _estimate_edge(
        self,
        confidence: float,
        regime: str,
        technique: str,
    ) -> float:
        """
        Estimate edge in bps based on confidence, regime, and technique.

        Simple heuristic (in production, use ML model):
        - Base edge = confidence * 20 bps
        - Regime bonus: TREND +5 bps, RANGE +2 bps, PANIC -5 bps
        - Technique bonus: varies by technique
        """
        base_edge = confidence * 20.0

        # Regime adjustment
        regime_bonus = {
            'TREND': 5.0,
            'RANGE': 2.0,
            'PANIC': -5.0,
        }.get(regime.upper(), 0.0)

        # Technique adjustment
        technique_bonus = {
            'TREND': 3.0,  # Good in trends
            'BREAKOUT': 5.0,  # High potential
            'RANGE': 2.0,  # Small edges
            'TAPE': 4.0,  # Microstructure edge
            'LEADER': 3.0,
            'SWEEP': 4.0,
        }.get(technique.upper(), 0.0)

        return base_edge + regime_bonus + technique_bonus

    def update_positions(self, symbol: str, current_price: float) -> None:
        """
        Update all positions for a symbol with current price.

        Args:
            symbol: Asset symbol
            current_price: Current price
        """
        self.current_prices[symbol] = current_price

        # Update all positions in both books
        for book_type in [BookType.SHORT_HOLD, BookType.LONG_HOLD]:
            positions = self.book_manager._get_book_positions(book_type, open_only=True)

            if positions:  # Check if positions list is not None/empty
                for position in positions:
                    if position and hasattr(position, 'symbol') and position.symbol == symbol:
                        if hasattr(position, 'position_id'):
                            self.book_manager.update_position_price(
                                position.position_id,
                                current_price,
                            )

    def simulate_pending_counterfactuals(self) -> int:
        """Simulate all pending counterfactuals with current prices."""
        return self.counterfactual_tracker.simulate_all_pending(self.current_prices)

    def get_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        # Book metrics
        book_summary = self.book_manager.get_summary()

        # Mode selector stats
        mode_stats = self.mode_selector.get_statistics()

        # Counterfactual stats
        cf_summary = self.counterfactual_tracker.get_summary()

        return {
            'coordinator': {
                'total_signals': self.total_signals_processed,
                'trades_executed': self.trades_executed,
                'blocked_consensus': self.trades_blocked_by_consensus,
                'blocked_gates': self.trades_blocked_by_gates,
                'blocked_heat': self.trades_blocked_by_heat,
                'execution_rate': (
                    self.trades_executed / self.total_signals_processed
                    if self.total_signals_processed > 0
                    else 0.0
                ),
            },
            'books': book_summary,
            'mode_selector': mode_stats,
            'counterfactuals': cf_summary,
        }

    def get_detailed_status(self) -> str:
        """Get human-readable status summary."""
        metrics = self.get_metrics()

        status = []
        status.append("=" * 70)
        status.append("TRADING COORDINATOR STATUS")
        status.append("=" * 70)

        # Overall
        coord = metrics['coordinator']
        status.append(f"\nSignals Processed: {coord['total_signals']}")
        status.append(f"Trades Executed: {coord['trades_executed']} ({coord['execution_rate']:.1%})")
        status.append(f"Blocked: Consensus={coord['blocked_consensus']}, Gates={coord['blocked_gates']}, Heat={coord['blocked_heat']}")

        # Books
        books = metrics['books']
        status.append(f"\n{'SCALP BOOK':-^70}")
        scalp = books['short_book']
        status.append(f"Positions: {scalp['positions']} | Heat: {scalp['heat']:.1%}")
        status.append(f"P&L: ${scalp['pnl']:.2f} | Win Rate: {scalp['win_rate']:.1%}")

        status.append(f"\n{'RUNNER BOOK':-^70}")
        runner = books['long_book']
        status.append(f"Positions: {runner['positions']} | Heat: {runner['heat']:.1%}")
        status.append(f"P&L: ${runner['pnl']:.2f} | Win Rate: {runner['win_rate']:.1%}")

        status.append(f"\n{'COMBINED':-^70}")
        combined = books['combined']
        status.append(f"Total P&L: ${combined['total_realized_pnl']:.2f}")
        status.append(f"Total Trades: {combined['total_trades']}")

        # Mode routing
        mode = metrics['mode_selector']
        status.append(f"\n{'MODE ROUTING':-^70}")
        status.append(f"Scalp: {mode['routed_to_scalp']} ({mode['scalp_ratio']:.0%})")
        status.append(f"Runner: {mode['routed_to_runner']} ({mode['runner_ratio']:.0%})")

        # Counterfactuals
        cf = metrics['counterfactuals']
        if cf.get('total_blocks', 0) > 0:
            status.append(f"\n{'GATE PERFORMANCE':-^70}")
            status.append(f"Blocks: {cf['total_blocks']}")
            status.append(f"Saved: ${cf['total_saved_usd']:.2f} | Missed: ${cf['total_missed_usd']:.2f}")
            status.append(f"Net Value: ${cf['net_value_usd']:.2f}")

        status.append(f"\n{'='*70}\n")

        return "\n".join(status)
