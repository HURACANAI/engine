"""
Mode Selector - Intelligent Signal Routing to Scalp vs Runner Books

Key Architecture Decision:
Some signals are naturally better for scalping (fast, microstructure-based)
Others are better for runners (high-conviction, trend-following)

Mode Selector Logic:
1. Analyze signal characteristics (technique, confidence, regime, urgency)
2. Determine preferred mode (SCALP, RUNNER, or BOTH)
3. Route to gate profiles accordingly
4. Respect heat limits per book

Signal-to-Mode Mapping:
┌─────────────────┬─────────────┬────────────────────────┐
│ Technique       │ Regime      │ Preferred Mode         │
├─────────────────┼─────────────┼────────────────────────┤
│ TAPE (micro)    │ Any         │ SCALP (fast, precise)  │
│ SWEEP (liqui)   │ Any         │ SCALP (fast, precise)  │
│ TREND (strong)  │ TREND       │ RUNNER (hold winners)  │
│ BREAKOUT        │ TREND       │ RUNNER (explosive)     │
│ RANGE (mean)    │ RANGE       │ SCALP (quick in/out)   │
│ LEADER (relstr) │ TREND       │ BOTH (depends on conf) │
└─────────────────┴─────────────┴────────────────────────┘

Examples:
    # Fast microstructure signal → SCALP
    TAPE engine, 0.68 confidence, RANGE regime
    → Try scalp gates only (target: £1-£2, quick exit)

    # High-conviction trend → RUNNER
    TREND engine, 0.82 confidence, TREND regime
    → Try runner gates first (target: £10+, let it run)

    # Medium signal → BOTH
    BREAKOUT engine, 0.65 confidence, TREND regime
    → Try runner first, fallback to scalp

Usage:
    selector = ModeSelector(
        dual_book_manager=book_manager,
        scalp_profile=scalp_profile,
        runner_profile=runner_profile,
    )

    # Route signal
    decision = selector.route_signal(
        technique='TREND',
        confidence=0.82,
        regime='TREND',
        edge_hat_bps=18.0,
        features={...},
    )

    if decision.approved:
        # Open position in recommended book
        book_manager.add_position(
            symbol='ETH-USD',
            book=decision.recommended_book,
            entry_price=2000.0,
            size=decision.recommended_size,
            direction='long',
        )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import structlog

from .gate_profiles import (
    ScalpGateProfile,
    RunnerGateProfile,
    HybridGateRouter,
    GateDecision,
    OrderType,
)
from .dual_book_manager import DualBookManager, BookType

logger = structlog.get_logger(__name__)


class PreferredMode(Enum):
    """Preferred trading mode for a signal."""

    SCALP_ONLY = "scalp_only"  # Fast techniques (TAPE, SWEEP)
    RUNNER_FIRST = "runner_first"  # High-conviction (TREND, BREAKOUT in TREND)
    BOTH = "both"  # Medium signals, try runner then scalp


@dataclass
class RoutingDecision:
    """Final routing decision with position sizing."""

    approved: bool  # Can we trade?
    recommended_book: Optional[BookType]
    recommended_size: float  # Position size in USD

    # Gate analysis
    gate_decision: Optional[GateDecision]

    # Mode selection
    preferred_mode: PreferredMode
    mode_reason: str

    # Constraints
    heat_available: float
    heat_limit_hit: bool

    # Details
    technique: str
    confidence: float
    regime: str


class ModeSelector:
    """
    Intelligently route signals to scalp vs runner books.

    Architecture:
    1. Determine preferred mode (technique + regime + confidence)
    2. Check heat limits for each book
    3. Route to appropriate gate profile(s)
    4. Calculate position size
    5. Return decision with book recommendation

    Key Features:
    - Technique-specific routing (TAPE → scalp, TREND → runner)
    - Regime-aware preferences (TREND regime → more runners)
    - Confidence-based fallback (high conf → runner, med → scalp)
    - Independent heat tracking per book
    """

    def __init__(
        self,
        dual_book_manager: DualBookManager,
        scalp_profile: Optional[ScalpGateProfile] = None,
        runner_profile: Optional[RunnerGateProfile] = None,
        hybrid_router: Optional[HybridGateRouter] = None,
    ):
        """
        Initialize mode selector.

        Args:
            dual_book_manager: Dual-book position manager
            scalp_profile: Scalp gate profile (or uses default)
            runner_profile: Runner gate profile (or uses default)
            hybrid_router: Hybrid router (or uses default)
        """
        self.book_manager = dual_book_manager

        # Gate profiles
        self.scalp_profile = scalp_profile or ScalpGateProfile()
        self.runner_profile = runner_profile or RunnerGateProfile()
        self.hybrid_router = hybrid_router or HybridGateRouter()

        # Routing statistics
        self.total_signals = 0
        self.routed_to_scalp = 0
        self.routed_to_runner = 0
        self.heat_blocked = 0
        self.gate_blocked = 0

        logger.info("mode_selector_initialized")

    def determine_preferred_mode(
        self,
        technique: str,
        confidence: float,
        regime: str,
    ) -> tuple[PreferredMode, str]:
        """
        Determine preferred mode based on signal characteristics.

        Args:
            technique: Trading technique (TREND, TAPE, etc.)
            confidence: Engine confidence (0-1)
            regime: Market regime (TREND, RANGE, PANIC)

        Returns:
            (preferred_mode, reason)
        """
        technique_upper = technique.upper()

        # TAPE and SWEEP → Always scalp (microstructure, fast)
        if technique_upper in ['TAPE', 'SWEEP']:
            return PreferredMode.SCALP_ONLY, f"{technique_upper} technique = fast scalp"

        # RANGE technique → Always scalp (mean reversion, quick)
        if technique_upper == 'RANGE':
            return PreferredMode.SCALP_ONLY, "RANGE technique = quick in/out"

        # TREND or BREAKOUT in TREND regime with high confidence → Runner first
        if technique_upper in ['TREND', 'BREAKOUT']:
            if regime.upper() == 'TREND' and confidence >= 0.70:
                return (
                    PreferredMode.RUNNER_FIRST,
                    f"{technique_upper} + TREND regime + high conf = let it run",
                )
            elif regime.upper() == 'TREND':
                return PreferredMode.BOTH, f"{technique_upper} + TREND = try runner, fallback scalp"
            else:
                return PreferredMode.SCALP_ONLY, f"{technique_upper} in {regime} = scalp only"

        # LEADER (relative strength) → Depends on confidence
        if technique_upper == 'LEADER':
            if confidence >= 0.75:
                return PreferredMode.RUNNER_FIRST, "LEADER + high conf = runner"
            else:
                return PreferredMode.BOTH, "LEADER + med conf = try both"

        # PANIC regime → Scalp only (too volatile for runners)
        if regime.upper() == 'PANIC':
            return PreferredMode.SCALP_ONLY, "PANIC regime = scalp only"

        # Default: Try both
        return PreferredMode.BOTH, "Default = try both modes"

    def route_signal(
        self,
        technique: str,
        confidence: float,
        regime: str,
        edge_hat_bps: float,
        features: Dict[str, float],
        symbol: str,
        order_type: OrderType = OrderType.MAKER,
        position_size_usd: float = 200.0,
        spread_bps: float = 8.0,
        liquidity_score: float = 0.75,
        urgency: str = 'moderate',
        **kwargs,
    ) -> RoutingDecision:
        """
        Route signal through mode selection and gate profiles.

        Args:
            technique: Trading technique
            confidence: Engine confidence
            regime: Market regime
            edge_hat_bps: Predicted edge
            features: Feature dict
            symbol: Asset symbol
            order_type: Order type
            position_size_usd: Desired position size
            spread_bps: Current spread
            liquidity_score: Liquidity score
            urgency: Urgency level
            **kwargs: Additional args for gates

        Returns:
            RoutingDecision with approval and book recommendation
        """
        self.total_signals += 1

        # 1. Determine preferred mode
        preferred_mode, mode_reason = self.determine_preferred_mode(
            technique=technique,
            confidence=confidence,
            regime=regime,
        )

        # 2. Check heat availability
        runner_heat = self.book_manager.get_book_heat(BookType.LONG_HOLD)
        scalp_heat = self.book_manager.get_book_heat(BookType.SHORT_HOLD)

        runner_can_add, runner_reason = self.book_manager.can_add_position(
            book=BookType.LONG_HOLD,
            size_usd=position_size_usd,
            symbol=symbol,
        )

        scalp_can_add, scalp_reason = self.book_manager.can_add_position(
            book=BookType.SHORT_HOLD,
            size_usd=position_size_usd,
            symbol=symbol,
        )

        # 3. Route based on preferred mode
        gate_decision = None

        if preferred_mode == PreferredMode.SCALP_ONLY:
            # Try scalp only
            if not scalp_can_add:
                self.heat_blocked += 1
                return RoutingDecision(
                    approved=False,
                    recommended_book=None,
                    recommended_size=0.0,
                    gate_decision=None,
                    preferred_mode=preferred_mode,
                    mode_reason=mode_reason,
                    heat_available=1.0 - scalp_heat,
                    heat_limit_hit=True,
                    technique=technique,
                    confidence=confidence,
                    regime=regime,
                )

            gate_decision = self.scalp_profile.check_all_gates(
                edge_hat_bps=edge_hat_bps,
                features=features,
                order_type=order_type,
                position_size_usd=position_size_usd,
                spread_bps=spread_bps,
                liquidity_score=liquidity_score,
                urgency=urgency,
            )

            if gate_decision.passes:
                self.routed_to_scalp += 1

        elif preferred_mode == PreferredMode.RUNNER_FIRST:
            # Try runner first, fallback to scalp if heat limited
            if runner_can_add:
                gate_decision = self.runner_profile.check_all_gates(
                    edge_hat_bps=edge_hat_bps,
                    features=features,
                    order_type=order_type,
                    position_size_usd=position_size_usd,
                    spread_bps=spread_bps,
                    liquidity_score=liquidity_score,
                    urgency=urgency,
                    **kwargs,
                )

                if gate_decision.passes:
                    self.routed_to_runner += 1
                elif scalp_can_add:
                    # Runner gates failed, try scalp
                    gate_decision = self.scalp_profile.check_all_gates(
                        edge_hat_bps=edge_hat_bps,
                        features=features,
                        order_type=order_type,
                        position_size_usd=position_size_usd,
                        spread_bps=spread_bps,
                        liquidity_score=liquidity_score,
                        urgency=urgency,
                    )

                    if gate_decision.passes:
                        self.routed_to_scalp += 1

            elif scalp_can_add:
                # Runner heat limited, try scalp
                gate_decision = self.scalp_profile.check_all_gates(
                    edge_hat_bps=edge_hat_bps,
                    features=features,
                    order_type=order_type,
                    position_size_usd=position_size_usd,
                    spread_bps=spread_bps,
                    liquidity_score=liquidity_score,
                    urgency=urgency,
                )

                if gate_decision.passes:
                    self.routed_to_scalp += 1

            else:
                # Both heat limited
                self.heat_blocked += 1
                return RoutingDecision(
                    approved=False,
                    recommended_book=None,
                    recommended_size=0.0,
                    gate_decision=None,
                    preferred_mode=preferred_mode,
                    mode_reason=mode_reason,
                    heat_available=0.0,
                    heat_limit_hit=True,
                    technique=technique,
                    confidence=confidence,
                    regime=regime,
                )

        else:  # BOTH
            # Use hybrid router (tries runner, then scalp)
            gate_decision = self.hybrid_router.route_signal(
                edge_hat_bps=edge_hat_bps,
                features=features,
                order_type=order_type,
                position_size_usd=position_size_usd,
                spread_bps=spread_bps,
                liquidity_score=liquidity_score,
                urgency=urgency,
                **kwargs,
            )

            # Check heat for recommended book
            if gate_decision.passes and gate_decision.recommended_book:
                can_add, reason = self.book_manager.can_add_position(
                    book=gate_decision.recommended_book,
                    size_usd=position_size_usd,
                    symbol=symbol,
                )

                if not can_add:
                    # Heat limited
                    self.heat_blocked += 1
                    gate_decision.passes = False
                    gate_decision.recommended_book = None
                else:
                    if gate_decision.recommended_book == BookType.SHORT_HOLD:
                        self.routed_to_scalp += 1
                    else:
                        self.routed_to_runner += 1

        # 4. Final decision
        if gate_decision and gate_decision.passes:
            approved = True
            recommended_book = gate_decision.recommended_book
            recommended_size = position_size_usd
            heat_limit_hit = False
        else:
            approved = False
            recommended_book = None
            recommended_size = 0.0
            heat_limit_hit = False

            if gate_decision and not gate_decision.passes:
                self.gate_blocked += 1

        heat_available = min(
            1.0 - runner_heat - self.book_manager.reserve_heat,
            1.0 - scalp_heat - self.book_manager.reserve_heat,
        )

        return RoutingDecision(
            approved=approved,
            recommended_book=recommended_book,
            recommended_size=recommended_size,
            gate_decision=gate_decision,
            preferred_mode=preferred_mode,
            mode_reason=mode_reason,
            heat_available=heat_available,
            heat_limit_hit=heat_limit_hit,
            technique=technique,
            confidence=confidence,
            regime=regime,
        )

    def get_statistics(self) -> Dict:
        """Get routing statistics."""
        return {
            'total_signals': self.total_signals,
            'routed_to_scalp': self.routed_to_scalp,
            'routed_to_runner': self.routed_to_runner,
            'heat_blocked': self.heat_blocked,
            'gate_blocked': self.gate_blocked,
            'scalp_ratio': (
                self.routed_to_scalp / self.total_signals if self.total_signals > 0 else 0.0
            ),
            'runner_ratio': (
                self.routed_to_runner / self.total_signals if self.total_signals > 0 else 0.0
            ),
            'approval_rate': (
                (self.routed_to_scalp + self.routed_to_runner) / self.total_signals
                if self.total_signals > 0
                else 0.0
            ),
        }


def run_mode_selector_example():
    """Example usage of mode selector."""
    from .dual_book_manager import DualBookManager, AssetProfile, BookType

    # Initialize system
    book_manager = DualBookManager(
        total_capital=10000.0,
        max_short_heat=0.40,
        max_long_heat=0.50,
    )

    # Set asset profile
    book_manager.set_asset_profile(
        'ETH-USD',
        AssetProfile(
            allowed_books=[BookType.SHORT_HOLD, BookType.LONG_HOLD],
            scalp_target_bps=100.0,
            runner_target_bps=800.0,
        ),
    )

    selector = ModeSelector(dual_book_manager=book_manager)

    # Test different signals
    test_signals = [
        {
            'name': 'Fast microstructure',
            'technique': 'TAPE',
            'confidence': 0.68,
            'regime': 'RANGE',
            'edge_hat_bps': 12.0,
        },
        {
            'name': 'High-conviction trend',
            'technique': 'TREND',
            'confidence': 0.82,
            'regime': 'TREND',
            'edge_hat_bps': 20.0,
        },
        {
            'name': 'Medium breakout',
            'technique': 'BREAKOUT',
            'confidence': 0.65,
            'regime': 'TREND',
            'edge_hat_bps': 15.0,
        },
    ]

    print(f"\n{'='*70}")
    print("MODE SELECTOR EXAMPLES")
    print(f"{'='*70}\n")

    for signal in test_signals:
        decision = selector.route_signal(
            technique=signal['technique'],
            confidence=signal['confidence'],
            regime=signal['regime'],
            edge_hat_bps=signal['edge_hat_bps'],
            features={
                'engine_conf': signal['confidence'],
                'regime': signal['regime'],
                'technique': signal['technique'],
            },
            symbol='ETH-USD',
            position_size_usd=200.0,
            spread_bps=8.0,
            liquidity_score=0.75,
        )

        print(f"Signal: {signal['name']}")
        print(f"  Technique: {signal['technique']} | Confidence: {signal['confidence']:.2f} | Regime: {signal['regime']}")
        print(f"  Preferred mode: {decision.preferred_mode.value}")
        print(f"  Reason: {decision.mode_reason}")
        print(f"  Approved: {decision.approved}")
        if decision.approved:
            print(f"  Book: {decision.recommended_book.value}")
            print(f"  Size: ${decision.recommended_size:.0f}")
        else:
            print(f"  Blocked: {'Heat limit' if decision.heat_limit_hit else 'Gates'}")
        print()

    # Statistics
    stats = selector.get_statistics()
    print(f"{'='*70}")
    print("ROUTING STATISTICS")
    print(f"{'='*70}")
    print(f"Total signals: {stats['total_signals']}")
    print(f"Routed to scalp: {stats['routed_to_scalp']} ({stats['scalp_ratio']:.0%})")
    print(f"Routed to runner: {stats['routed_to_runner']} ({stats['runner_ratio']:.0%})")
    print(f"Approval rate: {stats['approval_rate']:.0%}")


if __name__ == "__main__":
    run_mode_selector_example()
