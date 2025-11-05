"""
Ensemble Exit Strategy

Combines multiple exit systems intelligently with weighted voting.

Key Problems Solved:
1. **Single System Blindness**: One exit system misses signal that another catches
2. **Conflicting Signals**: Trailing stop says hold, exit detector says exit - which to trust?
3. **Edge Cases**: Each system has blind spots - ensemble covers all cases

Solution: Weighted Voting Across Exit Systems
- Each system votes with priority weight
- Combine votes intelligently
- Partial exits on moderate signals, full exit on strong signals

Example:
    3 Exit Systems Voting:

    1. Trailing Stop: HOLD (no breach)
    2. Exit Signal Detector: P2 WARNING (momentum weakening) - 2 votes
    3. Regime Exit Manager: EXIT_50PCT (regime deteriorating) - 3 votes

    Total Votes: 0 + 2 + 3 = 5 votes for exit

    Decision: 5 votes ≥ 4 threshold → EXIT 50%
    (Would need 6+ votes for full 100% exit)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ExitSystemType(Enum):
    """Types of exit systems."""

    TRAILING_STOP = "trailing_stop"
    EXIT_SIGNAL_DETECTOR = "exit_signal_detector"
    REGIME_EXIT_MANAGER = "regime_exit_manager"
    TP_LADDER = "tp_ladder"
    STOP_LOSS = "stop_loss"


class VoteDecision(Enum):
    """Exit decision from voting."""

    HOLD = "hold"  # No exit
    SCALE_OUT_25 = "scale_out_25"  # Exit 25%
    SCALE_OUT_50 = "scale_out_50"  # Exit 50%
    SCALE_OUT_75 = "scale_out_75"  # Exit 75%
    EXIT_ALL = "exit_all"  # Exit 100%


@dataclass
class ExitVote:
    """Vote from a single exit system."""

    system: ExitSystemType
    vote_strength: int  # 0-3 (0=hold, 1=weak exit, 2=moderate exit, 3=strong exit)
    recommended_exit_pct: float  # 0-1
    reason: str
    priority: int  # Priority level if applicable (1=highest)


@dataclass
class EnsembleExitDecision:
    """Final ensemble exit decision."""

    decision: VoteDecision
    exit_percentage: float  # 0-1
    total_votes: int
    vote_breakdown: Dict[ExitSystemType, ExitVote]
    confidence: float  # 0-1
    reasoning: str
    warnings: List[str]


class EnsembleExitStrategy:
    """
    Ensemble exit strategy combining multiple exit systems.

    Exit Systems Integrated:
    1. **Adaptive Trailing Stop** (Phase 2)
    2. **Exit Signal Detector** (Phase 2)
    3. **Regime Exit Manager** (Phase 2)
    4. **TP Ladder** (Phase 4 Wave 1)
    5. **Stop Loss** (baseline)

    Voting Logic:
    - Each system votes: 0 (hold) to 3 (strong exit)
    - P1 DANGER signals = 3 votes
    - P2 WARNING signals = 2 votes
    - P3 PROFIT signals = 1 vote
    - Regime exits = 2-3 votes based on severity
    - Trailing stop breach = 2 votes

    Decision Thresholds:
    - 0-2 votes: HOLD
    - 3-4 votes: SCALE_OUT_25 (exit 25%)
    - 5-6 votes: SCALE_OUT_50 (exit 50%)
    - 7-8 votes: SCALE_OUT_75 (exit 75%)
    - 9+ votes: EXIT_ALL (exit 100%)

    Usage:
        ensemble = EnsembleExitStrategy()

        # Collect votes from all systems
        votes = []

        # 1. Trailing stop vote
        if trailing_stop_breached:
            votes.append(ExitVote(
                system=ExitSystemType.TRAILING_STOP,
                vote_strength=2,
                recommended_exit_pct=1.0,
                reason="Trailing stop breached",
                priority=2,
            ))

        # 2. Exit signal vote
        if exit_signal:
            votes.append(ExitVote(
                system=ExitSystemType.EXIT_SIGNAL_DETECTOR,
                vote_strength=exit_signal.priority,  # 1-3
                recommended_exit_pct=1.0 if exit_signal.priority == 1 else 0.5,
                reason=exit_signal.reason,
                priority=exit_signal.priority,
            ))

        # 3. Regime exit vote
        if regime_signal:
            votes.append(ExitVote(
                system=ExitSystemType.REGIME_EXIT_MANAGER,
                vote_strength=regime_signal.priority,  # 1-3
                recommended_exit_pct=get_regime_exit_pct(regime_signal),
                reason=regime_signal.reason,
                priority=regime_signal.priority,
            ))

        # Make ensemble decision
        decision = ensemble.decide(votes, position_pnl_bps)

        if decision.decision != VoteDecision.HOLD:
            execute_partial_exit(
                size_pct=decision.exit_percentage,
                reason=decision.reasoning,
            )
    """

    def __init__(
        self,
        scale_out_25_threshold: int = 3,
        scale_out_50_threshold: int = 5,
        scale_out_75_threshold: int = 7,
        exit_all_threshold: int = 9,
        profit_protection_min_bps: float = 50.0,
    ):
        """
        Initialize ensemble exit strategy.

        Args:
            scale_out_25_threshold: Vote threshold for 25% exit
            scale_out_50_threshold: Vote threshold for 50% exit
            scale_out_75_threshold: Vote threshold for 75% exit
            exit_all_threshold: Vote threshold for 100% exit
            profit_protection_min_bps: Minimum profit to enable protective exits
        """
        self.thresholds = {
            'scale_25': scale_out_25_threshold,
            'scale_50': scale_out_50_threshold,
            'scale_75': scale_out_75_threshold,
            'exit_all': exit_all_threshold,
        }
        self.profit_protection_min = profit_protection_min_bps

        logger.info(
            "ensemble_exit_strategy_initialized",
            thresholds=self.thresholds,
        )

    def decide(
        self,
        votes: List[ExitVote],
        position_pnl_bps: float,
    ) -> EnsembleExitDecision:
        """
        Make ensemble exit decision based on votes.

        Args:
            votes: List of votes from exit systems
            position_pnl_bps: Current position P&L in bps

        Returns:
            EnsembleExitDecision with final recommendation
        """
        if not votes:
            # No votes = hold
            return EnsembleExitDecision(
                decision=VoteDecision.HOLD,
                exit_percentage=0.0,
                total_votes=0,
                vote_breakdown={},
                confidence=1.0,
                reasoning="No exit signals from any system",
                warnings=[],
            )

        # Calculate total votes
        total_votes = sum(vote.vote_strength for vote in votes)

        # Create vote breakdown
        vote_breakdown = {vote.system: vote for vote in votes}

        # Determine decision based on thresholds
        if total_votes >= self.thresholds['exit_all']:
            decision = VoteDecision.EXIT_ALL
            exit_pct = 1.0
            reasoning = f"Strong exit consensus: {total_votes} votes (threshold: {self.thresholds['exit_all']})"
        elif total_votes >= self.thresholds['scale_75']:
            decision = VoteDecision.SCALE_OUT_75
            exit_pct = 0.75
            reasoning = f"Significant exit signals: {total_votes} votes (threshold: {self.thresholds['scale_75']})"
        elif total_votes >= self.thresholds['scale_50']:
            decision = VoteDecision.SCALE_OUT_50
            exit_pct = 0.50
            reasoning = f"Moderate exit signals: {total_votes} votes (threshold: {self.thresholds['scale_50']})"
        elif total_votes >= self.thresholds['scale_25']:
            decision = VoteDecision.SCALE_OUT_25
            exit_pct = 0.25
            reasoning = f"Weak exit signals: {total_votes} votes (threshold: {self.thresholds['scale_25']})"
        else:
            decision = VoteDecision.HOLD
            exit_pct = 0.0
            reasoning = f"Insufficient votes to exit: {total_votes} votes (need {self.thresholds['scale_25']})"

        # Check profit protection override
        warnings = []
        if position_pnl_bps > self.profit_protection_min and exit_pct > 0:
            # Have profit - be more aggressive with exits
            if decision == VoteDecision.SCALE_OUT_25:
                # Upgrade to 50% if we have good profit
                if position_pnl_bps > 150:
                    decision = VoteDecision.SCALE_OUT_50
                    exit_pct = 0.50
                    warnings.append(
                        f"Upgraded exit to 50% due to profit protection "
                        f"(+{position_pnl_bps:.0f} bps)"
                    )

        # Calculate confidence (based on vote agreement)
        # If votes are unanimous in strength, confidence is high
        vote_strengths = [v.vote_strength for v in votes]
        if len(vote_strengths) > 1:
            vote_std = np.std(vote_strengths)
            vote_mean = np.mean(vote_strengths)
            confidence = 1.0 - min(vote_std / (vote_mean + 0.1), 0.5)  # Cap reduction at 50%
        else:
            confidence = 0.8  # Single vote

        # Add vote details to reasoning
        detailed_reasoning = [reasoning]
        detailed_reasoning.append("\nVote Breakdown:")
        for vote in votes:
            detailed_reasoning.append(
                f"  • {vote.system.value}: {vote.vote_strength} votes - {vote.reason}"
            )

        final_reasoning = "\n".join(detailed_reasoning)

        logger.info(
            "ensemble_decision_made",
            decision=decision.value,
            total_votes=total_votes,
            exit_pct=exit_pct,
            pnl_bps=position_pnl_bps,
            num_systems_voting=len(votes),
        )

        return EnsembleExitDecision(
            decision=decision,
            exit_percentage=exit_pct,
            total_votes=total_votes,
            vote_breakdown=vote_breakdown,
            confidence=confidence,
            reasoning=final_reasoning,
            warnings=warnings,
        )

    def create_vote_from_trailing_stop(
        self,
        stop_breached: bool,
        locked_profit_bps: float,
    ) -> Optional[ExitVote]:
        """Create vote from trailing stop system."""
        if not stop_breached:
            return None

        # Trailing stop breach = moderate exit signal
        return ExitVote(
            system=ExitSystemType.TRAILING_STOP,
            vote_strength=2,
            recommended_exit_pct=1.0,
            reason=f"Trailing stop breached (locked profit: +{locked_profit_bps:.0f} bps)",
            priority=2,
        )

    def create_vote_from_exit_signal(
        self,
        exit_signal,  # ExitSignal from exit_signal_detector
    ) -> Optional[ExitVote]:
        """Create vote from exit signal detector."""
        if exit_signal is None:
            return None

        # Map priority to vote strength
        # P1 DANGER = 3 votes
        # P2 WARNING = 2 votes
        # P3 PROFIT = 1 vote
        priority_to_votes = {1: 3, 2: 2, 3: 1}
        vote_strength = priority_to_votes.get(exit_signal.priority, 1)

        # P1 = 100% exit, P2 = 50% exit, P3 = 25% exit
        priority_to_exit_pct = {1: 1.0, 2: 0.5, 3: 0.25}
        exit_pct = priority_to_exit_pct.get(exit_signal.priority, 0.5)

        return ExitVote(
            system=ExitSystemType.EXIT_SIGNAL_DETECTOR,
            vote_strength=vote_strength,
            recommended_exit_pct=exit_pct,
            reason=f"Exit signal: {exit_signal.reason} (P{exit_signal.priority})",
            priority=exit_signal.priority,
        )

    def create_vote_from_regime_exit(
        self,
        regime_signal,  # RegimeExitSignal from regime_exit_manager
    ) -> Optional[ExitVote]:
        """Create vote from regime exit manager."""
        if regime_signal is None:
            return None

        # Map action to vote strength
        from src.cloud.training.models.regime_exit_manager import RegimeAction

        action_to_votes = {
            RegimeAction.EXIT_IMMEDIATELY: 3,
            RegimeAction.SCALE_OUT_HALF: 2,
            RegimeAction.TIGHTEN_STOPS: 1,
            RegimeAction.RELAX_STOPS: 0,
            RegimeAction.HOLD: 0,
        }

        action_to_exit_pct = {
            RegimeAction.EXIT_IMMEDIATELY: 1.0,
            RegimeAction.SCALE_OUT_HALF: 0.5,
            RegimeAction.TIGHTEN_STOPS: 0.0,
            RegimeAction.RELAX_STOPS: 0.0,
            RegimeAction.HOLD: 0.0,
        }

        vote_strength = action_to_votes.get(regime_signal.action, 0)
        exit_pct = action_to_exit_pct.get(regime_signal.action, 0.0)

        if vote_strength == 0:
            return None  # No exit recommended

        return ExitVote(
            system=ExitSystemType.REGIME_EXIT_MANAGER,
            vote_strength=vote_strength,
            recommended_exit_pct=exit_pct,
            reason=f"Regime exit: {regime_signal.reason}",
            priority=regime_signal.priority,
        )

    def create_vote_from_tp_ladder(
        self,
        tp_level_hit: bool,
        level_name: str,
        exit_pct: float,
    ) -> Optional[ExitVote]:
        """Create vote from TP ladder."""
        if not tp_level_hit:
            return None

        # TP levels are profit-taking, not danger signals
        # Give them 1 vote (informational)
        return ExitVote(
            system=ExitSystemType.TP_LADDER,
            vote_strength=1,
            recommended_exit_pct=exit_pct,
            reason=f"TP {level_name} hit",
            priority=3,
        )

    def create_vote_from_stop_loss(
        self,
        stop_hit: bool,
    ) -> Optional[ExitVote]:
        """Create vote from stop loss."""
        if not stop_hit:
            return None

        # Stop loss = absolute exit (3 votes)
        return ExitVote(
            system=ExitSystemType.STOP_LOSS,
            vote_strength=3,
            recommended_exit_pct=1.0,
            reason="Stop loss hit",
            priority=1,
        )

    def get_vote_summary(self, votes: List[ExitVote]) -> str:
        """Get human-readable vote summary."""
        if not votes:
            return "No exit votes"

        lines = []
        lines.append(f"Exit System Votes ({len(votes)} systems):")

        for vote in sorted(votes, key=lambda v: v.vote_strength, reverse=True):
            emoji = "🔴" if vote.vote_strength == 3 else "🟠" if vote.vote_strength == 2 else "🟡"
            lines.append(
                f"  {emoji} {vote.system.value}: {vote.vote_strength} votes "
                f"({vote.recommended_exit_pct:.0%} exit) - {vote.reason}"
            )

        total_votes = sum(v.vote_strength for v in votes)
        lines.append(f"\nTotal Votes: {total_votes}")

        return "\n".join(lines)
