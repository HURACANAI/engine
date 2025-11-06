"""
Meta-Labeling System

Meta-labeling asks a different question than traditional labeling:
- Traditional: "Will price go up or down?"
- Meta-label: "IF I take this trade, will it be profitable after costs?"

This is MUCH more useful because:
1. Accounts for transaction costs (fees, spread, slippage)
2. Accounts for stop-loss and timeout
3. Trains model to be cost-aware
4. Produces realistic win rates

Usage:
    # First, get raw labels from triple-barrier
    raw_labels = triple_barrier_labeler.label_dataframe(df)

    # Then, apply meta-labeling
    meta_labeler = MetaLabeler(cost_threshold_bps=5.0)
    meta_labels = meta_labeler.apply(raw_labels)

    # Only winners survive
    winners = [l for l in meta_labels if l.meta_label == 1]
"""

from typing import List

import structlog

from .label_schemas import LabeledTrade

logger = structlog.get_logger(__name__)


class MetaLabeler:
    """
    Apply meta-labeling to trades.

    Meta-label = 1 if:
    - P&L (after costs) > threshold
    - Otherwise 0

    This filters out unprofitable trades that would confuse your model.
    """

    def __init__(
        self,
        cost_threshold_bps: float = 0.0,
        min_pnl_bps: float = 0.0
    ):
        """
        Initialize meta-labeler.

        Args:
            cost_threshold_bps: Additional buffer beyond costs
                (e.g., 5 bps = must beat costs by at least 5 bps)
            min_pnl_bps: Minimum absolute P&L to be considered winner
        """
        self.cost_threshold = cost_threshold_bps
        self.min_pnl = min_pnl_bps

        logger.info(
            "meta_labeler_initialized",
            cost_threshold_bps=cost_threshold_bps,
            min_pnl_bps=min_pnl_bps
        )

    def apply(self, labeled_trades: List[LabeledTrade]) -> List[LabeledTrade]:
        """
        Apply meta-labeling to trades.

        This MODIFIES the meta_label field based on:
        - P&L net > cost_threshold
        - P&L net > min_pnl

        Args:
            labeled_trades: List of trades from triple-barrier labeler

        Returns:
            Same list with updated meta_label fields
        """
        if not labeled_trades:
            return []

        original_winners = sum(1 for t in labeled_trades if t.meta_label == 1)

        # Re-evaluate meta-labels
        for trade in labeled_trades:
            # Check if profitable after costs + threshold
            is_profitable = (
                trade.pnl_net_bps > self.cost_threshold and
                trade.pnl_net_bps > self.min_pnl
            )

            # Update meta-label
            trade.meta_label = 1 if is_profitable else 0

        new_winners = sum(1 for t in labeled_trades if t.meta_label == 1)

        logger.info(
            "meta_labeling_applied",
            total_trades=len(labeled_trades),
            original_winners=original_winners,
            new_winners=new_winners,
            filtered_out=original_winners - new_winners,
            new_win_rate=new_winners / len(labeled_trades) if labeled_trades else 0
        )

        return labeled_trades

    def apply_with_features(
        self,
        labeled_trades: List[LabeledTrade],
        feature_filter: callable
    ) -> List[LabeledTrade]:
        """
        Apply meta-labeling with additional feature-based filtering.

        Example:
            def my_filter(trade):
                # Only label as winner if confidence > 0.6
                return trade.confidence > 0.6

            meta_labeler.apply_with_features(trades, my_filter)

        Args:
            labeled_trades: List of trades
            feature_filter: Callable that takes LabeledTrade and returns bool

        Returns:
            List with updated meta-labels
        """
        # First apply standard meta-labeling
        labeled_trades = self.apply(labeled_trades)

        # Then apply feature filter
        for trade in labeled_trades:
            if trade.meta_label == 1:
                # Only keep as winner if passes feature filter
                if not feature_filter(trade):
                    trade.meta_label = 0

        logger.info(
            "feature_filter_applied",
            remaining_winners=sum(1 for t in labeled_trades if t.meta_label == 1)
        )

        return labeled_trades

    def get_label_distribution(self, labeled_trades: List[LabeledTrade]) -> dict:
        """
        Analyze meta-label distribution.

        Useful for understanding label balance.
        """
        if not labeled_trades:
            return {}

        total = len(labeled_trades)
        winners = sum(1 for t in labeled_trades if t.meta_label == 1)
        losers = total - winners

        # Break down by exit reason
        winner_tp = sum(
            1 for t in labeled_trades
            if t.meta_label == 1 and t.exit_reason.value == 'tp'
        )
        winner_timeout = sum(
            1 for t in labeled_trades
            if t.meta_label == 1 and t.exit_reason.value == 'timeout'
        )

        loser_sl = sum(
            1 for t in labeled_trades
            if t.meta_label == 0 and t.exit_reason.value == 'sl'
        )
        loser_timeout = sum(
            1 for t in labeled_trades
            if t.meta_label == 0 and t.exit_reason.value == 'timeout'
        )

        return {
            'total': total,
            'winners': winners,
            'losers': losers,
            'win_rate': winners / total if total > 0 else 0,
            'balance': min(winners, losers) / max(winners, losers) if max(winners, losers) > 0 else 0,
            'winner_breakdown': {
                'tp_exits': winner_tp,
                'timeout_exits': winner_timeout,
            },
            'loser_breakdown': {
                'sl_exits': loser_sl,
                'timeout_exits': loser_timeout,
            }
        }

    def suggest_threshold(
        self,
        labeled_trades: List[LabeledTrade],
        target_win_rate: float = 0.70
    ) -> float:
        """
        Suggest cost threshold to achieve target win rate.

        Usage:
            threshold = meta_labeler.suggest_threshold(trades, target_win_rate=0.75)
            meta_labeler.cost_threshold = threshold
            meta_labeler.apply(trades)

        Args:
            labeled_trades: List of trades
            target_win_rate: Desired win rate (e.g., 0.70 for 70%)

        Returns:
            Suggested cost threshold in bps
        """
        if not labeled_trades:
            return 0.0

        # Sort by net P&L
        sorted_trades = sorted(labeled_trades, key=lambda t: t.pnl_net_bps, reverse=True)

        # Find threshold that gives target win rate
        target_winners = int(len(sorted_trades) * target_win_rate)

        if target_winners < len(sorted_trades):
            suggested_threshold = sorted_trades[target_winners].pnl_net_bps

            logger.info(
                "threshold_suggestion",
                target_win_rate=target_win_rate,
                suggested_threshold_bps=suggested_threshold,
                resulting_winners=target_winners
            )

            return suggested_threshold

        return 0.0


def print_label_distribution(dist: dict) -> None:
    """Pretty-print label distribution."""
    print(f"""
╔═══════════════════════════════════════════╗
║       META-LABEL DISTRIBUTION             ║
╚═══════════════════════════════════════════╝

📊 Summary
──────────────────────────────────────────────
Total Trades:      {dist['total']:,}
Winners:           {dist['winners']:,} ({dist['win_rate']:.1%})
Losers:            {dist['losers']:,}
Balance:           {dist['balance']:.2f} (1.0 = perfect balance)

✅ Winner Breakdown
──────────────────────────────────────────────
TP Exits:          {dist['winner_breakdown']['tp_exits']:,}
Timeout Exits:     {dist['winner_breakdown']['timeout_exits']:,}

❌ Loser Breakdown
──────────────────────────────────────────────
SL Exits:          {dist['loser_breakdown']['sl_exits']:,}
Timeout Exits:     {dist['loser_breakdown']['timeout_exits']:,}

{'⚠️ Consider adjusting thresholds - labels are imbalanced!' if dist['balance'] < 0.5 else '✅ Good label balance!'}
""")
