"""
Bayesian Position Sizer

Uses Bayesian inference to estimate win probability and optimal position sizing.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from .kelly import calculate_kelly_with_uncertainty

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizeResult:
    """Position sizing result"""
    position_fraction: float  # Fraction of account [0-1]
    kelly_fraction: float
    confidence_multiplier: float
    regime_multiplier: float
    drawdown_multiplier: float

    # Beliefs
    win_probability: float
    win_probability_std: float
    avg_win_bps: float
    avg_loss_bps: float


class BayesianPositionSizer:
    """
    Bayesian Position Sizer

    Uses Bayesian inference to estimate trading parameters and
    calculate optimal position sizes.

    Example:
        sizer = BayesianPositionSizer()

        # Update beliefs from historical data
        sizer.update_beliefs(trades_df)

        # Calculate position size
        size = sizer.calculate_position_size(
            signal_confidence=0.75,
            current_regime="trending",
            account_balance=100000,
            current_drawdown_pct=5.0
        )

        print(f"Position: {size.position_fraction:.2%}")
        print(f"Win prob: {size.win_probability:.2%}")
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        max_position_fraction: float = 0.10,
        kelly_fraction: float = 0.25
    ):
        """
        Initialize Bayesian position sizer

        Args:
            prior_alpha: Prior alpha for Beta distribution (successes)
            prior_beta: Prior beta for Beta distribution (failures)
            max_position_fraction: Maximum position size [0-1]
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.max_position_fraction = max_position_fraction
        self.kelly_fraction = kelly_fraction

        # Posterior parameters (updated with data)
        self.alpha = prior_alpha
        self.beta = prior_beta

        # Win/loss statistics
        self.avg_win_bps = 50.0  # Default expected win
        self.avg_loss_bps = 40.0  # Default expected loss

        self.num_trades = 0

    def update_beliefs(
        self,
        trades_df: pd.DataFrame,
        win_col: str = "win",
        pnl_col: str = "pnl_bps"
    ) -> None:
        """
        Update Bayesian beliefs from historical trades

        Args:
            trades_df: DataFrame with historical trades
            win_col: Column indicating win (True/False or 1/0)
            pnl_col: Column with PnL in bps

        Example:
            sizer.update_beliefs(
                trades_df=historical_trades,
                win_col="win",
                pnl_col="pnl_bps"
            )
        """
        if len(trades_df) == 0:
            logger.warning("no_trades_to_update_beliefs")
            return

        # Count wins and losses
        wins = trades_df[win_col].sum()
        losses = len(trades_df) - wins

        # Update posterior
        self.alpha = self.prior_alpha + wins
        self.beta = self.prior_beta + losses

        # Calculate win/loss statistics
        winning_trades = trades_df[trades_df[win_col] == True]
        losing_trades = trades_df[trades_df[win_col] == False]

        if len(winning_trades) > 0:
            self.avg_win_bps = winning_trades[pnl_col].mean()
        else:
            self.avg_win_bps = 50.0

        if len(losing_trades) > 0:
            self.avg_loss_bps = abs(losing_trades[pnl_col].mean())
        else:
            self.avg_loss_bps = 40.0

        self.num_trades = len(trades_df)

        logger.info(
            "beliefs_updated",
            num_trades=len(trades_df),
            wins=wins,
            losses=losses,
            alpha=self.alpha,
            beta=self.beta,
            avg_win_bps=self.avg_win_bps,
            avg_loss_bps=self.avg_loss_bps
        )

    def calculate_position_size(
        self,
        signal_confidence: float,
        current_regime: str = "unknown",
        account_balance: float = 100000.0,
        current_drawdown_pct: float = 0.0,
        max_drawdown_threshold: float = 15.0
    ) -> PositionSizeResult:
        """
        Calculate position size

        Args:
            signal_confidence: Model confidence [0-1]
            current_regime: Current market regime
            account_balance: Current account balance
            current_drawdown_pct: Current drawdown percentage [0-100]
            max_drawdown_threshold: Max acceptable drawdown [0-100]

        Returns:
            PositionSizeResult
        """
        # 1. Estimate win probability from Bayesian posterior
        win_prob_mean = self.alpha / (self.alpha + self.beta)

        # Variance of Beta distribution
        win_prob_var = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        win_prob_std = np.sqrt(win_prob_var)

        # 2. Calculate base Kelly fraction
        kelly = calculate_kelly_with_uncertainty(
            win_probability=win_prob_mean,
            win_probability_std=win_prob_std,
            avg_win=self.avg_win_bps,
            avg_loss=self.avg_loss_bps
        )

        # Apply fractional Kelly (quarter Kelly by default)
        kelly = kelly * self.kelly_fraction

        # 3. Apply confidence multiplier
        # Higher confidence → larger position
        confidence_multiplier = 0.5 + (signal_confidence * 0.5)

        # 4. Apply regime multiplier
        regime_multiplier = self._get_regime_multiplier(current_regime)

        # 5. Apply drawdown multiplier
        # Reduce position size as drawdown increases
        if current_drawdown_pct >= max_drawdown_threshold:
            drawdown_multiplier = 0.0  # No new positions
        else:
            drawdown_multiplier = 1.0 - (current_drawdown_pct / max_drawdown_threshold)
            drawdown_multiplier = max(0.2, drawdown_multiplier)  # Min 20%

        # 6. Calculate final position size
        position_fraction = (
            kelly *
            confidence_multiplier *
            regime_multiplier *
            drawdown_multiplier
        )

        # Clip to max position size
        position_fraction = min(position_fraction, self.max_position_fraction)
        position_fraction = max(0.0, position_fraction)

        result = PositionSizeResult(
            position_fraction=position_fraction,
            kelly_fraction=kelly,
            confidence_multiplier=confidence_multiplier,
            regime_multiplier=regime_multiplier,
            drawdown_multiplier=drawdown_multiplier,
            win_probability=win_prob_mean,
            win_probability_std=win_prob_std,
            avg_win_bps=self.avg_win_bps,
            avg_loss_bps=self.avg_loss_bps
        )

        logger.debug(
            "position_size_calculated",
            position_fraction=position_fraction,
            kelly_fraction=kelly,
            signal_confidence=signal_confidence,
            regime=current_regime,
            drawdown_pct=current_drawdown_pct
        )

        return result

    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Get position size multiplier for regime

        Args:
            regime: Market regime

        Returns:
            Multiplier [0-1]
        """
        # Conservative multipliers for different regimes
        regime_multipliers = {
            "trending": 1.0,        # Full size in trending
            "choppy": 0.5,          # Half size in choppy
            "volatile": 0.3,        # Small size in volatile
            "low_liquidity": 0.2,   # Very small in low liquidity
            "unknown": 0.5          # Default to conservative
        }

        return regime_multipliers.get(regime, 0.5)

    def get_posterior_stats(self) -> dict:
        """
        Get posterior distribution statistics

        Returns:
            Dict with posterior stats
        """
        win_prob_mean = self.alpha / (self.alpha + self.beta)

        win_prob_var = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        win_prob_std = np.sqrt(win_prob_var)

        # 95% credible interval
        from scipy.stats import beta as beta_dist
        ci_lower = beta_dist.ppf(0.025, self.alpha, self.beta)
        ci_upper = beta_dist.ppf(0.975, self.alpha, self.beta)

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "win_prob_mean": win_prob_mean,
            "win_prob_std": win_prob_std,
            "win_prob_ci_lower": ci_lower,
            "win_prob_ci_upper": ci_upper,
            "avg_win_bps": self.avg_win_bps,
            "avg_loss_bps": self.avg_loss_bps,
            "num_trades": self.num_trades
        }

    def reset_beliefs(self) -> None:
        """Reset to prior beliefs"""
        self.alpha = self.prior_alpha
        self.beta = self.prior_beta
        self.avg_win_bps = 50.0
        self.avg_loss_bps = 40.0
        self.num_trades = 0

        logger.info("beliefs_reset_to_prior")


class RegimeSpecificSizer:
    """
    Regime-Specific Bayesian Sizer

    Maintains separate Bayesian beliefs for each regime.

    Example:
        sizer = RegimeSpecificSizer()

        # Update beliefs per regime
        trending_trades = trades_df[trades_df['regime'] == 'trending']
        sizer.update_regime_beliefs('trending', trending_trades)

        choppy_trades = trades_df[trades_df['regime'] == 'choppy']
        sizer.update_regime_beliefs('choppy', choppy_trades)

        # Calculate position size for current regime
        size = sizer.calculate_position_size(
            regime='trending',
            signal_confidence=0.75
        )
    """

    def __init__(self, max_position_fraction: float = 0.10):
        """
        Initialize regime-specific sizer

        Args:
            max_position_fraction: Maximum position size
        """
        self.max_position_fraction = max_position_fraction

        # Sizer per regime
        self.regime_sizers: dict[str, BayesianPositionSizer] = {}

    def update_regime_beliefs(
        self,
        regime: str,
        trades_df: pd.DataFrame,
        win_col: str = "win",
        pnl_col: str = "pnl_bps"
    ) -> None:
        """
        Update beliefs for specific regime

        Args:
            regime: Regime name
            trades_df: Trades in this regime
            win_col: Win column
            pnl_col: PnL column
        """
        if regime not in self.regime_sizers:
            self.regime_sizers[regime] = BayesianPositionSizer(
                max_position_fraction=self.max_position_fraction
            )

        self.regime_sizers[regime].update_beliefs(trades_df, win_col, pnl_col)

    def calculate_position_size(
        self,
        regime: str,
        signal_confidence: float,
        account_balance: float = 100000.0,
        current_drawdown_pct: float = 0.0
    ) -> PositionSizeResult:
        """
        Calculate position size for regime

        Args:
            regime: Current regime
            signal_confidence: Signal confidence
            account_balance: Account balance
            current_drawdown_pct: Current drawdown

        Returns:
            PositionSizeResult
        """
        if regime not in self.regime_sizers:
            # No data for this regime - use default sizer
            logger.warning(
                "no_regime_sizer_using_default",
                regime=regime
            )
            default_sizer = BayesianPositionSizer(
                max_position_fraction=self.max_position_fraction
            )
            return default_sizer.calculate_position_size(
                signal_confidence=signal_confidence,
                current_regime=regime,
                account_balance=account_balance,
                current_drawdown_pct=current_drawdown_pct
            )

        sizer = self.regime_sizers[regime]

        return sizer.calculate_position_size(
            signal_confidence=signal_confidence,
            current_regime=regime,
            account_balance=account_balance,
            current_drawdown_pct=current_drawdown_pct
        )

    def get_all_regime_stats(self) -> dict:
        """Get posterior stats for all regimes"""
        stats = {}

        for regime, sizer in self.regime_sizers.items():
            stats[regime] = sizer.get_posterior_stats()

        return stats
