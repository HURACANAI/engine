"""
Bayesian Position Sizing

Probabilistic position sizing based on Bayesian inference.

Key Features:
- Kelly Criterion with Bayesian parameter estimation
- Risk-aware position sizing
- Confidence-adjusted sizing
- Dynamic sizing based on regime
- Drawdown-based scaling

Usage:
    from models.position_sizing import BayesianPositionSizer

    sizer = BayesianPositionSizer()

    # Update beliefs from historical trades
    sizer.update_beliefs(
        trades_df=historical_trades,
        win_col="win",
        pnl_col="pnl_bps"
    )

    # Get position size for new signal
    position_size = sizer.calculate_position_size(
        signal_confidence=0.75,
        current_regime="trending",
        account_balance=100000,
        current_drawdown_pct=5.0
    )

    print(f"Position size: {position_size:.2%} of account")
"""

from .bayesian_sizer import (
    BayesianPositionSizer,
    PositionSizeResult
)
from .kelly import (
    calculate_kelly_fraction,
    calculate_half_kelly
)

__all__ = [
    # Sizer
    "BayesianPositionSizer",
    "PositionSizeResult",

    # Kelly
    "calculate_kelly_fraction",
    "calculate_half_kelly",
]
