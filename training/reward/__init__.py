"""
Enhanced RL Reward Function

Multi-objective reward function that balances:
- PnL (profit and loss)
- Risk (drawdown, volatility)
- Transaction costs (fees, slippage)
- Regime consistency
- Live feedback integration

Key Features:
- Configurable reward weights
- Regime-specific penalties
- Slippage and fee modeling
- Sharpe-aware rewards
- Live feedback integration

Usage:
    from training.reward import EnhancedRewardCalculator

    calculator = EnhancedRewardCalculator(
        pnl_weight=1.0,
        risk_penalty=0.3,
        cost_penalty=0.2,
        sharpe_bonus=0.5
    )

    # Calculate reward for a trade
    reward = calculator.calculate_reward(
        pnl_bps=25.0,
        max_drawdown_pct=5.0,
        total_fees_bps=3.0,
        slippage_bps=2.0,
        regime="trending"
    )

    # Integrate live feedback
    calculator.update_from_feedback(feedback_df)
"""

from .calculator import (
    EnhancedRewardCalculator,
    RewardComponents
)
from .config import RewardConfig

__all__ = [
    "EnhancedRewardCalculator",
    "RewardComponents",
    "RewardConfig",
]
