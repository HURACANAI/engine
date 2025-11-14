"""
Reward Configuration

Configuration for enhanced RL reward function.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """
    Reward Function Configuration

    Controls how different objectives are weighted in the reward calculation.

    Example:
        # Aggressive: PnL-focused
        aggressive_config = RewardConfig(
            pnl_weight=1.0,
            risk_penalty=0.1,
            cost_penalty=0.1,
            sharpe_bonus=0.3
        )

        # Conservative: Risk-aware
        conservative_config = RewardConfig(
            pnl_weight=0.7,
            risk_penalty=0.5,
            cost_penalty=0.3,
            sharpe_bonus=0.8
        )
    """

    # Core weights
    pnl_weight: float = 1.0          # Weight for PnL component
    risk_penalty: float = 0.3        # Penalty for drawdown/volatility
    cost_penalty: float = 0.2        # Penalty for fees/slippage
    sharpe_bonus: float = 0.5        # Bonus for risk-adjusted returns

    # Regime-specific modifiers
    regime_consistency_bonus: float = 0.1   # Bonus for trading in correct regime
    regime_mismatch_penalty: float = 0.5    # Penalty for wrong regime

    # Risk thresholds
    max_acceptable_drawdown_pct: float = 15.0
    drawdown_cliff_penalty: float = -10.0    # Applied if > max drawdown

    # Cost modeling
    expected_slippage_bps: float = 2.0       # Expected slippage per trade
    expected_fees_bps: float = 4.0           # Expected fees per trade

    # Sharpe calculation
    target_sharpe_ratio: float = 1.5         # Target Sharpe for bonus
    sharpe_scaling_factor: float = 1.0       # Scaling for Sharpe bonus

    # Live feedback integration
    use_live_feedback: bool = True
    feedback_learning_rate: float = 0.1      # How quickly to adapt from feedback

    # Regime-specific penalties (override defaults)
    regime_penalties: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Initialize regime penalties if not provided"""
        if self.regime_penalties is None:
            self.regime_penalties = {
                "trending": 0.0,       # No penalty in trending
                "choppy": 0.3,         # Moderate penalty in choppy
                "volatile": 0.5,       # High penalty in volatile
                "low_liquidity": 0.7,  # Very high penalty in low liquidity
            }

    def get_regime_penalty(self, regime: str) -> float:
        """
        Get penalty multiplier for a regime

        Args:
            regime: Regime name

        Returns:
            Penalty multiplier [0-1]
        """
        return self.regime_penalties.get(regime, 0.2)  # Default 0.2 penalty

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "pnl_weight": self.pnl_weight,
            "risk_penalty": self.risk_penalty,
            "cost_penalty": self.cost_penalty,
            "sharpe_bonus": self.sharpe_bonus,
            "regime_consistency_bonus": self.regime_consistency_bonus,
            "regime_mismatch_penalty": self.regime_mismatch_penalty,
            "max_acceptable_drawdown_pct": self.max_acceptable_drawdown_pct,
            "drawdown_cliff_penalty": self.drawdown_cliff_penalty,
            "expected_slippage_bps": self.expected_slippage_bps,
            "expected_fees_bps": self.expected_fees_bps,
            "target_sharpe_ratio": self.target_sharpe_ratio,
            "sharpe_scaling_factor": self.sharpe_scaling_factor,
            "use_live_feedback": self.use_live_feedback,
            "feedback_learning_rate": self.feedback_learning_rate,
            "regime_penalties": self.regime_penalties
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RewardConfig":
        """Create from dictionary"""
        return cls(**config_dict)


# Preset configurations

AGGRESSIVE_CONFIG = RewardConfig(
    pnl_weight=1.0,
    risk_penalty=0.1,
    cost_penalty=0.1,
    sharpe_bonus=0.3,
    regime_consistency_bonus=0.0,
    max_acceptable_drawdown_pct=25.0
)

BALANCED_CONFIG = RewardConfig(
    pnl_weight=1.0,
    risk_penalty=0.3,
    cost_penalty=0.2,
    sharpe_bonus=0.5,
    regime_consistency_bonus=0.1,
    max_acceptable_drawdown_pct=15.0
)

CONSERVATIVE_CONFIG = RewardConfig(
    pnl_weight=0.7,
    risk_penalty=0.5,
    cost_penalty=0.3,
    sharpe_bonus=0.8,
    regime_consistency_bonus=0.2,
    max_acceptable_drawdown_pct=10.0
)

SHARPE_OPTIMIZED_CONFIG = RewardConfig(
    pnl_weight=0.8,
    risk_penalty=0.4,
    cost_penalty=0.25,
    sharpe_bonus=1.0,
    regime_consistency_bonus=0.15,
    max_acceptable_drawdown_pct=12.0,
    target_sharpe_ratio=2.0
)
