"""
Enhanced Risk Manager v2

Integrates new position sizing and gate evaluation systems.

Upgrades from original risk manager:
- Bayesian position sizing (Item #15)
- Gate verdict integration (Item #16)
- Regime-aware risk limits
- Dynamic drawdown scaling
- Counterfactual analysis integration

Usage:
    from src.cloud.training.portfolio.risk_manager_v2 import EnhancedRiskManager

    risk_mgr = EnhancedRiskManager(
        max_position_pct=10.0,
        max_portfolio_leverage=3.0,
        use_bayesian_sizing=True
    )

    # Get position size with all new features
    position = risk_mgr.calculate_position_size(
        symbol="BTC",
        signal_confidence=0.75,
        current_regime="trending",
        model_gate_verdict=gate_verdict,
        account_balance=100000,
        current_drawdown_pct=5.0
    )

    print(f"Approved position: ${position.position_value:.2f}")
    print(f"Gate status: {position.gate_approved}")
"""

from dataclasses import dataclass
from typing import Optional

import structlog

from models.position_sizing import BayesianPositionSizer, RegimeSpecificSizer
from observability.decision_gates import GateVerdict

logger = structlog.get_logger(__name__)


@dataclass
class PositionAllocation:
    """Position allocation result"""
    symbol: str
    position_value: float  # Dollar value
    position_fraction: float  # Fraction of account
    position_size: float  # Number of coins

    # Risk metrics
    max_loss_value: float
    risk_pct: float

    # Gate integration
    gate_approved: bool
    gate_score: Optional[float] = None
    gate_warnings: Optional[list] = None

    # Sizing breakdown
    base_size: float = None
    confidence_multiplier: float = None
    regime_multiplier: float = None
    drawdown_multiplier: float = None


class EnhancedRiskManager:
    """
    Enhanced Risk Manager v2

    Integrates:
    - Bayesian position sizing (from Item #15)
    - Gate evaluation (from Item #16)
    - Regime awareness
    - Dynamic risk adjustment

    Example:
        risk_mgr = EnhancedRiskManager()

        position = risk_mgr.calculate_position_size(
            symbol="BTC",
            signal_confidence=0.8,
            current_regime="trending",
            model_gate_verdict=verdict,
            account_balance=100000
        )

        if position.gate_approved:
            execute_trade(position)
    """

    def __init__(
        self,
        max_position_pct: float = 10.0,
        max_portfolio_leverage: float = 2.0,
        max_drawdown_threshold: float = 15.0,
        use_bayesian_sizing: bool = True,
        require_gate_approval: bool = True
    ):
        """
        Initialize enhanced risk manager

        Args:
            max_position_pct: Maximum position as % of account
            max_portfolio_leverage: Maximum portfolio leverage
            max_drawdown_threshold: Maximum acceptable drawdown %
            use_bayesian_sizing: Use Bayesian position sizer
            require_gate_approval: Require gate approval for trades
        """
        self.max_position_pct = max_position_pct
        self.max_portfolio_leverage = max_portfolio_leverage
        self.max_drawdown_threshold = max_drawdown_threshold
        self.use_bayesian_sizing = use_bayesian_sizing
        self.require_gate_approval = require_gate_approval

        # Initialize position sizer
        if use_bayesian_sizing:
            self.position_sizer = BayesianPositionSizer(
                max_position_fraction=max_position_pct / 100.0
            )
            self.regime_sizer = RegimeSpecificSizer(
                max_position_fraction=max_position_pct / 100.0
            )
        else:
            self.position_sizer = None
            self.regime_sizer = None

    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        current_price: float,
        current_regime: str,
        account_balance: float,
        current_drawdown_pct: float = 0.0,
        model_gate_verdict: Optional[GateVerdict] = None,
        volatility: float = 0.02
    ) -> PositionAllocation:
        """
        Calculate position size with all enhancements

        Args:
            symbol: Trading symbol
            signal_confidence: Model confidence [0-1]
            current_price: Current market price
            current_regime: Current regime
            account_balance: Account balance
            current_drawdown_pct: Current drawdown percentage
            model_gate_verdict: Gate evaluation verdict
            volatility: Market volatility

        Returns:
            PositionAllocation
        """
        # Check gate approval first
        gate_approved = True
        gate_score = None
        gate_warnings = []

        if self.require_gate_approval and model_gate_verdict:
            gate_approved = model_gate_verdict.approved
            gate_score = model_gate_verdict.overall_score

            if not gate_approved:
                gate_warnings = model_gate_verdict.failed_gates

                logger.warning(
                    "position_rejected_by_gates",
                    symbol=symbol,
                    failed_gates=gate_warnings
                )

                # Return zero position
                return self._zero_position(
                    symbol=symbol,
                    gate_approved=False,
                    gate_score=gate_score,
                    gate_warnings=gate_warnings
                )

        # Calculate base position size
        if self.use_bayesian_sizing and self.regime_sizer:
            # Use regime-specific Bayesian sizing
            sizing_result = self.regime_sizer.calculate_position_size(
                regime=current_regime,
                signal_confidence=signal_confidence,
                account_balance=account_balance,
                current_drawdown_pct=current_drawdown_pct
            )

            position_fraction = sizing_result.position_fraction
            confidence_mult = sizing_result.confidence_multiplier
            regime_mult = sizing_result.regime_multiplier
            drawdown_mult = sizing_result.drawdown_multiplier
        else:
            # Simple fixed-fraction sizing
            base_fraction = self.max_position_pct / 100.0
            position_fraction = base_fraction * signal_confidence

            confidence_mult = signal_confidence
            regime_mult = 1.0
            drawdown_mult = 1.0 - (current_drawdown_pct / self.max_drawdown_threshold)
            drawdown_mult = max(0.2, min(1.0, drawdown_mult))

            position_fraction *= drawdown_mult

        # Apply hard limits
        position_fraction = min(position_fraction, self.max_position_pct / 100.0)
        position_fraction = max(0.0, position_fraction)

        # Calculate position value
        position_value = account_balance * position_fraction
        position_size = position_value / current_price if current_price > 0 else 0

        # Calculate risk metrics
        # Assume 2% stop loss
        stop_loss_pct = 2.0
        max_loss_value = position_value * (stop_loss_pct / 100.0)
        risk_pct = (max_loss_value / account_balance) * 100.0

        allocation = PositionAllocation(
            symbol=symbol,
            position_value=position_value,
            position_fraction=position_fraction,
            position_size=position_size,
            max_loss_value=max_loss_value,
            risk_pct=risk_pct,
            gate_approved=gate_approved,
            gate_score=gate_score,
            gate_warnings=gate_warnings,
            base_size=position_fraction,
            confidence_multiplier=confidence_mult,
            regime_multiplier=regime_mult,
            drawdown_multiplier=drawdown_mult
        )

        logger.info(
            "position_calculated",
            symbol=symbol,
            position_value=position_value,
            position_fraction=position_fraction,
            gate_approved=gate_approved,
            regime=current_regime
        )

        return allocation

    def _zero_position(
        self,
        symbol: str,
        gate_approved: bool,
        gate_score: Optional[float],
        gate_warnings: list
    ) -> PositionAllocation:
        """Return zero position allocation"""
        return PositionAllocation(
            symbol=symbol,
            position_value=0.0,
            position_fraction=0.0,
            position_size=0.0,
            max_loss_value=0.0,
            risk_pct=0.0,
            gate_approved=gate_approved,
            gate_score=gate_score,
            gate_warnings=gate_warnings
        )

    def update_regime_beliefs(
        self,
        regime: str,
        trades_df
    ):
        """
        Update Bayesian beliefs for regime

        Args:
            regime: Regime name
            trades_df: Historical trades in this regime
        """
        if self.regime_sizer:
            self.regime_sizer.update_regime_beliefs(
                regime=regime,
                trades_df=trades_df
            )

            logger.info(
                "regime_beliefs_updated",
                regime=regime,
                num_trades=len(trades_df)
            )
