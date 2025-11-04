"""
Master Orchestrator - Phase 4

The supreme controller that orchestrates ALL phases and features.

This is the apex of the system - it coordinates:
- Phase 1: Tactical Intelligence
- Phase 2: Portfolio Intelligence
- Phase 3: Advanced Intelligence
- Phase 4: Meta-Learning & Self-Optimization

The Master Orchestrator makes all final decisions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
import structlog

from ..models.adaptive_learner import AdaptiveLearner
from ..models.ensemble_predictor import EnsemblePredictor, PredictionSource
from ..models.meta_learner import MetaLearner
from ..models.self_diagnostic import SelfDiagnostic, SystemHealthReport
from .phase2_orchestrator import Phase2Orchestrator

logger = structlog.get_logger()


@dataclass
class MasterDecision:
    """Final trading decision from Master Orchestrator."""

    symbol: str
    action: str  # "enter", "exit", "hold"
    confidence: float
    position_size_gbp: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    health_status: str
    meta_config_id: str


class MasterOrchestrator:
    """
    Master orchestrator coordinating all system phases.

    This is the highest level of intelligence in the system.
    It makes final decisions after consulting all subsystems.
    """

    def __init__(
        self,
        total_capital_gbp: float = 10000.0,
        base_position_size_gbp: float = 100.0,
    ):
        """
        Initialize master orchestrator.

        Args:
            total_capital_gbp: Total capital
            base_position_size_gbp: Base position size
        """
        self.total_capital = total_capital_gbp

        # Initialize all orchestrators
        logger.info("initializing_phase2_orchestrator")
        self.phase2 = Phase2Orchestrator(
            total_capital_gbp=total_capital_gbp,
            base_position_size_gbp=base_position_size_gbp,
        )

        # Initialize Phase 3 modules
        logger.info("initializing_phase3_modules")
        self.ensemble = EnsemblePredictor(ema_alpha=0.05, min_agreement_threshold=0.6)
        self.adaptive_learner = AdaptiveLearner(
            base_learning_rate=0.05, min_learning_rate=0.01, max_learning_rate=0.20
        )

        # Initialize Phase 4 modules
        logger.info("initializing_phase4_modules")
        self.meta_learner = MetaLearner(exploration_rate=0.1, meta_learning_rate=0.01)
        self.diagnostics = SelfDiagnostic(
            performance_threshold=0.50, max_drawdown_threshold=0.20
        )

        # Register default hyperparameter configurations
        self._register_default_configs()

        # System state
        self.current_regime = "unknown"
        self.total_trades = 0
        self.winning_trades = 0
        self.current_drawdown = 0.0
        self.last_health_check = datetime.now()

        logger.info(
            "master_orchestrator_initialized",
            capital_gbp=total_capital_gbp,
            base_size_gbp=base_position_size_gbp,
        )

    def _register_default_configs(self) -> None:
        """Register default hyperparameter configurations for meta-learning."""
        configs = {
            "conservative": {
                "confidence_threshold": 0.60,
                "learning_rate": 0.03,
                "kelly_fraction": 0.20,
            },
            "balanced": {
                "confidence_threshold": 0.52,
                "learning_rate": 0.05,
                "kelly_fraction": 0.25,
            },
            "aggressive": {
                "confidence_threshold": 0.48,
                "learning_rate": 0.08,
                "kelly_fraction": 0.30,
            },
        }

        for config_id, params in configs.items():
            self.meta_learner.register_config(config_id, params)

    def evaluate_opportunity(
        self,
        symbol: str,
        df: pl.DataFrame,
        current_idx: int,
        base_confidence: float,
        asset_volatility_bps: float,
        current_regime: str,
    ) -> MasterDecision:
        """
        Master evaluation using all system intelligence.

        Args:
            symbol: Symbol to evaluate
            df: Price data
            current_idx: Current index
            base_confidence: Base confidence from Phase 1
            asset_volatility_bps: Asset volatility
            current_regime: Current market regime

        Returns:
            MasterDecision with all intelligence applied
        """
        self.current_regime = current_regime

        # 1. Check system health first
        health_report = self._run_health_check()

        if health_report.overall_status.value in ["critical", "failing"]:
            logger.warning(
                "trading_paused_poor_health",
                status=health_report.overall_status.value,
            )
            return MasterDecision(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                position_size_gbp=0.0,
                stop_loss=None,
                take_profit=None,
                reasoning=f"System health {health_report.overall_status.value} - pausing trading",
                health_status=health_report.overall_status.value,
                meta_config_id="none",
            )

        # 2. Select hyperparameter configuration from meta-learner
        config_id, config_params = self.meta_learner.select_config(current_regime)

        # 3. Get ensemble prediction (Phase 3)
        # For now, use base confidence as RL prediction
        # In full integration, would have actual predictions from all sources
        rl_pred = PredictionSource(
            source_name="rl_agent",
            prediction="buy",  # Simplified
            confidence=base_confidence,
            reasoning="RL agent signal",
        )

        regime_pred = PredictionSource(
            source_name="regime",
            prediction="buy" if current_regime == "trend" else "hold",
            confidence=0.7 if current_regime == "trend" else 0.4,
            reasoning=f"Regime is {current_regime}",
        )

        ensemble_result = self.ensemble.predict(
            rl_prediction=rl_pred,
            regime_prediction=regime_pred,
        )

        # 4. Apply adaptive learning rate
        performance_signal = self.winning_trades / max(1, self.total_trades)
        regime_changed = False  # Would track this in full system

        current_learning_rate = self.adaptive_learner.adapt(
            performance_signal, current_regime, regime_changed
        )

        # 5. Phase 2 evaluation (risk management, patterns, portfolio)
        phase2_decision = self.phase2.evaluate_entry(
            symbol=symbol,
            df=df,
            current_idx=current_idx,
            confidence=ensemble_result.ensemble_confidence,
            asset_volatility_bps=asset_volatility_bps,
            current_drawdown_pct=self.current_drawdown,
            win_rate=performance_signal,
        )

        # 6. Make master decision
        action = phase2_decision.action
        final_confidence = ensemble_result.ensemble_confidence

        # Apply meta-learning adjustments
        adjusted_confidence = self.meta_learner.suggest_hyperparameter_adjustment(
            "confidence_threshold",
            final_confidence,
            current_regime,
        )

        reasoning_parts = [
            f"Ensemble: {ensemble_result.reasoning}",
            f"Phase2: {phase2_decision.risk_reasoning}",
            f"Config: {config_id}",
            f"Health: {health_report.overall_status.value}",
        ]

        return MasterDecision(
            symbol=symbol,
            action=action,
            confidence=adjusted_confidence,
            position_size_gbp=phase2_decision.recommended_size_gbp,
            stop_loss=phase2_decision.stop_loss_price,
            take_profit=phase2_decision.take_profit_price,
            reasoning=" | ".join(reasoning_parts),
            health_status=health_report.overall_status.value,
            meta_config_id=config_id,
        )

    def _run_health_check(self) -> SystemHealthReport:
        """Run system health diagnostics."""
        performance_signal = self.winning_trades / max(1, self.total_trades)

        # Calculate learning efficiency
        # Simplified - would track actual improvement over time
        learning_efficiency = 0.05  # Placeholder

        health_report = self.diagnostics.diagnose(
            win_rate=performance_signal,
            current_drawdown=self.current_drawdown,
            learning_efficiency=learning_efficiency,
            feature_importance_stability=0.8,  # Placeholder
            num_trades=self.total_trades,
            data_quality_score=0.95,  # Placeholder
        )

        self.last_health_check = datetime.now()

        return health_report

    def record_trade_outcome(
        self,
        symbol: str,
        is_winner: bool,
        pnl_gbp: float,
        config_id: str,
    ) -> None:
        """
        Record trade outcome and update all learning systems.

        Args:
            symbol: Symbol traded
            is_winner: Whether trade won
            pnl_gbp: Profit/loss
            config_id: Config used for this trade
        """
        self.total_trades += 1
        if is_winner:
            self.winning_trades += 1

        # Update meta-learner
        performance = 1.0 if is_winner else 0.0
        self.meta_learner.update_config_performance(
            config_id, performance, self.current_regime
        )

        # Update ensemble weights
        # Would update individual source performances here

        logger.info(
            "trade_outcome_recorded",
            symbol=symbol,
            is_winner=is_winner,
            pnl_gbp=pnl_gbp,
            total_trades=self.total_trades,
            win_rate=self.winning_trades / self.total_trades,
        )

    def get_system_summary(self) -> Dict:
        """Get complete system summary."""
        win_rate = self.winning_trades / max(1, self.total_trades)

        return {
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "current_drawdown": self.current_drawdown,
            "current_regime": self.current_regime,
            "phase2_summary": self.phase2.get_portfolio_summary(),
            "meta_summary": self.meta_learner.get_meta_performance_summary(),
            "ensemble_stats": self.ensemble.get_source_stats(),
            "adaptive_learning": self.adaptive_learner.get_stats(),
        }

    def should_pause_trading(self) -> tuple[bool, str]:
        """Determine if trading should be paused."""
        # Check meta-learner
        should_reset, reset_reason = self.meta_learner.should_trigger_system_reset()
        if should_reset:
            return (True, f"Meta-learner: {reset_reason}")

        # Check drawdown
        if self.current_drawdown > 0.25:
            return (True, f"Excessive drawdown: {self.current_drawdown:.1%}")

        # Check Phase 2
        should_reduce, reduce_reason = self.phase2.should_reduce_exposure(
            self.current_drawdown * 100
        )
        if should_reduce:
            return (True, f"Phase2: {reduce_reason}")

        return (False, "All systems go")

    def get_complete_state(self) -> Dict:
        """Get complete system state for persistence."""
        return {
            "phase2": self.phase2.get_state(),
            "ensemble": self.ensemble.get_state(),
            "adaptive_learner": self.adaptive_learner.get_state(),
            "meta_learner": self.meta_learner.get_state(),
            "system_metrics": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "current_drawdown": self.current_drawdown,
                "current_regime": self.current_regime,
            },
        }

    def load_complete_state(self, state: Dict) -> None:
        """Load complete system state."""
        self.phase2.load_state(state.get("phase2", {}))
        self.ensemble.load_state(state.get("ensemble", {}))
        self.adaptive_learner.load_state(state.get("adaptive_learner", {}))
        self.meta_learner.load_state(state.get("meta_learner", {}))

        metrics = state.get("system_metrics", {})
        self.total_trades = metrics.get("total_trades", 0)
        self.winning_trades = metrics.get("winning_trades", 0)
        self.current_drawdown = metrics.get("current_drawdown", 0.0)
        self.current_regime = metrics.get("current_regime", "unknown")

        logger.info(
            "master_orchestrator_state_loaded",
            total_trades=self.total_trades,
            win_rate=self.winning_trades / max(1, self.total_trades),
        )
