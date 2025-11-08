"""
Strategy Design Hierarchy

Six-stage pipeline: Idea → Hypothesis → Rule → Backtest → Optimization → Live

Maps to Engine → Mechanic → Hamilton alignment:
- Idea generator → Model hypothesis
- Mechanic trains & validates
- Log Book stores metrics
- Council votes on robustness
- Hamilton executes top performer

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline stage"""
    IDEA = "idea"
    HYPOTHESIS = "hypothesis"
    RULE_DEFINITION = "rule_definition"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    LIVE_EXECUTION = "live_execution"


class StrategyStatus(Enum):
    """Strategy status"""
    DRAFT = "draft"
    TESTING = "testing"
    OPTIMIZING = "optimizing"
    APPROVED = "approved"
    LIVE = "live"
    PAUSED = "paused"
    REJECTED = "rejected"


@dataclass
class StrategyIdea:
    """Strategy idea"""
    idea_id: str
    description: str
    rationale: str
    expected_edge: str
    created_at: datetime
    created_by: str = "system"
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class StrategyHypothesis:
    """Strategy hypothesis"""
    hypothesis_id: str
    idea_id: str
    hypothesis: str
    testable_prediction: str
    expected_outcome: str
    created_at: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class StrategyRule:
    """Strategy rule definition"""
    rule_id: str
    hypothesis_id: str
    rule_definition: str
    entry_conditions: Dict[str, any]
    exit_conditions: Dict[str, any]
    position_sizing: Dict[str, any]
    risk_management: Dict[str, any]
    created_at: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest result"""
    backtest_id: str
    rule_id: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    backtest_period: str
    created_at: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Optimization result"""
    optimization_id: str
    backtest_id: str
    optimized_parameters: Dict[str, any]
    improvement_pct: float
    optimized_sharpe: float
    created_at: datetime
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class StrategyPipeline:
    """Complete strategy pipeline"""
    strategy_id: str
    idea: StrategyIdea
    hypothesis: Optional[StrategyHypothesis] = None
    rule: Optional[StrategyRule] = None
    backtest: Optional[BacktestResult] = None
    optimization: Optional[OptimizationResult] = None
    status: StrategyStatus = StrategyStatus.DRAFT
    current_stage: PipelineStage = PipelineStage.IDEA
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, any] = field(default_factory=dict)


class StrategyDesignPipeline:
    """
    Strategy Design Hierarchy - Six-Stage Pipeline.
    
    Stages:
    1. Idea Generator → Model hypothesis
    2. Mechanic trains & validates
    3. Log Book stores metrics
    4. Council votes on robustness
    5. Hamilton executes top performer
    
    Usage:
        pipeline = StrategyDesignPipeline()
        
        # Create idea
        idea = pipeline.create_idea(
            description="Momentum strategy",
            rationale="Price momentum persists",
            expected_edge="0.5% per trade"
        )
        
        # Create hypothesis
        hypothesis = pipeline.create_hypothesis(
            idea_id=idea.idea_id,
            hypothesis="Top decile momentum outperforms",
            testable_prediction="Sharpe > 1.5"
        )
        
        # Define rules
        rule = pipeline.define_rule(
            hypothesis_id=hypothesis.hypothesis_id,
            entry_conditions={"momentum_rank": "top_decile"},
            exit_conditions={"take_profit": 0.05, "stop_loss": 0.02}
        )
        
        # Backtest
        backtest = pipeline.run_backtest(rule_id=rule.rule_id)
        
        # Optimize
        optimization = pipeline.optimize(backtest_id=backtest.backtest_id)
        
        # Approve for live
        if optimization.improvement_pct > 0.1:
            pipeline.approve_for_live(strategy_id=...)
    """
    
    def __init__(self):
        """Initialize strategy design pipeline"""
        self.strategies: Dict[str, StrategyPipeline] = {}
        self.ideas: Dict[str, StrategyIdea] = {}
        self.hypotheses: Dict[str, StrategyHypothesis] = {}
        self.rules: Dict[str, StrategyRule] = {}
        self.backtests: Dict[str, BacktestResult] = {}
        self.optimizations: Dict[str, OptimizationResult] = {}
        
        logger.info("strategy_design_pipeline_initialized")
    
    def create_idea(
        self,
        description: str,
        rationale: str,
        expected_edge: str,
        created_by: str = "system",
        metadata: Optional[Dict[str, any]] = None
    ) -> StrategyIdea:
        """
        Create strategy idea.
        
        Args:
            description: Idea description
            rationale: Rationale for the idea
            expected_edge: Expected edge
            created_by: Creator identifier
            metadata: Optional metadata
        
        Returns:
            StrategyIdea
        """
        import uuid
        
        idea_id = str(uuid.uuid4())
        
        idea = StrategyIdea(
            idea_id=idea_id,
            description=description,
            rationale=rationale,
            expected_edge=expected_edge,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self.ideas[idea_id] = idea
        
        # Create strategy pipeline
        strategy_id = str(uuid.uuid4())
        pipeline = StrategyPipeline(
            strategy_id=strategy_id,
            idea=idea,
            status=StrategyStatus.DRAFT,
            current_stage=PipelineStage.IDEA
        )
        self.strategies[strategy_id] = pipeline
        
        logger.info(
            "strategy_idea_created",
            idea_id=idea_id,
            strategy_id=strategy_id,
            description=description
        )
        
        return idea
    
    def create_hypothesis(
        self,
        idea_id: str,
        hypothesis: str,
        testable_prediction: str,
        expected_outcome: str,
        metadata: Optional[Dict[str, any]] = None
    ) -> StrategyHypothesis:
        """
        Create strategy hypothesis.
        
        Args:
            idea_id: Idea ID
            hypothesis: Hypothesis statement
            testable_prediction: Testable prediction
            expected_outcome: Expected outcome
            metadata: Optional metadata
        
        Returns:
            StrategyHypothesis
        """
        import uuid
        
        if idea_id not in self.ideas:
            raise ValueError(f"Idea not found: {idea_id}")
        
        hypothesis_id = str(uuid.uuid4())
        
        strategy_hypothesis = StrategyHypothesis(
            hypothesis_id=hypothesis_id,
            idea_id=idea_id,
            hypothesis=hypothesis,
            testable_prediction=testable_prediction,
            expected_outcome=expected_outcome,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.hypotheses[hypothesis_id] = strategy_hypothesis
        
        # Update pipeline
        strategy = self._get_strategy_by_idea(idea_id)
        if strategy:
            strategy.hypothesis = strategy_hypothesis
            strategy.current_stage = PipelineStage.HYPOTHESIS
            strategy.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "strategy_hypothesis_created",
            hypothesis_id=hypothesis_id,
            idea_id=idea_id,
            hypothesis=hypothesis
        )
        
        return strategy_hypothesis
    
    def define_rule(
        self,
        hypothesis_id: str,
        entry_conditions: Dict[str, any],
        exit_conditions: Dict[str, any],
        position_sizing: Dict[str, any],
        risk_management: Dict[str, any],
        rule_definition: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> StrategyRule:
        """
        Define strategy rules.
        
        Args:
            hypothesis_id: Hypothesis ID
            entry_conditions: Entry conditions
            exit_conditions: Exit conditions
            position_sizing: Position sizing rules
            risk_management: Risk management rules
            rule_definition: Rule definition text (optional)
            metadata: Optional metadata
        
        Returns:
            StrategyRule
        """
        import uuid
        
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis not found: {hypothesis_id}")
        
        rule_id = str(uuid.uuid4())
        
        rule = StrategyRule(
            rule_id=rule_id,
            hypothesis_id=hypothesis_id,
            rule_definition=rule_definition or f"Rule for {hypothesis_id}",
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_sizing=position_sizing,
            risk_management=risk_management,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.rules[rule_id] = rule
        
        # Update pipeline
        hypothesis = self.hypotheses[hypothesis_id]
        strategy = self._get_strategy_by_idea(hypothesis.idea_id)
        if strategy:
            strategy.rule = rule
            strategy.current_stage = PipelineStage.RULE_DEFINITION
            strategy.status = StrategyStatus.TESTING
            strategy.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "strategy_rule_defined",
            rule_id=rule_id,
            hypothesis_id=hypothesis_id
        )
        
        return rule
    
    def run_backtest(
        self,
        rule_id: str,
        backtest_period: str = "180d",
        metadata: Optional[Dict[str, any]] = None
    ) -> BacktestResult:
        """
        Run backtest for strategy rule.
        
        Args:
            rule_id: Rule ID
            backtest_period: Backtest period
            metadata: Optional metadata
        
        Returns:
            BacktestResult
        """
        import uuid
        
        if rule_id not in self.rules:
            raise ValueError(f"Rule not found: {rule_id}")
        
        # In production, would call actual backtest engine
        # For now, return placeholder result
        backtest_id = str(uuid.uuid4())
        
        backtest = BacktestResult(
            backtest_id=backtest_id,
            rule_id=rule_id,
            sharpe_ratio=1.5,  # Placeholder
            total_return=0.15,  # Placeholder
            max_drawdown=0.05,  # Placeholder
            win_rate=0.55,  # Placeholder
            total_trades=100,  # Placeholder
            backtest_period=backtest_period,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.backtests[backtest_id] = backtest
        
        # Update pipeline
        rule = self.rules[rule_id]
        hypothesis = self.hypotheses[rule.hypothesis_id]
        strategy = self._get_strategy_by_idea(hypothesis.idea_id)
        if strategy:
            strategy.backtest = backtest
            strategy.current_stage = PipelineStage.BACKTEST
            strategy.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "strategy_backtest_complete",
            backtest_id=backtest_id,
            rule_id=rule_id,
            sharpe_ratio=backtest.sharpe_ratio
        )
        
        return backtest
    
    def optimize(
        self,
        backtest_id: str,
        optimization_method: str = "grid_search",
        metadata: Optional[Dict[str, any]] = None
    ) -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            backtest_id: Backtest ID
            optimization_method: Optimization method
            metadata: Optional metadata
        
        Returns:
            OptimizationResult
        """
        import uuid
        
        if backtest_id not in self.backtests:
            raise ValueError(f"Backtest not found: {backtest_id}")
        
        backtest = self.backtests[backtest_id]
        
        # In production, would run actual optimization
        # For now, return placeholder result
        optimization_id = str(uuid.uuid4())
        
        optimization = OptimizationResult(
            optimization_id=optimization_id,
            backtest_id=backtest_id,
            optimized_parameters={},  # Placeholder
            improvement_pct=0.1,  # 10% improvement
            optimized_sharpe=backtest.sharpe_ratio * 1.1,  # 10% better
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.optimizations[optimization_id] = optimization
        
        # Update pipeline
        rule = self.rules[backtest.rule_id]
        hypothesis = self.hypotheses[rule.hypothesis_id]
        strategy = self._get_strategy_by_idea(hypothesis.idea_id)
        if strategy:
            strategy.optimization = optimization
            strategy.current_stage = PipelineStage.OPTIMIZATION
            strategy.status = StrategyStatus.OPTIMIZING
            strategy.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "strategy_optimization_complete",
            optimization_id=optimization_id,
            backtest_id=backtest_id,
            improvement_pct=optimization.improvement_pct
        )
        
        return optimization
    
    def approve_for_live(
        self,
        strategy_id: str,
        council_approval: bool = True
    ) -> bool:
        """
        Approve strategy for live execution.
        
        Args:
            strategy_id: Strategy ID
            council_approval: Whether council approved
        
        Returns:
            True if approved
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        strategy = self.strategies[strategy_id]
        
        # Check if all stages complete
        if strategy.current_stage != PipelineStage.OPTIMIZATION:
            logger.warning(
                "strategy_not_ready_for_live",
                strategy_id=strategy_id,
                current_stage=strategy.current_stage.value
            )
            return False
        
        # Check if optimization passed
        if not strategy.optimization or strategy.optimization.improvement_pct <= 0:
            logger.warning(
                "strategy_optimization_failed",
                strategy_id=strategy_id
            )
            return False
        
        # Check council approval
        if not council_approval:
            logger.warning(
                "strategy_not_approved_by_council",
                strategy_id=strategy_id
            )
            return False
        
        # Approve for live
        strategy.status = StrategyStatus.APPROVED
        strategy.current_stage = PipelineStage.LIVE_EXECUTION
        strategy.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            "strategy_approved_for_live",
            strategy_id=strategy_id,
            sharpe_ratio=strategy.optimization.optimized_sharpe
        )
        
        return True
    
    def _get_strategy_by_idea(self, idea_id: str) -> Optional[StrategyPipeline]:
        """Get strategy pipeline by idea ID"""
        for strategy in self.strategies.values():
            if strategy.idea.idea_id == idea_id:
                return strategy
        return None
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyPipeline]:
        """Get strategy pipeline"""
        return self.strategies.get(strategy_id)
    
    def get_strategies_by_status(self, status: StrategyStatus) -> List[StrategyPipeline]:
        """Get strategies by status"""
        return [s for s in self.strategies.values() if s.status == status]

