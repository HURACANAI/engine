"""
Meta Learning - Contextual Bandit over Engines

Contextual bandit over engines with regime and liquidity as context.
Exploration budget per day and per symbol with caps.

Key Features:
- Contextual bandit algorithm
- Engine selection based on context
- Exploration vs exploitation
- Budget management
- Regime and liquidity as context

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ExplorationStrategy(Enum):
    """Exploration strategy"""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class Context:
    """Context for bandit decision"""
    regime: str
    liquidity_score: float
    volatility: float
    market_trend: str  # "up", "down", "sideways"
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class EnginePerformance:
    """Engine performance tracking"""
    engine_id: str
    total_rewards: float
    total_pulls: int
    avg_reward: float
    context_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)  # context -> {reward, pulls}


@dataclass
class BanditResult:
    """Bandit selection result"""
    selected_engine: str
    exploration: bool
    confidence: float
    context: Context
    exploration_budget_used: float


class MetaLearner:
    """
    Meta Learning - Contextual Bandit over Engines.
    
    Selects best engine based on context (regime, liquidity, etc.)
    with exploration budget management.
    
    Usage:
        learner = MetaLearner(
            engines=["trend", "range", "breakout", ...],
            exploration_budget_per_day=10.0,
            exploration_budget_per_symbol=2.0
        )
        
        result = learner.select_engine(
            context=Context(regime="trend", liquidity_score=0.8, ...),
            symbol="BTCUSDT"
        )
        
        # Update with reward
        learner.update(
            engine_id=result.selected_engine,
            context=context,
            reward=0.15,  # P&L or Sharpe ratio
            symbol="BTCUSDT"
        )
    """
    
    def __init__(
        self,
        engines: List[str],
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        epsilon: float = 0.1,  # Exploration probability
        exploration_budget_per_day: float = 10.0,
        exploration_budget_per_symbol: float = 2.0,
        learning_rate: float = 0.1,
        decay_factor: float = 0.99
    ):
        """
        Initialize meta learner.
        
        Args:
            engines: List of engine IDs
            exploration_strategy: Exploration strategy
            epsilon: Exploration probability (for epsilon-greedy)
            exploration_budget_per_day: Daily exploration budget
            exploration_budget_per_symbol: Per-symbol exploration budget
            learning_rate: Learning rate for updates
            decay_factor: Decay factor for exploration
        """
        self.engines = engines
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.exploration_budget_per_day = exploration_budget_per_day
        self.exploration_budget_per_symbol = exploration_budget_per_symbol
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        
        # Track engine performance
        self.engine_performance: Dict[str, EnginePerformance] = {
            engine: EnginePerformance(
                engine_id=engine,
                total_rewards=0.0,
                total_pulls=0,
                avg_reward=0.0
            )
            for engine in engines
        }
        
        # Track exploration budget
        self.daily_exploration_budget: Dict[str, float] = {}  # symbol -> budget used
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        logger.info(
            "meta_learner_initialized",
            num_engines=len(engines),
            exploration_strategy=exploration_strategy.value,
            exploration_budget_per_day=exploration_budget_per_day
        )
    
    def select_engine(
        self,
        context: Context,
        symbol: str
    ) -> BanditResult:
        """
        Select engine using contextual bandit.
        
        Args:
            context: Context (regime, liquidity, etc.)
            symbol: Trading symbol
        
        Returns:
            BanditResult with selected engine
        """
        # Reset daily budget if new day
        self._reset_daily_budget_if_needed()
        
        # Check exploration budget
        exploration_budget_remaining = self._get_exploration_budget_remaining(symbol)
        
        # Determine if exploring or exploiting
        exploring = False
        if exploration_budget_remaining > 0:
            if self.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
                exploring = np.random.random() < self.epsilon
            elif self.exploration_strategy == ExplorationStrategy.UCB:
                # UCB: explore if uncertainty is high
                exploring = self._should_explore_ucb(context)
            elif self.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
                # Thompson sampling: always uses probability, but we can force exploration
                exploring = exploration_budget_remaining > 0
        
        # Select engine
        if exploring:
            selected_engine = self._explore(context, symbol)
            exploration_budget_used = 1.0
        else:
            selected_engine = self._exploit(context)
            exploration_budget_used = 0.0
        
        # Update exploration budget
        if exploring:
            self._use_exploration_budget(symbol, exploration_budget_used)
        
        # Calculate confidence
        confidence = self._calculate_confidence(selected_engine, context)
        
        result = BanditResult(
            selected_engine=selected_engine,
            exploration=exploring,
            confidence=confidence,
            context=context,
            exploration_budget_used=exploration_budget_used
        )
        
        logger.debug(
            "engine_selected",
            engine=selected_engine,
            exploration=exploring,
            context_regime=context.regime,
            symbol=symbol
        )
        
        return result
    
    def update(
        self,
        engine_id: str,
        context: Context,
        reward: float,
        symbol: str
    ) -> None:
        """
        Update engine performance with reward.
        
        Args:
            engine_id: Engine ID
            context: Context
            reward: Reward (P&L, Sharpe ratio, etc.)
            symbol: Trading symbol
        """
        if engine_id not in self.engine_performance:
            return
        
        performance = self.engine_performance[engine_id]
        
        # Update overall performance
        performance.total_rewards += reward
        performance.total_pulls += 1
        performance.avg_reward = performance.total_rewards / performance.total_pulls
        
        # Update context-specific performance
        context_key = self._context_to_key(context)
        if context_key not in performance.context_performance:
            performance.context_performance[context_key] = {
                "reward": 0.0,
                "pulls": 0,
                "avg_reward": 0.0
            }
        
        context_perf = performance.context_performance[context_key]
        context_perf["reward"] += reward
        context_perf["pulls"] += 1
        context_perf["avg_reward"] = context_perf["reward"] / context_perf["pulls"]
        
        logger.debug(
            "engine_performance_updated",
            engine_id=engine_id,
            reward=reward,
            avg_reward=performance.avg_reward,
            context_key=context_key
        )
    
    def _explore(self, context: Context, symbol: str) -> str:
        """Explore: select random engine"""
        # Select random engine (excluding worst performers)
        available_engines = [
            engine for engine in self.engines
            if self.engine_performance[engine].total_pulls < 10  # Prefer less explored
        ]
        
        if not available_engines:
            available_engines = self.engines
        
        return np.random.choice(available_engines)
    
    def _exploit(self, context: Context) -> str:
        """Exploit: select best engine for context"""
        # Get context-specific performance
        context_key = self._context_to_key(context)
        
        best_engine = None
        best_reward = float('-inf')
        
        for engine_id, performance in self.engine_performance.items():
            # Get context-specific reward
            if context_key in performance.context_performance:
                reward = performance.context_performance[context_key]["avg_reward"]
            else:
                # Fallback to overall reward
                reward = performance.avg_reward
            
            # UCB bonus for exploration (if using UCB)
            if self.exploration_strategy == ExplorationStrategy.UCB:
                if performance.total_pulls > 0:
                    ucb_bonus = np.sqrt(2 * np.log(sum(p.total_pulls for p in self.engine_performance.values())) / performance.total_pulls)
                    reward += ucb_bonus
            
            if reward > best_reward:
                best_reward = reward
                best_engine = engine_id
        
        return best_engine or self.engines[0]
    
    def _should_explore_ucb(self, context: Context) -> bool:
        """Determine if should explore using UCB"""
        # Explore if there are engines with low pull counts
        min_pulls = min(p.total_pulls for p in self.engine_performance.values())
        return min_pulls < 5
    
    def _calculate_confidence(self, engine_id: str, context: Context) -> float:
        """Calculate confidence in engine selection"""
        if engine_id not in self.engine_performance:
            return 0.5
        
        performance = self.engine_performance[engine_id]
        context_key = self._context_to_key(context)
        
        # Get context-specific confidence
        if context_key in performance.context_performance:
            context_perf = performance.context_performance[context_key]
            if context_perf["pulls"] > 0:
                # Confidence based on number of pulls and consistency
                pulls = context_perf["pulls"]
                confidence = min(1.0, pulls / 20.0)  # More pulls = higher confidence
                return confidence
        
        # Fallback to overall confidence
        if performance.total_pulls > 0:
            confidence = min(1.0, performance.total_pulls / 50.0)
            return confidence
        
        return 0.5
    
    def _context_to_key(self, context: Context) -> str:
        """Convert context to key for performance tracking"""
        # Simplified key: regime + liquidity bin
        liquidity_bin = "high" if context.liquidity_score > 0.7 else "low" if context.liquidity_score < 0.3 else "medium"
        return f"{context.regime}_{liquidity_bin}"
    
    def _get_exploration_budget_remaining(self, symbol: str) -> float:
        """Get remaining exploration budget for symbol"""
        used = self.daily_exploration_budget.get(symbol, 0.0)
        return max(0.0, self.exploration_budget_per_symbol - used)
    
    def _use_exploration_budget(self, symbol: str, amount: float) -> None:
        """Use exploration budget"""
        if symbol not in self.daily_exploration_budget:
            self.daily_exploration_budget[symbol] = 0.0
        
        self.daily_exploration_budget[symbol] += amount
    
    def _reset_daily_budget_if_needed(self) -> None:
        """Reset daily budget if new day"""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            self.daily_exploration_budget = {}
            self.last_reset_date = current_date
            logger.info("exploration_budget_reset", date=current_date)
    
    def get_best_engine(self, context: Context) -> str:
        """Get best engine for context (without exploration)"""
        return self._exploit(context)
    
    def get_engine_stats(self) -> Dict[str, Dict[str, float]]:
        """Get engine performance statistics"""
        stats = {}
        for engine_id, performance in self.engine_performance.items():
            stats[engine_id] = {
                "total_rewards": performance.total_rewards,
                "total_pulls": performance.total_pulls,
                "avg_reward": performance.avg_reward,
                "num_contexts": len(performance.context_performance)
            }
        return stats
