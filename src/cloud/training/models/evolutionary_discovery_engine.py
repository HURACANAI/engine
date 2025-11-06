"""
Evolutionary / Auto-Discovery Engine

Genetic or reinforcement agents that invent new features or strategies.
Automatically discovers profitable trading patterns.

Key Features:
1. Feature discovery (genetic algorithms)
2. Strategy evolution (reinforcement learning)
3. Pattern discovery (auto-find profitable patterns)
4. Strategy mutation (evolve existing strategies)
5. Fitness-based selection (keep best strategies)

Best in: All regimes (continuously adapts)
Strategy: Automatically discover and evolve profitable strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import structlog
import random

logger = structlog.get_logger(__name__)


@dataclass
class DiscoveredStrategy:
    """Discovered trading strategy."""
    strategy_id: str
    strategy_type: str  # "feature_combination", "pattern", "rule"
    description: str
    fitness_score: float  # Performance score (higher = better)
    n_trades: int  # Number of trades
    win_rate: float  # Win rate
    sharpe_ratio: float  # Sharpe ratio
    is_active: bool  # Whether strategy is active
    created_at: datetime
    last_updated: datetime
    strategy_params: Dict[str, Any]  # Strategy parameters


@dataclass
class FeatureCombination:
    """Discovered feature combination."""
    features: List[str]  # List of feature names
    weights: List[float]  # Feature weights
    threshold: float  # Decision threshold
    fitness_score: float  # Performance score


class EvolutionaryDiscoveryEngine:
    """
    Evolutionary / Auto-Discovery Engine.
    
    Genetic or reinforcement agents that invent new features or strategies.
    Automatically discovers profitable trading patterns.
    
    Key Features:
    - Feature discovery (genetic algorithms)
    - Strategy evolution (reinforcement learning)
    - Pattern discovery
    - Strategy mutation
    - Fitness-based selection
    """
    
    def __init__(
        self,
        population_size: int = 50,  # Number of strategies in population
        mutation_rate: float = 0.1,  # Probability of mutation
        crossover_rate: float = 0.7,  # Probability of crossover
        elite_ratio: float = 0.2,  # Top 20% are elite
        min_fitness: float = 0.5,  # Minimum fitness to keep strategy
    ):
        """
        Initialize evolutionary discovery engine.
        
        Args:
            population_size: Number of strategies in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Top ratio that are elite
            min_fitness: Minimum fitness to keep strategy
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.min_fitness = min_fitness
        
        # Population of strategies
        self.strategies: List[DiscoveredStrategy] = []
        
        # Feature combinations
        self.feature_combinations: List[FeatureCombination] = []
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
        logger.info(
            "evolutionary_discovery_engine_initialized",
            population_size=population_size,
            mutation_rate=mutation_rate,
        )
    
    def discover_feature_combination(
        self,
        available_features: List[str],
        historical_data: Dict[str, np.ndarray],
        returns: np.ndarray,
    ) -> Optional[FeatureCombination]:
        """
        Discover profitable feature combination using genetic algorithm.
        
        Args:
            available_features: List of available features
            historical_data: Dict of {feature_name: feature_values}
            returns: Target returns
        
        Returns:
            FeatureCombination if discovered, None otherwise
        """
        if len(available_features) < 2:
            return None
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            # Random feature combination
            n_features = random.randint(2, min(5, len(available_features)))
            features = random.sample(available_features, n_features)
            weights = [random.uniform(-1.0, 1.0) for _ in features]
            threshold = random.uniform(-1.0, 1.0)
            
            population.append({
                "features": features,
                "weights": weights,
                "threshold": threshold,
            })
        
        # Evolve population
        for generation in range(10):  # 10 generations
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_feature_combination(
                    features=individual["features"],
                    weights=individual["weights"],
                    threshold=individual["threshold"],
                    historical_data=historical_data,
                    returns=returns,
                )
                fitness_scores.append(fitness)
            
            # Select elite
            elite_indices = np.argsort(fitness_scores)[-int(self.population_size * self.elite_ratio):]
            elite = [population[i] for i in elite_indices]
            
            # Create new population
            new_population = elite.copy()  # Keep elite
            
            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    child = self._crossover_feature_combination(parent1, parent2)
                else:
                    # Mutation
                    parent = random.choice(elite)
                    child = self._mutate_feature_combination(parent, available_features)
                
                new_population.append(child)
            
            population = new_population
        
        # Get best individual
        fitness_scores = []
        for individual in population:
            fitness = self._evaluate_feature_combination(
                features=individual["features"],
                weights=individual["weights"],
                threshold=individual["threshold"],
                historical_data=historical_data,
                returns=returns,
            )
            fitness_scores.append(fitness)
        
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        if best_fitness < self.min_fitness:
            # Not good enough
            return None
        
        # Create feature combination
        combination = FeatureCombination(
            features=best_individual["features"],
            weights=best_individual["weights"],
            threshold=best_individual["threshold"],
            fitness_score=best_fitness,
        )
        
        self.feature_combinations.append(combination)
        
        logger.info(
            "feature_combination_discovered",
            features=combination.features,
            fitness_score=best_fitness,
        )
        
        return combination
    
    def _evaluate_feature_combination(
        self,
        features: List[str],
        weights: List[float],
        threshold: float,
        historical_data: Dict[str, np.ndarray],
        returns: np.ndarray,
    ) -> float:
        """Evaluate fitness of feature combination."""
        try:
            # Calculate feature signal
            signal = np.zeros(len(returns))
            for i, feature_name in enumerate(features):
                if feature_name in historical_data:
                    feature_values = historical_data[feature_name]
                    if len(feature_values) == len(returns):
                        signal += feature_values * weights[i]
            
            # Normalize signal
            if np.std(signal) > 0:
                signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Generate predictions
            predictions = (signal > threshold).astype(int)
            
            # Calculate returns for predictions
            predicted_returns = returns * predictions
            
            # Calculate fitness (Sharpe ratio)
            if np.std(predicted_returns) > 0:
                sharpe = np.mean(predicted_returns) / np.std(predicted_returns)
                fitness = max(0.0, sharpe)  # Only positive Sharpe
            else:
                fitness = 0.0
            
            return fitness
        except Exception as e:
            logger.warning("feature_combination_evaluation_failed", error=str(e))
            return 0.0
    
    def _crossover_feature_combination(
        self,
        parent1: Dict,
        parent2: Dict,
    ) -> Dict:
        """Crossover two feature combinations."""
        # Combine features
        all_features = list(set(parent1["features"] + parent2["features"]))
        n_features = random.randint(2, min(5, len(all_features)))
        child_features = random.sample(all_features, n_features)
        
        # Average weights (if feature in both parents)
        child_weights = []
        for feature in child_features:
            if feature in parent1["features"] and feature in parent2["features"]:
                idx1 = parent1["features"].index(feature)
                idx2 = parent2["features"].index(feature)
                weight = (parent1["weights"][idx1] + parent2["weights"][idx2]) / 2.0
            elif feature in parent1["features"]:
                idx1 = parent1["features"].index(feature)
                weight = parent1["weights"][idx1]
            else:
                idx2 = parent2["features"].index(feature)
                weight = parent2["weights"][idx2]
            child_weights.append(weight)
        
        # Average threshold
        child_threshold = (parent1["threshold"] + parent2["threshold"]) / 2.0
        
        return {
            "features": child_features,
            "weights": child_weights,
            "threshold": child_threshold,
        }
    
    def _mutate_feature_combination(
        self,
        parent: Dict,
        available_features: List[str],
    ) -> Dict:
        """Mutate a feature combination."""
        child = parent.copy()
        
        # Mutate features
        if random.random() < self.mutation_rate:
            # Add or remove a feature
            if len(child["features"]) > 2 and random.random() < 0.5:
                # Remove a feature
                idx = random.randint(0, len(child["features"]) - 1)
                child["features"].pop(idx)
                child["weights"].pop(idx)
            else:
                # Add a feature
                new_feature = random.choice([f for f in available_features if f not in child["features"]])
                if new_feature:
                    child["features"].append(new_feature)
                    child["weights"].append(random.uniform(-1.0, 1.0))
        
        # Mutate weights
        for i in range(len(child["weights"])):
            if random.random() < self.mutation_rate:
                child["weights"][i] += random.uniform(-0.2, 0.2)
                child["weights"][i] = np.clip(child["weights"][i], -1.0, 1.0)
        
        # Mutate threshold
        if random.random() < self.mutation_rate:
            child["threshold"] += random.uniform(-0.2, 0.2)
            child["threshold"] = np.clip(child["threshold"], -1.0, 1.0)
        
        return child
    
    def get_best_strategies(
        self,
        n: int = 10,
    ) -> List[DiscoveredStrategy]:
        """Get best performing strategies."""
        active_strategies = [s for s in self.strategies if s.is_active]
        sorted_strategies = sorted(active_strategies, key=lambda x: x.fitness_score, reverse=True)
        return sorted_strategies[:n]

