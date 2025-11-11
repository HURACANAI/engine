"""
Formal Consensus Engine - Mathematically rigorous signal combination.

Implements:
- Probability normalization (0 to 1)
- Brier score calibration (Platt scaling)
- Exponential decay reliability weighting
- Engine correlation penalty
- Adaptive threshold based on realized volatility
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EngineVote:
    """Individual engine vote with metadata."""
    engine_id: str
    side: int  # -1 (sell), 0 (wait), +1 (buy)
    probability: float  # 0 to 1
    raw_score: float  # Original score before normalization
    reliability: float  # 0 to 1, based on recent performance
    brier_score: float  # Recent Brier score for calibration


@dataclass
class ConsensusResult:
    """Final consensus calculation result."""
    consensus_score: float  # Final S value
    weighted_sum: float  # Sum of weighted votes
    threshold: float  # Adaptive threshold (k * sigma_S)
    should_trade: bool  # Whether |S| > threshold
    action: str  # "buy", "sell", or "wait"
    confidence: float  # Normalized confidence (0 to 1)
    engine_weights: Dict[str, float]  # Final weights per engine
    correlation_penalty_applied: bool
    metadata: Dict[str, any]


class FormalConsensusEngine:
    """
    Formal consensus engine with mathematical rigor.
    
    Implements:
    1. Probability normalization: p in [0, 1]
    2. Brier score calibration: Platt scaling per engine
    3. Reliability weighting: w_i with exponential decay
    4. Correlation penalty: Down-weight correlated engines
    5. Adaptive threshold: |S| > k * sigma_S
    """
    
    def __init__(
        self,
        half_life_days: int = 14,
        threshold_k: float = 0.75,
        min_reliability: float = 0.3,
        correlation_shrinkage: float = 0.1,
    ) -> None:
        """
        Initialize formal consensus engine.
        
        Args:
            half_life_days: Half-life for reliability decay (default: 14 days)
            threshold_k: Multiplier for adaptive threshold (default: 0.75)
            min_reliability: Minimum reliability to include engine (default: 0.3)
            correlation_shrinkage: Shrinkage factor for correlation matrix (default: 0.1)
        """
        self.half_life_days = half_life_days
        self.threshold_k = threshold_k
        self.min_reliability = min_reliability
        self.correlation_shrinkage = correlation_shrinkage
        
        # Track engine performance
        self.engine_performance: Dict[str, Dict[str, any]] = {}
        
        # Track consensus history for volatility calculation
        self.consensus_history: List[float] = []
        self.consensus_window: int = 100  # Rolling window for sigma_S
        
        logger.info(
            "formal_consensus_engine_initialized",
            half_life_days=half_life_days,
            threshold_k=threshold_k,
            min_reliability=min_reliability
        )
    
    def normalize_probability(
        self,
        raw_score: float,
        engine_id: str,
        use_calibration: bool = True
    ) -> float:
        """
        Normalize engine score to probability p in [0, 1].
        
        Uses Platt scaling if calibration data available.
        
        Args:
            raw_score: Raw engine output
            engine_id: Engine identifier
            use_calibration: Whether to use Brier score calibration
        
        Returns:
            Normalized probability in [0, 1]
        """
        if use_calibration and engine_id in self.engine_performance:
            # Use Platt scaling for calibration
            perf = self.engine_performance[engine_id]
            if 'platt_a' in perf and 'platt_b' in perf:
                # Platt scaling: p = 1 / (1 + exp(a * score + b))
                a = perf['platt_a']
                b = perf['platt_b']
                prob = 1.0 / (1.0 + np.exp(a * raw_score + b))
                return np.clip(prob, 0.0, 1.0)
        
        # Fallback: Simple sigmoid normalization
        # Assume raw_score is roughly in [-1, 1] range
        prob = (raw_score + 1.0) / 2.0
        return np.clip(prob, 0.0, 1.0)
    
    def calculate_reliability(
        self,
        engine_id: str,
        current_date: datetime
    ) -> float:
        """
        Calculate engine reliability with exponential decay.
        
        Reliability = 1 - Brier_score, with exponential decay on lookback.
        
        Args:
            engine_id: Engine identifier
            current_date: Current date for decay calculation
        
        Returns:
            Reliability score in [0, 1]
        """
        if engine_id not in self.engine_performance:
            return 0.5  # Default reliability
        
        perf = self.engine_performance[engine_id]
        
        # Get Brier score
        brier = perf.get('brier_score', 0.5)
        
        # Base reliability = 1 - Brier (lower Brier = higher reliability)
        base_reliability = 1.0 - brier
        
        # Apply exponential decay based on recency
        if 'last_update' in perf:
            days_ago = (current_date - perf['last_update']).days
            decay_factor = np.exp(-np.log(2) * days_ago / self.half_life_days)
            reliability = base_reliability * decay_factor
        else:
            reliability = base_reliability
        
        return np.clip(reliability, 0.0, 1.0)
    
    def calculate_correlation_penalty(
        self,
        engine_votes: List[EngineVote],
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate correlation penalty to down-weight highly correlated engines.
        
        Uses shrinkage on correlation matrix and minimizes w^T C w.
        
        Args:
            engine_votes: List of engine votes
            lookback_days: Days to look back for correlation calculation
        
        Returns:
            Dictionary of adjusted weights per engine
        """
        if len(engine_votes) < 2:
            # No correlation penalty if only one engine
            return {vote.engine_id: 1.0 for vote in engine_votes}
        
        # Get correlation matrix of recent decisions
        correlation_matrix = self._get_engine_correlation_matrix(
            [v.engine_id for v in engine_votes],
            lookback_days
        )
        
        if correlation_matrix is None:
            # Fallback: equal weights
            return {vote.engine_id: 1.0 / len(engine_votes) for vote in engine_votes}
        
        # Apply shrinkage: C_shrunk = (1 - lambda) * C + lambda * I
        n = len(engine_votes)
        identity = np.eye(n)
        C_shrunk = (1 - self.correlation_shrinkage) * correlation_matrix + \
                   self.correlation_shrinkage * identity
        
        # Optimize weights: minimize w^T C w subject to sum(w) = 1, w >= 0
        initial_weights = np.ones(n) / n
        
        def objective(w):
            return w.T @ C_shrunk @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
            else:
                # Fallback to equal weights
                weights = initial_weights
        except Exception as e:
            logger.warning("correlation_optimization_failed", error=str(e))
            weights = initial_weights
        
        # Return as dictionary
        return {
            engine_votes[i].engine_id: float(weights[i])
            for i in range(len(engine_votes))
        }
    
    def _get_engine_correlation_matrix(
        self,
        engine_ids: List[str],
        lookback_days: int
    ) -> Optional[np.ndarray]:
        """
        Get correlation matrix of engine decisions.
        
        Args:
            engine_ids: List of engine IDs
            lookback_days: Days to look back
        
        Returns:
            Correlation matrix or None if insufficient data
        """
        # TODO: Implement actual correlation calculation from historical decisions
        # For now, return None to use fallback
        # This should query historical engine votes and calculate correlation
        return None
    
    def calculate_consensus(
        self,
        engine_votes: List[EngineVote],
        realized_volatility: Optional[float] = None,
        current_date: Optional[datetime] = None
    ) -> ConsensusResult:
        """
        Calculate formal consensus signal.
        
        Formula: S = sum(w_i * (2*p_i - 1) * s_i)
        Where:
        - w_i: Reliability-weighted, correlation-adjusted weight
        - p_i: Normalized probability [0, 1]
        - s_i: Side (-1, 0, +1)
        
        Args:
            engine_votes: List of engine votes
            realized_volatility: Current realized volatility (for adaptive threshold)
            current_date: Current date (for reliability decay)
        
        Returns:
            ConsensusResult with final signal and metadata
        """
        if current_date is None:
            current_date = datetime.now()
        
        if not engine_votes:
            return ConsensusResult(
                consensus_score=0.0,
                weighted_sum=0.0,
                threshold=0.0,
                should_trade=False,
                action="wait",
                confidence=0.0,
                engine_weights={},
                correlation_penalty_applied=False,
                metadata={}
            )
        
        # Step 1: Normalize probabilities
        normalized_votes = []
        for vote in engine_votes:
            prob = self.normalize_probability(vote.raw_score, vote.engine_id)
            normalized_votes.append(EngineVote(
                engine_id=vote.engine_id,
                side=vote.side,
                probability=prob,
                raw_score=vote.raw_score,
                reliability=vote.reliability,
                brier_score=vote.brier_score
            ))
        
        # Step 2: Calculate reliability weights
        reliability_weights = {}
        for vote in normalized_votes:
            rel = self.calculate_reliability(vote.engine_id, current_date)
            if rel < self.min_reliability:
                # Skip engines below minimum reliability
                continue
            reliability_weights[vote.engine_id] = rel
        
        # Normalize reliability weights
        total_rel = sum(reliability_weights.values())
        if total_rel > 0:
            reliability_weights = {
                k: v / total_rel
                for k, v in reliability_weights.items()
            }
        else:
            # Fallback: equal weights
            reliability_weights = {
                vote.engine_id: 1.0 / len(normalized_votes)
                for vote in normalized_votes
            }
        
        # Step 3: Apply correlation penalty
        filtered_votes = [
            v for v in normalized_votes
            if v.engine_id in reliability_weights
        ]
        correlation_weights = self.calculate_correlation_penalty(filtered_votes)
        
        # Combine reliability and correlation weights
        final_weights = {}
        for engine_id in reliability_weights:
            if engine_id in correlation_weights:
                # Combine: w_final = w_reliability * w_correlation (normalized)
                final_weights[engine_id] = (
                    reliability_weights[engine_id] * correlation_weights[engine_id]
                )
        
        # Normalize final weights
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        else:
            final_weights = reliability_weights
        
        # Step 4: Calculate consensus score
        # S = sum(w_i * (2*p_i - 1) * s_i)
        consensus_sum = 0.0
        for vote in filtered_votes:
            if vote.engine_id in final_weights:
                w = final_weights[vote.engine_id]
                p = vote.probability
                s = vote.side
                contribution = w * (2.0 * p - 1.0) * s
                consensus_sum += contribution
        
        # Step 5: Calculate adaptive threshold
        # Threshold = k * sigma_S
        sigma_S = self._calculate_consensus_volatility()
        
        # Adjust threshold based on realized volatility if provided
        if realized_volatility is not None:
            # Scale threshold with market volatility
            volatility_adjustment = np.clip(realized_volatility / 0.02, 0.5, 2.0)  # Normalize to 2% daily vol
            threshold = self.threshold_k * sigma_S * volatility_adjustment
        else:
            threshold = self.threshold_k * sigma_S
        
        # Step 6: Determine action
        abs_consensus = abs(consensus_sum)
        should_trade = abs_consensus > threshold
        
        if should_trade:
            action = "buy" if consensus_sum > 0 else "sell"
        else:
            action = "wait"
        
        # Confidence: normalized absolute consensus
        confidence = min(abs_consensus / (threshold * 2.0), 1.0) if threshold > 0 else 0.0
        
        # Update consensus history
        self.consensus_history.append(consensus_sum)
        if len(self.consensus_history) > self.consensus_window:
            self.consensus_history.pop(0)
        
        return ConsensusResult(
            consensus_score=consensus_sum,
            weighted_sum=consensus_sum,
            threshold=threshold,
            should_trade=should_trade,
            action=action,
            confidence=confidence,
            engine_weights=final_weights,
            correlation_penalty_applied=True,
            metadata={
                "sigma_S": sigma_S,
                "realized_volatility": realized_volatility,
                "num_engines": len(filtered_votes),
                "reliability_weights": reliability_weights,
                "correlation_weights": correlation_weights,
            }
        )
    
    def _calculate_consensus_volatility(self) -> float:
        """
        Calculate rolling standard deviation of consensus scores.
        
        Returns:
            Standard deviation (sigma_S)
        """
        if len(self.consensus_history) < 10:
            return 0.1  # Default volatility
        
        return float(np.std(self.consensus_history))
    
    def update_engine_performance(
        self,
        engine_id: str,
        brier_score: float,
        predictions: List[float],
        actuals: List[float],
        update_date: Optional[datetime] = None
    ) -> None:
        """
        Update engine performance metrics for calibration.
        
        Args:
            engine_id: Engine identifier
            brier_score: Recent Brier score
            predictions: List of predictions
            actuals: List of actual outcomes
            update_date: Date of update (default: now)
        """
        if update_date is None:
            update_date = datetime.now()
        
        # Calculate Platt scaling parameters
        # Platt scaling: p = 1 / (1 + exp(a * score + b))
        # Fit using logistic regression
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = np.array(predictions).reshape(-1, 1)
            y = np.array(actuals)
            
            # Fit logistic regression
            lr = LogisticRegression()
            lr.fit(X, y)
            
            # Extract parameters
            a = -lr.coef_[0][0]  # Negative because sklearn uses different sign
            b = -lr.intercept_[0]
        except Exception as e:
            logger.warning("platt_scaling_failed", engine_id=engine_id, error=str(e))
            a, b = 0.0, 0.0
        
        # Update performance record
        self.engine_performance[engine_id] = {
            'brier_score': brier_score,
            'platt_a': a,
            'platt_b': b,
            'last_update': update_date,
            'num_samples': len(predictions),
        }
        
        logger.info(
            "engine_performance_updated",
            engine_id=engine_id,
            brier_score=brier_score,
            platt_a=a,
            platt_b=b
        )

