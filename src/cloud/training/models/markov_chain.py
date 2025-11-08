"""Markov Chain modeling for state transitions."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class MarkovChainModel:
    """
    Markov Chain for state transitions:
    - Models state probabilities (trend → consolidation → reversal → breakout)
    - Stores transition probabilities in Brain Library
    - Updates hourly with Mechanic retrains
    - Predicts short-term state shifts
    """

    def __init__(
        self,
        states: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize Markov Chain model.
        
        Args:
            states: List of possible states (default: ['trend', 'consolidation', 'reversal', 'breakout'])
        """
        if states is None:
            states = ['trend', 'consolidation', 'reversal', 'breakout']
        
        self.states = states
        self.num_states = len(states)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.idx_to_state = {i: state for i, state in enumerate(states)}
        
        # Initialize transition matrix
        self.transition_matrix = np.ones((self.num_states, self.num_states)) / self.num_states
        
        logger.info(
            "markov_chain_initialized",
            states=states,
            num_states=self.num_states,
        )

    def initialize_matrix(self) -> np.ndarray:
        """
        Initialize transition matrix with uniform probabilities.
        
        Returns:
            Transition matrix
        """
        return np.ones((self.num_states, self.num_states)) / self.num_states

    def update_transitions(
        self,
        state_sequence: List[str],
    ) -> None:
        """
        Update transition matrix based on observed transitions.
        
        Args:
            state_sequence: Sequence of observed states
        """
        logger.info("updating_transitions", sequence_length=len(state_sequence))
        
        # Count transitions
        transition_counts = np.zeros((self.num_states, self.num_states))
        
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]
            
            if current_state in self.state_to_idx and next_state in self.state_to_idx:
                current_idx = self.state_to_idx[current_state]
                next_idx = self.state_to_idx[next_state]
                transition_counts[current_idx, next_idx] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        self.transition_matrix = transition_counts / row_sums
        
        logger.info("transitions_updated", transitions_observed=np.sum(transition_counts))

    def normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize transition matrix.
        
        Args:
            matrix: Transition matrix to normalize
            
        Returns:
            Normalized transition matrix
        """
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return matrix / row_sums

    def predict_next_state(
        self,
        current_state: str,
        n_steps: int = 1,
    ) -> Dict[str, any]:
        """
        Predict next state based on transition probabilities.
        
        Args:
            current_state: Current state
            n_steps: Number of steps ahead to predict
            
        Returns:
            Dictionary with predicted state and probabilities
        """
        if current_state not in self.state_to_idx:
            logger.warning("unknown_state", state=current_state)
            return {
                "predicted_state": None,
                "probabilities": {},
                "confidence": 0.0,
            }
        
        current_idx = self.state_to_idx[current_state]
        
        # Get transition probabilities for current state
        probabilities = self.transition_matrix[current_idx, :]
        
        # For n_steps > 1, multiply transition matrix
        if n_steps > 1:
            probabilities = np.linalg.matrix_power(self.transition_matrix, n_steps)[current_idx, :]
        
        # Get most likely next state
        next_idx = np.argmax(probabilities)
        predicted_state = self.idx_to_state[next_idx]
        confidence = float(probabilities[next_idx])
        
        # Create probabilities dictionary
        prob_dict = {
            self.idx_to_state[i]: float(probabilities[i])
            for i in range(self.num_states)
        }
        
        result = {
            "predicted_state": predicted_state,
            "probabilities": prob_dict,
            "confidence": confidence,
        }
        
        logger.debug(
            "state_prediction",
            current_state=current_state,
            predicted_state=predicted_state,
            confidence=confidence,
        )
        
        return result

    def get_stationary_distribution(self) -> Dict[str, float]:
        """
        Calculate stationary distribution of Markov chain.
        
        Returns:
            Dictionary mapping states to stationary probabilities
        """
        # Find eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize
        stationary = stationary / stationary.sum()
        
        # Create dictionary
        stationary_dict = {
            self.idx_to_state[i]: float(stationary[i])
            for i in range(self.num_states)
        }
        
        return stationary_dict

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get transition matrix as DataFrame.
        
        Returns:
            Transition matrix DataFrame
        """
        return pd.DataFrame(
            self.transition_matrix,
            index=self.states,
            columns=self.states,
        )

    def fit(
        self,
        state_sequence: List[str],
    ) -> Dict[str, any]:
        """
        Fit Markov chain to state sequence.
        
        Args:
            state_sequence: Sequence of observed states
            
        Returns:
            Training results
        """
        logger.info("fitting_markov_chain", sequence_length=len(state_sequence))
        
        # Update transitions
        self.update_transitions(state_sequence)
        
        # Calculate stationary distribution
        stationary = self.get_stationary_distribution()
        
        return {
            "status": "success",
            "transition_matrix": self.transition_matrix.tolist(),
            "stationary_distribution": stationary,
            "num_transitions": len(state_sequence) - 1,
        }

