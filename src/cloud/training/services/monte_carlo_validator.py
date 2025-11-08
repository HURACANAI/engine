"""Monte Carlo simulation for model validation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class MonteCarloValidator:
    """
    Monte Carlo simulation for model validation:
    - Runs 1,000 simulated futures
    - Tests model stability under randomness
    - Selects models with stable Sharpe ratios
    - Rejects models that collapse under randomness
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        noise_level: float = 0.01,  # 1% noise
        stability_threshold: float = 0.1,  # 10% coefficient of variation
    ) -> None:
        """
        Initialize Monte Carlo validator.
        
        Args:
            n_simulations: Number of simulations to run
            noise_level: Level of noise to add (as fraction of std)
            stability_threshold: Threshold for stability (coefficient of variation)
        """
        self.n_simulations = n_simulations
        self.noise_level = noise_level
        self.stability_threshold = stability_threshold
        
        logger.info(
            "monte_carlo_validator_initialized",
            n_simulations=n_simulations,
            noise_level=noise_level,
            stability_threshold=stability_threshold,
        )

    def add_noise(
        self,
        data: np.ndarray,
        noise_level: Optional[float] = None,
    ) -> np.ndarray:
        """
        Add random noise to data.
        
        Args:
            data: Input data
            noise_level: Noise level (default: self.noise_level)
            
        Returns:
            Noisy data
        """
        if noise_level is None:
            noise_level = self.noise_level
        
        # Calculate data std
        data_std = np.std(data)
        
        # Generate noise
        noise = np.random.normal(0, noise_level * data_std, data.shape)
        
        # Add noise
        noisy_data = data + noise
        
        return noisy_data

    def validate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_simulations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate model using Monte Carlo simulation.
        
        Args:
            model: Model to validate
            X_test: Test features
            y_test: Test targets
            n_simulations: Number of simulations (default: self.n_simulations)
            
        Returns:
            Validation results dictionary
        """
        if n_simulations is None:
            n_simulations = self.n_simulations
        
        logger.info("monte_carlo_validation_started", n_simulations=n_simulations)
        
        sharpe_ratios = []
        returns_list = []
        
        for i in range(n_simulations):
            try:
                # Add random noise to test data
                X_noisy = self.add_noise(X_test)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    predictions = model.predict(X_noisy)
                else:
                    logger.warning("model_does_not_have_predict_method")
                    break
                
                # Calculate returns (prediction error as proxy)
                if len(predictions.shape) > 1:
                    predictions = predictions.flatten()
                
                returns = predictions - y_test
                returns_list.append(returns)
                
                # Calculate Sharpe ratio
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                    sharpe_ratios.append(sharpe)
                else:
                    sharpe_ratios.append(0.0)
                
            except Exception as e:
                logger.warning("simulation_failed", simulation=i, error=str(e))
                continue
        
        if len(sharpe_ratios) == 0:
            return {
                "status": "failed",
                "error": "All simulations failed",
            }
        
        # Calculate statistics
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        min_sharpe = np.min(sharpe_ratios)
        max_sharpe = np.max(sharpe_ratios)
        
        # Calculate coefficient of variation (stability metric)
        if mean_sharpe != 0:
            coefficient_of_variation = abs(std_sharpe / mean_sharpe)
        else:
            coefficient_of_variation = float('inf')
        
        # Determine stability
        is_stable = coefficient_of_variation < self.stability_threshold
        
        # Calculate percentiles
        p25 = np.percentile(sharpe_ratios, 25)
        p50 = np.percentile(sharpe_ratios, 50)
        p75 = np.percentile(sharpe_ratios, 75)
        
        result = {
            "status": "stable" if is_stable else "unstable",
            "mean_sharpe": float(mean_sharpe),
            "std_sharpe": float(std_sharpe),
            "min_sharpe": float(min_sharpe),
            "max_sharpe": float(max_sharpe),
            "coefficient_of_variation": float(coefficient_of_variation),
            "stability_threshold": self.stability_threshold,
            "is_stable": is_stable,
            "percentiles": {
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
            },
            "n_simulations": len(sharpe_ratios),
        }
        
        if is_stable:
            logger.info(
                "monte_carlo_validation_passed",
                mean_sharpe=mean_sharpe,
                coefficient_of_variation=coefficient_of_variation,
            )
        else:
            logger.warning(
                "monte_carlo_validation_failed",
                mean_sharpe=mean_sharpe,
                coefficient_of_variation=coefficient_of_variation,
                threshold=self.stability_threshold,
            )
        
        return result

    def stress_test(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_levels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Stress test model with different noise levels.
        
        Args:
            model: Model to test
            X_test: Test features
            y_test: Test targets
            noise_levels: List of noise levels to test
            
        Returns:
            Stress test results
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        logger.info("stress_test_started", noise_levels=noise_levels)
        
        results = {}
        
        for noise_level in noise_levels:
            original_noise_level = self.noise_level
            self.noise_level = noise_level
            
            validation_result = self.validate_model(model, X_test, y_test, n_simulations=100)
            results[f"noise_{noise_level}"] = validation_result
            
            self.noise_level = original_noise_level
        
        return {
            "status": "success",
            "results": results,
        }

