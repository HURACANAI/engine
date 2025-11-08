"""Hyperparameter optimization using Bayesian optimization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization:
    - Bayesian optimization (Optuna)
    - Grid search (for discrete parameters)
    - Logs all runs to Brain Library
    - Automatically re-runs monthly or after market shifts
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,  # seconds
        study_name: Optional[str] = None,
    ) -> None:
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum time for optimization (seconds)
            study_name: Name of the study
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        
        logger.info(
            "hyperparameter_optimizer_initialized",
            n_trials=n_trials,
            timeout=timeout,
            study_name=study_name,
        )

    def optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
    ) -> Dict[str, Any]:
        """
        Optimize LSTM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            input_dim: Input dimension
            
        Returns:
            Best hyperparameters and study results
        """
        try:
            import optuna  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("optuna_not_available", using_default_hyperparameters=True)
            return {
                "status": "failed",
                "error": "Optuna not installed",
                "default_params": {
                    "lstm_units": 128,
                    "num_layers": 2,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                },
            }
        
        from ..models.standardized_lstm import StandardizedLSTM
        from .comprehensive_evaluation import ComprehensiveEvaluation
        
        evaluator = ComprehensiveEvaluation()
        
        def objective(trial):
            # Suggest hyperparameters
            lstm_units = trial.suggest_int("lstm_units", 64, 256)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
            
            # Create model
            model = StandardizedLSTM(
                input_dim=input_dim,
                lstm_units=lstm_units,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
            )
            
            # Train model
            try:
                training_result = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=50,
                    batch_size=32,
                )
                
                if training_result.get("status") == "failed":
                    return float('inf')
                
                # Evaluate on validation set
                predictions = model.predict(X_val)
                returns = predictions.flatten() - y_val
                metrics = evaluator.evaluate_model(
                    predictions=predictions.flatten(),
                    actuals=y_val,
                    returns=returns,
                )
                
                # Return negative Sharpe ratio (Optuna minimizes)
                return -metrics.get("sharpe_ratio", 0.0)
                
            except Exception as e:
                logger.warning("trial_failed", trial=trial.number, error=str(e))
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=self.study_name or "lstm_optimization",
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(
            "hyperparameter_optimization_complete",
            best_params=best_params,
            best_value=best_value,
        )
        
        return {
            "status": "success",
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
        }

    def optimize_hybrid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
    ) -> Dict[str, Any]:
        """
        Optimize Hybrid CNN-LSTM hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            input_dim: Input dimension
            
        Returns:
            Best hyperparameters and study results
        """
        try:
            import optuna  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("optuna_not_available", using_default_hyperparameters=True)
            return {
                "status": "failed",
                "error": "Optuna not installed",
            }
        
        from ..models.hybrid_cnn_lstm import HybridCNNLSTM
        from .comprehensive_evaluation import ComprehensiveEvaluation
        
        evaluator = ComprehensiveEvaluation()
        
        def objective(trial):
            # Suggest hyperparameters
            cnn_filters = trial.suggest_int("cnn_filters", 32, 128)
            lstm_units = trial.suggest_int("lstm_units", 64, 256)
            num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
            
            # Create model
            model = HybridCNNLSTM(
                input_dim=input_dim,
                cnn_filters=cnn_filters,
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dropout_rate=dropout_rate,
            )
            
            # Train model
            try:
                training_result = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=50,
                    batch_size=32,
                )
                
                if training_result.get("status") == "failed":
                    return float('inf')
                
                # Evaluate on validation set
                predictions = model.predict(X_val)
                returns = predictions.flatten() - y_val
                metrics = evaluator.evaluate_model(
                    predictions=predictions.flatten(),
                    actuals=y_val,
                    returns=returns,
                )
                
                # Return negative Sharpe ratio
                return -metrics.get("sharpe_ratio", 0.0)
                
            except Exception as e:
                logger.warning("trial_failed", trial=trial.number, error=str(e))
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=self.study_name or "hybrid_optimization",
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(
            "hyperparameter_optimization_complete",
            best_params=best_params,
            best_value=best_value,
        )
        
        return {
            "status": "success",
            "best_params": best_params,
            "best_value": best_value,
            "study": study,
        }

