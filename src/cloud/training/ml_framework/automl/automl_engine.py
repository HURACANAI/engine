"""
AutoML Engine - Automated Model Selection and Hyperparameter Optimization

Automates model selection and hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..base import BaseModel, ModelConfig, ModelMetrics

logger = structlog.get_logger(__name__)

# Try to import Optuna (optional)
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna_not_available_automl_will_not_work")


class AutoMLEngine:
    """
    AutoML engine for automated model selection and hyperparameter optimization.
    
    Features:
    - Automated model selection
    - Hyperparameter optimization
    - Cross-validation
    - Best model selection
    """
    
    def __init__(
        self,
        models: List[BaseModel],
        objective_metric: str = "sharpe_ratio",
        n_trials: int = 100,
        cv_folds: int = 5,
    ):
        """
        Initialize AutoML engine.
        
        Args:
            models: List of models to optimize
            objective_metric: Metric to optimize
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.models = models
        self.objective_metric = objective_metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        self.study: Optional[Any] = None
        self.best_model: Optional[BaseModel] = None
        self.best_params: Optional[Dict[str, Any]] = None
        
        logger.info(
            "automl_engine_initialized",
            num_models=len(models),
            objective_metric=objective_metric,
            n_trials=n_trials,
        )
    
    def optimize(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Optimize models and hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Tuple of (best_model, best_params)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for AutoML. Install it with: pip install optuna")
        
        logger.info("starting_automl_optimization")
        
        # Create study
        self.study = optuna.create_study(
            direction="maximize",  # Maximize Sharpe ratio
            sampler=TPESampler(),
        )
        
        # Objective function
        def objective(trial: Any) -> float:
            # Select model
            model_idx = trial.suggest_categorical("model", list(range(len(self.models))))
            model = self.models[model_idx]
            
            # Suggest hyperparameters
            hyperparams = self._suggest_hyperparameters(trial, model)
            model.config.hyperparameters.update(hyperparams)
            
            # Train and evaluate
            metrics = model.fit(X_train, y_train, X_val, y_val)
            
            # Return objective metric
            metric_value = getattr(metrics, self.objective_metric, 0.0)
            return metric_value
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Get best model
        best_trial = self.study.best_trial
        best_model_idx = best_trial.params["model"]
        self.best_model = self.models[best_model_idx]
        self.best_params = best_trial.params
        
        # Update best model with best hyperparameters
        self.best_model.config.hyperparameters.update(self.best_params)
        
        logger.info(
            "automl_optimization_complete",
            best_model=self.best_model.config.name,
            best_score=best_trial.value,
        )
        
        return self.best_model, self.best_params
    
    def _suggest_hyperparameters(self, trial: Any, model: BaseModel) -> Dict[str, Any]:
        """Suggest hyperparameters for a model."""
        hyperparams = {}
        model_name = model.config.name
        
        if "random_forest" in model_name:
            hyperparams["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            hyperparams["max_depth"] = trial.suggest_int("max_depth", 5, 20)
            hyperparams["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        
        elif "xgboost" in model_name:
            hyperparams["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            hyperparams["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            hyperparams["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            hyperparams["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        
        elif "lstm" in model_name or "gru" in model_name:
            hyperparams["hidden_units"] = trial.suggest_int("hidden_units", 32, 128)
            hyperparams["num_layers"] = trial.suggest_int("num_layers", 1, 3)
            hyperparams["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
            hyperparams["learning_rate"] = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        
        return hyperparams
    
    def get_best_model(self) -> Optional[BaseModel]:
        """Get best model."""
        return self.best_model
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        if self.study is None:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                "trial": trial.number,
                "value": trial.value,
                "params": trial.params,
            })
        
        return history

