"""
ML Engine Orchestrator

Main orchestration script that coordinates all ML framework components:
- Data ingestion → preprocessing → model training → prediction → feedback logging
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from .base import BaseModel, ModelConfig, ModelMetrics
from .baseline import (
    KNNModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    SVMModel,
)
from .clustering import KMeansClustering
from .core import DecisionTreeModel, RandomForestModel, XGBoostModel
from .feedback import FeedbackConfig, ModelFeedback
from .feature_selection import FeatureSelector
from .meta import EnsembleBlender, EnsembleConfig
from .neural import GRUModel, LSTMModel
from .preprocessing import PreprocessingConfig, PreprocessingPipeline
from .validation import BiasVarianceDiagnostics, CrossValidator, create_train_val_test_split
from .visualizer import ModelVisualizer

logger = structlog.get_logger(__name__)


@dataclass
class MLEngineConfig:
    """Configuration for ML Engine Orchestrator."""
    
    config_path: Path
    preprocessing: PreprocessingConfig
    ensemble: EnsembleConfig
    feedback: FeedbackConfig
    models: Dict[str, ModelConfig]
    storage_path: Path = Path("models/ml_framework")
    log_level: str = "INFO"


class MLEngineOrchestrator:
    """
    Main orchestrator for the ML framework.
    
    Coordinates:
    1. Data ingestion
    2. Preprocessing
    3. Model training (baseline, core, neural)
    4. Ensemble blending
    5. Prediction
    6. Feedback and auto-tuning
    """
    
    def __init__(self, config_path: Path | str):
        """
        Initialize ML Engine Orchestrator.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        self.config = self._load_config(config_dict, config_path)
        self.preprocessing = PreprocessingPipeline(self.config.preprocessing)
        self.ensemble = EnsembleBlender(self.config.ensemble)
        self.feedback = ModelFeedback(self.config.feedback)
        self.models: Dict[str, BaseModel] = {}
        self.is_preprocessing_fitted = False
        self.feature_selector: Optional[FeatureSelector] = None
        self.cross_validator: Optional[CrossValidator] = None
        self.visualizer: Optional[ModelVisualizer] = None
        
        logger.info(
            "ml_engine_orchestrator_initialized",
            config_path=str(config_path),
            num_models=len(self.config.models),
        )
    
    def _load_config(self, config_dict: Dict[str, Any], config_path: Path) -> MLEngineConfig:
        """Load configuration from dictionary."""
        # Preprocessing config
        prep_dict = config_dict.get("preprocessing", {})
        preprocessing = PreprocessingConfig(**prep_dict)
        
        # Ensemble config
        ensemble_dict = config_dict.get("ensemble", {})
        ensemble = EnsembleConfig(**ensemble_dict)
        
        # Feedback config
        feedback_dict = config_dict.get("feedback", {})
        feedback = FeedbackConfig(**feedback_dict)
        
        # Model configs
        models = {}
        
        # Baseline models
        baseline_dict = config_dict.get("baseline_models", {})
        for name, model_dict in baseline_dict.items():
            if model_dict.get("enabled", False):
                model_config = ModelConfig(
                    name=name,
                    enabled=True,
                    model_type=model_dict.get("model_type", "regression"),
                    hyperparameters=model_dict.get("hyperparameters", {}),
                )
                models[name] = model_config
        
        # Core models
        core_dict = config_dict.get("core_models", {})
        for name, model_dict in core_dict.items():
            if model_dict.get("enabled", False):
                model_config = ModelConfig(
                    name=name,
                    enabled=True,
                    model_type=model_dict.get("model_type", "regression"),
                    hyperparameters=model_dict.get("hyperparameters", {}),
                )
                models[name] = model_config
        
        # Neural models
        neural_dict = config_dict.get("neural_models", {})
        for name, model_dict in neural_dict.items():
            if model_dict.get("enabled", False):
                model_config = ModelConfig(
                    name=name,
                    enabled=True,
                    model_type=model_dict.get("model_type", "regression"),
                    hyperparameters=model_dict.get("hyperparameters", {}),
                )
                models[name] = model_config
        
        # Clustering models
        clustering_dict = config_dict.get("clustering_models", {})
        for name, model_dict in clustering_dict.items():
            if model_dict.get("enabled", False):
                model_config = ModelConfig(
                    name=name,
                    enabled=True,
                    model_type="clustering",  # Special type for clustering
                    hyperparameters=model_dict.get("hyperparameters", {}),
                )
                models[name] = model_config
        
        # Storage path
        storage_path = Path(config_dict.get("storage", {}).get("base_path", "models/ml_framework"))
        
        # Log level
        log_level = config_dict.get("logging", {}).get("level", "INFO")
        
        return MLEngineConfig(
            config_path=config_path,
            preprocessing=preprocessing,
            ensemble=ensemble,
            feedback=feedback,
            models=models,
            storage_path=storage_path,
            log_level=log_level,
        )
    
    def _create_model(self, name: str, config: ModelConfig) -> BaseModel:
        """Create model instance from config."""
        # Baseline models
        if name == "linear_regression":
            return LinearRegressionModel(config)
        elif name == "logistic_regression":
            return LogisticRegressionModel(config)
        elif name == "knn":
            return KNNModel(config)
        elif name == "svm":
            return SVMModel(config)
        
        # Core models
        elif name == "decision_tree":
            return DecisionTreeModel(config)
        elif name == "random_forest":
            return RandomForestModel(config)
        elif name == "xgboost":
            return XGBoostModel(config)
        
        # Neural models
        elif name == "lstm":
            return LSTMModel(config)
        elif name == "gru":
            return GRUModel(config)
        
        # Clustering models
        elif name == "kmeans":
            return KMeansClustering(config)
        
        else:
            raise ValueError(f"Unknown model type: {name}")
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, ModelMetrics]:
        """
        Train all enabled models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of model names to metrics
        """
        logger.info("training_all_models", num_models=len(self.config.models))
        
        # Fit preprocessing
        if not self.is_preprocessing_fitted:
            X_train_processed = self.preprocessing.fit_transform(X_train)
            self.is_preprocessing_fitted = True
        else:
            X_train_processed = self.preprocessing.transform(X_train)
        
        if X_val is not None:
            X_val_processed = self.preprocessing.transform(X_val)
        else:
            X_val_processed = None
        
        # Train each model
        results = {}
        
        for name, model_config in self.config.models.items():
            if not model_config.enabled:
                continue
            
            try:
                logger.info("training_model", model_name=name)
                
                # Create model
                model = self._create_model(name, model_config)
                
                # Train
                metrics = model.fit(
                    X_train_processed,
                    y_train,
                    X_val_processed,
                    y_val,
                )
                
                # Store model
                self.models[name] = model
                results[name] = metrics
                
                # Add to ensemble
                if self.config.ensemble.enabled:
                    self.ensemble.add_model(name, model)
                
                # Record performance
                if self.config.feedback.enabled:
                    self.feedback.record_performance(name, metrics, sample_count=len(X_train))
                
                # Save model
                model_path = self.config.storage_path / f"{name}_model.pkl"
                model.save(model_path)
                
                logger.info("model_trained", model_name=name, **metrics.to_dict())
                
            except Exception as e:
                logger.error("model_training_failed", model_name=name, error=str(e))
                continue
        
        logger.info("all_models_trained", num_trained=len(results))
        return results
    
    def predict(
        self,
        X: pd.DataFrame,
        use_ensemble: bool = True,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            use_ensemble: Whether to use ensemble (True) or best single model (False)
            
        Returns:
            Predictions array
        """
        if not self.models:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        # Preprocess
        X_processed = self.preprocessing.transform(X)
        
        # Predict
        if use_ensemble and self.config.ensemble.enabled:
            predictions = self.ensemble.predict(X_processed)
        else:
            # Use best single model (by Sharpe ratio)
            best_model_name = max(
                self.models.keys(),
                key=lambda name: self.feedback.get_performance_summary(name)["avg_sharpe"]
                if self.feedback.get_performance_summary(name)
                else 0.0,
            )
            predictions = self.models[best_model_name].predict(X_processed)
        
        return predictions
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, ModelMetrics]:
        """
        Evaluate all models.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Dictionary of model names to metrics
        """
        logger.info("evaluating_all_models")
        
        # Preprocess
        X_processed = self.preprocessing.transform(X)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                metrics = model.evaluate(X_processed, y)
                results[name] = metrics
                
                # Update feedback
                if self.config.feedback.enabled:
                    self.feedback.record_performance(name, metrics, sample_count=len(X))
                
                logger.info("model_evaluated", model_name=name, **metrics.to_dict())
                
            except Exception as e:
                logger.error("model_evaluation_failed", model_name=name, error=str(e))
                continue
        
        return results
    
    def auto_tune(self) -> None:
        """Automatically tune models based on feedback."""
        if not self.config.feedback.enabled:
            return
        
        logger.info("starting_auto_tune")
        
        # Get retrain queue
        retrain_queue = self.feedback.get_retrain_queue()
        if retrain_queue:
            logger.info("models_to_retrain", models=retrain_queue)
            # TODO: Implement retraining logic
        
        # Get prune candidates
        prune_candidates = self.feedback.get_prune_candidates()
        if prune_candidates:
            logger.warning("models_to_prune", models=prune_candidates)
            # TODO: Implement pruning logic
        
        # Recalculate ensemble weights
        if self.config.ensemble.enabled and self.config.feedback.auto_reweight_enabled:
            # Weights are automatically recalculated when performance is updated
            logger.info("ensemble_weights_updated", weights=self.ensemble.get_weights())
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "models": self.feedback.get_all_performance_summaries(),
            "ensemble_weights": self.ensemble.get_weights() if self.config.ensemble.enabled else {},
            "retrain_queue": self.feedback.get_retrain_queue(),
            "prune_candidates": self.feedback.get_prune_candidates(),
        }
        return report
    
    def save_state(self, path: Path) -> None:
        """Save orchestrator state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble
        if self.config.ensemble.enabled:
            ensemble_path = path / "ensemble.pkl"
            self.ensemble.save(ensemble_path)
        
        # Save feedback
        if self.config.feedback.enabled:
            self.feedback.save_to_database()
        
        logger.info("orchestrator_state_saved", path=str(path))
    
    def load_state(self, path: Path) -> None:
        """Load orchestrator state."""
        # Load ensemble
        if self.config.ensemble.enabled:
            ensemble_path = path / "ensemble.pkl"
            if ensemble_path.exists():
                self.ensemble.load(ensemble_path)
        
        # Load feedback
        if self.config.feedback.enabled:
            self.feedback.load_from_database()
        
        logger.info("orchestrator_state_loaded", path=str(path))

