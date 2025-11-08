"""
Unified ML Pipeline - Complete Integration

Integrates all ML layers: preprocessing → baselines → core learners → meta-layer → feedback → MLOps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from ..automl.automl_engine import AutoMLEngine
from ..baselines.ab_testing import ABTestingFramework
from ..feedback import ModelFeedback
from ..meta import EnsembleBlender
from ..mlops.drift_detector import DriftDetector
from ..model_registry import ModelMetadata, ModelRegistry, get_registry
from ..preprocessing.enhanced_preprocessing import EnhancedPreprocessor
from ..orchestrator import MLEngineOrchestrator

logger = structlog.get_logger(__name__)


class UnifiedMLPipeline:
    """
    Unified ML pipeline that integrates all layers.
    
    Pipeline flow:
    1. Pre-processing (EDA, cleaning, normalization, feature engineering)
    2. Baselines (Linear/Logistic Regression, classifiers)
    3. Core learners (Random Forest, XGBoost, CNN, LSTM, GRU, Transformer, GAN, RL)
    4. Meta-layer (ensemble stacking, AutoML)
    5. Feedback loop (A/B testing, drift detection, automated retraining)
    6. MLOps (version control, monitoring, distributed training)
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize unified ML pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Initialize components
        self.orchestrator = MLEngineOrchestrator(config_path)
        self.preprocessor = EnhancedPreprocessor()
        self.registry = get_registry()
        self.ab_tester = ABTestingFramework()
        self.drift_detector = DriftDetector()
        self.feedback = self.orchestrator.feedback
        self.ensemble = self.orchestrator.ensemble
        
        logger.info("unified_ml_pipeline_initialized")
    
    def run_complete_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run complete ML pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("starting_complete_ml_pipeline")
        
        results = {}
        
        # 1. Pre-processing
        logger.info("step_1_preprocessing")
        X_train_processed = self.preprocessor.process(X_train, fit=True)
        if X_val is not None:
            X_val_processed = self.preprocessor.process(X_val, fit=False)
        if X_test is not None:
            X_test_processed = self.preprocessor.process(X_test, fit=False)
        
        results["preprocessing"] = {
            "train_shape": X_train_processed.shape,
            "feature_stats": self.preprocessor.feature_stats,
        }
        
        # 2. Train models
        logger.info("step_2_training_models")
        training_results = self.orchestrator.train_all_models(
            X_train_processed,
            y_train,
            X_val_processed if X_val is not None else None,
            y_val if y_val is not None else None,
        )
        results["training"] = training_results
        
        # 3. Register models with metadata
        logger.info("step_3_registering_models")
        for name, model in self.orchestrator.models.items():
            metadata = self.registry.create_metadata_from_model(model)
            self.registry.register_model(model, metadata)
        
        # 4. A/B testing (if validation data provided)
        if X_val is not None and y_val is not None:
            logger.info("step_4_ab_testing")
            ab_results = self._perform_ab_testing(X_val_processed, y_val)
            results["ab_testing"] = ab_results
        
        # 5. Ensemble predictions
        logger.info("step_5_ensemble_predictions")
        if X_test is not None:
            predictions = self.orchestrator.predict(X_test_processed, use_ensemble=True)
            results["predictions"] = predictions
        
        # 6. Drift detection
        logger.info("step_6_drift_detection")
        if X_test is not None:
            self.drift_detector.set_reference(X_train_processed)
            drift_results = self.drift_detector.detect_data_drift(X_test_processed)
            results["drift_detection"] = drift_results
            
            # Check if retraining is needed
            should_retrain = self.drift_detector.should_retrain(X_test_processed)
            results["should_retrain"] = should_retrain
        
        # 7. Feedback loop
        logger.info("step_7_feedback_loop")
        if X_val is not None and y_val is not None:
            evaluation_results = self.orchestrator.evaluate(X_val_processed, y_val)
            results["evaluation"] = evaluation_results
            
            # Update feedback
            for name, metrics in evaluation_results.items():
                self.feedback.record_performance(name, metrics)
            
            # Auto-tune
            self.orchestrator.auto_tune()
        
        # 8. Performance report
        logger.info("step_8_performance_report")
        performance_report = self.orchestrator.get_performance_report()
        results["performance_report"] = performance_report
        
        logger.info("complete_ml_pipeline_finished")
        return results
    
    def _perform_ab_testing(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Perform A/B testing on models."""
        # Get predictions from different models
        model_results = {}
        
        for name, model in self.orchestrator.models.items():
            try:
                predictions = model.predict(X_val)
                
                # Calculate metrics
                from ..base import ModelMetrics
                metrics = model.evaluate(X_val, y_val)
                
                model_results[name] = {
                    "predictions": predictions,
                    "metrics": metrics,
                }
            except Exception as e:
                logger.warning("model_prediction_failed_for_ab_test", model_name=name, error=str(e))
        
        # Compare models
        if len(model_results) >= 2:
            model_names = list(model_results.keys())
            model_a = model_names[0]
            model_b = model_names[1]
            
            # Compare Sharpe ratios
            sharpe_a = np.array([model_results[model_a]["metrics"].sharpe_ratio])
            sharpe_b = np.array([model_results[model_b]["metrics"].sharpe_ratio])
            
            ab_result = self.ab_tester.t_test(
                sharpe_a,
                sharpe_b,
                test_name=f"{model_a}_vs_{model_b}",
                metric_name="sharpe_ratio",
            )
            
            return {
                "comparison": {
                    "model_a": model_a,
                    "model_b": model_b,
                    "result": ab_result,
                }
            }
        
        return {}
    
    def get_models_for_regime(self, regime: str) -> List[str]:
        """
        Get models suitable for a market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            List of model names
        """
        return self.registry.get_models_by_regime(regime)
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models."""
        return self.registry.get_all_models_info()

