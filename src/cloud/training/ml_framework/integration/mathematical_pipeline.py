"""
Mathematical Pipeline - Huracan Engine Integration

Integrates mathematical reasoning into the complete ML pipeline.
Every prediction is traceable to mathematical principles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from ..mathematics.huracan_core import HuracanCore
from ..model_registry import get_registry
from .unified_pipeline import UnifiedMLPipeline

logger = structlog.get_logger(__name__)


class MathematicalPipeline(UnifiedMLPipeline):
    """
    Mathematical pipeline that extends UnifiedMLPipeline with mathematical reasoning.
    
    Every prediction, decision, and output is traceable to mathematical principles.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize mathematical pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        
        # Initialize Huracan Core
        self.huracan_core = HuracanCore()
        
        logger.info("mathematical_pipeline_initialized")
    
    def run_mathematical_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run complete pipeline with mathematical reasoning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with pipeline results and mathematical reasoning
        """
        logger.info("starting_mathematical_pipeline")
        
        results = {}
        
        # 1. Understand data mathematically
        logger.info("step_1_mathematical_data_understanding")
        data_understanding = self.huracan_core.understand_data(X_train, y_train)
        results["data_understanding"] = data_understanding
        
        # 2. Pre-processing
        logger.info("step_2_preprocessing")
        X_train_processed = self.preprocessor.process(X_train, fit=True)
        if X_val is not None:
            X_val_processed = self.preprocessor.process(X_val, fit=False)
        if X_test is not None:
            X_test_processed = self.preprocessor.process(X_test, fit=False)
        
        # 3. Train models with mathematical reasoning
        logger.info("step_3_training_with_mathematical_reasoning")
        training_results = {}
        
        for name, model_config in self.orchestrator.config.models.items():
            if not model_config.enabled:
                continue
            
            try:
                model = self.orchestrator._create_model(name, model_config)
                
                # Train with mathematical reasoning
                train_result = self.huracan_core.train_with_reasoning(
                    model,
                    X_train_processed.values if isinstance(X_train_processed, pd.DataFrame) else X_train_processed,
                    y_train.values if isinstance(y_train, pd.Series) else y_train,
                    X_val_processed.values if X_val is not None and isinstance(X_val_processed, pd.DataFrame) else (X_val_processed if X_val is not None else X_train_processed.values),
                    y_val.values if y_val is not None and isinstance(y_val, pd.Series) else (y_val if y_val is not None else y_train.values),
                )
                
                training_results[name] = train_result
                self.orchestrator.models[name] = model
                
                # Register with metadata
                from ..model_registry import ModelMetadata
                metadata = self.registry.create_metadata_from_model(model)
                self.registry.register_model(model, metadata)
                
                logger.info(
                    "model_trained_with_reasoning",
                    model_name=name,
                    generalization_score=train_result["prediction_reasoning"]["generalization_score"],
                )
                
            except Exception as e:
                logger.error("model_training_failed", model_name=name, error=str(e))
                continue
        
        results["training"] = training_results
        
        # 4. Make predictions with mathematical reasoning
        logger.info("step_4_predictions_with_mathematical_reasoning")
        if X_test is not None:
            predictions_with_reasoning = {}
            
            for name, model in self.orchestrator.models.items():
                try:
                    prediction_reasoning = self.huracan_core.predict_with_reasoning(
                        model,
                        X_test_processed.values if isinstance(X_test_processed, pd.DataFrame) else X_test_processed,
                        y_test.values if y_test is not None and isinstance(y_test, pd.Series) else (y_test if y_test is not None else None),
                    )
                    
                    predictions_with_reasoning[name] = {
                        "prediction": prediction_reasoning.prediction,
                        "confidence_interval": prediction_reasoning.confidence_interval,
                        "confidence_score": prediction_reasoning.confidence_score,
                        "uncertainty_measure": prediction_reasoning.uncertainty_measure,
                        "bias_estimate": prediction_reasoning.bias_estimate,
                        "variance_estimate": prediction_reasoning.variance_estimate,
                        "generalization_score": prediction_reasoning.generalization_score,
                        "explanation": self.huracan_core.explain_decision(prediction_reasoning),
                        "mathematical_trace": self.huracan_core.get_mathematical_trace(prediction_reasoning),
                    }
                except Exception as e:
                    logger.error("prediction_with_reasoning_failed", model_name=name, error=str(e))
                    continue
            
            results["predictions"] = predictions_with_reasoning
        
        # 5. Continuous learning cycle
        logger.info("step_5_continuous_learning_cycle")
        if X_val is not None and y_val is not None:
            learning_results = {}
            
            for name, model in self.orchestrator.models.items():
                try:
                    learning_result = self.huracan_core.continuous_learning_iteration(
                        model,
                        X_train_processed.values if isinstance(X_train_processed, pd.DataFrame) else X_train_processed,
                        y_train.values if isinstance(y_train, pd.Series) else y_train,
                        X_val_processed.values if isinstance(X_val_processed, pd.DataFrame) else X_val_processed,
                        y_val.values if isinstance(y_val, pd.Series) else y_val,
                        iteration=0,
                    )
                    
                    learning_results[name] = learning_result
                except Exception as e:
                    logger.error("continuous_learning_failed", model_name=name, error=str(e))
                    continue
            
            results["continuous_learning"] = learning_results
            
            # Get learning summary
            learning_summary = self.huracan_core.learning_cycle.get_learning_summary()
            results["learning_summary"] = learning_summary
        
        logger.info("mathematical_pipeline_complete")
        
        return results
    
    def explain_prediction(
        self,
        model_name: str,
        X: pd.DataFrame | np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> str:
        """
        Explain a prediction in plain reasoning.
        
        Args:
            model_name: Name of the model
            X: Features to predict on
            y_true: True values (optional)
            
        Returns:
            Plain-language explanation
        """
        if model_name not in self.orchestrator.models:
            return f"Model {model_name} not found."
        
        model = self.orchestrator.models[model_name]
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Get prediction with reasoning
        prediction_reasoning = self.huracan_core.predict_with_reasoning(model, X_array, y_true)
        
        # Generate explanation
        explanation = self.huracan_core.explain_decision(prediction_reasoning)
        
        return explanation
    
    def get_mathematical_trace(
        self,
        model_name: str,
        X: pd.DataFrame | np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get full mathematical trace for a prediction.
        
        Args:
            model_name: Name of the model
            X: Features to predict on
            y_true: True values (optional)
            
        Returns:
            List of mathematical reasoning steps
        """
        if model_name not in self.orchestrator.models:
            return []
        
        model = self.orchestrator.models[model_name]
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Get prediction with reasoning
        prediction_reasoning = self.huracan_core.predict_with_reasoning(model, X_array, y_true)
        
        # Get mathematical trace
        trace = self.huracan_core.get_mathematical_trace(prediction_reasoning)
        
        return trace

