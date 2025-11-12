"""Enhanced training service with Brain Library integration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary
from ..brain.feature_importance_analyzer import FeatureImportanceAnalyzer
from ..brain.model_comparison import ModelComparisonFramework
from ..brain.model_versioning import ModelVersioning
from ..config.settings import EngineSettings
from .comprehensive_evaluation import ComprehensiveEvaluation

logger = structlog.get_logger(__name__)


class BrainIntegratedTraining:
    """
    Enhanced training service that integrates Brain Library components:
    - Model comparison
    - Feature importance analysis
    - Model versioning
    - Automatic rollback
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
        settings: Optional[EngineSettings] = None,
    ) -> None:
        """
        Initialize Brain-integrated training service.
        
        Args:
            brain_library: Brain Library instance
            settings: Engine settings
        """
        self.brain = brain_library
        self.settings = settings
        
        # Initialize Brain Library components
        self.feature_analyzer = FeatureImportanceAnalyzer(brain_library)
        self.model_comparison = ModelComparisonFramework(brain_library)
        self.model_versioning = ModelVersioning(brain_library)
        self.evaluator = ComprehensiveEvaluation(brain_library)
        
        logger.info("brain_integrated_training_initialized")

    def train_with_brain_integration(
        self,
        symbol: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        base_model: Any,
        model_type: str = "lightgbm",
    ) -> Dict[str, Any]:
        """
        Train model with Brain Library integration.
        
        Steps:
        1. Train base model
        2. Analyze feature importance
        3. Compare with other models (if available)
        4. Register model version
        5. Check for rollback
        
        Args:
            symbol: Trading symbol
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            base_model: Base model to train
            model_type: Model type identifier
            
        Returns:
            Dictionary with training results
        """
        logger.info("brain_integrated_training_started", symbol=symbol, model_type=model_type)
        
        # Step 1: Train base model
        try:
            base_model.fit(X_train.values, y_train.values)
            logger.info("base_model_trained", symbol=symbol)
        except Exception as e:
            logger.error("base_model_training_failed", symbol=symbol, error=str(e))
            return {"status": "failed", "error": str(e)}
        
        # Step 2: Get predictions for evaluation
        try:
            train_predictions = base_model.predict(X_train.values)
            test_predictions = base_model.predict(X_test.values)
        except Exception as e:
            logger.error("prediction_failed", symbol=symbol, error=str(e))
            return {"status": "failed", "error": f"Prediction failed: {str(e)}"}
        
        # Step 3: Calculate comprehensive metrics
        # Calculate returns from predictions (prediction error as proxy)
        test_returns = test_predictions - y_test.values
        
        # Use comprehensive evaluation
        metrics = self.evaluator.evaluate_model(
            predictions=test_predictions,
            actuals=y_test.values,
            returns=test_returns,
        )
        
        # Step 4: Analyze feature importance
        try:
            importance_results = self.feature_analyzer.analyze_feature_importance(
                symbol=symbol,
                model=base_model,
                X=X_train.values,
                y=y_train.values,
                feature_names=feature_names,
                methods=['shap', 'permutation'] if len(feature_names) <= 50 else ['correlation'],
            )
            logger.info("feature_importance_analyzed", symbol=symbol)
        except Exception as e:
            logger.warning("feature_importance_analysis_failed", symbol=symbol, error=str(e))
            importance_results = {}
        
        # Step 5: Compare with other models (if multiple models available)
        comparison_results = {}
        try:
            # For now, just store this model's metrics
            # In future, can compare with other model types
            self.brain.store_model_comparison(
                comparison_date=datetime.now(tz=timezone.utc),
                symbol=symbol,
                model_type=model_type,
                metrics=metrics,
            )
            comparison_results[model_type] = metrics
            logger.info("model_comparison_stored", symbol=symbol, model_type=model_type)
        except Exception as e:
            logger.warning("model_comparison_failed", symbol=symbol, error=str(e))
        
        # Step 6: Register model version
        model_id = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version = 1
        
        try:
            # Get previous version number
            previous_model = self.brain.get_active_model(symbol, model_type)
            if previous_model:
                # Get manifest to find version
                manifest = self.brain.get_model_manifest(previous_model["model_id"])
                if manifest:
                    version = manifest["version"] + 1
            
            # Store model manifest
            self.model_versioning.register_model_version(
                model_id=model_id,
                symbol=symbol,
                version=version,
                hyperparameters=self._extract_hyperparameters(base_model),
                dataset_id=f"dataset_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                feature_set=feature_names,
                training_metrics=metrics,
                validation_metrics=metrics,
            )
            
            # Store model metrics
            self.brain.store_model_metrics(
                model_id=model_id,
                evaluation_date=datetime.now(tz=timezone.utc),
                symbol=symbol,
                metrics=metrics,
            )
            
            logger.info("model_version_registered", symbol=symbol, model_id=model_id, version=version)
        except Exception as e:
            logger.warning("model_versioning_failed", symbol=symbol, error=str(e))
        
        # Step 7: Check for rollback
        rollback_occurred = False
        try:
            rollback_occurred = self.model_versioning.check_and_rollback(
                model_id=model_id,
                symbol=symbol,
                new_metrics=metrics,
            )
            if rollback_occurred:
                logger.warning("model_rollback_occurred", symbol=symbol, model_id=model_id)
        except Exception as e:
            logger.warning("rollback_check_failed", symbol=symbol, error=str(e))
        
        # Step 8: Register in model registry (if not rolled back)
        if not rollback_occurred:
            try:
                # Get best model info
                best_model_info = self.brain.get_best_model(symbol)
                if best_model_info:
                    composite_score = best_model_info.get("composite_score", 0.0)
                else:
                    # Calculate composite score
                    composite_score = (
                        0.4 * metrics.get("sharpe_ratio", 0.0) +
                        0.3 * metrics.get("profit_factor", 0.0) +
                        0.2 * (1.0 - abs(metrics.get("max_drawdown", 0.0))) +
                        0.1 * metrics.get("accuracy", 0.0)
                    )
                
                self.brain.register_model(
                    model_id=model_id,
                    symbol=symbol,
                    model_type=model_type,
                    version=version,
                    composite_score=composite_score,
                    hyperparameters=self._extract_hyperparameters(base_model),
                    dataset_id=f"dataset_{symbol}_{datetime.now().strftime('%Y%m%d')}",
                    feature_set=feature_names,
                )
                logger.info("model_registered", symbol=symbol, model_id=model_id)
            except Exception as e:
                logger.warning("model_registration_failed", symbol=symbol, error=str(e))
        
        return {
            "status": "rollback" if rollback_occurred else "success",
            "model_id": model_id,
            "version": version,
            "metrics": metrics,
            "importance_results": importance_results,
            "comparison_results": comparison_results,
            "rollback_occurred": rollback_occurred,
        }

    def _calculate_metrics(
        self,
        y_train: np.ndarray,
        train_predictions: np.ndarray,
        y_test: np.ndarray,
        test_predictions: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics (legacy method - use ComprehensiveEvaluation instead)."""
        # Calculate returns (prediction error as proxy)
        test_returns = test_predictions - y_test
        
        # Use comprehensive evaluation
        return self.evaluator.evaluate_model(
            predictions=test_predictions,
            actuals=y_test,
            returns=test_returns,
        )

    def _extract_hyperparameters(self, model: Any) -> Dict[str, Any]:
        """Extract hyperparameters from model, cleaning NaN values for JSON serialization."""
        import numpy as np
        
        def clean_for_json(obj: Any) -> Any:
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj if not (isinstance(item, float) and (np.isnan(item) or np.isinf(item)))]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                val = float(obj)
                # Replace NaN and Inf with None
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [clean_for_json(item) for item in obj.tolist()]
            elif obj is None:
                return None
            else:
                # Try to serialize, if it fails return string representation
                try:
                    import json
                    json.dumps(obj, allow_nan=False)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        try:
            if hasattr(model, 'get_params'):
                params = model.get_params()
                # Clean NaN values from hyperparameters
                return clean_for_json(params)
            elif hasattr(model, '__dict__'):
                # Extract relevant attributes
                params = {}
                for key, value in model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        try:
                            # Try to serialize
                            import json
                            json.dumps(value, allow_nan=False)
                            params[key] = clean_for_json(value)
                        except (TypeError, ValueError):
                            params[key] = str(value)
                return clean_for_json(params)
            else:
                return {}
        except Exception as e:
            logger.warning("hyperparameter_extraction_failed", error=str(e))
            return {}

    def get_top_features(
        self,
        symbol: str,
        top_n: int = 20,
    ) -> List[str]:
        """
        Get top features for a symbol from Brain Library.
        
        Args:
            symbol: Trading symbol
            top_n: Number of top features to return
            
        Returns:
            List of top feature names
        """
        return self.feature_analyzer.get_top_features_for_symbol(
            symbol=symbol,
            top_n=top_n,
            method='shap',
        )

    def get_active_model(
        self,
        symbol: str,
        model_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get active model for a symbol from Brain Library.
        
        Args:
            symbol: Trading symbol
            model_type: Optional model type filter
            
        Returns:
            Active model information or None
        """
        return self.brain.get_active_model(symbol, model_type)

