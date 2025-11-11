"""
Multi-Model Trainer - Train Multiple Techniques in Parallel

Trains multiple models simultaneously with different techniques and merges them:
- XGBoost (gradient boosting)
- Random Forest (ensemble trees)
- LightGBM (gradient boosting)
- Logistic Regression (linear)
- Neural Network (deep learning, optional)

Uses Ray for parallel training and ensemble methods for combining predictions.

Usage:
    trainer = MultiModelTrainer(
        techniques=['xgboost', 'random_forest', 'lightgbm', 'logistic'],
        ensemble_method='weighted_voting',
    )
    
    # Train all models in parallel
    results = trainer.train_parallel(X_train, y_train, X_val, y_val)
    
    # Get ensemble prediction
    prediction = trainer.predict_ensemble(X_test)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import structlog  # type: ignore
import pickle
from datetime import datetime

# ML imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import ray  # type: ignore
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

logger = structlog.get_logger(__name__)


@dataclass
class ModelResult:
    """Result from training a single model."""
    model_name: str
    model: Any
    train_score: float
    val_score: float
    cv_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time_seconds: float = 0.0
    is_classification: bool = False


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    prediction: np.ndarray
    confidence: float
    model_contributions: Dict[str, float]
    ensemble_method: str


class MultiModelTrainer:
    """
    Train multiple models in parallel and combine them using ensemble methods.
    
    Supports:
    - Parallel training using Ray
    - Multiple ensemble methods (weighted voting, stacking, dynamic weighting)
    - Performance tracking by regime
    - Automatic model selection based on performance
    """

    def __init__(
        self,
        techniques: List[str] = None,
        ensemble_method: str = 'weighted_voting',  # 'weighted_voting', 'stacking', 'dynamic'
        use_ray: bool = True,
        is_classification: bool = False,
        n_jobs: int = -1,
    ):
        """
        Initialize multi-model trainer.
        
        Args:
            techniques: List of techniques to train ['xgboost', 'random_forest', 'lightgbm', 'logistic']
            ensemble_method: How to combine models ('weighted_voting', 'stacking', 'dynamic')
            use_ray: Whether to use Ray for parallel training
            is_classification: True for classification, False for regression
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        if techniques is None:
            techniques = ['xgboost', 'random_forest', 'lightgbm', 'logistic']
        
        self.techniques = techniques
        self.ensemble_method = ensemble_method
        self.use_ray = use_ray and HAS_RAY
        self.is_classification = is_classification
        self.n_jobs = n_jobs
        
        # Trained models
        self.models: Dict[str, Any] = {}
        self.model_results: Dict[str, ModelResult] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Ensemble weights (learned from performance)
        self.model_weights: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_by_regime: Dict[str, Dict[str, float]] = {}
        
        # Meta-model for stacking
        self.meta_model: Optional[Any] = None
        
        logger.info(
            "multi_model_trainer_initialized",
            techniques=techniques,
            ensemble_method=ensemble_method,
            use_ray=self.use_ray,
            is_classification=is_classification,
        )

    def train_parallel(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> Dict[str, ModelResult]:
        """
        Train all models in parallel.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            regimes: Market regimes for each sample (optional)
            
        Returns:
            Dictionary of ModelResult for each technique
        """
        logger.info(
            "starting_parallel_training",
            num_techniques=len(self.techniques),
            train_samples=len(X_train),
            use_ray=self.use_ray,
        )
        
        if self.use_ray and HAS_RAY:
            results = self._train_parallel_ray(X_train, y_train, X_val, y_val)
        else:
            results = self._train_sequential(X_train, y_train, X_val, y_val)
        
        # Store results
        self.model_results = results
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(results)
        
        # Track performance by regime if available
        if regimes is not None and X_val is not None and y_val is not None:
            self._track_performance_by_regime(X_val, y_val, regimes)
        
        logger.info(
            "parallel_training_complete",
            num_models=len(results),
            best_model=max(results.items(), key=lambda x: x[1].val_score)[0] if results else None,
        )
        
        return results

    def _train_parallel_ray(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict[str, ModelResult]:
        """Train models in parallel using Ray."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create remote training tasks
        futures = []
        for technique in self.techniques:
            future = _train_model_remote.remote(
                technique,
                X_train,
                y_train,
                X_val,
                y_val,
                self.is_classification,
                self.n_jobs,
            )
            futures.append((technique, future))
        
        # Collect results
        results = {}
        for technique, future in futures:
            try:
                result = ray.get(future)
                results[technique] = result
                self.models[technique] = result.model
            except Exception as e:
                logger.error("model_training_failed", technique=technique, error=str(e))
        
        return results

    def _train_sequential(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> Dict[str, ModelResult]:
        """Train models sequentially."""
        results = {}
        
        for technique in self.techniques:
            try:
                result = self._train_single_model(
                    technique,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
                results[technique] = result
                self.models[technique] = result.model
            except Exception as e:
                logger.error("model_training_failed", technique=technique, error=str(e))
        
        return results

    def _train_single_model(
        self,
        technique: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> ModelResult:
        """Train a single model."""
        start_time = datetime.now()
        
        # Initialize model based on technique
        model = self._create_model(technique)
        
        # Scale features if needed
        if technique in ['logistic']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            self.scalers[technique] = scaler
            
            if X_val is not None:
                X_val_scaled = scaler.transform(X_val)
                X_val_final = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
            else:
                X_val_final = None
        else:
            X_train_final = X_train
            X_val_final = X_val
        
        # Train model with early stopping if supported
        if technique in ['xgboost', 'lightgbm']:
            # Early stopping for XGBoost/LightGBM
            if X_val_final is not None and y_val is not None:
                if technique == 'xgboost':
                    model.fit(
                        X_train_final, y_train,
                        eval_set=[(X_val_final, y_val)],
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                elif technique == 'lightgbm':
                    model.fit(
                        X_train_final, y_train,
                        eval_set=[(X_val_final, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                    )
            else:
                model.fit(X_train_final, y_train)
        else:
            model.fit(X_train_final, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_final)
        train_score = self._calculate_score(y_train, train_pred)
        
        val_score = None
        if X_val_final is not None and y_val is not None:
            val_pred = model.predict(X_val_final)
            val_score = self._calculate_score(y_val, val_pred)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) == 1:
                feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
            else:
                feature_importance = dict(zip(X_train.columns, np.abs(model.coef_[0])))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelResult(
            model_name=technique,
            model=model,
            train_score=train_score,
            val_score=val_score,
            feature_importance=feature_importance,
            training_time_seconds=training_time,
            is_classification=self.is_classification,
        )

    def _create_model(self, technique: str) -> Any:
        """Create a model instance for the given technique."""
        if technique == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not available")
            if self.is_classification:
                return xgb.XGBClassifier(
                    n_estimators=1000,  # Large number for early stopping
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    eval_metric='logloss',
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=1000,  # Large number for early stopping
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    eval_metric='rmse',
                )
        
        elif technique == 'random_forest':
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn not available")
            if self.is_classification:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=self.n_jobs,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=self.n_jobs,
                )
        
        elif technique == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM not available")
            if self.is_classification:
                return lgb.LGBMClassifier(
                    n_estimators=1000,  # Large number for early stopping
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbosity=-1,
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=1000,  # Large number for early stopping
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    verbosity=-1,
                )
        
        elif technique == 'logistic':
            if not HAS_SKLEARN:
                raise ImportError("scikit-learn not available")
            if not self.is_classification:
                raise ValueError("Logistic regression only for classification")
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=self.n_jobs,
            )
        
        else:
            raise ValueError(f"Unknown technique: {technique}")

    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate appropriate score for task."""
        if self.is_classification:
            # Use AUC for classification
            try:
                return roc_auc_score(y_true, y_pred)
            except Exception as e:
                # Fallback to accuracy if AUC calculation fails
                logger.debug("auc_calculation_failed", error=str(e))
                return (y_true == (y_pred > 0.5)).mean()
        else:
            # Use R² for regression
            return r2_score(y_true, y_pred)

    def _calculate_ensemble_weights(self, results: Dict[str, ModelResult]):
        """Calculate ensemble weights based on validation performance."""
        if not results:
            return
        
        # Weight by validation score (higher = better)
        total_score = 0.0
        for result in results.values():
            if result.val_score is not None:
                # Normalize to positive (add 1 for R², use as-is for AUC)
                score = max(0.0, result.val_score) if self.is_classification else max(0.0, result.val_score + 1.0)
                total_score += score
        
        if total_score > 0:
            for technique, result in results.items():
                if result.val_score is not None:
                    score = max(0.0, result.val_score) if self.is_classification else max(0.0, result.val_score + 1.0)
                    self.model_weights[technique] = score / total_score
                else:
                    self.model_weights[technique] = 0.0
        else:
            # Equal weights if no valid scores
            equal_weight = 1.0 / len(results)
            for technique in results.keys():
                self.model_weights[technique] = equal_weight
        
        logger.info("ensemble_weights_calculated", weights=self.model_weights)

    def _track_performance_by_regime(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        regimes: pd.Series,
    ):
        """Track model performance by market regime."""
        for regime in regimes.unique():
            regime_mask = regimes == regime
            X_regime = X_val[regime_mask]
            y_regime = y_val[regime_mask]
            
            if len(y_regime) == 0:
                continue
            
            regime_performance = {}
            for technique, model in self.models.items():
                # Scale if needed
                if technique in self.scalers:
                    X_regime_scaled = self.scalers[technique].transform(X_regime)
                    X_regime_final = pd.DataFrame(X_regime_scaled, columns=X_regime.columns, index=X_regime.index)
                else:
                    X_regime_final = X_regime
                
                pred = model.predict(X_regime_final)
                score = self._calculate_score(y_regime, pred)
                regime_performance[technique] = score
            
            self.performance_by_regime[regime] = regime_performance
        
        logger.info("performance_by_regime_tracked", regimes=list(self.performance_by_regime.keys()))

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        regime: Optional[str] = None,
    ) -> EnsembleResult:
        """
        Get ensemble prediction from all models.
        
        Args:
            X: Features to predict
            regime: Current market regime (for dynamic weighting)
            
        Returns:
            EnsembleResult with combined prediction
        """
        if not self.models:
            raise ValueError("No models trained. Call train_parallel() first.")
        
        # Get predictions from all models
        predictions = {}
        for technique, model in self.models.items():
            # Scale if needed
            if technique in self.scalers:
                X_scaled = self.scalers[technique].transform(X)
                X_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_final = X
            
            pred = model.predict(X_final)
            predictions[technique] = pred
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_voting':
            ensemble_pred, contributions = self._weighted_voting(predictions, regime)
        elif self.ensemble_method == 'stacking':
            ensemble_pred, contributions = self._stacking_predict(predictions, X)
        elif self.ensemble_method == 'dynamic':
            ensemble_pred, contributions = self._dynamic_weighting(predictions, regime)
        else:
            # Simple average
            ensemble_pred, contributions = self._simple_average(predictions)
        
        # Calculate confidence (agreement between models)
        if len(predictions) > 1:
            pred_array = np.array(list(predictions.values()))
            confidence = 1.0 - np.std(pred_array, axis=0).mean()
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.5
        
        return EnsembleResult(
            prediction=ensemble_pred,
            confidence=confidence,
            model_contributions=contributions,
            ensemble_method=self.ensemble_method,
        )

    def _weighted_voting(
        self,
        predictions: Dict[str, np.ndarray],
        regime: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Weighted voting ensemble."""
        # Get weights (regime-specific if available)
        if regime and regime in self.performance_by_regime:
            # Use regime-specific weights
            regime_perf = self.performance_by_regime[regime]
            total_perf = sum(regime_perf.values())
            if total_perf > 0:
                weights = {k: v / total_perf for k, v in regime_perf.items()}
            else:
                weights = self.model_weights
        else:
            weights = self.model_weights
        
        # Weighted average
        weighted_sum = None
        for technique, pred in predictions.items():
            weight = weights.get(technique, 0.0)
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
        
        return weighted_sum, weights

    def _stacking_predict(
        self,
        predictions: Dict[str, np.ndarray],
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Stacking ensemble (requires meta-model)."""
        if self.meta_model is None:
            # Fallback to weighted voting if meta-model not trained
            logger.warning("meta_model_not_trained_falling_back_to_weighted_voting")
            return self._weighted_voting(predictions, None)
        
        # Stack predictions as features
        stacked_features = np.column_stack(list(predictions.values()))
        
        # Meta-model prediction
        ensemble_pred = self.meta_model.predict(stacked_features)
        
        # Contributions based on meta-model coefficients
        contributions = {}
        if hasattr(self.meta_model, 'coef_'):
            coef = self.meta_model.coef_
            if len(coef.shape) == 1:
                for i, technique in enumerate(predictions.keys()):
                    contributions[technique] = abs(coef[i]) if i < len(coef) else 0.0
            else:
                for i, technique in enumerate(predictions.keys()):
                    contributions[technique] = abs(coef[0, i]) if i < len(coef[0]) else 0.0
        else:
            # Equal contributions if no coefficients
            equal = 1.0 / len(predictions)
            contributions = {k: equal for k in predictions.keys()}
        
        return ensemble_pred, contributions

    def _dynamic_weighting(
        self,
        predictions: Dict[str, np.ndarray],
        regime: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Dynamic weighting based on recent performance."""
        # Use regime-specific weights if available
        return self._weighted_voting(predictions, regime)

    def _simple_average(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Simple average of all predictions."""
        pred_array = np.array(list(predictions.values()))
        ensemble_pred = pred_array.mean(axis=0)
        
        equal_weight = 1.0 / len(predictions)
        contributions = {k: equal_weight for k in predictions.keys()}
        
        return ensemble_pred, contributions

    def train_stacking_meta_model(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Train meta-model for stacking ensemble."""
        if not self.models:
            raise ValueError("Train base models first")
        
        # Get base model predictions on validation set
        base_predictions = {}
        for technique, model in self.models.items():
            if technique in self.scalers:
                X_val_scaled = self.scalers[technique].transform(X_val)
                X_val_final = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
            else:
                X_val_final = X_val
            
            pred = model.predict(X_val_final)
            base_predictions[technique] = pred
        
        # Stack predictions as features
        stacked_X = np.column_stack(list(base_predictions.values()))
        
        # Train meta-model
        if self.is_classification:
            self.meta_model = LogisticRegression(random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
        
        self.meta_model.fit(stacked_X, y_val)
        
        logger.info("stacking_meta_model_trained")

    def get_best_model(self) -> Optional[str]:
        """Get the best performing model."""
        if not self.model_results:
            return None
        
        best = max(self.model_results.items(), key=lambda x: x[1].val_score if x[1].val_score is not None else -1)
        return best[0] if best[1].val_score is not None else None

    def save_models(self, path: str):
        """Save all trained models."""
        import os
        os.makedirs(path, exist_ok=True)
        
        for technique, model in self.models.items():
            model_path = os.path.join(path, f"{technique}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for technique, scaler in self.scalers.items():
            scaler_path = os.path.join(path, f"{technique}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save meta-model
        if self.meta_model:
            meta_path = os.path.join(path, "meta_model.pkl")
            with open(meta_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
        
        # Save weights and metadata
        metadata = {
            'model_weights': self.model_weights,
            'performance_by_regime': self.performance_by_regime,
            'ensemble_method': self.ensemble_method,
            'is_classification': self.is_classification,
        }
        metadata_path = os.path.join(path, "ensemble_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("models_saved", path=path)
    
    def load_models(self, path: str) -> bool:
        """
        Load saved models from disk.
        
        Args:
            path: Directory containing saved models
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        import os
        from pathlib import Path
        
        model_path = Path(path)
        if not model_path.exists():
            logger.warning("model_path_not_found", path=path)
            return False
        
        try:
            # Load models
            for model_file in model_path.glob("*_model.pkl"):
                technique = model_file.stem.replace("_model", "")
                with open(model_file, 'rb') as f:
                    self.models[technique] = pickle.load(f)
            
            # Load scalers
            for scaler_file in model_path.glob("*_scaler.pkl"):
                technique = scaler_file.stem.replace("_scaler", "")
                with open(scaler_file, 'rb') as f:
                    self.scalers[technique] = pickle.load(f)
            
            # Load meta-model
            meta_path = model_path / "meta_model.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
            
            # Load ensemble metadata
            metadata_path = model_path / "ensemble_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.model_weights = metadata.get('model_weights', {})
                    self.performance_by_regime = metadata.get('performance_by_regime', {})
                    self.ensemble_method = metadata.get('ensemble_method', 'weighted_voting')
                    self.is_classification = metadata.get('is_classification', False)
            
            logger.info("models_loaded", path=path, n_models=len(self.models))
            return True
        except Exception as e:
            logger.error("failed_to_load_models", path=path, error=str(e))
            return False


# Ray remote function for parallel training
if HAS_RAY:
    @ray.remote
    def _train_model_remote(
        technique: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        is_classification: bool,
        n_jobs: int,
    ) -> ModelResult:
        """Ray remote function to train a single model."""
        trainer = MultiModelTrainer(
            techniques=[technique],
            is_classification=is_classification,
            n_jobs=n_jobs,
            use_ray=False,  # Don't use Ray inside Ray
        )
        results = trainer._train_sequential(X_train, y_train, X_val, y_val)
        return results[technique]

