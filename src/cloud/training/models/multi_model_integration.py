"""
Multi-Model Training Integration for Orchestration

This module provides a drop-in replacement for single model training
in the daily retrain pipeline.

Replace:
    model = LGBMRegressor(**hyperparams)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

With:
    trainer, results = train_multi_model_ensemble(...)
    ensemble_result = predict_with_ensemble(trainer, X_test, regime)
    predictions = ensemble_result.prediction
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import structlog

from ..models.multi_model_trainer import MultiModelTrainer, ModelResult, EnsembleResult
from ..models.dynamic_ensemble_combiner import DynamicEnsembleCombiner

logger = structlog.get_logger(__name__)


def train_multi_model_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    techniques: Optional[List[str]] = None,
    ensemble_method: str = 'weighted_voting',
    is_classification: bool = False,
    hyperparams: Optional[Dict[str, Any]] = None,  # Hyperparameters from config
    fixed_ensemble_weights: Optional[Dict[str, float]] = None,  # Fixed ensemble weights
    use_fixed_weights: bool = False,  # Use fixed weights instead of performance-based
) -> Tuple[MultiModelTrainer, Dict[str, ModelResult]]:
    """
    Train multi-model ensemble.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        regimes: Market regimes for each sample
        techniques: List of techniques to train
        ensemble_method: Ensemble method ('weighted_voting', 'stacking', 'dynamic')
        is_classification: True for classification, False for regression
        
    Returns:
        (trainer, results) tuple
    """
    if techniques is None:
        techniques = ['xgboost', 'random_forest', 'lightgbm']
    
    trainer = MultiModelTrainer(
        techniques=techniques,
        ensemble_method=ensemble_method,
        is_classification=is_classification,
        use_ray=True,
        hyperparams=hyperparams,
        fixed_ensemble_weights=fixed_ensemble_weights,
        use_fixed_weights=use_fixed_weights,
    )
    
    # Train all models in parallel
    results = trainer.train_parallel(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        regimes=regimes,
    )
    
    # Train stacking meta-model if using stacking
    if ensemble_method == 'stacking' and X_val is not None and y_val is not None:
        trainer.train_stacking_meta_model(X_val, y_val)
    
    logger.info(
        "multi_model_ensemble_trained",
        num_models=len(results),
        best_model=trainer.get_best_model(),
        ensemble_method=ensemble_method,
    )
    
    return trainer, results


def predict_with_ensemble(
    trainer: MultiModelTrainer,
    X: pd.DataFrame,
    regime: Optional[str] = None,
    combiner: Optional[DynamicEnsembleCombiner] = None,
) -> EnsembleResult:
    """
    Get ensemble prediction.
    
    Args:
        trainer: Trained MultiModelTrainer
        X: Features to predict
        regime: Current market regime
        combiner: Dynamic ensemble combiner (for dynamic weighting)
        
    Returns:
        EnsembleResult with combined prediction
    """
    if trainer.ensemble_method == 'dynamic' and combiner:
        # Get dynamic weights from combiner
        model_names = list(trainer.models.keys())
        predictions = {}
        
        for name, model in trainer.models.items():
            # Scale if needed
            if name in trainer.scalers:
                X_scaled = trainer.scalers[name].transform(X)
                X_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_final = X
            
            pred = model.predict(X_final)
            predictions[name] = pred
        
        # Get dynamic weights
        weights = combiner.get_weights(
            current_regime=regime or 'UNKNOWN',
            model_names=model_names,
            predictions=predictions,
        )
        
        # Combine
        ensemble_pred = combiner.combine(predictions, weights)
        
        # Calculate confidence
        pred_array = np.array(list(predictions.values()))
        confidence = 1.0 - np.std(pred_array, axis=0).mean()
        confidence = max(0.0, min(1.0, confidence))
        
        return EnsembleResult(
            prediction=ensemble_pred,
            confidence=confidence,
            model_contributions=weights,
            ensemble_method='dynamic',
        )
    else:
        # Use trainer's built-in ensemble method
        return trainer.predict_ensemble(X, regime=regime)


def replace_single_model_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    techniques: Optional[List[str]] = None,
    ensemble_method: str = 'weighted_voting',
) -> Tuple[np.ndarray, MultiModelTrainer, Dict[str, ModelResult]]:
    """
    Drop-in replacement for single model training.
    
    This function can replace the single model training in orchestration.py:
    
    OLD:
        model = LGBMRegressor(**hyperparams)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    NEW:
        predictions, trainer, results = replace_single_model_training(
            X_train, y_train, X_test, X_val, y_val, regimes
        )
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features to predict
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        regimes: Market regimes (optional)
        techniques: List of techniques (optional)
        ensemble_method: Ensemble method (optional)
        
    Returns:
        (predictions, trainer, results) tuple
    """
    # Train ensemble
    trainer, results = train_multi_model_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        regimes=regimes,
        techniques=techniques,
        ensemble_method=ensemble_method,
        is_classification=False,
    )
    
    # Get ensemble prediction
    # Determine regime from test data if not provided
    current_regime = None
    if regimes is not None and len(regimes) > 0:
        # Use most common regime in validation set
        current_regime = regimes.mode()[0] if len(regimes.mode()) > 0 else None
    
    ensemble_result = trainer.predict_ensemble(X_test, regime=current_regime)
    predictions = ensemble_result.prediction
    
    logger.info(
        "single_model_replaced_with_ensemble",
        num_models=len(results),
        ensemble_method=ensemble_method,
        confidence=ensemble_result.confidence,
    )
    
    return predictions, trainer, results
