"""
Example: Multi-Model Training Integration

Shows how to integrate multi-model training into the daily retrain pipeline.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.cloud.training.models.multi_model_trainer import MultiModelTrainer, ModelResult, EnsembleResult
from src.cloud.training.models.dynamic_ensemble_combiner import DynamicEnsembleCombiner
from src.cloud.training.models.multi_model_integration import train_multi_model_ensemble, predict_with_ensemble


def example_basic_multi_model_training():
    """Example: Basic multi-model training"""
    # Simulated data
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = pd.Series(np.random.randn(1000))
    X_val = pd.DataFrame(np.random.randn(200, 20))
    y_val = pd.Series(np.random.randn(200))
    
    # Initialize trainer
    trainer = MultiModelTrainer(
        techniques=['xgboost', 'random_forest', 'lightgbm'],
        ensemble_method='weighted_voting',
        is_classification=False,
    )
    
    # Train all models in parallel
    results = trainer.train_parallel(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    
    # Get ensemble prediction
    X_test = pd.DataFrame(np.random.randn(100, 20))
    ensemble_result = trainer.predict_ensemble(X_test)
    
    print(f"Ensemble prediction shape: {ensemble_result.prediction.shape}")
    print(f"Confidence: {ensemble_result.confidence:.3f}")
    print(f"Model contributions: {ensemble_result.model_contributions}")
    
    # Get best model
    best_model = trainer.get_best_model()
    print(f"Best model: {best_model}")


def example_dynamic_ensemble_weighting():
    """Example: Dynamic ensemble weighting with regime awareness"""
    # Initialize combiner
    combiner = DynamicEnsembleCombiner(
        lookback_trades=50,
        min_trades_for_weighting=10,
    )
    
    # Simulate some trades
    models = ['xgboost', 'random_forest', 'lightgbm']
    
    # Update performance for each model
    for i in range(30):
        for model in models:
            won = np.random.rand() > 0.3  # 70% win rate
            profit_bps = np.random.randn() * 50 + (100 if won else -50)
            regime = np.random.choice(['TREND', 'RANGE', 'PANIC'])
            
            combiner.update_performance(
                model_name=model,
                won=won,
                profit_bps=profit_bps,
                regime=regime,
            )
    
    # Get dynamic weights
    predictions = {
        'xgboost': np.random.randn(100),
        'random_forest': np.random.randn(100),
        'lightgbm': np.random.randn(100),
    }
    
    weights = combiner.get_weights(
        current_regime='TREND',
        model_names=models,
        predictions=predictions,
    )
    
    print(f"Dynamic weights: {weights}")
    
    # Combine predictions
    ensemble_pred = combiner.combine(predictions, weights)
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")
    
    # Get performance summary
    summary = combiner.get_performance_summary()
    print(f"Performance summary: {summary}")


def example_integration_with_daily_retrain():
    """Example: How to integrate into daily retrain pipeline"""
    # This is what you would do in orchestration.py
    
    # Simulated data from walk-forward splits
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = pd.Series(np.random.randn(1000))
    X_val = pd.DataFrame(np.random.randn(200, 20))
    y_val = pd.Series(np.random.randn(200))
    regimes = pd.Series(np.random.choice(['TREND', 'RANGE', 'PANIC'], size=200))
    
    # Train multi-model ensemble
    trainer, results = train_multi_model_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        regimes=regimes,
        techniques=['xgboost', 'random_forest', 'lightgbm'],
        ensemble_method='weighted_voting',
        is_classification=False,
    )
    
    # Use for predictions
    X_test = pd.DataFrame(np.random.randn(100, 20))
    ensemble_result = predict_with_ensemble(
        trainer=trainer,
        X=X_test,
        regime='TREND',
    )
    
    # Use predictions for trading
    predictions = ensemble_result.prediction
    confidence = ensemble_result.confidence
    
    print(f"Predictions: {predictions[:5]}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Model contributions: {ensemble_result.model_contributions}")


def example_stacking_ensemble():
    """Example: Stacking ensemble with meta-model"""
    # Simulated data
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = pd.Series(np.random.randn(1000))
    X_val = pd.DataFrame(np.random.randn(200, 20))
    y_val = pd.Series(np.random.randn(200))
    
    # Initialize trainer with stacking
    trainer = MultiModelTrainer(
        techniques=['xgboost', 'random_forest', 'lightgbm'],
        ensemble_method='stacking',
        is_classification=False,
    )
    
    # Train base models
    results = trainer.train_parallel(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    
    # Train stacking meta-model
    trainer.train_stacking_meta_model(X_val, y_val)
    
    # Get ensemble prediction (uses meta-model)
    X_test = pd.DataFrame(np.random.randn(100, 20))
    ensemble_result = trainer.predict_ensemble(X_test)
    
    print(f"Stacking ensemble prediction shape: {ensemble_result.prediction.shape}")
    print(f"Confidence: {ensemble_result.confidence:.3f}")


def example_regime_specific_performance():
    """Example: Track performance by regime"""
    # Simulated data with regimes
    X_train = pd.DataFrame(np.random.randn(1000, 20))
    y_train = pd.Series(np.random.randn(1000))
    X_val = pd.DataFrame(np.random.randn(200, 20))
    y_val = pd.Series(np.random.randn(200))
    regimes = pd.Series(np.random.choice(['TREND', 'RANGE', 'PANIC'], size=200))
    
    # Train with regime tracking
    trainer = MultiModelTrainer(
        techniques=['xgboost', 'random_forest', 'lightgbm'],
        ensemble_method='weighted_voting',
    )
    
    results = trainer.train_parallel(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        regimes=regimes,
    )
    
    # Check performance by regime
    print("Performance by regime:")
    for regime, perf in trainer.performance_by_regime.items():
        print(f"\n{regime}:")
        for model, score in perf.items():
            print(f"  {model}: {score:.3f}")
    
    # Get prediction with regime-specific weighting
    X_test = pd.DataFrame(np.random.randn(100, 20))
    ensemble_result = trainer.predict_ensemble(X_test, regime='TREND')
    
    print(f"\nEnsemble prediction (TREND regime): {ensemble_result.prediction[:5]}")


if __name__ == '__main__':
    print("Multi-Model Training Examples")
    print("=" * 80)
    
    print("\n1. Basic Multi-Model Training")
    print("-" * 80)
    example_basic_multi_model_training()
    
    print("\n2. Dynamic Ensemble Weighting")
    print("-" * 80)
    example_dynamic_ensemble_weighting()
    
    print("\n3. Integration with Daily Retrain")
    print("-" * 80)
    example_integration_with_daily_retrain()
    
    print("\n4. Stacking Ensemble")
    print("-" * 80)
    example_stacking_ensemble()
    
    print("\n5. Regime-Specific Performance")
    print("-" * 80)
    example_regime_specific_performance()
    
    print("\n✓ All examples complete!")

