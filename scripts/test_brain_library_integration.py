#!/usr/bin/env python3
"""
Test script for Brain Library integration.

This script demonstrates how to use Brain Library components
with the Engine training pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Import Brain Library components
from src.cloud.training.brain.brain_library import BrainLibrary
from src.cloud.training.brain.feature_importance_analyzer import FeatureImportanceAnalyzer
from src.cloud.training.brain.model_comparison import ModelComparisonFramework
from src.cloud.training.brain.model_versioning import ModelVersioning
from src.cloud.training.services.brain_integrated_training import BrainIntegratedTraining
from src.cloud.training.services.model_selector import ModelSelector
from src.cloud.training.services.nightly_feature_analysis import NightlyFeatureAnalysis
from src.cloud.training.services.data_collector import DataCollector
from src.cloud.training.config.settings import EngineSettings


def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> Dict[str, Any]:
    """Create sample training data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target (with some relationship to features)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # Feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Split into train/test
    split_idx = int(n_samples * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "X_test": pd.DataFrame(X_test, columns=feature_names),
        "y_train": pd.Series(y_train),
        "y_test": pd.Series(y_test),
        "feature_names": feature_names,
    }


def test_brain_library():
    """Test Brain Library basic functionality."""
    print("=" * 60)
    print("Testing Brain Library")
    print("=" * 60)
    
    # Get database DSN from environment or settings
    dsn = os.getenv("DATABASE_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        try:
            settings = EngineSettings.load()
            dsn = settings.postgres.dsn if settings.postgres else None
        except Exception:
            dsn = None
    
    if not dsn:
        print("⚠️  No database DSN found. Skipping Brain Library tests.")
        print("   Set DATABASE_DSN environment variable or configure in settings.")
        return False
    
    try:
        # Initialize Brain Library
        print("\n1. Initializing Brain Library...")
        brain = BrainLibrary(dsn=dsn, use_pool=True)
        print("   ✅ Brain Library initialized")
        
        # Test storing feature importance
        print("\n2. Testing feature importance storage...")
        feature_rankings = [
            {"feature_name": "feature_0", "importance_score": 0.95},
            {"feature_name": "feature_1", "importance_score": 0.85},
            {"feature_name": "feature_2", "importance_score": 0.75},
        ]
        brain.store_feature_importance(
            analysis_date=datetime.now(tz=timezone.utc),
            symbol="BTC/USDT",
            feature_rankings=feature_rankings,
            method="shap",
        )
        print("   ✅ Feature importance stored")
        
        # Test getting top features
        top_features = brain.get_top_features("BTC/USDT", top_n=5)
        print(f"   ✅ Retrieved {len(top_features)} top features")
        
        # Test model comparison
        print("\n3. Testing model comparison...")
        metrics = {
            "accuracy": 0.85,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 1.8,
            "max_drawdown": 0.1,
            "profit_factor": 1.5,
        }
        brain.store_model_comparison(
            comparison_date=datetime.now(tz=timezone.utc),
            symbol="BTC/USDT",
            model_type="lightgbm",
            metrics=metrics,
        )
        print("   ✅ Model comparison stored")
        
        # Test getting best model
        best_model = brain.get_best_model("BTC/USDT")
        if best_model:
            print(f"   ✅ Best model: {best_model['model_type']} (score: {best_model['composite_score']:.4f})")
        
        print("\n✅ Brain Library tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Brain Library test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_brain_integrated_training():
    """Test Brain-integrated training."""
    print("\n" + "=" * 60)
    print("Testing Brain-Integrated Training")
    print("=" * 60)
    
    dsn = os.getenv("DATABASE_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        try:
            settings = EngineSettings.load()
            dsn = settings.postgres.dsn if settings.postgres else None
        except Exception:
            dsn = None
    
    if not dsn:
        print("⚠️  No database DSN found. Skipping Brain-integrated training tests.")
        return False
    
    try:
        # Initialize Brain Library
        brain = BrainLibrary(dsn=dsn, use_pool=True)
        settings = EngineSettings.load() if dsn else None
        
        # Initialize training service
        print("\n1. Initializing Brain-Integrated Training...")
        brain_training = BrainIntegratedTraining(brain, settings)
        print("   ✅ Brain-Integrated Training initialized")
        
        # Create sample data
        print("\n2. Creating sample data...")
        data = create_sample_data(n_samples=1000, n_features=20)
        print(f"   ✅ Created data: {len(data['X_train'])} train, {len(data['X_test'])} test samples")
        
        # Train model
        print("\n3. Training model with Brain Library integration...")
        model = LGBMRegressor(n_estimators=10, random_state=42)
        
        result = brain_training.train_with_brain_integration(
            symbol="BTC/USDT",
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            feature_names=data["feature_names"],
            base_model=model,
            model_type="lightgbm",
        )
        
        print(f"   ✅ Training complete: {result['status']}")
        print(f"   ✅ Model ID: {result['model_id']}")
        print(f"   ✅ Metrics: Sharpe={result['metrics'].get('sharpe_ratio', 0):.4f}")
        
        # Get top features
        print("\n4. Getting top features...")
        top_features = brain_training.get_top_features("BTC/USDT", top_n=5)
        print(f"   ✅ Top features: {top_features[:3]}...")
        
        print("\n✅ Brain-Integrated Training tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Brain-Integrated Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selector():
    """Test model selector."""
    print("\n" + "=" * 60)
    print("Testing Model Selector")
    print("=" * 60)
    
    dsn = os.getenv("DATABASE_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        try:
            settings = EngineSettings.load()
            dsn = settings.postgres.dsn if settings.postgres else None
        except Exception:
            dsn = None
    
    if not dsn:
        print("⚠️  No database DSN found. Skipping Model Selector tests.")
        return False
    
    try:
        # Initialize Brain Library
        brain = BrainLibrary(dsn=dsn, use_pool=True)
        
        # Initialize model selector
        print("\n1. Initializing Model Selector...")
        model_selector = ModelSelector(brain)
        print("   ✅ Model Selector initialized")
        
        # Test model selection
        print("\n2. Testing model selection...")
        selected_model = model_selector.select_model_for_symbol(
            symbol="BTC/USDT",
            volatility_regime="high",
        )
        
        if selected_model:
            print(f"   ✅ Selected model: {selected_model['model_type']}")
        else:
            print("   ⚠️  No active model found (this is OK if no models have been trained)")
        
        # Test model confidence
        print("\n3. Testing model confidence...")
        confidence = model_selector.get_model_confidence("BTC/USDT")
        print(f"   ✅ Model confidence: {confidence:.4f}")
        
        print("\n✅ Model Selector tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model Selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collector():
    """Test data collector."""
    print("\n" + "=" * 60)
    print("Testing Data Collector")
    print("=" * 60)
    
    dsn = os.getenv("DATABASE_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        try:
            settings = EngineSettings.load()
            dsn = settings.postgres.dsn if settings.postgres else None
        except Exception:
            dsn = None
    
    if not dsn:
        print("⚠️  No database DSN found. Skipping Data Collector tests.")
        return False
    
    try:
        # Initialize Brain Library
        brain = BrainLibrary(dsn=dsn, use_pool=True)
        
        # Initialize data collector
        print("\n1. Initializing Data Collector...")
        data_collector = DataCollector(brain, exchanges=['binance'])
        print("   ✅ Data Collector initialized")
        
        # Test liquidation features
        print("\n2. Testing liquidation features...")
        features = data_collector.get_liquidation_features("BTC/USDT", hours=24)
        print(f"   ✅ Liquidation features: {features}")
        
        print("\n✅ Data Collector tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data Collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧠 Brain Library Integration Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test Brain Library
    results["brain_library"] = test_brain_library()
    
    # Test Brain-Integrated Training
    results["brain_integrated_training"] = test_brain_integrated_training()
    
    # Test Model Selector
    results["model_selector"] = test_model_selector()
    
    # Test Data Collector
    results["data_collector"] = test_data_collector()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

