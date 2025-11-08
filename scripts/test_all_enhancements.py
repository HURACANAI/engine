#!/usr/bin/env python3
"""Quick test script for all enhancement components."""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from src.cloud.training.services.baseline_comparison import BaselineComparison
        print("✅ BaselineComparison")
    except Exception as e:
        print(f"❌ BaselineComparison: {e}")
        return False
    
    try:
        from src.cloud.training.services.data_integrity_verifier import DataIntegrityVerifier
        print("✅ DataIntegrityVerifier")
    except Exception as e:
        print(f"❌ DataIntegrityVerifier: {e}")
        return False
    
    try:
        from src.cloud.training.services.feature_pruner import FeaturePruner
        print("✅ FeaturePruner")
    except Exception as e:
        print(f"❌ FeaturePruner: {e}")
        return False
    
    try:
        from src.cloud.training.services.adaptive_trainer import AdaptiveTrainer
        print("✅ AdaptiveTrainer")
    except Exception as e:
        print(f"❌ AdaptiveTrainer: {e}")
        return False
    
    try:
        from src.cloud.training.services.error_resolver import ErrorResolver
        print("✅ ErrorResolver")
    except Exception as e:
        print(f"❌ ErrorResolver: {e}")
        return False
    
    try:
        from src.cloud.training.services.architecture_benchmarker import ArchitectureBenchmarker
        print("✅ ArchitectureBenchmarker")
    except Exception as e:
        print(f"❌ ArchitectureBenchmarker: {e}")
        return False
    
    try:
        from src.cloud.training.services.regime_detector import RegimeDetector
        print("✅ RegimeDetector")
    except Exception as e:
        print(f"❌ RegimeDetector: {e}")
        return False
    
    try:
        from src.cloud.training.services.hyperparameter_optimizer import HyperparameterOptimizer
        print("✅ HyperparameterOptimizer")
    except Exception as e:
        print(f"❌ HyperparameterOptimizer: {e}")
        return False
    
    try:
        from src.cloud.training.services.class_balancer import ClassBalancer
        print("✅ ClassBalancer")
    except Exception as e:
        print(f"❌ ClassBalancer: {e}")
        return False
    
    try:
        from src.cloud.training.models.hybrid_cnn_lstm import HybridCNNLSTM
        print("✅ HybridCNNLSTM")
    except Exception as e:
        print(f"❌ HybridCNNLSTM: {e}")
        return False
    
    try:
        from src.cloud.training.models.markov_chain import MarkovChainModel
        print("✅ MarkovChainModel")
    except Exception as e:
        print(f"❌ MarkovChainModel: {e}")
        return False
    
    try:
        from src.cloud.training.services.monte_carlo_validator import MonteCarloValidator
        print("✅ MonteCarloValidator")
    except Exception as e:
        print(f"❌ MonteCarloValidator: {e}")
        return False
    
    try:
        from src.cloud.training.models.stat_arb import StatisticalArbitrage
        print("✅ StatisticalArbitrage")
    except Exception as e:
        print(f"❌ StatisticalArbitrage: {e}")
        return False
    
    try:
        from src.cloud.training.services.hft_executor import HFTExecutor
        print("✅ HFTExecutor")
    except Exception as e:
        print(f"❌ HFTExecutor: {e}")
        return False
    
    try:
        from src.cloud.training.services.real_time_monitor import RealTimeMonitor
        print("✅ RealTimeMonitor")
    except Exception as e:
        print(f"❌ RealTimeMonitor: {e}")
        return False
    
    try:
        from src.cloud.training.services.ai_collaborator import AICollaborator
        print("✅ AICollaborator")
    except Exception as e:
        print(f"❌ AICollaborator: {e}")
        return False
    
    try:
        from src.cloud.training.services.meta_agent import MetaAgent
        print("✅ MetaAgent")
    except Exception as e:
        print(f"❌ MetaAgent: {e}")
        return False
    
    try:
        from src.cloud.training.features.liquidation_features import LiquidationFeatures
        print("✅ LiquidationFeatures")
    except Exception as e:
        print(f"❌ LiquidationFeatures: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test Markov Chain
        from src.cloud.training.models.markov_chain import MarkovChainModel
        model = MarkovChainModel()
        state_sequence = ['trend', 'consolidation', 'trend', 'breakout']
        result = model.fit(state_sequence)
        assert result["status"] == "success"
        print("✅ MarkovChainModel basic functionality")
    except Exception as e:
        print(f"❌ MarkovChainModel: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test Regime Detector
        import pandas as pd
        import numpy as np
        from src.cloud.training.services.regime_detector import RegimeDetector
        detector = RegimeDetector()
        data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100),
        })
        result = detector.detect_regime('BTC/USDT', data)
        assert "volatility_regime" in result
        print("✅ RegimeDetector basic functionality")
    except Exception as e:
        print(f"❌ RegimeDetector: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test Statistical Arbitrage
        import pandas as pd
        import numpy as np
        from src.cloud.training.models.stat_arb import StatisticalArbitrage
        arb = StatisticalArbitrage()
        asset1 = pd.Series(np.random.randn(100))
        asset2 = pd.Series(np.random.randn(100))
        result = arb.analyze_pair(asset1, asset2)
        assert "status" in result
        print("✅ StatisticalArbitrage basic functionality")
    except Exception as e:
        print(f"❌ StatisticalArbitrage: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Testing All Enhancement Components")
    print("=" * 70)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("❌ SOME FUNCTIONALITY TESTS FAILED")
            print("=" * 70)
            sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("❌ SOME IMPORT TESTS FAILED")
        print("=" * 70)
        sys.exit(1)

