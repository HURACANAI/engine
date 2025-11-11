#!/usr/bin/env python3
"""
Comprehensive Feature Testing Script - Fixed Version

Tests each feature individually with correct API calls.
Run with: python test_all_features_fixed.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import traceback
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test results tracking
test_results: Dict[str, Dict[str, any]] = {}


def test_result(feature_name: str, test_name: str, passed: bool, error: str = None):
    """Record test result."""
    if feature_name not in test_results:
        test_results[feature_name] = {}
    test_results[feature_name][test_name] = {
        "passed": passed,
        "error": error
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if error:
        print(f"    Error: {error}")


def test_return_converter():
    """Test Return Converter feature."""
    print("\n" + "="*80)
    print("TESTING: Return Converter")
    print("="*80)
    
    try:
        import pandas as pd
        import polars as pl
        import numpy as np
        from src.cloud.training.datasets.return_converter import ReturnConverter
        
        # Test 1: Basic initialization
        converter = ReturnConverter()
        test_result("ReturnConverter", "Initialization", True)
        
        # Test 2: Create sample data (polars DataFrame)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df_pandas = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        df = pl.from_pandas(df_pandas)
        test_result("ReturnConverter", "Sample data creation", True)
        
        # Test 3: Convert to returns (drop method)
        converter_drop = ReturnConverter(fill_method='drop')
        result = converter_drop.convert(
            price_data=df,
            price_column='close',
            symbol='BTC/USD',
            timestamp_column='timestamp'
        )
        assert 'raw_returns' in result.columns
        assert 'log_returns' in result.columns
        test_result("ReturnConverter", "Convert to returns (drop)", True)
        
        # Test 4: Convert to returns (forward fill)
        converter_ffill = ReturnConverter(fill_method='forward')
        result = converter_ffill.convert(
            price_data=df,
            price_column='close',
            symbol='BTC/USD',
            timestamp_column='timestamp'
        )
        assert 'raw_returns' in result.columns
        test_result("ReturnConverter", "Convert to returns (forward fill)", True)
        
        print("✅ Return Converter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("ReturnConverter", "Error", False, str(e))
        print(f"❌ Return Converter: FAILED - {e}")
        traceback.print_exc()
        return False


def test_mechanic_utils():
    """Test Mechanic Utils."""
    print("\n" + "="*80)
    print("TESTING: Mechanic Utils")
    print("="*80)
    
    try:
        import numpy as np
        import pandas as pd
        from src.cloud.training.services.mechanic_utils import Mechanic
        
        # Test 1: Geometric link with numpy array
        returns = np.array([0.01, 0.02, -0.01, 0.03])
        result = Mechanic.geometric_link(returns)
        assert isinstance(result, np.ndarray)
        test_result("MechanicUtils", "Geometric link (numpy)", True)
        
        # Test 2: Geometric link with DataFrame
        df = pd.DataFrame({'returns': [0.01, 0.02, -0.01, 0.03]})
        result = Mechanic.geometric_link(df, return_column='returns')
        assert 'geometric_return' in result.columns
        test_result("MechanicUtils", "Geometric link (DataFrame)", True)
        
        # Test 3: Annualize return
        annual_ret = Mechanic.annualize_return(returns, periods_per_year=252)
        assert isinstance(annual_ret, float)
        test_result("MechanicUtils", "Annualize return", True)
        
        # Test 4: Annualize volatility
        annual_vol = Mechanic.annualize_volatility(returns, periods_per_year=252)
        assert isinstance(annual_vol, float)
        assert annual_vol >= 0
        test_result("MechanicUtils", "Annualize volatility", True)
        
        # Test 5: Create wealth index (returns array directly for numpy)
        wealth = Mechanic.create_wealth_index(returns, initial_value=1000.0)
        assert len(wealth) == len(returns)
        # First value is initial_value * (1 + first_return)
        assert wealth[0] == 1000.0 * (1 + returns[0])
        test_result("MechanicUtils", "Create wealth index", True)
        
        # Test 6: Calculate drawdowns (needs DataFrame with wealth_index column)
        df_wealth = pd.DataFrame({'wealth_index': wealth})
        df_with_dd = Mechanic.calc_drawdowns(df_wealth, wealth_column='wealth_index')
        assert 'drawdown' in df_with_dd.columns
        assert 'previous_peak' in df_with_dd.columns
        test_result("MechanicUtils", "Calculate drawdowns", True)
        
        # Test 7: Get max drawdown
        max_dd, start_idx, recovery_idx = Mechanic.get_max_drawdown(df_wealth, wealth_column='wealth_index')
        assert max_dd >= 0
        assert start_idx >= 0
        assert recovery_idx >= 0
        test_result("MechanicUtils", "Get max drawdown", True)
        
        print("✅ Mechanic Utils: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("MechanicUtils", "Error", False, str(e))
        print(f"❌ Mechanic Utils: FAILED - {e}")
        traceback.print_exc()
        return False


def test_world_model():
    """Test World Model."""
    print("\n" + "="*80)
    print("TESTING: World Model")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from src.cloud.training.models.world_model import WorldModel
        
        # Test 1: Initialization
        world_model = WorldModel(state_dim=32, lookback_days=30)
        test_result("WorldModel", "Initialization", True)
        
        # Test 2: Create sample data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.rand(1000) * 1000
        })
        test_result("WorldModel", "Sample data creation", True)
        
        # Test 3: Build state (not predict_next_state directly)
        state = world_model.build_state(df, symbol='BTC/USD', timestamp_column='timestamp')
        assert state.state_vector is not None
        assert len(state.state_vector) == 32
        test_result("WorldModel", "Build state", True)
        
        # Test 4: Predict next state
        next_state = world_model.predict_next_state(state, horizon_days=1)
        assert next_state.state_vector is not None
        assert len(next_state.state_vector) == 32
        test_result("WorldModel", "Predict next state", True)
        
        print("✅ World Model: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("WorldModel", "Error", False, str(e))
        print(f"❌ World Model: FAILED - {e}")
        traceback.print_exc()
        return False


def test_training_inference_split():
    """Test Training/Inference Split."""
    print("\n" + "="*80)
    print("TESTING: Training/Inference Split")
    print("="*80)
    
    try:
        from src.cloud.training.models.training_inference_split import TrainingInferenceSplit
        
        # Test 1: Initialization
        split = TrainingInferenceSplit()
        test_result("TrainingInferenceSplit", "Initialization", True)
        
        # Test 2: Set training mode
        split.set_training_mode()
        assert split.is_training_mode()
        test_result("TrainingInferenceSplit", "Set training mode", True)
        
        # Test 3: Set inference mode (will fail without stable checkpoint, but that's expected)
        try:
            split.set_inference_mode()
            test_result("TrainingInferenceSplit", "Set inference mode", False, "Should fail without stable checkpoint")
        except RuntimeError:
            test_result("TrainingInferenceSplit", "Set inference mode (validation)", True)
        
        print("✅ Training/Inference Split: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("TrainingInferenceSplit", "Error", False, str(e))
        print(f"❌ Training/Inference Split: FAILED - {e}")
        traceback.print_exc()
        return False


def test_feature_autoencoder():
    """Test Feature Autoencoder."""
    print("\n" + "="*80)
    print("TESTING: Feature Autoencoder")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.models.feature_autoencoder import FeatureAutoencoder
        
        # Test 1: Initialization
        autoencoder = FeatureAutoencoder(input_dim=10, latent_dim=5)
        test_result("FeatureAutoencoder", "Initialization", True)
        
        # Test 2: Create sample features
        features = np.random.randn(100, 10)
        test_result("FeatureAutoencoder", "Sample features creation", True)
        
        # Test 3: Train autoencoder
        autoencoder.train(features, epochs=5, batch_size=32)
        test_result("FeatureAutoencoder", "Train autoencoder", True)
        
        # Test 4: Encode features (returns EncodedFeatures object)
        encoded = autoencoder.encode(features)
        assert hasattr(encoded, 'encoded_features')
        assert encoded.encoded_features.shape[1] == 5
        test_result("FeatureAutoencoder", "Encode features", True)
        
        print("✅ Feature Autoencoder: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("FeatureAutoencoder", "Error", False, str(e))
        print(f"❌ Feature Autoencoder: FAILED - {e}")
        traceback.print_exc()
        return False


def test_adaptive_loss():
    """Test Adaptive Loss."""
    print("\n" + "="*80)
    print("TESTING: Adaptive Loss")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.models.adaptive_loss import AdaptiveLoss
        
        # Test 1: Initialization
        adaptive_loss = AdaptiveLoss()
        test_result("AdaptiveLoss", "Initialization", True)
        
        # Test 2: Update metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        metrics = adaptive_loss.update_metrics(y_pred, y_true, regime='trending')
        assert isinstance(metrics, dict)
        test_result("AdaptiveLoss", "Update metrics", True)
        
        # Test 3: Get best loss
        best_loss = adaptive_loss.get_best_loss('trending')
        assert best_loss is not None
        test_result("AdaptiveLoss", "Get best loss", True)
        
        print("✅ Adaptive Loss: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("AdaptiveLoss", "Error", False, str(e))
        print(f"❌ Adaptive Loss: FAILED - {e}")
        traceback.print_exc()
        return False


def test_data_integrity_checkpoint():
    """Test Data Integrity Checkpoint."""
    print("\n" + "="*80)
    print("TESTING: Data Integrity Checkpoint")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from src.cloud.training.datasets.data_integrity_checkpoint import DataIntegrityCheckpoint
        
        # Test 1: Initialization
        checkpoint = DataIntegrityCheckpoint()
        test_result("DataIntegrityCheckpoint", "Initialization", True)
        
        # Test 2: Validate clean data (requires symbol parameter)
        clean_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.rand(100) * 1000,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h')
        })
        result = checkpoint.validate(clean_data, symbol='BTC/USD')
        assert result.overall_score >= 0
        test_result("DataIntegrityCheckpoint", "Validate clean data", True)
        
        print("✅ Data Integrity Checkpoint: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("DataIntegrityCheckpoint", "Error", False, str(e))
        print(f"❌ Data Integrity Checkpoint: FAILED - {e}")
        traceback.print_exc()
        return False


def test_reward_evaluator():
    """Test Regime-Aware Reward Evaluator."""
    print("\n" + "="*80)
    print("TESTING: Regime-Aware Reward Evaluator")
    print("="*80)
    
    try:
        from src.cloud.training.agents.reward_evaluator import RegimeAwareRewardEvaluator
        
        # Test 1: Initialization
        evaluator = RegimeAwareRewardEvaluator()
        test_result("RewardEvaluator", "Initialization", True)
        
        # Test 2: Evaluate reward (method is 'evaluate', not 'evaluate_reward')
        reward_signal = evaluator.evaluate(
            prediction=0.05,
            actual=0.03,
            regime='trending',
            confidence=0.8,
            symbol='BTC/USD'
        )
        assert hasattr(reward_signal, 'reward')
        assert hasattr(reward_signal, 'regime')
        test_result("RewardEvaluator", "Evaluate reward", True)
        
        print("✅ Reward Evaluator: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("RewardEvaluator", "Error", False, str(e))
        print(f"❌ Reward Evaluator: FAILED - {e}")
        traceback.print_exc()
        return False


def test_data_drift_detector():
    """Test Data Drift Detector."""
    print("\n" + "="*80)
    print("TESTING: Data Drift Detector")
    print("="*80)
    
    try:
        import pandas as pd
        import numpy as np
        from src.cloud.training.datasets.data_drift_detector import DataDriftDetector
        
        # Test 1: Initialization
        detector = DataDriftDetector(drift_threshold=0.2)
        test_result("DataDriftDetector", "Initialization", True)
        
        # Test 2: Set baseline (requires DataFrame with symbol)
        baseline_df = pd.DataFrame({
            'close': np.random.randn(1000),
            'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='h')
        })
        detector.set_baseline(baseline_df, symbol='BTC/USD', price_column='close')
        test_result("DataDriftDetector", "Set baseline", True)
        
        # Test 3: Detect drift (requires DataFrame with symbol)
        new_data = pd.DataFrame({
            'close': np.random.randn(100),
            'timestamp': pd.date_range(start='2024-02-01', periods=100, freq='h')
        })
        report = detector.detect_drift(new_data, symbol='BTC/USD', price_column='close')
        assert hasattr(report, 'metrics')
        assert hasattr(report.metrics, 'should_retrain')
        test_result("DataDriftDetector", "Detect drift", True)
        
        print("✅ Data Drift Detector: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("DataDriftDetector", "Error", False, str(e))
        print(f"❌ Data Drift Detector: FAILED - {e}")
        traceback.print_exc()
        return False


def test_multi_exchange_aggregator():
    """Test Multi-Exchange Orderbook Aggregator."""
    print("\n" + "="*80)
    print("TESTING: Multi-Exchange Orderbook Aggregator")
    print("="*80)
    
    try:
        from src.cloud.training.orderbook.multi_exchange_aggregator import (
            MultiExchangeOrderbookAggregator,
            OrderBookSnapshot,
            OrderBookLevel,
        )
        from datetime import datetime
        
        # Test 1: Initialization (no exchanges parameter)
        aggregator = MultiExchangeOrderbookAggregator()
        test_result("MultiExchangeAggregator", "Initialization", True)
        
        # Test 2: Create sample orderbook
        bids = [OrderBookLevel(price=100.0, size=1.0), OrderBookLevel(price=99.9, size=2.0)]
        asks = [OrderBookLevel(price=100.1, size=1.0), OrderBookLevel(price=100.2, size=2.0)]
        orderbook = OrderBookSnapshot(
            symbol='BTC/USD',
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
        test_result("MultiExchangeAggregator", "Sample orderbook creation", True)
        
        # Test 3: Add exchange orderbook
        aggregator.add_exchange_orderbook(
            exchange='binance',
            snapshot=orderbook,
            latency_ms=50.0
        )
        test_result("MultiExchangeAggregator", "Add exchange orderbook", True)
        
        # Test 4: Aggregate orderbooks
        aggregated = aggregator.aggregate(symbol='BTC/USD')
        assert aggregated.symbol == 'BTC/USD'
        assert aggregated.best_bid > 0
        assert aggregated.best_ask > 0
        test_result("MultiExchangeAggregator", "Aggregate orderbooks", True)
        
        print("✅ Multi-Exchange Aggregator: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("MultiExchangeAggregator", "Error", False, str(e))
        print(f"❌ Multi-Exchange Aggregator: FAILED - {e}")
        traceback.print_exc()
        return False


def test_fee_latency_calibration():
    """Test Fee/Latency Calibration."""
    print("\n" + "="*80)
    print("TESTING: Fee/Latency Calibration")
    print("="*80)
    
    try:
        from src.cloud.training.services.fee_latency_calibration import FeeLatencyCalibrator
        
        # Test 1: Initialization (no exchanges parameter)
        calibrator = FeeLatencyCalibrator()
        test_result("FeeLatencyCalibrator", "Initialization", True)
        
        # Test 2: Record execution
        calibrator.record_execution(
            exchange='binance',
            symbol='BTC/USD',
            order_type='maker',
            size_usd=1000.0,
            actual_fee_bps=2.0,
            actual_latency_ms=50.0,
            price=50000.0
        )
        test_result("FeeLatencyCalibrator", "Record execution", True)
        
        # Test 3: Get fee calibration
        fee_cal = calibrator.get_fee_calibration('binance')
        # May be None if not enough samples
        test_result("FeeLatencyCalibrator", "Get fee calibration", True)
        
        # Test 4: Get latency calibration
        latency_cal = calibrator.get_latency_calibration('binance')
        # May be None if not enough samples
        test_result("FeeLatencyCalibrator", "Get latency calibration", True)
        
        print("✅ Fee/Latency Calibrator: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("FeeLatencyCalibrator", "Error", False, str(e))
        print(f"❌ Fee/Latency Calibrator: FAILED - {e}")
        traceback.print_exc()
        return False


def test_spread_threshold_manager():
    """Test Spread Threshold Manager."""
    print("\n" + "="*80)
    print("TESTING: Spread Threshold Manager")
    print("="*80)
    
    try:
        from src.cloud.training.execution.spread_threshold_manager import (
            SpreadThresholdManager,
            OrderStatus
        )
        from datetime import datetime
        import uuid
        
        # Test 1: Initialization (correct parameter names)
        manager = SpreadThresholdManager(
            max_spread_bps=10.0,
            auto_cancel=True,
            max_order_age_seconds=60
        )
        test_result("SpreadThresholdManager", "Initialization", True)
        
        # Test 2: Place order (requires order_id parameter)
        order_id = str(uuid.uuid4())
        order = manager.place_order(
            order_id=order_id,
            symbol='BTC/USD',
            side='buy',
            price=50000.0,
            size=0.1,
            exchange='binance',
            spread_threshold_bps=10.0
        )
        assert order.status == OrderStatus.PENDING
        test_result("SpreadThresholdManager", "Place order", True)
        
        # Test 3: Update spread (no exchange parameter)
        manager.update_spread('BTC/USD', best_bid=49999.0, best_ask=50001.0)
        test_result("SpreadThresholdManager", "Update spread", True)
        
        # Test 4: Monitor orders
        monitored = manager.monitor_orders()
        assert isinstance(monitored, list)
        test_result("SpreadThresholdManager", "Monitor orders", True)
        
        # Test 5: Cancel order
        cancelled = manager.cancel_order(order_id, reason="test")
        assert cancelled == True
        test_result("SpreadThresholdManager", "Cancel order", True)
        
        print("✅ Spread Threshold Manager: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("SpreadThresholdManager", "Error", False, str(e))
        print(f"❌ Spread Threshold Manager: FAILED - {e}")
        traceback.print_exc()
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_features = len(test_results)
    passed_features = 0
    failed_features = 0
    
    for feature_name, tests in test_results.items():
        all_passed = all(test['passed'] for test in tests.values())
        if all_passed:
            passed_features += 1
            status = "✅ PASS"
        else:
            failed_features += 1
            status = "❌ FAIL"
        
        print(f"\n{status}: {feature_name}")
        for test_name, result in tests.items():
            test_status = "✅" if result['passed'] else "❌"
            print(f"  {test_status} {test_name}")
            if result['error']:
                print(f"    → {result['error']}")
    
    print("\n" + "="*80)
    print(f"Total Features: {total_features}")
    print(f"✅ Passed: {passed_features}")
    print(f"❌ Failed: {failed_features}")
    print("="*80)


def main():
    """Run all feature tests."""
    print("="*80)
    print("COMPREHENSIVE FEATURE TESTING - FIXED VERSION")
    print("="*80)
    print("Testing all features individually with correct APIs...")
    
    # Run all tests
    tests = [
        test_return_converter,
        test_mechanic_utils,
        test_world_model,
        test_training_inference_split,
        test_feature_autoencoder,
        test_adaptive_loss,
        test_data_integrity_checkpoint,
        test_reward_evaluator,
        test_data_drift_detector,
        test_multi_exchange_aggregator,
        test_fee_latency_calibration,
        test_spread_threshold_manager,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_func.__name__}: {e}")
            traceback.print_exc()
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    main()

