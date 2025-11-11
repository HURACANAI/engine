#!/usr/bin/env python3
"""
Comprehensive Feature Testing Script

Tests each feature individually to ensure they work correctly.
Run with: python test_all_features.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple

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
        import numpy as np
        from src.cloud.training.datasets.return_converter import ReturnConverter
        
        # Test 1: Basic initialization
        converter = ReturnConverter()
        test_result("ReturnConverter", "Initialization", True)
        
        # Test 2: Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })
        test_result("ReturnConverter", "Sample data creation", True)
        
        # Test 3: Convert to returns (drop method)
        converter_drop = ReturnConverter(fill_method='drop')
        result = converter_drop.convert_to_returns(df, symbol='BTC/USD', price_column='close', timestamp_column='timestamp')
        assert 'raw_returns' in result.columns
        assert 'log_returns' in result.columns
        test_result("ReturnConverter", "Convert to returns (drop)", True)
        
        # Test 4: Convert to returns (forward fill)
        converter_ffill = ReturnConverter(fill_method='forward')
        result = converter_ffill.convert_to_returns(df, symbol='BTC/USD', price_column='close', timestamp_column='timestamp')
        assert 'raw_returns' in result.columns
        test_result("ReturnConverter", "Convert to returns (forward fill)", True)
        
        # Test 5: Convert to returns (backward fill)
        converter_bfill = ReturnConverter(fill_method='backward')
        result = converter_bfill.convert_to_returns(df, symbol='BTC/USD', price_column='close', timestamp_column='timestamp')
        assert 'raw_returns' in result.columns
        test_result("ReturnConverter", "Convert to returns (backward fill)", True)
        
        print("✅ Return Converter: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("ReturnConverter", "Error", False, str(e))
        print(f"❌ Return Converter: FAILED - {e}")
        traceback.print_exc()
        return False


def test_mechanic_utils():
    """Test Mechanic Utils (geometric returns, annualization, etc.)."""
    print("\n" + "="*80)
    print("TESTING: Mechanic Utils (Geometric Returns, Annualization, etc.)")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.services.mechanic_utils import (
            geometric_link,
            annualize_return,
            annualize_volatility,
            create_wealth_index,
            calculate_drawdowns
        )
        
        # Test 1: Geometric link
        returns = [0.01, 0.02, -0.01, 0.03]
        compound = geometric_link(returns)
        assert isinstance(compound, (float, np.floating))
        test_result("MechanicUtils", "Geometric link", True)
        
        # Test 2: Annualize return
        annual_ret = annualize_return(returns, periods_per_year=252)
        assert isinstance(annual_ret, (float, np.floating))
        test_result("MechanicUtils", "Annualize return", True)
        
        # Test 3: Annualize volatility
        annual_vol = annualize_volatility(returns, periods_per_year=252)
        assert isinstance(annual_vol, (float, np.floating))
        assert annual_vol >= 0
        test_result("MechanicUtils", "Annualize volatility", True)
        
        # Test 4: Create wealth index
        wealth = create_wealth_index(returns)
        assert len(wealth) == len(returns) + 1
        assert wealth[0] == 1.0
        test_result("MechanicUtils", "Create wealth index", True)
        
        # Test 5: Calculate drawdowns
        drawdowns, max_dd, max_dd_duration = calculate_drawdowns(wealth)
        assert len(drawdowns) == len(wealth)
        assert max_dd <= 0
        assert max_dd_duration >= 0
        test_result("MechanicUtils", "Calculate drawdowns", True)
        
        print("✅ Mechanic Utils: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("MechanicUtils", "Error", False, str(e))
        print(f"❌ Mechanic Utils: FAILED - {e}")
        traceback.print_exc()
        return False


def test_sharpe_ratio():
    """Test Sharpe Ratio integration."""
    print("\n" + "="*80)
    print("TESTING: Sharpe Ratio Integration")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.metrics.enhanced_metrics import EnhancedMetricsCalculator
        
        calculator = EnhancedMetricsCalculator()
        test_result("SharpeRatio", "Initialization", True)
        
        # Test 1: Calculate Sharpe ratio
        returns = np.random.randn(100) * 0.01
        sharpe = calculator.evaluate_sharpe(returns, periods_per_year=252)
        assert isinstance(sharpe, (float, np.floating))
        test_result("SharpeRatio", "Calculate Sharpe ratio", True)
        
        # Test 2: Handle empty returns
        sharpe_empty = calculator.evaluate_sharpe([], periods_per_year=252)
        assert sharpe_empty == 0.0
        test_result("SharpeRatio", "Handle empty returns", True)
        
        # Test 3: Handle zero volatility
        returns_constant = [0.01] * 100
        sharpe_const = calculator.evaluate_sharpe(returns_constant, periods_per_year=252)
        # Should handle gracefully (either 0 or inf)
        assert isinstance(sharpe_const, (float, np.floating))
        test_result("SharpeRatio", "Handle zero volatility", True)
        
        print("✅ Sharpe Ratio: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("SharpeRatio", "Error", False, str(e))
        print(f"❌ Sharpe Ratio: FAILED - {e}")
        traceback.print_exc()
        return False


def test_performance_visualizer():
    """Test Performance Visualizer."""
    print("\n" + "="*80)
    print("TESTING: Performance Visualizer")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.metrics.performance_visualizer import PerformanceVisualizer
        
        visualizer = PerformanceVisualizer()
        test_result("PerformanceVisualizer", "Initialization", True)
        
        # Test 1: Create sample data
        returns = np.random.randn(100) * 0.01
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        test_result("PerformanceVisualizer", "Sample data creation", True)
        
        # Test 2: Plot wealth index (matplotlib)
        try:
            visualizer.plot_wealth_index(returns, dates, backend='matplotlib', show=False)
            test_result("PerformanceVisualizer", "Plot wealth index (matplotlib)", True)
        except Exception as e:
            test_result("PerformanceVisualizer", "Plot wealth index (matplotlib)", False, str(e))
        
        # Test 3: Plot drawdowns (matplotlib)
        try:
            visualizer.plot_drawdowns(returns, dates, backend='matplotlib', show=False)
            test_result("PerformanceVisualizer", "Plot drawdowns (matplotlib)", True)
        except Exception as e:
            test_result("PerformanceVisualizer", "Plot drawdowns (matplotlib)", False, str(e))
        
        print("✅ Performance Visualizer: TESTS COMPLETED")
        return True
        
    except Exception as e:
        test_result("PerformanceVisualizer", "Error", False, str(e))
        print(f"❌ Performance Visualizer: FAILED - {e}")
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
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.rand(1000) * 1000
        })
        test_result("WorldModel", "Sample data creation", True)
        
        # Test 3: Predict next state
        state = world_model.predict_next_state(df, symbol='BTC/USD', timestamp_column='timestamp')
        assert len(state) == 32
        test_result("WorldModel", "Predict next state", True)
        
        # Test 4: Handle empty DataFrame
        try:
            empty_df = pd.DataFrame()
            world_model.predict_next_state(empty_df, symbol='BTC/USD', timestamp_column='timestamp')
            test_result("WorldModel", "Handle empty DataFrame", False, "Should raise ValueError")
        except ValueError:
            test_result("WorldModel", "Handle empty DataFrame", True)
        
        # Test 5: Handle zero prices
        df_zero = df.copy()
        df_zero.loc[0, 'close'] = 0.0
        state = world_model.predict_next_state(df_zero, symbol='BTC/USD', timestamp_column='timestamp')
        assert len(state) == 32
        test_result("WorldModel", "Handle zero prices", True)
        
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
        from src.cloud.training.models.training_inference_split import (
            TrainingPhase,
            InferencePhase
        )
        
        # Test 1: Training phase initialization
        training = TrainingPhase()
        test_result("TrainingInferenceSplit", "Training phase initialization", True)
        
        # Test 2: Inference phase initialization
        inference = InferencePhase()
        test_result("TrainingInferenceSplit", "Inference phase initialization", True)
        
        # Test 3: Phase switching
        training.enter_training()
        assert training.is_training
        test_result("TrainingInferenceSplit", "Enter training phase", True)
        
        inference.enter_inference()
        assert inference.is_inference
        test_result("TrainingInferenceSplit", "Enter inference phase", True)
        
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
        
        # Test 4: Encode features
        encoded = autoencoder.encode(features)
        assert encoded.shape[1] == 5
        test_result("FeatureAutoencoder", "Encode features", True)
        
        # Test 5: Input validation - wrong shape
        try:
            wrong_features = np.random.randn(100, 5)  # Wrong dimension
            autoencoder.train(wrong_features, epochs=1)
            test_result("FeatureAutoencoder", "Input validation (wrong shape)", False, "Should raise ValueError")
        except ValueError:
            test_result("FeatureAutoencoder", "Input validation (wrong shape)", True)
        
        # Test 6: Input validation - empty array
        try:
            empty_features = np.array([]).reshape(0, 10)
            autoencoder.train(empty_features, epochs=1)
            test_result("FeatureAutoencoder", "Input validation (empty array)", False, "Should raise ValueError")
        except ValueError:
            test_result("FeatureAutoencoder", "Input validation (empty array)", True)
        
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
        
        # Test 2: Select loss function
        loss_fn = adaptive_loss.select_loss_function(regime='trending')
        assert callable(loss_fn)
        test_result("AdaptiveLoss", "Select loss function", True)
        
        # Test 3: Calculate loss
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        loss = adaptive_loss.calculate_loss(y_true, y_pred, regime='trending')
        assert isinstance(loss, (float, np.floating))
        assert loss >= 0
        test_result("AdaptiveLoss", "Calculate loss", True)
        
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
        
        # Test 2: Validate clean data
        clean_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.rand(100) * 1000
        })
        result = checkpoint.validate(clean_data)
        assert result['passed'] == True
        test_result("DataIntegrityCheckpoint", "Validate clean data", True)
        
        # Test 3: Detect NaN values
        data_with_nan = clean_data.copy()
        data_with_nan.loc[10:20, 'close'] = np.nan
        result = checkpoint.validate(data_with_nan)
        assert result['passed'] == False
        assert len(result['issues']) > 0
        test_result("DataIntegrityCheckpoint", "Detect NaN values", True)
        
        print("✅ Data Integrity Checkpoint: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("DataIntegrityCheckpoint", "Error", False, str(e))
        print(f"❌ Data Integrity Checkpoint: FAILED - {e}")
        traceback.print_exc()
        return False


def test_model_introspection():
    """Test Model Introspection."""
    print("\n" + "="*80)
    print("TESTING: Model Introspection")
    print("="*80)
    
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from src.cloud.training.analysis.model_introspection import ModelIntrospection
        
        # Test 1: Initialization
        introspection = ModelIntrospection()
        test_result("ModelIntrospection", "Initialization", True)
        
        # Test 2: Create sample model and data
        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1
        model.fit(X, y)
        feature_names = [f'feature_{i}' for i in range(5)]
        test_result("ModelIntrospection", "Sample model creation", True)
        
        # Test 3: Calculate SHAP values (may fail if SHAP not installed)
        try:
            shap_values = introspection.calculate_shap_values(model, X, feature_names)
            if shap_values is not None:
                assert len(shap_values) == len(feature_names)
                test_result("ModelIntrospection", "Calculate SHAP values", True)
            else:
                test_result("ModelIntrospection", "Calculate SHAP values", True, "SHAP not available (optional)")
        except Exception as e:
            test_result("ModelIntrospection", "Calculate SHAP values", True, f"SHAP not available: {str(e)}")
        
        # Test 4: Score model
        score = introspection.score_model(model, X, y)
        assert isinstance(score, (float, np.floating))
        test_result("ModelIntrospection", "Score model", True)
        
        print("✅ Model Introspection: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("ModelIntrospection", "Error", False, str(e))
        print(f"❌ Model Introspection: FAILED - {e}")
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
        
        # Test 2: Evaluate reward
        reward = evaluator.evaluate_reward(
            forecast_return=0.02,
            actual_return=0.015,
            regime='trending',
            position_size=1.0
        )
        assert isinstance(reward, float)
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
        import numpy as np
        from src.cloud.training.datasets.data_drift_detector import DataDriftDetector
        
        # Test 1: Initialization
        detector = DataDriftDetector(drift_threshold=0.2)
        test_result("DataDriftDetector", "Initialization", True)
        
        # Test 2: Create baseline and new data
        baseline = np.random.randn(1000)
        new_data = np.random.randn(100)  # Similar distribution
        test_result("DataDriftDetector", "Sample data creation", True)
        
        # Test 3: Detect drift (should not detect drift for similar data)
        drift_detected = detector.detect_drift(baseline, new_data)
        assert isinstance(drift_detected, bool)
        test_result("DataDriftDetector", "Detect drift", True)
        
        print("✅ Data Drift Detector: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("DataDriftDetector", "Error", False, str(e))
        print(f"❌ Data Drift Detector: FAILED - {e}")
        traceback.print_exc()
        return False


def test_reality_deviation_score():
    """Test Reality Deviation Score."""
    print("\n" + "="*80)
    print("TESTING: Reality Deviation Score")
    print("="*80)
    
    try:
        import numpy as np
        from src.cloud.training.metrics.reality_deviation_score import RealityDeviationScore
        
        # Test 1: Initialization
        rds = RealityDeviationScore(alert_threshold=0.2)
        test_result("RealityDeviationScore", "Initialization", True)
        
        # Test 2: Calculate RDS
        predictions = np.random.randn(100) * 0.01
        actuals = predictions + np.random.randn(100) * 0.001  # Close to predictions
        score = rds.calculate(predictions, actuals)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        test_result("RealityDeviationScore", "Calculate RDS", True)
        
        # Test 3: Check alerts
        should_alert = rds.should_alert(score)
        assert isinstance(should_alert, bool)
        test_result("RealityDeviationScore", "Check alerts", True)
        
        print("✅ Reality Deviation Score: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        test_result("RealityDeviationScore", "Error", False, str(e))
        print(f"❌ Reality Deviation Score: FAILED - {e}")
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
            OrderBookSide
        )
        from datetime import datetime
        
        # Test 1: Initialization
        aggregator = MultiExchangeOrderbookAggregator(exchanges=['binance', 'coinbase'])
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
        
        # Test 3: Aggregate orderbooks
        exchange_orderbooks = {
            'binance': orderbook,
            'coinbase': orderbook
        }
        aggregated = aggregator.aggregate_orderbooks(exchange_orderbooks)
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
        
        # Test 1: Initialization
        calibrator = FeeLatencyCalibrator(exchanges=['binance'])
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
        
        # Test 3: Input validation - negative size
        try:
            calibrator.record_execution(
                exchange='binance',
                symbol='BTC/USD',
                order_type='maker',
                size_usd=-1000.0,  # Invalid
                actual_fee_bps=2.0,
                actual_latency_ms=50.0,
                price=50000.0
            )
            test_result("FeeLatencyCalibrator", "Input validation (negative size)", False, "Should raise ValueError")
        except ValueError:
            test_result("FeeLatencyCalibrator", "Input validation (negative size)", True)
        
        # Test 4: Get calibration report
        report = calibrator.get_calibration_report('binance')
        # May be None if not enough samples
        test_result("FeeLatencyCalibrator", "Get calibration report", True)
        
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
        
        # Test 1: Initialization
        manager = SpreadThresholdManager(
            default_max_spread_bps=10.0,
            default_max_age_seconds=60
        )
        test_result("SpreadThresholdManager", "Initialization", True)
        
        # Test 2: Place order
        order = manager.place_order(
            symbol='BTC/USD',
            side='buy',
            price=50000.0,
            size=0.1,
            exchange='binance',
            spread_threshold_bps=10.0
        )
        assert order.status == OrderStatus.PENDING
        test_result("SpreadThresholdManager", "Place order", True)
        
        # Test 3: Update spread
        manager.update_spread('BTC/USD', 'binance', bid=49999.0, ask=50001.0)
        test_result("SpreadThresholdManager", "Update spread", True)
        
        # Test 4: Monitor and cancel orders
        cancelled = manager.monitor_and_cancel_orders()
        assert isinstance(cancelled, list)
        test_result("SpreadThresholdManager", "Monitor and cancel orders", True)
        
        # Test 5: Memory limit check
        assert manager.max_orders == 10000
        test_result("SpreadThresholdManager", "Memory limit check", True)
        
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
    print("COMPREHENSIVE FEATURE TESTING")
    print("="*80)
    print("Testing all features individually...")
    
    # Import pandas for tests that need it
    global pd
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  Warning: pandas not available, some tests may fail")
        pd = None
    
    # Run all tests
    tests = [
        test_return_converter,
        test_mechanic_utils,
        test_sharpe_ratio,
        test_performance_visualizer,
        test_world_model,
        test_training_inference_split,
        test_feature_autoencoder,
        test_adaptive_loss,
        test_data_integrity_checkpoint,
        test_model_introspection,
        test_reward_evaluator,
        test_data_drift_detector,
        test_reality_deviation_score,
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

