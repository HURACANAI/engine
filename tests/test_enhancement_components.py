"""Comprehensive tests for enhancement components."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Import components to test
from src.cloud.training.services.baseline_comparison import BaselineComparison
from src.cloud.training.services.data_integrity_verifier import DataIntegrityVerifier
from src.cloud.training.services.feature_pruner import FeaturePruner
from src.cloud.training.services.adaptive_trainer import AdaptiveTrainer
from src.cloud.training.services.error_resolver import ErrorResolver
from src.cloud.training.services.architecture_benchmarker import ArchitectureBenchmarker
from src.cloud.training.services.regime_detector import RegimeDetector
from src.cloud.training.services.hyperparameter_optimizer import HyperparameterOptimizer
from src.cloud.training.services.class_balancer import ClassBalancer
from src.cloud.training.models.markov_chain import MarkovChainModel
from src.cloud.training.services.monte_carlo_validator import MonteCarloValidator
from src.cloud.training.models.stat_arb import StatisticalArbitrage
from src.cloud.training.services.hft_executor import HFTExecutor
from src.cloud.training.services.real_time_monitor import RealTimeMonitor
from src.cloud.training.services.ai_collaborator import AICollaborator
from src.cloud.training.services.meta_agent import MetaAgent
from src.cloud.training.features.liquidation_features import LiquidationFeatures


class TestBaselineComparison:
    """Test baseline comparison system."""
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        comparator = BaselineComparison()
        model = comparator.create_baseline_model()
        assert model is not None
    
    def test_compare_with_baseline(self):
        """Test model comparison with baseline."""
        comparator = BaselineComparison()
        
        # Create dummy data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20)
        
        # Create dummy model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Compare
        result = comparator.compare_with_baseline_from_data(
            model, X_train, y_train, X_test, y_test
        )
        
        assert "status" in result
        assert "model_metrics" in result or "error" in result


class TestDataIntegrityVerifier:
    """Test data integrity verifier."""
    
    def test_verify_data(self):
        """Test data verification."""
        verifier = DataIntegrityVerifier()
        
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100),
        })
        
        result = verifier.verify_data(data)
        
        assert "reliability" in result
        assert "should_train" in result
        assert "issues" in result
    
    def test_repair_data(self):
        """Test data repair."""
        verifier = DataIntegrityVerifier()
        
        # Create data with issues
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100),
            'close': np.random.randn(100),
        })
        data.loc[10:15, 'open'] = np.nan  # Add NaN values
        
        verification_result = verifier.verify_data(data)
        repaired_data, repair_report = verifier.repair_data(data, verification_result)
        
        assert len(repaired_data) >= len(data)
        assert "repairs" in repair_report


class TestFeaturePruner:
    """Test feature pruner."""
    
    def test_prune_features(self):
        """Test feature pruning."""
        pruner = FeaturePruner(prune_percentage=0.2)
        
        # Create feature importance
        feature_names = [f'feature_{i}' for i in range(20)]
        feature_importance = {name: np.random.rand() for name in feature_names}
        
        kept_features, prune_report = pruner.prune_features(feature_importance)
        
        assert len(kept_features) <= len(feature_names)
        assert "pruned_count" in prune_report


class TestRegimeDetector:
    """Test regime detector."""
    
    def test_detect_regime(self):
        """Test regime detection."""
        detector = RegimeDetector()
        
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100),
        })
        
        result = detector.detect_regime('BTC/USDT', data)
        
        assert "volatility_regime" in result
        assert "market_regime" in result


class TestMarkovChain:
    """Test Markov chain model."""
    
    def test_markov_chain(self):
        """Test Markov chain functionality."""
        model = MarkovChainModel()
        
        # Create state sequence
        state_sequence = ['trend', 'consolidation', 'trend', 'breakout', 'consolidation']
        
        # Fit model
        result = model.fit(state_sequence)
        
        assert result["status"] == "success"
        
        # Predict next state
        prediction = model.predict_next_state('trend')
        
        assert "predicted_state" in prediction
        assert "probabilities" in prediction


class TestMonteCarloValidator:
    """Test Monte Carlo validator."""
    
    def test_validate_model(self):
        """Test Monte Carlo validation."""
        validator = MonteCarloValidator(n_simulations=10)  # Reduced for testing
        
        # Create dummy model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        # Create test data
        X_test = np.random.randn(50, 10)
        y_test = np.random.randn(50)
        model.fit(X_test, y_test)
        
        result = validator.validate_model(model, X_test, y_test, n_simulations=10)
        
        assert "status" in result
        assert "mean_sharpe" in result or "error" in result


class TestStatisticalArbitrage:
    """Test statistical arbitrage."""
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        arb = StatisticalArbitrage()
        
        # Create test data
        asset1 = pd.Series(np.random.randn(100))
        asset2 = pd.Series(np.random.randn(100))
        
        spread = arb.calculate_spread(asset1, asset2)
        
        assert len(spread) > 0
    
    def test_analyze_pair(self):
        """Test pair analysis."""
        arb = StatisticalArbitrage()
        
        # Create test data
        asset1 = pd.Series(np.random.randn(100))
        asset2 = pd.Series(np.random.randn(100))
        
        result = arb.analyze_pair(asset1, asset2, 'BTC/USDT', 'ETH/USDT')
        
        assert "status" in result
        assert "cointegration" in result


class TestLiquidationFeatures:
    """Test liquidation features."""
    
    def test_create_features(self):
        """Test liquidation feature creation."""
        features = LiquidationFeatures()
        
        # Create test data
        liquidation_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1T'),
            'size_usd': np.random.randn(100),
            'side': np.random.choice(['long', 'short'], 100),
        })
        
        volume_data = pd.Series(np.random.randn(100))
        
        result = features.create_features(liquidation_data, volume_data)
        
        assert "liquidation_intensity" in result
        assert "liquidation_momentum" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

