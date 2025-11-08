"""Multi-architecture benchmarking system."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

from .comprehensive_evaluation import ComprehensiveEvaluation
from ..models.hybrid_cnn_lstm import HybridCNNLSTM
from ..models.standardized_lstm import StandardizedLSTM

logger = structlog.get_logger(__name__)


class ArchitectureBenchmarker:
    """
    Benchmarks multiple architectures:
    - LSTM
    - CNN-LSTM hybrid
    - Transformer
    - XGBoost baseline
    
    Selects best architecture per market regime.
    """

    def __init__(
        self,
        evaluator: Optional[ComprehensiveEvaluation] = None,
    ) -> None:
        """
        Initialize architecture benchmarker.
        
        Args:
            evaluator: Comprehensive evaluation instance
        """
        self.evaluator = evaluator or ComprehensiveEvaluation()
        logger.info("architecture_benchmarker_initialized")

    def benchmark_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
    ) -> Dict[str, Any]:
        """Benchmark LSTM architecture."""
        logger.info("benchmarking_lstm", input_dim=input_dim)
        
        try:
            model = StandardizedLSTM(
                input_dim=input_dim,
                lstm_units=128,
                num_layers=2,
                dropout_rate=0.2,
            )
            
            # Train
            training_result = model.fit(
                X_train, y_train,
                X_test, y_test,
                epochs=50,
                batch_size=32,
            )
            
            # Predict
            predictions = model.predict(X_test)
            
            # Evaluate
            returns = predictions.flatten() - y_test
            metrics = self.evaluator.evaluate_model(
                predictions=predictions.flatten(),
                actuals=y_test,
                returns=returns,
            )
            
            return {
                "architecture": "lstm",
                "metrics": metrics,
                "training_result": training_result,
            }
        except Exception as e:
            logger.error("lstm_benchmark_failed", error=str(e))
            return {
                "architecture": "lstm",
                "status": "failed",
                "error": str(e),
            }

    def benchmark_hybrid(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
    ) -> Dict[str, Any]:
        """Benchmark CNN-LSTM hybrid architecture."""
        logger.info("benchmarking_hybrid", input_dim=input_dim)
        
        try:
            model = HybridCNNLSTM(
                input_dim=input_dim,
                cnn_filters=64,
                lstm_units=128,
                num_lstm_layers=2,
                dropout_rate=0.2,
            )
            
            # Train
            training_result = model.fit(
                X_train, y_train,
                X_test, y_test,
                epochs=50,
                batch_size=32,
            )
            
            # Predict
            predictions = model.predict(X_test)
            
            # Evaluate
            returns = predictions.flatten() - y_test
            metrics = self.evaluator.evaluate_model(
                predictions=predictions.flatten(),
                actuals=y_test,
                returns=returns,
            )
            
            return {
                "architecture": "hybrid_cnn_lstm",
                "metrics": metrics,
                "training_result": training_result,
            }
        except Exception as e:
            logger.error("hybrid_benchmark_failed", error=str(e))
            return {
                "architecture": "hybrid_cnn_lstm",
                "status": "failed",
                "error": str(e),
            }

    def benchmark_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Benchmark XGBoost baseline."""
        logger.info("benchmarking_xgboost")
        
        try:
            import xgboost as xgb  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("xgboost_not_available")
            return {
                "architecture": "xgboost",
                "status": "failed",
                "error": "XGBoost not installed",
            }
        
        try:
            # Reshape to 2D if needed
            if len(X_train.shape) == 3:
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
            else:
                X_train_2d = X_train
                X_test_2d = X_test
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
            )
            
            # Train
            model.fit(X_train_2d, y_train)
            
            # Predict
            predictions = model.predict(X_test_2d)
            
            # Evaluate
            returns = predictions - y_test
            metrics = self.evaluator.evaluate_model(
                predictions=predictions,
                actuals=y_test,
                returns=returns,
            )
            
            return {
                "architecture": "xgboost",
                "metrics": metrics,
                "training_result": {"status": "success"},
            }
        except Exception as e:
            logger.error("xgboost_benchmark_failed", error=str(e))
            return {
                "architecture": "xgboost",
                "status": "failed",
                "error": str(e),
            }

    def benchmark_all(
        self,
        symbol: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
    ) -> Dict[str, Any]:
        """
        Benchmark all architectures.
        
        Args:
            symbol: Trading symbol
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            input_dim: Input dimension
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("benchmarking_all_architectures", symbol=symbol)
        
        results = {}
        
        # Benchmark LSTM
        results["lstm"] = self.benchmark_lstm(X_train, y_train, X_test, y_test, input_dim)
        
        # Benchmark Hybrid
        results["hybrid"] = self.benchmark_hybrid(X_train, y_train, X_test, y_test, input_dim)
        
        # Benchmark XGBoost
        results["xgboost"] = self.benchmark_xgboost(X_train, y_train, X_test, y_test)
        
        # Select best architecture
        best = self.select_best(results)
        
        return {
            "symbol": symbol,
            "results": results,
            "best_architecture": best,
        }

    def select_best(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select best architecture based on composite score.
        
        Args:
            results: Dictionary of architecture results
            
        Returns:
            Best architecture information
        """
        best_score = -float('inf')
        best_architecture = None
        best_metrics = None
        
        for arch_name, arch_result in results.items():
            if arch_result.get("status") == "failed":
                continue
            
            metrics = arch_result.get("metrics", {})
            
            # Calculate composite score
            sharpe = metrics.get("sharpe_ratio", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            max_drawdown = metrics.get("max_drawdown", 1.0)
            accuracy = metrics.get("accuracy", 0.0)
            
            composite_score = (
                0.4 * sharpe +
                0.3 * profit_factor +
                0.2 * (1.0 - min(max_drawdown, 1.0)) +
                0.1 * accuracy
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_architecture = arch_name
                best_metrics = metrics
        
        logger.info("best_architecture_selected", architecture=best_architecture, score=best_score)
        
        return {
            "architecture": best_architecture,
            "composite_score": best_score,
            "metrics": best_metrics,
        }

