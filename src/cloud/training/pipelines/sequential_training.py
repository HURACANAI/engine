"""Sequential training pipeline with automated preprocessing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class SequentialTrainingPipeline:
    """
    Automated sequential training process:
    1. Data preprocessing
    2. Windowing and reshaping
    3. Scaling (per-asset normalization)
    4. Validation split (forward-only)
    5. Model training
    """

    def __init__(
        self,
        window_size: int = 64,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        scale_per_asset: bool = True,
    ) -> None:
        """
        Initialize sequential training pipeline.
        
        Args:
            window_size: Size of time window
            train_split: Training data split (0.0-1.0)
            val_split: Validation data split (0.0-1.0)
            test_split: Test data split (0.0-1.0)
            scale_per_asset: Whether to scale per asset
        """
        self.window_size = window_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.scale_per_asset = scale_per_asset
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        logger.info(
            "sequential_training_pipeline_initialized",
            window_size=window_size,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
    ) -> pd.DataFrame:
        """
        Preprocess data.
        
        Args:
            data: Raw data DataFrame
            target_column: Target column name
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("preprocessing_data", rows=len(data), columns=list(data.columns))
        
        # Copy data
        processed = data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(processed['timestamp']):
                processed['timestamp'] = pd.to_datetime(processed['timestamp'])
            processed = processed.sort_values('timestamp').reset_index(drop=True)
        
        # Remove NaN values (forward fill, then backward fill)
        processed = processed.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN rows
        processed = processed.dropna()
        
        logger.info("preprocessing_complete", rows=len(processed))
        
        return processed

    def create_windows(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time series data.
        
        Args:
            data: Preprocessed DataFrame
            target_column: Target column name
            
        Returns:
            Tuple of (X_windows, y_targets)
        """
        logger.info("creating_windows", window_size=self.window_size, data_length=len(data))
        
        # Get feature columns (exclude timestamp and target)
        feature_columns = [col for col in data.columns if col not in ['timestamp', target_column]]
        
        if not feature_columns:
            # If no features, use all numeric columns except target
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col != target_column]
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Create windows
        X_windows = []
        y_targets = []
        
        for i in range(self.window_size, len(X)):
            X_window = X[i - self.window_size:i]
            y_target = y[i]
            
            X_windows.append(X_window)
            y_targets.append(y_target)
        
        X_windows = np.array(X_windows)
        y_targets = np.array(y_targets)
        
        logger.info("windows_created", num_windows=len(X_windows), window_shape=X_windows.shape)
        
        return X_windows, y_targets

    def scale_per_asset(
        self,
        X: np.ndarray,
        fit: bool = False,
        scaler: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Scale data per asset (feature-wise normalization).
        
        Args:
            X: Input data (samples, timesteps, features)
            fit: Whether to fit the scaler
            scaler: Optional pre-fitted scaler
            
        Returns:
            Tuple of (scaled_X, scaler)
        """
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("sklearn_not_available", message="sklearn not installed - skipping scaling")
            return X, None
        
        if scaler is None:
            scaler = StandardScaler()
        
        # Reshape for scaling (flatten time and samples, keep features)
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_scaled = scaler.fit_transform(X_flat)
        else:
            X_scaled = scaler.transform(X_flat)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled, scaler

    def forward_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data forward-only (no shuffling).
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        total_samples = len(X)
        
        # Calculate split indices
        train_end = int(total_samples * self.train_split)
        val_end = int(total_samples * (self.train_split + self.val_split))
        
        # Split data
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logger.info(
            "forward_split_complete",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def run_pipeline(
        self,
        symbol: str,
        data: pd.DataFrame,
        model: Any,
        target_column: str = "close",
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Run complete sequential training pipeline.
        
        Args:
            symbol: Trading symbol
            data: Raw data DataFrame
            model: Model to train
            target_column: Target column name
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training results dictionary
        """
        logger.info("sequential_training_pipeline_started", symbol=symbol)
        
        # Step 1: Preprocess
        processed_data = self.preprocess(data, target_column=target_column)
        
        # Step 2: Create windows
        X_windows, y_targets = self.create_windows(processed_data, target_column=target_column)
        
        # Step 3: Scale (per-asset normalization)
        if self.scale_per_asset:
            X_scaled, scaler = self.scale_per_asset(X_windows, fit=True)
        else:
            X_scaled = X_windows
            scaler = None
        
        # Step 4: Forward split
        X_train, y_train, X_val, y_val, X_test, y_test = self.forward_split(X_scaled, y_targets)
        
        # Step 5: Train model
        try:
            if hasattr(model, 'fit'):
                # For Keras/TensorFlow models
                training_result = model.fit(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            else:
                # For sklearn models (reshape to 2D)
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_val_2d = X_val.reshape(X_val.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                
                model.fit(X_train_2d, y_train)
                training_result = {"status": "success"}
            
            # Get predictions
            if hasattr(model, 'predict'):
                if len(X_test.shape) == 3:  # 3D array (samples, timesteps, features)
                    predictions = model.predict(X_test)
                else:
                    X_test_2d = X_test.reshape(X_test.shape[0], -1)
                    predictions = model.predict(X_test_2d)
            else:
                predictions = None
            
            result = {
                "status": "success",
                "symbol": symbol,
                "training_result": training_result,
                "predictions": predictions,
                "actuals": y_test,
                "scaler": scaler,
                "data_info": {
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "window_size": self.window_size,
                },
            }
            
            logger.info("sequential_training_pipeline_complete", symbol=symbol)
            
            return result
            
        except Exception as e:
            logger.error("sequential_training_pipeline_failed", symbol=symbol, error=str(e))
            return {
                "status": "failed",
                "symbol": symbol,
                "error": str(e),
            }

