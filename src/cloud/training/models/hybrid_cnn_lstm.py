"""Hybrid CNN-LSTM model for pattern extraction and sequence memory."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class HybridCNNLSTM:
    """
    Hybrid model combining CNN (pattern extraction) + LSTM (sequence memory).
    
    Architecture:
    1. CNN layers: Extract momentum and micro-patterns
    2. LSTM layers: Capture time dependencies
    3. Attention mechanism: Weight important features
    4. Dense output: Final prediction
    """

    def __init__(
        self,
        input_dim: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        lstm_units: int = 128,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        attention_dim: int = 64,
        output_dim: int = 1,
        use_bidirectional: bool = True,
    ) -> None:
        """
        Initialize hybrid CNN-LSTM model.
        
        Args:
            input_dim: Input feature dimension
            cnn_filters: Number of CNN filters
            cnn_kernel_size: CNN kernel size
            cnn_pool_size: CNN pooling size
            lstm_units: Number of LSTM units per layer
            num_lstm_layers: Number of stacked LSTM layers
            dropout_rate: Dropout rate
            attention_dim: Attention mechanism dimension
            output_dim: Output dimension
            use_bidirectional: Whether to use bidirectional LSTM
        """
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_pool_size = cnn_pool_size
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.use_bidirectional = use_bidirectional
        
        # Model will be initialized when fit is called
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        logger.info(
            "hybrid_cnn_lstm_initialized",
            input_dim=input_dim,
            cnn_filters=cnn_filters,
            lstm_units=lstm_units,
            num_lstm_layers=num_lstm_layers,
        )

    def _create_model(self) -> Any:
        """
        Create hybrid CNN-LSTM model.
        
        Architecture:
        - Input layer
        - CNN layers (pattern extraction)
        - LSTM layers (sequence memory)
        - Attention mechanism
        - Dense output layer
        """
        try:
            import tensorflow as tf  # type: ignore[reportMissingImports]
            from tensorflow import keras  # type: ignore[reportMissingImports]
            from tensorflow.keras import layers  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("tensorflow_not_available", message="TensorFlow not installed - using placeholder")
            return None
        
        # Input layer
        inputs = keras.Input(shape=(None, self.input_dim))
        
        # CNN layers for pattern extraction
        x = layers.Conv1D(
            filters=self.cnn_filters,
            kernel_size=self.cnn_kernel_size,
            activation='relu',
            padding='same',
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=self.cnn_pool_size)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second CNN layer
        x = layers.Conv1D(
            filters=self.cnn_filters * 2,
            kernel_size=self.cnn_kernel_size,
            activation='relu',
            padding='same',
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=self.cnn_pool_size)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # LSTM layers for sequence memory
        for i in range(self.num_lstm_layers):
            lstm_layer = layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                return_state=False,
            )
            
            if self.use_bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer)
            
            x = lstm_layer(x)
            
            # Layer normalization
            x = layers.LayerNormalization()(x)
            
            # Dropout (except on last layer)
            if i < self.num_lstm_layers - 1:
                x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        attention_output = self._attention_layer(x, self.attention_dim)
        
        # Dense output layer
        outputs = layers.Dense(self.output_dim, activation='linear')(attention_output)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae'],
        )
        
        return model

    def _attention_layer(self, inputs: Any, attention_dim: int) -> Any:
        """
        Create attention layer.
        
        Args:
            inputs: Input tensor
            attention_dim: Attention dimension
            
        Returns:
            Attention-weighted output
        """
        try:
            from tensorflow import keras  # type: ignore[reportMissingImports]
            from tensorflow.keras import layers  # type: ignore[reportMissingImports]
        except ImportError:
            return inputs
        
        # Attention mechanism
        attention = layers.Dense(attention_dim, activation='tanh')(inputs)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Multiply()([inputs, attention])
        attention = layers.GlobalAveragePooling1D()(attention)
        
        return attention

    def _scale_inputs(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Scale inputs using z-score normalization.
        
        Args:
            X: Input data
            fit: Whether to fit the scaler
            
        Returns:
            Scaled data
        """
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("sklearn_not_available", message="sklearn not installed - skipping scaling")
            return X
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Reshape for scaling (flatten time and features)
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            X_scaled = self.scaler.transform(X_flat)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the hybrid CNN-LSTM model.
        
        Args:
            X: Training features (samples, timesteps, features)
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history and metrics
        """
        logger.info("hybrid_cnn_lstm_training_started", samples=len(X), epochs=epochs)
        
        # Scale inputs
        X_scaled = self._scale_inputs(X, fit=True)
        if X_val is not None:
            X_val_scaled = self._scale_inputs(X_val, fit=False)
        else:
            X_val_scaled = None
        
        # Create model
        if self.model is None:
            self.model = self._create_model()
        
        if self.model is None:
            logger.error("model_creation_failed", message="Could not create hybrid CNN-LSTM model")
            return {"status": "failed", "error": "Model creation failed"}
        
        # Prepare validation data
        validation_data = None
        if X_val_scaled is not None and y_val is not None:
            validation_data = (X_val_scaled, y_val)
        
        # Train model
        try:
            history = self.model.fit(
                X_scaled,
                y,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                shuffle=False,  # Important for time series
            )
            
            # Get final metrics
            train_loss = history.history['loss'][-1]
            val_loss = history.history.get('val_loss', [None])[-1]
            
            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss else None,
                "epochs": epochs,
            }
            
            logger.info("hybrid_cnn_lstm_training_complete", **metrics)
            
            return {
                "status": "success",
                "metrics": metrics,
                "history": history.history,
            }
        except Exception as e:
            logger.error("hybrid_cnn_lstm_training_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (samples, timesteps, features)
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Scale inputs
        X_scaled = self._scale_inputs(X, fit=False)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        
        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "hybrid_cnn_lstm",
            "input_dim": self.input_dim,
            "cnn_filters": self.cnn_filters,
            "lstm_units": self.lstm_units,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout_rate": self.dropout_rate,
            "attention_dim": self.attention_dim,
            "use_bidirectional": self.use_bidirectional,
            "purpose": "Pattern extraction (CNN) + sequence memory (LSTM)",
            "ideal_dataset_shape": "(samples, timesteps, features)",
            "feature_requirements": ["time_series_data"],
            "output_schema": {"predictions": "array"},
            "market_regimes": ["trending", "ranging", "volatile"],
        }

