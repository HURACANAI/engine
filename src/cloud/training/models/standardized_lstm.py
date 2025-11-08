"""Standardized LSTM architecture with attention mechanism."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class StandardizedLSTM:
    """
    Standardized bidirectional stacked LSTM with:
    - Dropout layers
    - Layer normalization
    - Attention mechanism
    - Automatic input scaling
    """

    def __init__(
        self,
        input_dim: int,
        lstm_units: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        attention_dim: int = 64,
        output_dim: int = 1,
        use_bidirectional: bool = True,
    ) -> None:
        """
        Initialize standardized LSTM.
        
        Args:
            input_dim: Input feature dimension
            lstm_units: Number of LSTM units per layer
            num_layers: Number of stacked LSTM layers
            dropout_rate: Dropout rate (0.2-0.3 recommended)
            attention_dim: Attention mechanism dimension
            output_dim: Output dimension
            use_bidirectional: Whether to use bidirectional LSTM
        """
        self.input_dim = input_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.use_bidirectional = use_bidirectional
        
        # Model will be initialized when fit is called
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        logger.info(
            "standardized_lstm_initialized",
            input_dim=input_dim,
            lstm_units=lstm_units,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_bidirectional=use_bidirectional,
        )

    def _create_model(self) -> Any:
        """
        Create standardized LSTM model.
        
        Architecture:
        - Input layer
        - Bidirectional stacked LSTMs with dropout
        - Layer normalization
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
        
        # Bidirectional stacked LSTMs with dropout
        x = inputs
        for i in range(self.num_layers):
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
            
            # Dropout
            if i < self.num_layers - 1:  # No dropout on last layer before attention
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
        Train the LSTM model.
        
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
        logger.info("lstm_training_started", samples=len(X), epochs=epochs)
        
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
            logger.error("model_creation_failed", message="Could not create LSTM model")
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
            
            logger.info("lstm_training_complete", **metrics)
            
            return {
                "status": "success",
                "metrics": metrics,
                "history": history.history,
            }
        except Exception as e:
            logger.error("lstm_training_failed", error=str(e))
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

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.
        
        Args:
            X: Input features (samples, timesteps, features)
            
        Returns:
            Attention weights
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # This would extract attention weights from the model
        # For now, return placeholder
        logger.debug("getting_attention_weights", samples=len(X))
        return np.ones((len(X), X.shape[1])) / X.shape[1]  # Uniform weights as placeholder

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "standardized_lstm",
            "input_dim": self.input_dim,
            "lstm_units": self.lstm_units,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "attention_dim": self.attention_dim,
            "use_bidirectional": self.use_bidirectional,
            "purpose": "Time series prediction with attention mechanism",
            "ideal_dataset_shape": "(samples, timesteps, features)",
            "feature_requirements": ["time_series_data"],
            "output_schema": {"predictions": "array"},
            "market_regimes": ["trending", "ranging", "volatile"],
        }

