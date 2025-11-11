"""
Feature Autoencoder - Automatic Signal Extraction

Deep learning replaces manual feature engineering by automatically extracting signals.
Before forecasting, feed all raw candle and derived indicators through an autoencoder
to compress into latent features. Store encodings in Brain Library for reuse.

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import structlog

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = structlog.get_logger(__name__)


@dataclass
class EncodedFeatures:
    """Encoded features from autoencoder."""
    original_features: np.ndarray
    encoded_features: np.ndarray  # Latent representation
    reconstruction: np.ndarray
    reconstruction_error: float


class FeatureAutoencoder:
    """
    Autoencoder for automatic feature extraction.
    
    Compresses raw features into latent representations that capture
    the most important signal patterns.
    
    Usage:
        autoencoder = FeatureAutoencoder(input_dim=100, latent_dim=32)
        
        # Train on historical data
        autoencoder.train(features_df)
        
        # Encode new features
        encoded = autoencoder.encode(features_df)
        
        # Use encoded features for forecasting
        predictions = model.predict(encoded.encoded_features)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        use_pytorch: bool = True
    ):
        """
        Initialize feature autoencoder.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent representation (default: 32)
            hidden_dims: Hidden layer dimensions (default: [64, 32])
            use_pytorch: Whether to use PyTorch (default: True)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.use_pytorch = use_pytorch and HAS_TORCH
        
        self.encoder = None
        self.decoder = None
        self.is_trained = False
        
        if self.use_pytorch:
            self._build_pytorch_model()
        else:
            self._build_simple_model()
        
        logger.info(
            "feature_autoencoder_initialized",
            input_dim=input_dim,
            latent_dim=latent_dim,
            use_pytorch=self.use_pytorch
        )
    
    def _build_pytorch_model(self) -> None:
        """Build PyTorch autoencoder model."""
        if not HAS_TORCH:
            logger.warning("pytorch_not_available_falling_back_to_simple")
            self._build_simple_model()
            return
        
        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info("pytorch_autoencoder_built")
    
    def _build_simple_model(self) -> None:
        """Build simple autoencoder using sklearn or numpy."""
        # For now, use PCA-like compression
        # In production, use sklearn's MLPRegressor or similar
        logger.info("simple_autoencoder_built_using_pca_like_compression")
    
    def train(
        self,
        features: pl.DataFrame | pd.DataFrame | np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> None:
        """
        Train autoencoder on features.
        
        Args:
            features: Feature matrix (samples x features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        # Convert to numpy
        if isinstance(features, pl.DataFrame):
            X = features.to_numpy()
        elif isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension {X.shape[1]} does not match input_dim {self.input_dim}"
            )
        
        # Normalize features
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std
        
        if self.use_pytorch and HAS_TORCH:
            self._train_pytorch(X_norm, epochs, batch_size, learning_rate)
        else:
            self._train_simple(X_norm)
        
        self.is_trained = True
        logger.info(
            "autoencoder_trained",
            epochs=epochs,
            samples=X.shape[0]
        )
    
    def _train_pytorch(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> None:
        """Train PyTorch autoencoder."""
        if not HAS_TORCH:
            return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(device)
        self.decoder.to(device)
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            total_loss = 0.0
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = X_tensor[batch_indices]
                
                # Forward pass
                encoded = self.encoder(batch)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (len(X) // batch_size)
                logger.debug("training_epoch", epoch=epoch+1, loss=avg_loss)
    
    def _train_simple(self, X: np.ndarray) -> None:
        """Train simple autoencoder (PCA-like)."""
        # Store for simple encoding/decoding
        self.X_train = X
        logger.info("simple_autoencoder_trained_using_pca_like_compression")
    
    def encode(
        self,
        features: pl.DataFrame | pd.DataFrame | np.ndarray
    ) -> EncodedFeatures:
        """
        Encode features into latent representation.
        
        Args:
            features: Feature matrix to encode
        
        Returns:
            EncodedFeatures object with original, encoded, and reconstructed features
        """
        if not self.is_trained:
            raise RuntimeError("Autoencoder must be trained before encoding")
        
        # Convert to numpy
        if isinstance(features, pl.DataFrame):
            X = features.to_numpy()
        elif isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features
        
        # Normalize
        X_norm = (X - self.feature_mean) / self.feature_std
        
        if self.use_pytorch and HAS_TORCH:
            return self._encode_pytorch(X_norm, X)
        else:
            return self._encode_simple(X_norm, X)
    
    def _encode_pytorch(
        self,
        X_norm: np.ndarray,
        X_original: np.ndarray
    ) -> EncodedFeatures:
        """Encode using PyTorch model."""
        if not HAS_TORCH:
            return self._encode_simple(X_norm, X_original)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(device)
            encoded = self.encoder(X_tensor).cpu().numpy()
            reconstructed = self.decoder(self.encoder(X_tensor)).cpu().numpy()
        
        # Denormalize reconstruction
        reconstructed_denorm = reconstructed * self.feature_std + self.feature_mean
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((X_original - reconstructed_denorm) ** 2)
        
        return EncodedFeatures(
            original_features=X_original,
            encoded_features=encoded,
            reconstruction=reconstructed_denorm,
            reconstruction_error=reconstruction_error
        )
    
    def _encode_simple(
        self,
        X_norm: np.ndarray,
        X_original: np.ndarray
    ) -> EncodedFeatures:
        """Encode using simple compression."""
        # Simple PCA-like compression (first N components)
        # In production, use actual trained model
        if X_norm.shape[1] > self.latent_dim:
            encoded = X_norm[:, :self.latent_dim]
            # Simple reconstruction (pad with zeros)
            reconstructed_norm = np.pad(
                encoded,
                ((0, 0), (0, X_norm.shape[1] - self.latent_dim)),
                mode='constant'
            )
        else:
            encoded = X_norm
            reconstructed_norm = X_norm
        
        # Denormalize
        reconstructed = reconstructed_norm * self.feature_std + self.feature_mean
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((X_original - reconstructed) ** 2)
        
        return EncodedFeatures(
            original_features=X_original,
            encoded_features=encoded,
            reconstruction=reconstructed,
            reconstruction_error=reconstruction_error
        )
    
    def save(self, path: str) -> None:
        """Save autoencoder to disk."""
        if self.use_pytorch and HAS_TORCH:
            torch.save({
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
                'feature_mean': self.feature_mean,
                'feature_std': self.feature_std
            }, path)
        else:
            # Save simple model
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'input_dim': self.input_dim,
                    'latent_dim': self.latent_dim,
                    'feature_mean': self.feature_mean,
                    'feature_std': self.feature_std,
                    'X_train': getattr(self, 'X_train', None)
                }, f)
        
        logger.info("autoencoder_saved", path=path)
    
    def load(self, path: str) -> None:
        """Load autoencoder from disk."""
        if self.use_pytorch and HAS_TORCH:
            checkpoint = torch.load(path)
            self.input_dim = checkpoint['input_dim']
            self.latent_dim = checkpoint['latent_dim']
            self.hidden_dims = checkpoint['hidden_dims']
            self._build_pytorch_model()
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.feature_mean = checkpoint['feature_mean']
            self.feature_std = checkpoint['feature_std']
        else:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.input_dim = data['input_dim']
                self.latent_dim = data['latent_dim']
                self.feature_mean = data['feature_mean']
                self.feature_std = data['feature_std']
                self.X_train = data.get('X_train')
        
        self.is_trained = True
        logger.info("autoencoder_loaded", path=path)

