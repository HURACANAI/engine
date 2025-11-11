"""
Shared Encoder for Cross-Coin Learning

Trains a shared feature encoder on all coins to reuse patterns without coupling.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import structlog

logger = structlog.get_logger(__name__)


class SharedEncoder:
    """Shared feature encoder trained on all coins."""
    
    def __init__(
        self,
        encoder_type: str = "pca",  # "pca" or "autoencoder"
        n_components: int = 50,
        random_state: int = 42,
    ):
        """Initialize shared encoder.
        
        Args:
            encoder_type: Type of encoder ("pca" or "autoencoder")
            n_components: Number of components/features
            random_state: Random state for reproducibility
        """
        self.encoder_type = encoder_type
        self.n_components = n_components
        self.random_state = random_state
        
        self.encoder: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.trained_at: Optional[datetime] = None
        
        logger.info("shared_encoder_initialized", encoder_type=encoder_type, n_components=n_components)
    
    def fit(
        self,
        all_features: Dict[str, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit encoder on all coin features.
        
        Args:
            all_features: Dictionary mapping symbol to feature DataFrame
            feature_names: List of feature names (optional)
        """
        logger.info("fitting_shared_encoder", symbols=len(all_features))
        
        # Combine all features
        combined_features = []
        for symbol, features_df in all_features.items():
            if features_df.empty:
                continue
            
            # Extract feature values
            if feature_names:
                feature_values = features_df[feature_names].values
            else:
                feature_values = features_df.values
            
            combined_features.append(feature_values)
        
        if not combined_features:
            raise ValueError("No features provided for training")
        
        # Concatenate all features
        X_all = np.vstack(combined_features)
        
        logger.info("combined_features", total_samples=X_all.shape[0], total_features=X_all.shape[1])
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)
        
        # Fit encoder
        if self.encoder_type == "pca":
            self.encoder = PCA(n_components=self.n_components, random_state=self.random_state)
            self.encoder.fit(X_scaled)
        elif self.encoder_type == "autoencoder":
            # TODO: Implement autoencoder
            raise NotImplementedError("Autoencoder not yet implemented")
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_all.shape[1])]
        self.trained_at = datetime.now(timezone.utc)
        
        # Log explained variance
        if self.encoder_type == "pca":
            explained_variance = self.encoder.explained_variance_ratio_.sum()
            logger.info("encoder_fitted", 
                       explained_variance=explained_variance,
                       n_components=self.n_components)
    
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using encoder.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Encoded features as numpy array
        """
        if self.encoder is None or self.scaler is None:
            raise ValueError("Encoder not fitted")
        
        # Extract feature values
        if self.feature_names:
            feature_values = features[self.feature_names].values
        else:
            feature_values = features.values
        
        # Scale and transform
        X_scaled = self.scaler.transform(feature_values)
        X_encoded = self.encoder.transform(X_scaled)
        
        return X_encoded
    
    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit and transform features (for single coin).
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Encoded features as numpy array
        """
        # This assumes encoder is already fitted on all coins
        return self.transform(features)
    
    def save(self, path: str) -> None:
        """Save encoder to disk.
        
        Args:
            path: Path to save encoder
        """
        encoder_data = {
            "encoder": self.encoder,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "encoder_type": self.encoder_type,
            "n_components": self.n_components,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        logger.info("encoder_saved", path=path)
    
    def load(self, path: str) -> None:
        """Load encoder from disk.
        
        Args:
            path: Path to load encoder from
        """
        with open(path, 'rb') as f:
            encoder_data = pickle.load(f)
        
        self.encoder = encoder_data["encoder"]
        self.scaler = encoder_data["scaler"]
        self.feature_names = encoder_data["feature_names"]
        self.encoder_type = encoder_data["encoder_type"]
        self.n_components = encoder_data["n_components"]
        self.trained_at = datetime.fromisoformat(encoder_data["trained_at"]) if encoder_data.get("trained_at") else None
        
        logger.info("encoder_loaded", path=path)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from encoder.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.encoder is None:
            return {}
        
        if self.encoder_type == "pca":
            # For PCA, use explained variance ratio
            importance = {}
            for i, var_ratio in enumerate(self.encoder.explained_variance_ratio_):
                importance[f"component_{i}"] = float(var_ratio)
            return importance
        else:
            return {}

