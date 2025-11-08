"""
Clustering Models for Unsupervised Learning

Implements K-Means clustering for market regime detection and volatility clustering.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base import BaseModel, ModelConfig

logger = structlog.get_logger(__name__)


class KMeansClustering(BaseModel):
    """
    K-Means clustering for unsupervised learning.
    
    Used for:
    - Market regime detection (bullish, bearish, neutral)
    - Volatility clustering
    - Feature space exploration
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default hyperparameters
        if "n_clusters" not in config.hyperparameters:
            config.hyperparameters["n_clusters"] = 3
        if "random_state" not in config.hyperparameters:
            config.hyperparameters["random_state"] = 42
        if "n_init" not in config.hyperparameters:
            config.hyperparameters["n_init"] = 10
        if "max_iter" not in config.hyperparameters:
            config.hyperparameters["max_iter"] = 300
        
        self.n_clusters = config.hyperparameters["n_clusters"]
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.silhouette_score: float = 0.0
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,  # Not used for clustering
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train K-Means model."""
        logger.info("training_kmeans", samples=len(X), n_clusters=self.n_clusters)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        # Create model
        self.model = KMeans(**self.config.hyperparameters)
        
        # Train
        self.model.fit(X)
        self.is_trained = True
        
        # Get cluster labels and centers
        self.cluster_labels = self.model.labels_
        self.cluster_centers = self.model.cluster_centers_
        
        # Calculate silhouette score
        self.silhouette_score = silhouette_score(X, self.cluster_labels)
        
        logger.info(
            "kmeans_trained",
            n_clusters=self.n_clusters,
            silhouette_score=self.silhouette_score,
        )
        
        # Return dummy metrics (clustering doesn't have traditional metrics)
        from .base import ModelMetrics
        return ModelMetrics(
            accuracy=self.silhouette_score,  # Use silhouette as proxy
        )
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> ModelMetrics:
        """Evaluate clustering quality."""
        predictions = self.predict(X)
        
        # Calculate silhouette score
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        silhouette = silhouette_score(X, predictions)
        
        from .base import ModelMetrics
        return ModelMetrics(
            accuracy=silhouette,  # Use silhouette as proxy
        )
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting centers")
        return self.cluster_centers
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels for training data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting labels")
        return self.cluster_labels
    
    def get_cluster_statistics(self, X: pd.DataFrame | np.ndarray) -> Dict[str, Any]:
        """Get statistics for each cluster."""
        predictions = self.predict(X)
        
        stats = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = predictions == cluster_id
            cluster_data = X[cluster_mask] if isinstance(X, np.ndarray) else X.iloc[cluster_mask]
            
            stats[f"cluster_{cluster_id}"] = {
                "size": int(np.sum(cluster_mask)),
                "percentage": float(np.sum(cluster_mask) / len(predictions) * 100),
                "center": self.cluster_centers[cluster_id].tolist() if self.cluster_centers is not None else None,
            }
        
        return stats
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("kmeans_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        self.cluster_centers = self.model.cluster_centers_
        logger.info("kmeans_loaded", path=str(path))

