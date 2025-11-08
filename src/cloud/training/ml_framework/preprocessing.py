"""
Pre-processing Layer

Handles data normalization, feature engineering, and PCA-based dimensionality reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = structlog.get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Normalization
    normalize: bool = True
    scaler_type: str = "standard"  # "standard" or "minmax"
    
    # PCA
    use_pca: bool = True
    pca_variance_threshold: float = 0.95  # Keep 95% variance
    
    # Feature engineering
    engineer_features: bool = True
    
    # Missing data
    handle_missing: bool = True
    missing_strategy: str = "mean"  # "mean", "median", "most_frequent", "constant"
    
    # Outliers
    handle_outliers: bool = True
    outlier_method: str = "clip"  # "clip", "remove", "winsorize"
    outlier_threshold: float = 3.0  # Z-score threshold
    
    # Timestamp alignment
    align_timestamps: bool = True


class PreprocessingPipeline:
    """
    Preprocessing pipeline for financial time-series data.
    
    Handles:
    - Data normalization (StandardScaler or MinMaxScaler)
    - Feature engineering (moving averages, RSI, MACD, Bollinger bands, etc.)
    - PCA-based dimensionality reduction
    - Missing data handling
    - Outlier detection and handling
    - Timestamp alignment
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.pca: Optional[PCA] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_names: Optional[List[str]] = None
        self.pca_components: Optional[int] = None
        
        logger.info(
            "preprocessing_pipeline_initialized",
            normalize=config.normalize,
            use_pca=config.use_pca,
            engineer_features=config.engineer_features,
        )
    
    def fit(self, X: pd.DataFrame) -> PreprocessingPipeline:
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X: Training features
            
        Returns:
            Self for chaining
        """
        logger.info("fitting_preprocessing_pipeline", samples=len(X), features=len(X.columns))
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing data
        if self.config.handle_missing:
            self.imputer = SimpleImputer(strategy=self.config.missing_strategy)
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
        else:
            X_processed = X.copy()
        
        # Handle outliers
        if self.config.handle_outliers:
            X_processed = self._handle_outliers(X_processed)
        
        # Feature engineering (if enabled)
        if self.config.engineer_features:
            X_processed = self._engineer_features(X_processed)
        
        # Normalization
        if self.config.normalize:
            if self.config.scaler_type == "standard":
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index,
            )
        else:
            X_scaled = X_processed
        
        # PCA
        if self.config.use_pca:
            self.pca = PCA(n_components=self.config.pca_variance_threshold)
            X_pca = self.pca.fit_transform(X_scaled)
            self.pca_components = self.pca.n_components_
            
            logger.info(
                "pca_fitted",
                original_features=len(X_scaled.columns),
                pca_components=self.pca_components,
                explained_variance=self.pca.explained_variance_ratio_.sum(),
            )
        else:
            self.pca_components = len(X_scaled.columns)
        
        logger.info("preprocessing_pipeline_fitted", final_features=self.pca_components)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if self.imputer is None or self.scaler is None:
            raise ValueError("Pipeline must be fitted before transformation")
        
        # Handle missing data
        X_processed = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index,
        )
        
        # Handle outliers
        if self.config.handle_outliers:
            X_processed = self._handle_outliers(X_processed)
        
        # Feature engineering
        if self.config.engineer_features:
            X_processed = self._engineer_features(X_processed)
        
        # Normalization
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index,
        )
        
        # PCA
        if self.config.use_pca and self.pca is not None:
            X_transformed = pd.DataFrame(
                self.pca.transform(X_scaled),
                columns=[f"pca_{i}" for i in range(self.pca.n_components_)],
                index=X_scaled.index,
            )
        else:
            X_transformed = X_scaled
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        X_clean = X.copy()
        
        if self.config.outlier_method == "clip":
            # Clip outliers to threshold
            for col in X_clean.columns:
                if X_clean[col].dtype in [np.float64, np.int64]:
                    mean = X_clean[col].mean()
                    std = X_clean[col].std()
                    if std > 0:
                        lower = mean - self.config.outlier_threshold * std
                        upper = mean + self.config.outlier_threshold * std
                        X_clean[col] = X_clean[col].clip(lower=lower, upper=upper)
        
        elif self.config.outlier_method == "remove":
            # Remove rows with outliers (more aggressive)
            for col in X_clean.columns:
                if X_clean[col].dtype in [np.float64, np.int64]:
                    mean = X_clean[col].mean()
                    std = X_clean[col].std()
                    if std > 0:
                        z_scores = np.abs((X_clean[col] - mean) / std)
                        X_clean = X_clean[z_scores < self.config.outlier_threshold]
        
        elif self.config.outlier_method == "winsorize":
            # Winsorize (cap at percentiles)
            for col in X_clean.columns:
                if X_clean[col].dtype in [np.float64, np.int64]:
                    lower = X_clean[col].quantile(0.01)
                    upper = X_clean[col].quantile(0.99)
                    X_clean[col] = X_clean[col].clip(lower=lower, upper=upper)
        
        return X_clean
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing ones.
        
        This extends the existing FeatureRecipe with additional engineering.
        """
        X_eng = X.copy()
        
        # Look for common financial columns
        if "close" in X_eng.columns:
            # Moving averages (if close price exists)
            for window in [5, 10, 20, 50]:
                if f"close_ma_{window}" not in X_eng.columns:
                    X_eng[f"close_ma_{window}"] = X_eng["close"].rolling(window=window).mean()
            
            # RSI (if not already present)
            if "rsi" not in X_eng.columns:
                X_eng["rsi"] = self._calculate_rsi(X_eng["close"])
            
            # MACD (if not already present)
            if "macd" not in X_eng.columns:
                macd, signal = self._calculate_macd(X_eng["close"])
                X_eng["macd"] = macd
                X_eng["macd_signal"] = signal
                X_eng["macd_diff"] = macd - signal
            
            # Bollinger Bands (if not already present)
            if "bb_upper" not in X_eng.columns:
                upper, lower, width = self._calculate_bollinger_bands(X_eng["close"])
                X_eng["bb_upper"] = upper
                X_eng["bb_lower"] = lower
                X_eng["bb_width"] = width
        
        # Volatility features
        if "close" in X_eng.columns:
            for window in [10, 20, 30]:
                if f"volatility_{window}" not in X_eng.columns:
                    X_eng[f"volatility_{window}"] = X_eng["close"].pct_change().rolling(window=window).std()
        
        # Momentum features
        if "close" in X_eng.columns:
            for period in [1, 3, 5, 10]:
                if f"momentum_{period}" not in X_eng.columns:
                    X_eng[f"momentum_{period}"] = X_eng["close"].pct_change(periods=period)
        
        # Fill NaN values created by rolling windows
        X_eng = X_eng.fillna(method="bfill").fillna(method="ffill").fillna(0)
        
        return X_eng
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.fillna(0), signal.fillna(0)
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma
        return upper.fillna(prices), lower.fillna(prices), width.fillna(0)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get PCA component importance.
        
        Returns:
            Dictionary mapping PCA components to explained variance
        """
        if self.pca is None:
            return None
        
        importance = {}
        for i, variance in enumerate(self.pca.explained_variance_ratio_):
            importance[f"pca_{i}"] = float(variance)
        
        return importance

