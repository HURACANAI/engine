"""
Data Understanding - Mathematical Analysis

Understand data using covariance matrices, PCA, and statistical measures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.decomposition import PCA

logger = structlog.get_logger(__name__)


class DataUnderstanding:
    """
    Mathematical data understanding using statistics and linear algebra.
    
    Methods:
    - Covariance matrix analysis
    - PCA decomposition
    - Trend detection
    - Volatility analysis
    - Correlation analysis
    """
    
    def __init__(self):
        """Initialize data understanding module."""
        logger.info("data_understanding_initialized")
    
    def analyze_covariance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze covariance matrix.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with covariance analysis
        """
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Calculate covariance matrix
        covariance_matrix = np.cov(X_centered.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Explained variance
        explained_variance = eigenvalues / np.sum(eigenvalues)
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        analysis = {
            "covariance_matrix": covariance_matrix.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "explained_variance": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components_95": int(n_components_95),
            "total_variance": float(np.sum(eigenvalues)),
        }
        
        logger.info(
            "covariance_analysis_complete",
            n_features=X.shape[1],
            n_components_95=n_components_95,
        )
        
        return analysis
    
    def pca_decomposition(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Perform PCA decomposition.
        
        Args:
            X: Feature matrix
            n_components: Number of components (None = auto based on variance)
            variance_threshold: Variance threshold for auto selection
            
        Returns:
            Dictionary with PCA results
        """
        if n_components is None:
            # Auto-select based on variance threshold
            pca = PCA()
            pca.fit(X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        
        results = {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "X_transformed": X_transformed.tolist(),
            "pca_object": pca,  # Keep for inverse transform
        }
        
        logger.info(
            "pca_decomposition_complete",
            n_components=n_components,
            total_variance=float(np.sum(pca.explained_variance_ratio_)),
        )
        
        return results
    
    def detect_trends(
        self,
        series: pd.Series | np.ndarray,
        window: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect trends using linear regression.
        
        Args:
            series: Time series data
            window: Rolling window size
            
        Returns:
            Dictionary with trend analysis
        """
        if isinstance(series, pd.Series):
            series = series.values
        
        # Linear regression on time index
        n = len(series)
        x = np.arange(n)
        
        # Calculate slope (trend)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        # Rolling mean
        rolling_mean = pd.Series(series).rolling(window=window).mean().values
        
        # Trend strength (R-squared)
        trend_strength = r_value ** 2
        
        # Detrended series
        trend_line = slope * x + intercept
        detrended = series - trend_line
        
        analysis = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(trend_strength),
            "p_value": float(p_value),
            "std_error": float(std_err),
            "trend_direction": "upward" if slope > 0 else "downward" if slope < 0 else "flat",
            "trend_strength": "strong" if trend_strength > 0.7 else "moderate" if trend_strength > 0.3 else "weak",
            "rolling_mean": rolling_mean.tolist(),
            "detrended_series": detrended.tolist(),
        }
        
        logger.info(
            "trend_analysis_complete",
            slope=slope,
            trend_strength=trend_strength,
            trend_direction=analysis["trend_direction"],
        )
        
        return analysis
    
    def analyze_volatility(
        self,
        returns: pd.Series | np.ndarray,
        window: int = 20,
    ) -> Dict[str, Any]:
        """
        Analyze volatility using statistical measures.
        
        Args:
            returns: Returns series
            window: Rolling window size
            
        Returns:
            Dictionary with volatility analysis
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        # Rolling volatility (standard deviation)
        returns_series = pd.Series(returns)
        rolling_vol = returns_series.rolling(window=window).std().values
        
        # Annualized volatility
        annualized_vol = np.std(returns) * np.sqrt(252)  # Assuming daily returns
        
        # Volatility clustering (autocorrelation of squared returns)
        squared_returns = returns ** 2
        autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        
        analysis = {
            "volatility_mean": float(np.mean(rolling_vol)),
            "volatility_std": float(np.std(rolling_vol)),
            "annualized_volatility": float(annualized_vol),
            "volatility_clustering": float(autocorr),
            "rolling_volatility": rolling_vol.tolist(),
            "has_clustering": abs(autocorr) > 0.1,
        }
        
        logger.info(
            "volatility_analysis_complete",
            annualized_vol=annualized_vol,
            has_clustering=analysis["has_clustering"],
        )
        
        return analysis
    
    def analyze_correlations(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze feature correlations.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Dictionary with correlation analysis
        """
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        "feature_i": feature_names[i] if feature_names else f"feature_{i}",
                        "feature_j": feature_names[j] if feature_names else f"feature_{j}",
                        "correlation": float(corr),
                    })
        
        analysis = {
            "correlation_matrix": correlation_matrix.tolist(),
            "mean_correlation": float(np.mean(correlation_matrix[correlation_matrix != 1.0])),
            "max_correlation": float(np.max(correlation_matrix[correlation_matrix != 1.0])),
            "min_correlation": float(np.min(correlation_matrix)),
            "high_correlation_pairs": high_corr_pairs,
        }
        
        logger.info(
            "correlation_analysis_complete",
            num_high_corr_pairs=len(high_corr_pairs),
        )
        
        return analysis
    
    def comprehensive_analysis(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive mathematical analysis of data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            feature_names: Names of features
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("starting_comprehensive_data_analysis")
        
        results = {}
        
        # Covariance analysis
        results["covariance"] = self.analyze_covariance(X, feature_names)
        
        # PCA decomposition
        results["pca"] = self.pca_decomposition(X, variance_threshold=0.95)
        
        # Correlation analysis
        results["correlations"] = self.analyze_correlations(X, feature_names)
        
        # Target analysis (if available)
        if y is not None:
            results["target_statistics"] = {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "skewness": float(stats.skew(y)),
                "kurtosis": float(stats.kurtosis(y)),
            }
            
            # Trend analysis if y is time series
            results["target_trend"] = self.detect_trends(y)
            
            # Volatility analysis if y is returns
            if len(y) > 1:
                returns = np.diff(y) / y[:-1]
                results["target_volatility"] = self.analyze_volatility(returns)
        
        logger.info("comprehensive_data_analysis_complete")
        
        return results

