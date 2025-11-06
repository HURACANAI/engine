"""
Advanced Feature Scaling

Multiple scaling methods for different use cases:
- RobustScaler: Handles outliers better
- QuantileTransformer: Normal distribution
- PowerTransformer: Handles skewness
- Regime-aware scaling: Different scaling per regime

Source: scikit-learn preprocessing best practices
Expected Impact: Better handling of outliers, improved model performance
"""

from typing import Dict, Optional, Any
import structlog  # type: ignore
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import (
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        QuantileTransformer,
        PowerTransformer,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = structlog.get_logger(__name__)


class AdvancedFeatureScaler:
    """
    Advanced feature scaling with multiple methods.
    
    Methods:
    - StandardScaler: Mean/std (standard)
    - RobustScaler: Median/IQR (handles outliers)
    - QuantileTransformer: Normal distribution
    - PowerTransformer: Handles skewness
    - Regime-aware: Different scaling per regime
    """

    def __init__(
        self,
        method: str = 'robust',  # 'standard', 'robust', 'quantile', 'power', 'regime_aware'
    ):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        self.method = method
        self.scalers: Dict[str, Any] = {}
        
        logger.info("advanced_feature_scaler_initialized", method=method)

    def fit_transform(
        self,
        X: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fit scaler and transform features.
        
        Args:
            X: Feature DataFrame
            regimes: Market regimes (for regime-aware scaling)
            
        Returns:
            Scaled DataFrame
        """
        if self.method == 'regime_aware' and regimes is not None:
            return self._fit_transform_regime_aware(X, regimes)
        
        # Standard scaling
        scaler = self._create_scaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store scaler
        self.scalers['default'] = scaler
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def transform(
        self,
        X: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if self.method == 'regime_aware' and regimes is not None:
            return self._transform_regime_aware(X, regimes)
        
        scaler = self.scalers.get('default')
        if scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _create_scaler(self):
        """Create scaler based on method."""
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'robust':
            return RobustScaler()
        elif self.method == 'quantile':
            return QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.method == 'power':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            return RobustScaler()  # Default

    def _fit_transform_regime_aware(
        self,
        X: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform with regime-aware scaling."""
        X_scaled = X.copy()
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_data = X[mask]
            
            if len(regime_data) > 0:
                scaler = self._create_scaler()
                regime_scaled = scaler.fit_transform(regime_data)
                X_scaled.loc[mask] = regime_scaled
                self.scalers[regime] = scaler
        
        return X_scaled

    def _transform_regime_aware(
        self,
        X: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """Transform with regime-aware scaling."""
        X_scaled = X.copy()
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_data = X[mask]
            
            if len(regime_data) > 0:
                scaler = self.scalers.get(regime)
                if scaler is None:
                    logger.warning("scaler_not_found_for_regime", regime=regime)
                    continue
                
                regime_scaled = scaler.transform(regime_data)
                X_scaled.loc[mask] = regime_scaled
        
        return X_scaled

