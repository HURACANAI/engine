"""
Enhanced Pre-processing - EDA, Feature Engineering, Trend Decomposition

Comprehensive preprocessing for financial time-series data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = structlog.get_logger(__name__)


class EnhancedPreprocessor:
    """
    Enhanced preprocessing pipeline for financial data.
    
    Features:
    - EDA and data quality checks
    - Feature cleaning and normalization
    - Outlier detection and handling
    - Feature engineering (family size, income bands, etc.)
    - Rolling-window normalization
    - Trend decomposition
    - Feature lagging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or {}
        self.scalers: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info("enhanced_preprocessor_initialized")
    
    def eda_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform Exploratory Data Analysis (EDA).
        
        Args:
            df: Input dataframe
            
        Returns:
            EDA results dictionary
        """
        logger.info("performing_eda_analysis", shape=df.shape)
        
        eda_results = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "numeric_stats": df.describe().to_dict(),
        }
        
        # Detect outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = {
                "count": ((df[col] < lower_bound) | (df[col] > upper_bound)).sum(),
                "percentage": ((df[col] < lower_bound) | (df[col] > upper_bound)).sum() / len(df) * 100,
            }
        
        eda_results["outliers"] = outliers
        
        logger.info("eda_analysis_complete", **{k: v for k, v in eda_results.items() if k != "numeric_stats"})
        return eda_results
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features (handle missing values, duplicates, etc.).
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("cleaning_features", initial_shape=df.shape)
        
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        missing_strategy = self.config.get("missing_strategy", "forward_fill")
        
        if missing_strategy == "forward_fill":
            df_clean = df_clean.ffill().bfill()
        elif missing_strategy == "mean":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif missing_strategy == "median":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif missing_strategy == "drop":
            df_clean = df_clean.dropna()
        
        logger.info("features_cleaned", final_shape=df_clean.shape)
        return df_clean
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers in data.
        
        Args:
            df: Input dataframe
            method: Detection method ("iqr", "zscore", "isolation_forest")
            threshold: Detection threshold
            
        Returns:
            Dictionary of column names to outlier indices
        """
        logger.info("detecting_outliers", method=method)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df[col]))
                outlier_mask = z_scores > threshold
            
            else:
                continue
            
            outliers[col] = np.where(outlier_mask)[0]
        
        logger.info("outliers_detected", num_outlier_cols=len(outliers))
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "clip",
        outliers: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Handle outliers.
        
        Args:
            df: Input dataframe
            method: Handling method ("clip", "remove", "winsorize")
            outliers: Pre-detected outliers (optional)
            
        Returns:
            Dataframe with outliers handled
        """
        logger.info("handling_outliers", method=method)
        
        df_handled = df.copy()
        
        if outliers is None:
            outliers = self.detect_outliers(df)
        
        numeric_cols = df_handled.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in outliers:
                continue
            
            if method == "clip":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_handled[col] = df_handled[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == "remove":
                df_handled = df_handled.drop(df_handled.index[outliers[col]])
            
            elif method == "winsorize":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df_handled[col] = df_handled[col].clip(lower=lower, upper=upper)
        
        logger.info("outliers_handled", final_shape=df_handled.shape)
        return df_handled
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features (family size, income bands, etc.).
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("engineering_features", initial_cols=len(df.columns))
        
        df_eng = df.copy()
        
        # Financial-specific feature engineering
        if "close" in df_eng.columns:
            # Returns
            df_eng["returns"] = df_eng["close"].pct_change()
            df_eng["log_returns"] = np.log(df_eng["close"] / df_eng["close"].shift(1))
            
            # Volatility
            for window in [5, 10, 20, 30]:
                df_eng[f"volatility_{window}"] = df_eng["returns"].rolling(window).std()
            
            # Moving averages
            for window in [5, 10, 20, 50, 200]:
                df_eng[f"sma_{window}"] = df_eng["close"].rolling(window).mean()
                df_eng[f"ema_{window}"] = df_eng["close"].ewm(span=window).mean()
            
            # RSI
            delta = df_eng["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_eng["rsi"] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df_eng["close"].ewm(span=12).mean()
            ema_26 = df_eng["close"].ewm(span=26).mean()
            df_eng["macd"] = ema_12 - ema_26
            df_eng["macd_signal"] = df_eng["macd"].ewm(span=9).mean()
            df_eng["macd_histogram"] = df_eng["macd"] - df_eng["macd_signal"]
            
            # Bollinger Bands
            sma_20 = df_eng["close"].rolling(20).mean()
            std_20 = df_eng["close"].rolling(20).std()
            df_eng["bb_upper"] = sma_20 + (std_20 * 2)
            df_eng["bb_lower"] = sma_20 - (std_20 * 2)
            df_eng["bb_width"] = (df_eng["bb_upper"] - df_eng["bb_lower"]) / sma_20
        
        # Fill NaN values created by rolling operations
        df_eng = df_eng.bfill().ffill().fillna(0)
        
        logger.info("features_engineered", final_cols=len(df_eng.columns))
        return df_eng
    
    def rolling_window_normalize(
        self,
        df: pd.DataFrame,
        window: int = 60,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Apply rolling-window normalization.
        
        Args:
            df: Input dataframe
            window: Rolling window size
            method: Normalization method ("zscore", "minmax")
            
        Returns:
            Normalized dataframe
        """
        logger.info("applying_rolling_window_normalization", window=window, method=method)
        
        df_norm = df.copy()
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "zscore":
                mean = df_norm[col].rolling(window).mean()
                std = df_norm[col].rolling(window).std()
                df_norm[col] = (df_norm[col] - mean) / (std + 1e-8)
            elif method == "minmax":
                min_val = df_norm[col].rolling(window).min()
                max_val = df_norm[col].rolling(window).max()
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-8)
        
        # Fill NaN values
        df_norm = df_norm.bfill().ffill().fillna(0)
        
        logger.info("rolling_window_normalization_complete")
        return df_norm
    
    def trend_decomposition(
        self,
        series: pd.Series,
        period: int = 20,
    ) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend and seasonal components.
        
        Args:
            series: Input time series
            period: Seasonal period
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        logger.info("decomposing_trend", period=period)
        
        # Simple moving average for trend
        trend = series.rolling(window=period, center=True).mean()
        
        # Detrend
        detrended = series - trend
        
        # Seasonal component (simplified)
        seasonal = detrended.groupby(detrended.index % period).mean()
        seasonal = seasonal.reindex(detrended.index, method="nearest")
        
        # Residual
        residual = detrended - seasonal
        
        decomposition = {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
        }
        
        logger.info("trend_decomposition_complete")
        return decomposition
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lags: List[int] = [1, 2, 3, 5, 10],
    ) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: Input dataframe
            columns: Columns to lag (None = all numeric columns)
            lags: List of lag periods
            
        Returns:
            Dataframe with lagged features
        """
        logger.info("creating_lagged_features", lags=lags)
        
        df_lagged = df.copy()
        
        if columns is None:
            columns = df_lagged.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            for lag in lags:
                df_lagged[f"{col}_lag_{lag}"] = df_lagged[col].shift(lag)
        
        # Fill NaN values
        df_lagged = df_lagged.bfill().ffill().fillna(0)
        
        logger.info("lagged_features_created", num_new_features=len(columns) * len(lags))
        return df_lagged
    
    def normalize(
        self,
        df: pd.DataFrame,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize features.
        
        Args:
            df: Input dataframe
            method: Normalization method ("standard", "robust", "minmax")
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Normalized dataframe
        """
        logger.info("normalizing_features", method=method, fit=fit)
        
        df_norm = df.copy()
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if fit:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                df_norm[col] = scaler.fit_transform(df_norm[[col]])
                self.scalers[col] = scaler
                self.feature_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
            else:
                if col in self.scalers:
                    df_norm[col] = self.scalers[col].transform(df_norm[[col]])
                else:
                    logger.warning("scaler_not_found_for_column", column=col)
        
        logger.info("normalization_complete")
        return df_norm
    
    def process(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit (True for training, False for inference)
            
        Returns:
            Processed dataframe
        """
        logger.info("starting_preprocessing_pipeline", fit=fit)
        
        # EDA (only for training)
        if fit:
            eda_results = self.eda_analysis(df)
            logger.info("eda_results", **{k: v for k, v in eda_results.items() if k != "numeric_stats"})
        
        # Clean features
        df_processed = self.clean_features(df)
        
        # Handle outliers
        df_processed = self.handle_outliers(df_processed)
        
        # Engineer features
        df_processed = self.engineer_features(df_processed)
        
        # Create lagged features
        df_processed = self.create_lagged_features(df_processed)
        
        # Normalize
        df_processed = self.normalize(df_processed, fit=fit)
        
        logger.info("preprocessing_pipeline_complete", final_shape=df_processed.shape)
        return df_processed

