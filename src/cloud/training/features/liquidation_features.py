"""Liquidation-based feature engineering."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class LiquidationFeatures:
    """
    Creates liquidation-based features:
    - liquidation_intensity = sum(liquidations) / rolling_volume
    - liquidation_momentum = rolling sum of liquidations
    - liquidation_imbalance = long / short ratio
    - cascade_indicator = binary flag for cascade events
    """

    def __init__(
        self,
        rolling_window: int = 60,  # 60 minutes
        cascade_window: int = 5,  # 5 minutes
        cascade_threshold: int = 10,  # 10 liquidations
    ) -> None:
        """
        Initialize liquidation features.
        
        Args:
            rolling_window: Rolling window for calculations (minutes)
            cascade_window: Window for cascade detection (minutes)
            cascade_threshold: Threshold for cascade detection
        """
        self.rolling_window = rolling_window
        self.cascade_window = cascade_window
        self.cascade_threshold = cascade_threshold
        
        logger.info(
            "liquidation_features_initialized",
            rolling_window=rolling_window,
            cascade_window=cascade_window,
            cascade_threshold=cascade_threshold,
        )

    def calculate_intensity(
        self,
        liquidation_data: pd.DataFrame,
        volume_data: pd.Series,
    ) -> pd.Series:
        """
        Calculate liquidation intensity.
        
        Args:
            liquidation_data: Liquidation data DataFrame
            volume_data: Volume data Series
            
        Returns:
            Liquidation intensity series
        """
        # Sum liquidations per time period
        if 'timestamp' in liquidation_data.columns and 'size_usd' in liquidation_data.columns:
            liquidation_data['timestamp'] = pd.to_datetime(liquidation_data['timestamp'])
            liquidation_sum = liquidation_data.groupby(
                pd.Grouper(key='timestamp', freq='1T')
            )['size_usd'].sum()
        else:
            # Assume liquidation_data is already aggregated
            liquidation_sum = liquidation_data['size_usd'] if 'size_usd' in liquidation_data.columns else pd.Series()
        
        # Align with volume data
        liquidation_sum = liquidation_sum.reindex(volume_data.index, fill_value=0)
        
        # Calculate rolling sum of liquidations
        rolling_liquidations = liquidation_sum.rolling(window=self.rolling_window).sum()
        
        # Calculate rolling volume
        rolling_volume = volume_data.rolling(window=self.rolling_window).sum()
        
        # Calculate intensity
        intensity = rolling_liquidations / (rolling_volume + 1e-8)
        
        return intensity

    def calculate_momentum(
        self,
        liquidation_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate liquidation momentum (rolling sum).
        
        Args:
            liquidation_data: Liquidation data DataFrame
            
        Returns:
            Liquidation momentum series
        """
        if 'timestamp' in liquidation_data.columns and 'size_usd' in liquidation_data.columns:
            liquidation_data['timestamp'] = pd.to_datetime(liquidation_data['timestamp'])
            liquidation_sum = liquidation_data.groupby(
                pd.Grouper(key='timestamp', freq='1T')
            )['size_usd'].sum()
        else:
            liquidation_sum = liquidation_data['size_usd'] if 'size_usd' in liquidation_data.columns else pd.Series()
        
        # Calculate rolling sum
        momentum = liquidation_sum.rolling(window=self.rolling_window).sum()
        
        return momentum

    def calculate_imbalance(
        self,
        liquidation_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate liquidation imbalance (long / short ratio).
        
        Args:
            liquidation_data: Liquidation data DataFrame with 'side' column
            
        Returns:
            Liquidation imbalance series
        """
        if 'side' not in liquidation_data.columns:
            logger.warning("liquidation_data_missing_side_column")
            return pd.Series()
        
        if 'timestamp' in liquidation_data.columns:
            liquidation_data['timestamp'] = pd.to_datetime(liquidation_data['timestamp'])
            
            # Group by timestamp and side
            long_liquidations = liquidation_data[liquidation_data['side'] == 'long'].groupby(
                pd.Grouper(key='timestamp', freq='1T')
            ).size()
            short_liquidations = liquidation_data[liquidation_data['side'] == 'short'].groupby(
                pd.Grouper(key='timestamp', freq='1T')
            ).size()
        else:
            long_liquidations = (liquidation_data['side'] == 'long').sum()
            short_liquidations = (liquidation_data['side'] == 'short').sum()
        
        # Calculate rolling sums
        rolling_long = long_liquidations.rolling(window=self.rolling_window).sum()
        rolling_short = short_liquidations.rolling(window=self.rolling_window).sum()
        
        # Calculate imbalance (long / short)
        imbalance = rolling_long / (rolling_short + 1e-8)
        
        return imbalance

    def detect_cascades(
        self,
        liquidation_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Detect liquidation cascades.
        
        Args:
            liquidation_data: Liquidation data DataFrame
            
        Returns:
            Cascade indicator series (1 = cascade, 0 = no cascade)
        """
        if 'timestamp' in liquidation_data.columns:
            liquidation_data['timestamp'] = pd.to_datetime(liquidation_data['timestamp'])
            liquidation_count = liquidation_data.groupby(
                pd.Grouper(key='timestamp', freq='1T')
            ).size()
        else:
            liquidation_count = pd.Series([len(liquidation_data)], index=[pd.Timestamp.now()])
        
        # Calculate rolling sum
        rolling_count = liquidation_count.rolling(window=self.cascade_window).sum()
        
        # Detect cascades (threshold exceeded)
        cascade_indicator = (rolling_count >= self.cascade_threshold).astype(int)
        
        return cascade_indicator

    def create_features(
        self,
        liquidation_data: pd.DataFrame,
        volume_data: pd.Series,
    ) -> Dict[str, pd.Series]:
        """
        Create all liquidation features.
        
        Args:
            liquidation_data: Liquidation data DataFrame
            volume_data: Volume data Series
            
        Returns:
            Dictionary of feature series
        """
        logger.info("creating_liquidation_features")
        
        features = {}
        
        # Liquidation intensity
        features["liquidation_intensity"] = self.calculate_intensity(liquidation_data, volume_data)
        
        # Liquidation momentum
        features["liquidation_momentum"] = self.calculate_momentum(liquidation_data)
        
        # Liquidation imbalance
        features["liquidation_imbalance"] = self.calculate_imbalance(liquidation_data)
        
        # Cascade indicator
        features["cascade_indicator"] = self.detect_cascades(liquidation_data)
        
        logger.info("liquidation_features_created", num_features=len(features))
        
        return features

