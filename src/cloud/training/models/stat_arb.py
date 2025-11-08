"""Statistical arbitrage module for pairs trading."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class StatisticalArbitrage:
    """
    Statistical arbitrage for correlated assets:
    - Computes z-scores of normalized spreads
    - Cointegration tests for stationarity
    - Mean-reversion triggers when deviation > k·σ
    - Pairs trading (ETH/BTC, SOL/AVAX)
    """

    def __init__(
        self,
        lookback_window: int = 60,  # 60 periods
        z_score_threshold: float = 2.0,  # 2 standard deviations
        cointegration_p_value: float = 0.05,
    ) -> None:
        """
        Initialize statistical arbitrage.
        
        Args:
            lookback_window: Lookback window for calculations
            z_score_threshold: Threshold for z-score triggers
            cointegration_p_value: P-value threshold for cointegration test
        """
        self.lookback_window = lookback_window
        self.z_score_threshold = z_score_threshold
        self.cointegration_p_value = cointegration_p_value
        
        logger.info(
            "statistical_arbitrage_initialized",
            lookback_window=lookback_window,
            z_score_threshold=z_score_threshold,
            cointegration_p_value=cointegration_p_value,
        )

    def calculate_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        normalize: bool = True,
    ) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
            normalize: Whether to normalize the spread
            
        Returns:
            Spread series
        """
        # Align series
        asset1_aligned, asset2_aligned = asset1.align(asset2, join='inner')
        
        # Calculate spread
        spread = asset1_aligned - asset2_aligned
        
        # Normalize if requested
        if normalize:
            spread_mean = spread.rolling(window=self.lookback_window).mean()
            spread_std = spread.rolling(window=self.lookback_window).std()
            spread = (spread - spread_mean) / (spread_std + 1e-8)
        
        return spread

    def calculate_z_score(
        self,
        spread: pd.Series,
    ) -> pd.Series:
        """
        Calculate z-score of spread.
        
        Args:
            spread: Spread series
            
        Returns:
            Z-score series
        """
        spread_mean = spread.rolling(window=self.lookback_window).mean()
        spread_std = spread.rolling(window=self.lookback_window).std()
        
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        
        return z_score

    def test_cointegration(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
    ) -> Dict[str, Any]:
        """
        Test for cointegration between two assets.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
            
        Returns:
            Cointegration test results
        """
        try:
            from statsmodels.tsa.stattools import coint  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("statsmodels_not_available", using_simple_correlation=True)
            # Fallback to simple correlation
            correlation = asset1.corr(asset2)
            return {
                "is_cointegrated": abs(correlation) > 0.7,
                "correlation": correlation,
                "p_value": 1.0 - abs(correlation),
                "method": "correlation",
            }
        
        # Align series
        asset1_aligned, asset2_aligned = asset1.align(asset2, join='inner')
        
        # Remove NaN values
        valid_mask = ~(asset1_aligned.isna() | asset2_aligned.isna())
        asset1_clean = asset1_aligned[valid_mask]
        asset2_clean = asset2_aligned[valid_mask]
        
        if len(asset1_clean) < 10:
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "error": "Insufficient data",
            }
        
        # Perform cointegration test
        try:
            score, p_value, _ = coint(asset1_clean, asset2_clean)
            
            is_cointegrated = p_value < self.cointegration_p_value
            
            return {
                "is_cointegrated": is_cointegrated,
                "p_value": float(p_value),
                "score": float(score),
                "threshold": self.cointegration_p_value,
            }
        except Exception as e:
            logger.warning("cointegration_test_failed", error=str(e))
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "error": str(e),
            }

    def detect_opportunity(
        self,
        z_score: float,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect trading opportunity based on z-score.
        
        Args:
            z_score: Current z-score
            threshold: Z-score threshold (default: self.z_score_threshold)
            
        Returns:
            Trading signal dictionary
        """
        if threshold is None:
            threshold = self.z_score_threshold
        
        if z_score > threshold:
            # Asset1 overvalued, Asset2 undervalued
            return {
                "signal": "short_asset1_long_asset2",
                "confidence": min(abs(z_score) / threshold, 2.0),  # Cap at 2.0
                "z_score": z_score,
                "reason": f"Spread {z_score:.2f} standard deviations above mean",
            }
        elif z_score < -threshold:
            # Asset1 undervalued, Asset2 overvalued
            return {
                "signal": "long_asset1_short_asset2",
                "confidence": min(abs(z_score) / threshold, 2.0),
                "z_score": z_score,
                "reason": f"Spread {z_score:.2f} standard deviations below mean",
            }
        else:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "z_score": z_score,
                "reason": f"Spread within {threshold} standard deviations",
            }

    def analyze_pair(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        asset1_name: str = "asset1",
        asset2_name: str = "asset2",
    ) -> Dict[str, Any]:
        """
        Analyze pair for statistical arbitrage opportunities.
        
        Args:
            asset1: First asset price series
            asset2: Second asset price series
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            
        Returns:
            Analysis results dictionary
        """
        logger.info("analyzing_pair", asset1=asset1_name, asset2=asset2_name)
        
        # Test cointegration
        cointegration_result = self.test_cointegration(asset1, asset2)
        
        if not cointegration_result.get("is_cointegrated", False):
            return {
                "status": "not_cointegrated",
                "cointegration": cointegration_result,
                "signal": "no_trade",
                "reason": "Assets are not cointegrated",
            }
        
        # Calculate spread
        spread = self.calculate_spread(asset1, asset2, normalize=True)
        
        # Calculate z-score
        z_score_series = self.calculate_z_score(spread)
        current_z_score = z_score_series.iloc[-1] if len(z_score_series) > 0 else 0.0
        
        # Detect opportunity
        opportunity = self.detect_opportunity(current_z_score)
        
        # Calculate spread statistics
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        result = {
            "status": "cointegrated",
            "cointegration": cointegration_result,
            "spread_statistics": {
                "mean": float(spread_mean),
                "std": float(spread_std),
                "current": float(spread.iloc[-1]) if len(spread) > 0 else 0.0,
            },
            "z_score": {
                "current": float(current_z_score),
                "threshold": self.z_score_threshold,
            },
            "opportunity": opportunity,
            "asset1": asset1_name,
            "asset2": asset2_name,
        }
        
        logger.info(
            "pair_analysis_complete",
            asset1=asset1_name,
            asset2=asset2_name,
            signal=opportunity["signal"],
            z_score=current_z_score,
        )
        
        return result

    def create_pairs(
        self,
        assets: Dict[str, pd.Series],
        min_correlation: float = 0.7,
    ) -> List[Tuple[str, str]]:
        """
        Create trading pairs from assets based on correlation.
        
        Args:
            assets: Dictionary of asset name to price series
            min_correlation: Minimum correlation for pairing
            
        Returns:
            List of (asset1, asset2) tuples
        """
        logger.info("creating_pairs", num_assets=len(assets))
        
        pairs = []
        asset_names = list(assets.keys())
        
        # Calculate correlation matrix
        for i, asset1_name in enumerate(asset_names):
            for asset2_name in asset_names[i + 1:]:
                asset1 = assets[asset1_name]
                asset2 = assets[asset2_name]
                
                # Align and calculate correlation
                asset1_aligned, asset2_aligned = asset1.align(asset2, join='inner')
                correlation = asset1_aligned.corr(asset2_aligned)
                
                if not pd.isna(correlation) and abs(correlation) >= min_correlation:
                    pairs.append((asset1_name, asset2_name))
                    logger.debug(
                        "pair_created",
                        asset1=asset1_name,
                        asset2=asset2_name,
                        correlation=correlation,
                    )
        
        logger.info("pairs_created", num_pairs=len(pairs))
        
        return pairs

