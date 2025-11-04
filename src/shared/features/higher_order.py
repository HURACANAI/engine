"""
Higher-Order Feature Engineering

Creates advanced features that capture non-linear relationships:
1. Feature interactions (cross-products)
2. Polynomial features (squared, cubed)
3. Time-lagged features (temporal patterns)
4. Ratio features (comparative metrics)

These features help ML models detect complex patterns that linear features miss.
"""

from typing import List, Optional, Tuple

import polars as pl
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class HigherOrderFeatureBuilder:
    """
    Builds higher-order features from base features.

    Transforms 68 base features → 100+ features by adding:
    - Feature interactions (e.g., btc_beta * market_divergence)
    - Polynomial features (e.g., rsi^2, momentum^3)
    - Time-lagged features (e.g., rsi_lag_1, rsi_lag_3)
    - Ratio features (e.g., volume / volume_mean)
    """

    def __init__(
        self,
        enable_interactions: bool = True,
        enable_polynomials: bool = True,
        enable_time_lags: bool = True,
        enable_ratios: bool = True,
        max_lag: int = 5,
    ):
        """
        Initialize feature builder.

        Args:
            enable_interactions: Create feature cross-products
            enable_polynomials: Create squared/cubed features
            enable_time_lags: Create lagged features
            enable_ratios: Create ratio features
            max_lag: Maximum lag for time-lagged features
        """
        self.enable_interactions = enable_interactions
        self.enable_polynomials = enable_polynomials
        self.enable_time_lags = enable_time_lags
        self.enable_ratios = enable_ratios
        self.max_lag = max_lag

        logger.info(
            "higher_order_feature_builder_initialized",
            interactions=enable_interactions,
            polynomials=enable_polynomials,
            time_lags=enable_time_lags,
            ratios=enable_ratios,
            max_lag=max_lag,
        )

    def build(self, frame: pl.DataFrame) -> pl.DataFrame:
        """
        Build all higher-order features.

        Args:
            frame: DataFrame with base features

        Returns:
            DataFrame with base + higher-order features
        """
        result = frame

        if self.enable_interactions:
            result = self._add_interactions(result)

        if self.enable_polynomials:
            result = self._add_polynomials(result)

        if self.enable_time_lags:
            result = self._add_time_lags(result)

        if self.enable_ratios:
            result = self._add_ratios(result)

        logger.info(
            "higher_order_features_built",
            original_features=len(frame.columns),
            total_features=len(result.columns),
            added_features=len(result.columns) - len(frame.columns),
        )

        return result

    def _add_interactions(self, frame: pl.DataFrame) -> pl.DataFrame:
        """
        Add feature interaction terms.

        Key interactions that capture market dynamics:
        - Beta * Divergence (amplified divergence)
        - Trend * Volume (trend confirmation)
        - Compression * Volatility (breakout potential)
        - Regime * Momentum (regime-aligned momentum)
        """
        interactions = []

        # Cross-asset interactions
        if "btc_beta" in frame.columns and "btc_divergence" in frame.columns:
            interactions.append(
                (pl.col("btc_beta") * pl.col("btc_divergence")).alias("btc_beta_x_divergence")
            )

        if "btc_correlation" in frame.columns and "btc_relative_momentum" in frame.columns:
            interactions.append(
                (pl.col("btc_correlation") * pl.col("btc_relative_momentum")).alias("btc_corr_x_momentum")
            )

        # Trend interactions
        if "trend_strength" in frame.columns and "vol_jump_z" in frame.columns:
            interactions.append(
                (pl.col("trend_strength") * pl.col("vol_jump_z")).alias("trend_x_volume")
            )

        if "trend_strength" in frame.columns and "adx" in frame.columns:
            interactions.append(
                (pl.col("trend_strength") * pl.col("adx")).alias("trend_x_adx")
            )

        # Breakout interactions
        if "compression" in frame.columns and "atr" in frame.columns:
            interactions.append(
                (pl.col("compression") * pl.col("atr")).alias("compression_x_volatility")
            )

        if "ignition_score" in frame.columns and "breakout_quality" in frame.columns:
            interactions.append(
                (pl.col("ignition_score") * pl.col("breakout_quality") / 100.0).alias("breakout_power")
            )

        # Momentum interactions
        for momentum_window in [1, 3, 5]:
            ret_col = f"ret_{momentum_window}"
            zscore_col = f"zscore_ret_{momentum_window}"

            if ret_col in frame.columns and "rsi_14" in frame.columns:
                interactions.append(
                    (pl.col(ret_col) * (pl.col("rsi_14") - 50) / 50).alias(f"ret_{momentum_window}_x_rsi")
                )

            if zscore_col in frame.columns and "adx" in frame.columns:
                interactions.append(
                    (pl.col(zscore_col) * pl.col("adx") / 50).alias(f"zscore_{momentum_window}_x_adx")
                )

        # Regime interactions (if regime detector is used)
        if "vol_regime" in frame.columns and "momentum_slope" in frame.columns:
            interactions.append(
                (pl.col("vol_regime") * pl.col("momentum_slope")).alias("regime_x_momentum")
            )

        # Market structure interactions
        if "leader_bias" in frame.columns and "rs_score" in frame.columns:
            interactions.append(
                (pl.col("leader_bias") * pl.col("rs_score") / 100).alias("leader_x_rs")
            )

        # Apply interactions
        if interactions:
            frame = frame.with_columns(interactions)

        return frame

    def _add_polynomials(self, frame: pl.DataFrame) -> pl.DataFrame:
        """
        Add polynomial features (squared, cubed).

        Captures non-linear relationships:
        - RSI^2: Extreme RSI values have non-linear impact
        - Momentum^3: Strong momentum accelerates
        - Volatility^2: Vol clustering
        """
        polynomials = []

        # RSI polynomials (non-linear at extremes)
        for rsi_period in [7, 14]:
            rsi_col = f"rsi_{rsi_period}"
            if rsi_col in frame.columns:
                # Centered around 50 (neutral)
                rsi_centered = pl.col(rsi_col) - 50

                polynomials.append(
                    (rsi_centered ** 2).alias(f"{rsi_col}_squared")
                )

        # Momentum polynomials
        for window in [1, 3, 5]:
            ret_col = f"ret_{window}"
            if ret_col in frame.columns:
                polynomials.append(
                    (pl.col(ret_col) ** 2).alias(f"{ret_col}_squared")
                )
                polynomials.append(
                    (pl.col(ret_col) ** 3).alias(f"{ret_col}_cubed")
                )

        # Volatility polynomials
        if "atr" in frame.columns:
            polynomials.append(
                (pl.col("atr") ** 2).alias("atr_squared")
            )

        for window in [30, 60]:
            sigma_col = f"realized_sigma_{window}"
            if sigma_col in frame.columns:
                polynomials.append(
                    (pl.col(sigma_col) ** 2).alias(f"{sigma_col}_squared")
                )

        # ADX polynomial (trend strength non-linearity)
        if "adx" in frame.columns:
            polynomials.append(
                ((pl.col("adx") / 50) ** 2).alias("adx_squared")
            )

        # Divergence polynomials
        if "btc_divergence" in frame.columns:
            polynomials.append(
                (pl.col("btc_divergence") ** 2).alias("btc_divergence_squared")
            )

        # Compression polynomial
        if "compression" in frame.columns:
            polynomials.append(
                (pl.col("compression") ** 2).alias("compression_squared")
            )

        # Apply polynomials
        if polynomials:
            frame = frame.with_columns(polynomials)

        return frame

    def _add_time_lags(self, frame: pl.DataFrame) -> pl.DataFrame:
        """
        Add time-lagged features.

        Creates temporal patterns:
        - rsi_lag_1, rsi_lag_3, rsi_lag_5
        - momentum_lag_1, momentum_lag_3
        - btc_divergence_lag_1, btc_divergence_lag_3

        Allows model to see "momentum of momentum" and detect accelerating trends.
        """
        lagged_features = []

        # Define key features to lag
        features_to_lag = [
            # Momentum
            "ret_1", "ret_3", "ret_5",
            "zscore_ret_1", "zscore_ret_3",
            # RSI
            "rsi_7", "rsi_14",
            # Trend
            "trend_strength", "momentum_slope", "ema_slope",
            # Volatility
            "atr", "realized_sigma_30",
            # Cross-asset
            "btc_beta", "btc_divergence", "btc_correlation",
            # Revuelto
            "ignition_score", "compression", "micro_score",
        ]

        # Lag periods: 1, 3, 5
        lag_periods = [1, 3, 5] if self.max_lag >= 5 else [1, 3]

        for feature in features_to_lag:
            if feature not in frame.columns:
                continue

            for lag in lag_periods:
                if lag > self.max_lag:
                    break

                lagged_features.append(
                    pl.col(feature).shift(lag).alias(f"{feature}_lag_{lag}")
                )

        # Apply lags
        if lagged_features:
            frame = frame.with_columns(lagged_features)

        return frame

    def _add_ratios(self, frame: pl.DataFrame) -> pl.DataFrame:
        """
        Add ratio features.

        Comparative metrics that normalize by reference values:
        - volume / volume_mean (volume surge)
        - spread / volatility (normalized spread)
        - beta / correlation (leverage factor)
        """
        ratios = []

        # Volume ratios
        for window in [240]:
            vol_mean_col = f"volume_mean_{window}"
            if vol_mean_col in frame.columns and "volume" in frame.columns:
                ratios.append(
                    (pl.col("volume") / (pl.col(vol_mean_col) + 1e-9)).alias(f"volume_to_mean_{window}")
                )

        # Spread to volatility ratio
        if "spread_bps" in frame.columns and "atr" in frame.columns:
            ratios.append(
                (pl.col("spread_bps") / (pl.col("atr") + 1e-9)).alias("spread_to_atr")
            )

        # Beta to correlation ratio (leverage factor)
        if "btc_beta" in frame.columns and "btc_correlation" in frame.columns:
            ratios.append(
                (pl.col("btc_beta") / (pl.col("btc_correlation").abs() + 0.1)).alias("btc_leverage_factor")
            )

        # Momentum to volatility ratio (signal-to-noise)
        for window in [1, 3, 5]:
            ret_col = f"ret_{window}"
            sigma_col = "atr"

            if ret_col in frame.columns and sigma_col in frame.columns:
                ratios.append(
                    (pl.col(ret_col).abs() / (pl.col(sigma_col) + 1e-9)).alias(f"ret_{window}_to_atr")
                )

        # Close to VWAP ratio
        if "close" in frame.columns and "vwap" in frame.columns:
            ratios.append(
                ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 1e-9)).alias("close_to_vwap_ratio")
            )

        # Price position normalized by range
        if "price_position" in frame.columns and "bb_width" in frame.columns:
            ratios.append(
                (pl.col("price_position") / (pl.col("bb_width") + 1e-9)).alias("position_to_bb_width")
            )

        # Apply ratios
        if ratios:
            frame = frame.with_columns(ratios)

        return frame

    def get_feature_names(self, base_frame: pl.DataFrame) -> List[str]:
        """
        Get list of all feature names that will be created.

        Args:
            base_frame: Base feature dataframe

        Returns:
            List of feature names
        """
        # Build features to get names
        full_frame = self.build(base_frame)
        return full_frame.columns


def build_higher_order_features(
    frame: pl.DataFrame,
    enable_interactions: bool = True,
    enable_polynomials: bool = True,
    enable_time_lags: bool = True,
    enable_ratios: bool = True,
    max_lag: int = 5,
) -> pl.DataFrame:
    """
    Convenience function to build higher-order features.

    Args:
        frame: Base feature dataframe
        enable_interactions: Create feature cross-products
        enable_polynomials: Create squared/cubed features
        enable_time_lags: Create lagged features
        enable_ratios: Create ratio features
        max_lag: Maximum lag for time-lagged features

    Returns:
        DataFrame with higher-order features added
    """
    builder = HigherOrderFeatureBuilder(
        enable_interactions=enable_interactions,
        enable_polynomials=enable_polynomials,
        enable_time_lags=enable_time_lags,
        enable_ratios=enable_ratios,
        max_lag=max_lag,
    )

    return builder.build(frame)
