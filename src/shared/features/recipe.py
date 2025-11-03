"""Shared feature recipe ensuring parity across Engine components."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np
import polars as pl


def _rolling_zscore(column: pl.Expr, window: int) -> pl.Expr:
    mean = column.rolling_mean(window, min_periods=max(1, window // 2))
    std = column.rolling_std(window, min_periods=max(1, window // 2))
    return ((column - mean) / std).fill_null(0.0).alias(f"zscore_{window}")


def _ema(column: pl.Expr, span: int) -> pl.Expr:
    alpha = 2.0 / (span + 1.0)
    return column.ewm_mean(alpha=alpha, adjust=False).alias(f"ema_{span}")


def _rsi(column: pl.Expr, period: int) -> pl.Expr:
    delta = column.diff()
    up = pl.when(delta > 0).then(delta).otherwise(0.0)
    down = pl.when(delta < 0).then(-delta).otherwise(0.0)
    roll_up = up.ewm_mean(alpha=1.0 / period, adjust=False)
    roll_down = down.ewm_mean(alpha=1.0 / period, adjust=False)
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    clipped = rsi.clip(lower_bound=0.0, upper_bound=100.0)
    return clipped.alias(f"rsi_{period}")


def _atr(high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 14) -> pl.Expr:
    prev_close = close.shift(1)
    tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    return tr.ewm_mean(alpha=1.0 / period, adjust=False).alias("atr")


@dataclass
class FeatureRecipe:
    """Produces momentum, volatility, liquidity, and temporal features."""

    momentum_windows: Sequence[int] = (1, 3, 5)
    ema_pairs: Sequence[tuple[int, int]] = ((5, 21), (8, 34))
    rsi_periods: Sequence[int] = (7, 14)
    volatility_windows: Sequence[int] = (30, 60)
    liquidity_windows: Sequence[int] = (240,)

    def build(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "ts" not in frame.columns:
            raise ValueError("Expected timestamp column 'ts'")
        ordered = frame.sort("ts")
        base = ordered.with_columns(
            [
                pl.col("close").pct_change(n).alias(f"ret_{n}") for n in self.momentum_windows
            ]
        )
        zscore_features = [
            _rolling_zscore(pl.col(f"ret_{n}"), window=60) for n in self.momentum_windows
        ]
        ema_features = []
        for fast, slow in self.ema_pairs:
            ema_fast = _ema(pl.col("close"), fast)
            ema_slow = _ema(pl.col("close"), slow)
            ema_features.extend([ema_fast, ema_slow, (ema_fast - ema_slow).alias(f"ema_diff_{fast}_{slow}")])

        rsi_features = [_rsi(pl.col("close"), period) for period in self.rsi_periods]

        volatility_features = [
            pl.col("close").pct_change().rolling_std(window, min_periods=window // 2).alias(f"realized_sigma_{window}")
            for window in self.volatility_windows
        ]

        atr_feature = _atr(pl.col("high"), pl.col("low"), pl.col("close"))
        range_feature = ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct")

        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        cum_typical_volume = (typical_price * pl.col("volume")).cumsum()
        cum_volume = pl.col("volume").cumsum()
        vwap = (cum_typical_volume / (cum_volume + 1e-9)).alias("vwap")
        drift = ((pl.col("close") - vwap) / vwap).alias("close_to_vwap")

        slope_feature = (
            pl.col("close").pct_change().rolling_mean(window=15, min_periods=10).alias("micro_trend")
        )

        tod_seconds = pl.col("ts").dt.hour() * 3600 + pl.col("ts").dt.minute() * 60 + pl.col("ts").dt.second()
        tod_fraction = (tod_seconds.cast(pl.Float64) / 86400.0).alias("tod_fraction")
        tod_sin = (pl.col("tod_fraction") * 2 * pi).sin().alias("tod_sin")
        tod_cos = (pl.col("tod_fraction") * 2 * pi).cos().alias("tod_cos")

        liquidity_features = []
        for window in self.liquidity_windows:
            liquidity_features.extend(
                [
                    pl.col("volume").rolling_mean(window, min_periods=window // 2).alias(f"volume_mean_{window}"),
                    pl.col("volume").rolling_std(window, min_periods=window // 2).alias(f"volume_std_{window}"),
                ]
            )

        coin_age = (
            (pl.col("ts") - pl.col("ts").first()).dt.total_days().alias("coin_age_days")
        )

        feature_frame = base.with_columns(
            [
                *zscore_features,
                *ema_features,
                *rsi_features,
                *volatility_features,
                atr_feature,
                range_feature,
                vwap,
                drift,
                slope_feature,
                tod_fraction,
                tod_sin,
                tod_cos,
                *liquidity_features,
                coin_age,
            ]
        )

        filled = feature_frame.fill_null(strategy="forward").fill_null(strategy="backward").fill_nan(0.0)
        return filled
