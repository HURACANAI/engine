"""Shared feature recipe ensuring parity across Engine components."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np
import polars as pl


def _rolling_zscore(column: pl.Expr, window: int) -> pl.Expr:
    mean = column.rolling_mean(window_size=window, min_periods=max(1, window // 2))
    std = column.rolling_std(window_size=window, min_periods=max(1, window // 2))
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


def _adx(high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 14) -> pl.Expr:
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (0-100), where >25 indicates strong trend.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # Calculate +DM and -DM
    plus_dm = pl.when((high - prev_high) > (prev_low - low)).then(
        pl.max_horizontal(high - prev_high, pl.lit(0.0))
    ).otherwise(0.0)

    minus_dm = pl.when((prev_low - low) > (high - prev_high)).then(
        pl.max_horizontal(prev_low - low, pl.lit(0.0))
    ).otherwise(0.0)

    # Calculate True Range
    tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())

    # Smooth with EMA
    alpha = 1.0 / period
    atr_smooth = tr.ewm_mean(alpha=alpha, adjust=False)
    plus_di = (plus_dm.ewm_mean(alpha=alpha, adjust=False) / (atr_smooth + 1e-9)) * 100
    minus_di = (minus_dm.ewm_mean(alpha=alpha, adjust=False) / (atr_smooth + 1e-9)) * 100

    # Calculate DX and ADX
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.ewm_mean(alpha=alpha, adjust=False)

    return adx.alias("adx")


def _bollinger_bands(close: pl.Expr, period: int = 20, std_dev: float = 2.0) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Calculate Bollinger Bands and width.
    Returns (upper_band, lower_band, bb_width_pct)
    """
    ma = close.rolling_mean(window_size=period, min_periods=period // 2)
    std = close.rolling_std(window_size=period, min_periods=period // 2)

    upper_band = (ma + std_dev * std).alias("bb_upper")
    lower_band = (ma - std_dev * std).alias("bb_lower")
    bb_width = ((upper_band - lower_band) / ma).alias("bb_width")

    return upper_band, lower_band, bb_width


def _volatility_regime(close: pl.Expr, short_window: int = 10, long_window: int = 50) -> pl.Expr:
    """
    Calculate volatility regime indicator.
    Returns ratio of short-term volatility to long-term volatility.
    > 1.5 = high volatility regime, < 0.7 = low volatility regime.
    """
    short_vol = close.pct_change().rolling_std(window_size=short_window, min_periods=short_window // 2)
    long_vol = close.pct_change().rolling_std(window_size=long_window, min_periods=long_window // 2)

    # Ratio of short-term to long-term volatility
    vol_regime = (short_vol / (long_vol + 1e-9)).alias("volatility_regime")

    return vol_regime


def _price_position_in_range(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int = 14) -> pl.Expr:
    """
    Calculate where price is positioned within recent range.
    0 = at bottom of range, 1 = at top of range.
    """
    rolling_high = high.rolling_max(window_size=window, min_periods=window // 2)
    rolling_low = low.rolling_min(window_size=window, min_periods=window // 2)

    # Position as percentage of range
    range_size = rolling_high - rolling_low
    position = ((close - rolling_low) / (range_size + 1e-9)).clip(0.0, 1.0)

    return position.alias("price_position_in_range")


# ============================================================================
# REVUELTO-SPECIFIC FEATURES (From Revuelto Bot Analysis)
# ============================================================================

def _ignition_score(high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Breakout ignition score (0-100).
    Measures breakout initiation strength combining price thrust and volume.
    """
    # Price thrust: how much price is breaking out of recent range
    recent_high = high.rolling_max(window_size=window, min_periods=window // 2)
    recent_low = low.rolling_min(window_size=window, min_periods=window // 2)
    range_size = recent_high - recent_low

    thrust = ((close - recent_high) / (range_size + 1e-9)).clip(0.0, 1.0)

    # Volume surge component
    avg_volume = volume.rolling_mean(window_size=window, min_periods=window // 2)
    vol_surge = (volume / (avg_volume + 1e-9)).clip(0.0, 3.0) / 3.0

    # Combine: 60% price thrust + 40% volume
    ignition = (0.6 * thrust + 0.4 * vol_surge) * 100.0

    return ignition.alias("ignition")


def _compression_score(high: pl.Expr, low: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Range compression score (0-1).
    1 = maximum compression (tight range), 0 = wide range.
    Used for range trading and breakout setup detection.
    """
    # Current range
    current_range = high - low

    # Historical average range
    avg_range = current_range.rolling_mean(window_size=window, min_periods=window // 2)

    # Compression: inverse of range expansion
    # Small range relative to average = high compression
    compression = 1.0 - (current_range / (avg_range + 1e-9)).clip(0.0, 2.0) / 2.0

    return compression.alias("compression_score")


def _nr7_density(high: pl.Expr, low: pl.Expr, window: int = 20) -> pl.Expr:
    """
    NR7 (Narrow Range 7) density score.
    Counts how many of the last N bars are NR7 (smallest range in 7 bars).
    High density = high compression, breakout imminent.
    """
    # Calculate range
    bar_range = high - low

    # For each bar, check if it's the smallest in last 7 bars
    rolling_min_7 = bar_range.rolling_min(window_size=7, min_periods=7)
    is_nr7 = (bar_range == rolling_min_7).cast(pl.Float64)

    # Density: fraction of last N bars that are NR7
    density = is_nr7.rolling_mean(window_size=window, min_periods=window // 2)

    return density.alias("nr7_density")


def _micro_score(close: pl.Expr, volume: pl.Expr, window: int = 10) -> pl.Expr:
    """
    Microstructure score (0-100).
    Measures short-term order flow and momentum quality.
    Used for tape reading / microstructure analysis.
    """
    # Short-term momentum
    returns = close.pct_change()
    momentum = returns.rolling_mean(window_size=window, min_periods=window // 2)

    # Volume quality (higher is better)
    avg_volume = volume.rolling_mean(window_size=window, min_periods=window // 2)
    vol_quality = (volume / (avg_volume + 1e-9)).clip(0.0, 2.0) / 2.0

    # Combine
    micro = (np.sign(momentum) * vol_quality).clip(-1.0, 1.0)
    micro_score = (micro + 1.0) * 50.0  # Scale to 0-100

    return micro_score.alias("micro_score")


def _uptick_ratio(close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Uptick ratio (0-1).
    Fraction of recent price changes that were positive.
    > 0.6 = strong buying, < 0.4 = strong selling.
    """
    # Price changes
    price_change = close.diff()

    # Upticks (positive changes)
    upticks = (price_change > 0).cast(pl.Float64)

    # Ratio
    uptick_ratio = upticks.rolling_mean(window_size=window, min_periods=window // 2)

    return uptick_ratio.alias("uptick_ratio")


def _leader_bias(close: pl.Expr, benchmark: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Leader bias score (-1 to 1).
    Measures if asset is leading or lagging benchmark/sector.
    Positive = leading (relative strength), negative = lagging.

    Note: benchmark should be a sector index or market index price.
    For now, using self-comparison as placeholder.
    """
    # Relative strength: asset returns vs benchmark returns
    asset_returns = close.pct_change()
    benchmark_returns = benchmark.pct_change()

    # Relative performance
    relative_perf = asset_returns - benchmark_returns

    # Smooth it
    leader_bias = relative_perf.rolling_mean(window_size=window, min_periods=window // 2)

    # Clip to reasonable range
    return leader_bias.clip(-0.1, 0.1).alias("leader_bias")


def _rs_score(close: pl.Expr, benchmark: pl.Expr, short_window: int = 10, med_window: int = 20) -> pl.Expr:
    """
    Composite Relative Strength score (0-100).
    Combines short and medium-term relative strength.
    > 70 = strong leader, < 30 = weak laggard.
    """
    # Short-term RS
    asset_ret_short = (close / close.shift(short_window)) - 1.0
    bench_ret_short = (benchmark / benchmark.shift(short_window)) - 1.0
    rs_short = asset_ret_short - bench_ret_short

    # Medium-term RS
    asset_ret_med = (close / close.shift(med_window)) - 1.0
    bench_ret_med = (benchmark / benchmark.shift(med_window)) - 1.0
    rs_med = asset_ret_med - bench_ret_med

    # Combine (60% short, 40% medium)
    rs_combined = 0.6 * rs_short + 0.4 * rs_med

    # Normalize to 0-100 scale (tanh to keep in reasonable range)
    rs_score = (pl.Expr.tanh(rs_combined * 10.0) + 1.0) * 50.0

    return rs_score.alias("rs_score")


def _breakout_thrust(high: pl.Expr, close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Breakout thrust score (0-1).
    Measures momentum of breakout above recent highs.
    """
    recent_high = high.rolling_max(window_size=window, min_periods=window // 2)

    # How far above recent high
    thrust = (close - recent_high) / (recent_high + 1e-9)

    # Clip and normalize to 0-1
    return thrust.clip(0.0, 0.1) * 10.0


def _breakout_quality(close: pl.Expr, volume: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Breakout quality score (0-1).
    High quality breakout = strong price move + high volume.
    """
    # Price momentum
    returns = close.pct_change()
    momentum = returns.rolling_sum(window_size=5, min_periods=3)

    # Volume confirmation
    avg_volume = volume.rolling_mean(window_size=window, min_periods=window // 2)
    vol_confirm = (volume / (avg_volume + 1e-9)).clip(0.0, 3.0) / 3.0

    # Quality = momentum * volume confirmation
    quality = (momentum.abs() * vol_confirm).clip(0.0, 1.0)

    return quality.alias("breakout_quality")


def _trend_strength(close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Trend strength score (-1 to 1).
    -1 = strong downtrend, 0 = no trend, 1 = strong uptrend.
    """
    # Linear regression slope
    # Simplified: use EMA slope as proxy
    ema = close.ewm_mean(alpha=2.0 / (window + 1.0), adjust=False)
    ema_prev = ema.shift(1)

    slope = (ema - ema_prev) / (close + 1e-9)

    # Normalize using tanh
    strength = pl.Expr.tanh(slope * 100.0)

    return strength.alias("trend_strength")


def _ema_slope(close: pl.Expr, period: int = 20) -> pl.Expr:
    """
    EMA slope indicator.
    Positive = upward trending, negative = downward trending.
    """
    ema = close.ewm_mean(alpha=2.0 / (period + 1.0), adjust=False)
    ema_prev = ema.shift(1)

    slope = (ema - ema_prev) / (ema_prev + 1e-9)

    return slope.alias("ema_slope")


def _momentum_slope(close: pl.Expr, short: int = 5, long: int = 20) -> pl.Expr:
    """
    Momentum slope (rate of change of momentum).
    Measures acceleration/deceleration.
    """
    # Short-term momentum
    momentum_short = close.pct_change(n=short)

    # Momentum trend (is momentum increasing or decreasing?)
    momentum_slope = momentum_short.diff(n=long)

    return momentum_slope.alias("momentum_slope")


def _htf_bias(close: pl.Expr, htf_period: int = 100) -> pl.Expr:
    """
    Higher TimeFrame bias score (0-1).
    0.5 = neutral, >0.5 = bullish HTF, <0.5 = bearish HTF.
    """
    # HTF moving average (represents higher timeframe trend)
    htf_ma = close.rolling_mean(window_size=htf_period, min_periods=htf_period // 2)

    # Current price relative to HTF MA
    bias = (close / (htf_ma + 1e-9)).clip(0.8, 1.2)

    # Normalize to 0-1 (1.0 = at MA, >1 = above MA, <1 = below MA)
    htf_bias = (bias - 0.8) / 0.4  # Maps [0.8, 1.2] to [0, 1]

    return htf_bias.alias("htf_bias")


def _spread_bps(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """
    Estimated bid-ask spread in basis points.
    Using high-low range as proxy for spread.
    """
    # Simple spread estimate: (high - low) / close
    spread = (high - low) / (close + 1e-9)

    # Convert to basis points
    spread_bps = spread * 10000.0

    return spread_bps.alias("spread_bps")


def _vol_jump_z(volume: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Volume jump Z-score.
    How many standard deviations above normal is current volume?
    > 2.0 = significant volume spike.
    """
    avg_volume = volume.rolling_mean(window_size=window, min_periods=window // 2)
    std_volume = volume.rolling_std(window_size=window, min_periods=window // 2)

    z_score = (volume - avg_volume) / (std_volume + 1e-9)

    return z_score.alias("vol_jump_z")


def _compression_rank(high: pl.Expr, low: pl.Expr, window: int = 50) -> pl.Expr:
    """
    Compression percentile rank (0-1).
    Where does current compression sit in historical distribution?
    1 = tightest range in recent history.
    """
    # Current range
    current_range = high - low

    # Percentile rank of current range (inverted for compression)
    # Lower range = higher compression rank
    rank = current_range.rolling_quantile(quantile=0.5, window_size=window, min_periods=window // 2)

    # Invert so tight range = high score
    compression_rank = 1.0 - (current_range / (rank + 1e-9)).clip(0.0, 2.0) / 2.0

    return compression_rank.alias("compression_rank")


def _mean_revert_bias(close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Mean reversion bias score (-1 to 1).
    How far from mean and likely to revert?
    Positive = above mean (expect reversion down), negative = below mean (expect reversion up).
    """
    ma = close.rolling_mean(window_size=window, min_periods=window // 2)
    std = close.rolling_std(window_size=window, min_periods=window // 2)

    # Z-score (how far from mean in standard deviations)
    z_score = (close - ma) / (std + 1e-9)

    # Clip to reasonable range [-3, 3] and normalize to [-1, 1]
    bias = z_score.clip(-3.0, 3.0) / 3.0

    return bias.alias("mean_revert_bias")


def _pullback_depth(close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Pullback depth from recent high (0-1).
    0 = at recent high, 1 = at recent low (deep pullback).
    """
    recent_high = close.rolling_max(window_size=window, min_periods=window // 2)
    recent_low = close.rolling_min(window_size=window, min_periods=window // 2)

    range_size = recent_high - recent_low

    # Where in the range (inverted from price_position for pullback perspective)
    pullback = (recent_high - close) / (range_size + 1e-9)

    return pullback.clip(0.0, 1.0).alias("pullback_depth")


def _kurtosis(close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Rolling kurtosis (tail risk measure).
    > 3 = fat tails (high tail risk), < 3 = thin tails.
    """
    # Returns
    returns = close.pct_change()

    # Mean and std
    mean = returns.rolling_mean(window_size=window, min_periods=window // 2)
    std = returns.rolling_std(window_size=window, min_periods=window // 2)

    # Standardized returns
    z = (returns - mean) / (std + 1e-9)

    # Fourth moment (kurtosis)
    kurtosis = (z ** 4).rolling_mean(window_size=window, min_periods=window // 2)

    return kurtosis.alias("kurtosis")


# ========================================
# CROSS-ASSET FEATURES
# (Market Structure / Leadership Features)
# ========================================


def _beta_vs_benchmark(close: pl.Expr, benchmark_close: pl.Expr, window: int = 30) -> pl.Expr:
    """
    Rolling beta vs benchmark (e.g., BTC).
    Beta ≈ StdDev(asset) / StdDev(benchmark) when correlation is high

    Beta > 1: Asset moves more than benchmark (amplified)
    Beta = 1: Asset moves with benchmark
    Beta < 1: Asset moves less than benchmark
    """
    # Returns
    asset_returns = close.pct_change()
    benchmark_returns = benchmark_close.pct_change()

    # Simplified beta: ratio of volatilities
    # This approximates true beta when correlation is high (which is typical for crypto)
    std_asset = asset_returns.rolling_std(window_size=window, min_periods=window // 2)
    std_benchmark = benchmark_returns.rolling_std(window_size=window, min_periods=window // 2)

    beta = std_asset / (std_benchmark + 1e-9)

    return beta.fill_null(1.0).alias("beta")  # Default beta = 1


def _cross_asset_correlation(close: pl.Expr, benchmark_close: pl.Expr, window: int = 30) -> pl.Expr:
    """
    Rolling correlation with benchmark.
    Approximated using normalized returns similarity.

    1 = perfect positive correlation
    0 = no correlation
    -1 = perfect negative correlation
    """
    asset_returns = close.pct_change()
    benchmark_returns = benchmark_close.pct_change()

    # Normalize returns (z-score)
    asset_mean = asset_returns.rolling_mean(window_size=window, min_periods=window // 2)
    asset_std = asset_returns.rolling_std(window_size=window, min_periods=window // 2)
    asset_z = (asset_returns - asset_mean) / (asset_std + 1e-9)

    bench_mean = benchmark_returns.rolling_mean(window_size=window, min_periods=window // 2)
    bench_std = benchmark_returns.rolling_std(window_size=window, min_periods=window // 2)
    bench_z = (benchmark_returns - bench_mean) / (bench_std + 1e-9)

    # Approximate correlation as rolling mean of product of z-scores
    # This approximates Pearson correlation
    correlation = (asset_z * bench_z).rolling_mean(window_size=window, min_periods=window // 2)

    return correlation.clip(-1.0, 1.0).fill_null(0.0).alias("correlation")


def _market_divergence(close: pl.Expr, benchmark_close: pl.Expr, short_window: int = 5, long_window: int = 20) -> pl.Expr:
    """
    Market divergence score (-1 to 1).
    Detects when asset is diverging from benchmark.

    Positive = asset outperforming benchmark (bullish divergence)
    Negative = asset underperforming benchmark (bearish divergence)
    Near 0 = moving in sync
    """
    # Short-term relative strength
    asset_ret_short = close.pct_change(short_window)
    bench_ret_short = benchmark_close.pct_change(short_window)
    rs_short = asset_ret_short - bench_ret_short

    # Long-term relative strength
    asset_ret_long = close.pct_change(long_window)
    bench_ret_long = benchmark_close.pct_change(long_window)
    rs_long = asset_ret_long - bench_ret_long

    # Divergence: difference between short-term and long-term RS
    # If short-term RS > long-term RS → accelerating outperformance
    divergence = rs_short - rs_long

    # Normalize to [-1, 1]
    divergence = divergence.clip(-0.2, 0.2) / 0.2  # Clip to ±20% divergence

    return divergence.alias("market_divergence")


def _leader_volatility(benchmark_close: pl.Expr, window: int = 20) -> pl.Expr:
    """
    Leader (benchmark) volatility.
    Higher leader volatility = higher spillover risk.
    """
    benchmark_returns = benchmark_close.pct_change()
    vol = benchmark_returns.rolling_std(window_size=window, min_periods=window // 2)

    return vol.fill_null(0.02).alias("leader_vol")  # Default 2% volatility


def _cross_asset_momentum(close: pl.Expr, benchmark_close: pl.Expr, window: int = 10) -> pl.Expr:
    """
    Momentum relative to benchmark.
    Positive = outperforming benchmark recently
    Negative = underperforming benchmark recently
    """
    asset_momentum = close.pct_change(window)
    benchmark_momentum = benchmark_close.pct_change(window)

    relative_momentum = asset_momentum - benchmark_momentum

    return relative_momentum.fill_null(0.0).alias("cross_momentum")


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
        zscore_features = []
        for n in self.momentum_windows:
            mean = pl.col(f"ret_{n}").rolling_mean(window_size=60, min_periods=30)
            std = pl.col(f"ret_{n}").rolling_std(window_size=60, min_periods=30)
            zscore = ((pl.col(f"ret_{n}") - mean) / std).fill_null(0.0).alias(f"zscore_ret_{n}")
            zscore_features.append(zscore)
        ema_features = []
        for fast, slow in self.ema_pairs:
            ema_fast = _ema(pl.col("close"), fast)
            ema_slow = _ema(pl.col("close"), slow)
            ema_features.extend([ema_fast, ema_slow, (ema_fast - ema_slow).alias(f"ema_diff_{fast}_{slow}")])

        rsi_features = [_rsi(pl.col("close"), period) for period in self.rsi_periods]

        volatility_features = [
            pl.col("close").pct_change().rolling_std(window_size=window, min_periods=window // 2).alias(f"realized_sigma_{window}")
            for window in self.volatility_windows
        ]

        atr_feature = _atr(pl.col("high"), pl.col("low"), pl.col("close"))
        adx_feature = _adx(pl.col("high"), pl.col("low"), pl.col("close"))
        bb_upper, bb_lower, bb_width = _bollinger_bands(pl.col("close"), period=20, std_dev=2.0)
        range_feature = ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct")

        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        cum_typical_volume = (typical_price * pl.col("volume")).cumsum()
        cum_volume = pl.col("volume").cumsum()
        vwap = (cum_typical_volume / (cum_volume + 1e-9)).alias("vwap")
        drift = ((pl.col("close") - vwap) / vwap).alias("close_to_vwap")

        slope_feature = (
            pl.col("close").pct_change().rolling_mean(window_size=15, min_periods=10).alias("micro_trend")
        )

        tod_seconds = pl.col("ts").dt.hour() * 3600 + pl.col("ts").dt.minute() * 60 + pl.col("ts").dt.second()
        tod_fraction_expr = tod_seconds.cast(pl.Float64) / 86400.0
        tod_fraction = tod_fraction_expr.alias("tod_fraction")
        tod_sin = (tod_fraction_expr * 2 * pi).sin().alias("tod_sin")
        tod_cos = (tod_fraction_expr * 2 * pi).cos().alias("tod_cos")

        # Day of week features (0=Monday, 6=Sunday)
        dow = pl.col("ts").dt.weekday().alias("day_of_week")
        dow_sin = ((dow.cast(pl.Float64) / 7.0) * 2 * pi).sin().alias("dow_sin")
        dow_cos = ((dow.cast(pl.Float64) / 7.0) * 2 * pi).cos().alias("dow_cos")

        # Enhanced volatility features
        vol_regime = _volatility_regime(pl.col("close"), short_window=10, long_window=50)
        price_position = _price_position_in_range(pl.col("high"), pl.col("low"), pl.col("close"), window=14)

        liquidity_features = []
        for window in self.liquidity_windows:
            liquidity_features.extend(
                [
                    pl.col("volume").rolling_mean(window_size=window, min_periods=window // 2).alias(f"volume_mean_{window}"),
                    pl.col("volume").rolling_std(window_size=window, min_periods=window // 2).alias(f"volume_std_{window}"),
                ]
            )

        coin_age = (
            (pl.col("ts") - pl.col("ts").first()).dt.total_days().alias("coin_age_days")
        )

        # ========================================
        # REVUELTO FEATURES - Battle-Tested Alpha
        # ========================================

        # Breakout Features
        ignition = _ignition_score(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), window=20)
        breakout_thrust = _breakout_thrust(pl.col("high"), pl.col("close"), window=20)
        breakout_qual = _breakout_quality(pl.col("close"), pl.col("volume"), window=20)

        # Compression/Range Features
        compression = _compression_score(pl.col("high"), pl.col("low"), window=20)
        nr7_dens = _nr7_density(pl.col("high"), pl.col("low"), window=20)
        compress_rank = _compression_rank(pl.col("high"), pl.col("low"), window=50)
        mean_rev = _mean_revert_bias(pl.col("close"), window=20)
        pullback = _pullback_depth(pl.col("close"), window=20)

        # Microstructure Features
        micro = _micro_score(pl.col("close"), pl.col("volume"), window=10)
        uptick = _uptick_ratio(pl.col("close"), window=20)
        spread = _spread_bps(pl.col("high"), pl.col("low"), pl.col("close"))

        # Relative Strength Features (using close as benchmark for now)
        leader = _leader_bias(pl.col("close"), pl.col("close").rolling_mean(100), window=20)
        rs = _rs_score(pl.col("close"), pl.col("close").rolling_mean(100), short_window=10, med_window=20)

        # Trend Features
        trend_str = _trend_strength(pl.col("close"), window=20)
        ema_slope_feat = _ema_slope(pl.col("close"), period=20)
        momentum_slope_feat = _momentum_slope(pl.col("close"), short=5, long=20)
        htf = _htf_bias(pl.col("close"), htf_period=100)

        # Volume Features
        vol_jump = _vol_jump_z(pl.col("volume"), window=20)

        # Distribution Features
        kurt = _kurtosis(pl.col("close"), window=20)

        feature_frame = base.with_columns(
            [
                *zscore_features,
                *ema_features,
                *rsi_features,
                *volatility_features,
                atr_feature,
                adx_feature,
                bb_upper,
                bb_lower,
                bb_width,
                range_feature,
                vwap,
                drift,
                slope_feature,
                tod_fraction,
                tod_sin,
                tod_cos,
                dow,
                dow_sin,
                dow_cos,
                vol_regime,
                price_position,
                *liquidity_features,
                coin_age,
                # Revuelto Features
                ignition,
                breakout_thrust,
                breakout_qual,
                compression,
                nr7_dens,
                compress_rank,
                mean_rev,
                pullback,
                micro,
                uptick,
                spread,
                leader,
                rs,
                trend_str,
                ema_slope_feat,
                momentum_slope_feat,
                htf,
                vol_jump,
                kurt,
            ]
        )

        filled = feature_frame.fill_null(strategy="forward").fill_null(strategy="backward").fill_nan(0.0)
        return filled

    def build_with_market_context(
        self,
        frame: pl.DataFrame,
        btc_frame: Optional[pl.DataFrame] = None,
        eth_frame: Optional[pl.DataFrame] = None,
        sol_frame: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Build features WITH cross-asset market context.

        This method adds cross-asset features based on market leaders (BTC, ETH, SOL).
        Use this when you have benchmark data available for more intelligent features.

        Args:
            frame: Asset price data (must have: ts, close, high, low, volume)
            btc_frame: BTC price data (must have: ts, close) - optional
            eth_frame: ETH price data (must have: ts, close) - optional
            sol_frame: SOL price data (must have: ts, close) - optional

        Returns:
            DataFrame with all features including cross-asset features
        """
        # First, calculate all base features
        base_features = self.build(frame)

        # If no benchmark data provided, return base features only
        if btc_frame is None and eth_frame is None and sol_frame is None:
            return base_features

        # Merge benchmark data on timestamp
        # Start with base features
        result = base_features

        # Add BTC features if available
        if btc_frame is not None and "close" in btc_frame.columns:
            # Ensure btc_frame has ts column
            if "ts" not in btc_frame.columns:
                raise ValueError("btc_frame must have 'ts' column")

            # Join on timestamp
            btc_data = btc_frame.select([pl.col("ts"), pl.col("close").alias("btc_close")])

            result = result.join(btc_data, on="ts", how="left")

            # Calculate BTC features
            btc_beta = _beta_vs_benchmark(pl.col("close"), pl.col("btc_close"), window=30)
            btc_corr = _cross_asset_correlation(pl.col("close"), pl.col("btc_close"), window=30)
            btc_divergence = _market_divergence(pl.col("close"), pl.col("btc_close"), short_window=5, long_window=20)
            btc_vol = _leader_volatility(pl.col("btc_close"), window=20)
            btc_momentum = _cross_asset_momentum(pl.col("close"), pl.col("btc_close"), window=10)

            result = result.with_columns([
                btc_beta.alias("btc_beta"),
                btc_corr.alias("btc_correlation"),
                btc_divergence.alias("btc_divergence"),
                btc_vol.alias("btc_volatility"),
                btc_momentum.alias("btc_relative_momentum"),
            ])

        # Add ETH features if available
        if eth_frame is not None and "close" in eth_frame.columns:
            if "ts" not in eth_frame.columns:
                raise ValueError("eth_frame must have 'ts' column")

            eth_data = eth_frame.select([pl.col("ts"), pl.col("close").alias("eth_close")])

            result = result.join(eth_data, on="ts", how="left")

            eth_beta = _beta_vs_benchmark(pl.col("close"), pl.col("eth_close"), window=30)
            eth_corr = _cross_asset_correlation(pl.col("close"), pl.col("eth_close"), window=30)
            eth_divergence = _market_divergence(pl.col("close"), pl.col("eth_close"), short_window=5, long_window=20)
            eth_vol = _leader_volatility(pl.col("eth_close"), window=20)
            eth_momentum = _cross_asset_momentum(pl.col("close"), pl.col("eth_close"), window=10)

            result = result.with_columns([
                eth_beta.alias("eth_beta"),
                eth_corr.alias("eth_correlation"),
                eth_divergence.alias("eth_divergence"),
                eth_vol.alias("eth_volatility"),
                eth_momentum.alias("eth_relative_momentum"),
            ])

        # Add SOL features if available (important for Solana ecosystem coins)
        if sol_frame is not None and "close" in sol_frame.columns:
            if "ts" not in sol_frame.columns:
                raise ValueError("sol_frame must have 'ts' column")

            sol_data = sol_frame.select([pl.col("ts"), pl.col("close").alias("sol_close")])

            result = result.join(sol_data, on="ts", how="left")

            sol_beta = _beta_vs_benchmark(pl.col("close"), pl.col("sol_close"), window=30)
            sol_corr = _cross_asset_correlation(pl.col("close"), pl.col("sol_close"), window=30)
            sol_divergence = _market_divergence(pl.col("close"), pl.col("sol_close"), short_window=5, long_window=20)
            sol_vol = _leader_volatility(pl.col("sol_close"), window=20)
            sol_momentum = _cross_asset_momentum(pl.col("close"), pl.col("sol_close"), window=10)

            result = result.with_columns([
                sol_beta.alias("sol_beta"),
                sol_corr.alias("sol_correlation"),
                sol_divergence.alias("sol_divergence"),
                sol_vol.alias("sol_volatility"),
                sol_momentum.alias("sol_relative_momentum"),
            ])

        # Drop the benchmark close columns (we don't need them as features)
        cols_to_drop = []
        if "btc_close" in result.columns:
            cols_to_drop.append("btc_close")
        if "eth_close" in result.columns:
            cols_to_drop.append("eth_close")
        if "sol_close" in result.columns:
            cols_to_drop.append("sol_close")

        if cols_to_drop:
            result = result.drop(cols_to_drop)

        # Fill nulls
        result = result.fill_null(strategy="forward").fill_null(strategy="backward").fill_nan(0.0)

        return result
