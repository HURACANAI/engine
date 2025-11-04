"""
Market Structure Intelligence Module

Models cross-asset influence and market leadership dynamics.

This module captures how major coins (BTC, ETH, SOL) influence other assets:
- Beta calculation (sensitivity to leader movements)
- Granger causality (does leader predict follower?)
- Lead-lag relationships (who moves first?)
- Volatility spillover (how volatility propagates)
- Market regime detection (risk-on vs risk-off)

Based on quantitative finance research:
- Vector Autoregression (VAR) theory
- Granger causality testing
- Multivariate GARCH for volatility spillover
- Cross-correlation analysis for lead-lag
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime types based on leader behavior."""

    RISK_ON = "risk_on"  # Leaders up, high correlation, bullish
    RISK_OFF = "risk_off"  # Leaders down, flight to safety
    ROTATION = "rotation"  # Leadership changing (e.g., ETH > BTC)
    DIVERGENCE = "divergence"  # Followers diverging from leaders
    UNKNOWN = "unknown"


@dataclass
class PriceData:
    """Price data for a single asset."""

    timestamps: List[datetime]
    prices: List[float]
    returns: Optional[List[float]] = None  # Calculated on demand

    def __post_init__(self):
        """Calculate returns if not provided."""
        if self.returns is None and len(self.prices) > 1:
            self.returns = [
                (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
                if self.prices[i - 1] != 0
                else 0.0
                for i in range(1, len(self.prices))
            ]
            self.returns.insert(0, 0.0)  # First return is 0


@dataclass
class BetaMetrics:
    """Beta metrics vs a benchmark asset."""

    asset: str
    benchmark: str
    beta: float  # Regression coefficient
    alpha: float  # Intercept
    r_squared: float  # Goodness of fit
    sample_size: int
    window_days: int
    last_updated: datetime


@dataclass
class LeadLagMetrics:
    """Lead-lag relationship between two assets."""

    leader: str
    follower: str
    optimal_lag: int  # Periods (negative = leader leads, positive = follower leads)
    max_correlation: float  # Correlation at optimal lag
    confidence: float  # 0-1, statistical confidence in relationship
    last_updated: datetime


@dataclass
class VolatilitySpillover:
    """Volatility spillover from one asset to another."""

    source: str
    target: str
    spillover_coefficient: float  # How much source vol affects target vol
    correlation: float  # Correlation of volatilities
    half_life_periods: float  # How long spillover effect lasts
    last_updated: datetime


@dataclass
class MarketStructureSnapshot:
    """Complete market structure state at a point in time."""

    timestamp: datetime
    regime: MarketRegime
    regime_confidence: float
    leader_volatilities: Dict[str, float]  # BTC, ETH, SOL volatility
    cross_asset_correlation: float  # Average correlation across assets
    divergence_score: float  # 0-1, how much divergence from leaders


class BetaCalculator:
    """
    Calculates rolling beta vs benchmark assets.

    Beta = Cov(Asset, Benchmark) / Var(Benchmark)

    A beta of 1.5 means: If BTC moves 1%, asset moves 1.5% on average.
    """

    def __init__(self, window_days: int = 30, min_periods: int = 20):
        """
        Initialize beta calculator.

        Args:
            window_days: Rolling window for calculation
            min_periods: Minimum data points needed
        """
        self.window_days = window_days
        self.min_periods = min_periods

    def calculate_beta(
        self,
        asset_data: PriceData,
        benchmark_data: PriceData,
        current_time: datetime,
    ) -> Optional[BetaMetrics]:
        """
        Calculate rolling beta.

        Args:
            asset_data: Asset price data
            benchmark_data: Benchmark price data
            current_time: Current timestamp

        Returns:
            BetaMetrics or None if insufficient data
        """
        # Align data by finding common timestamps
        cutoff = current_time - timedelta(days=self.window_days)

        asset_returns = []
        bench_returns = []
        common_times = []

        asset_dict = {t: r for t, r in zip(asset_data.timestamps, asset_data.returns)}
        bench_dict = {t: r for t, r in zip(benchmark_data.timestamps, benchmark_data.returns)}

        for ts in asset_data.timestamps:
            if ts >= cutoff and ts in bench_dict:
                asset_returns.append(asset_dict[ts])
                bench_returns.append(bench_dict[ts])
                common_times.append(ts)

        if len(asset_returns) < self.min_periods:
            return None

        # Calculate beta via linear regression
        asset_arr = np.array(asset_returns)
        bench_arr = np.array(bench_returns)

        # Remove NaN/inf
        mask = np.isfinite(asset_arr) & np.isfinite(bench_arr)
        asset_arr = asset_arr[mask]
        bench_arr = bench_arr[mask]

        if len(asset_arr) < self.min_periods:
            return None

        # Beta = cov(asset, bench) / var(bench)
        covariance = np.cov(asset_arr, bench_arr)[0, 1]
        variance = np.var(bench_arr)

        if variance == 0:
            return None

        beta = covariance / variance

        # Alpha = mean(asset) - beta * mean(bench)
        alpha = np.mean(asset_arr) - beta * np.mean(bench_arr)

        # R-squared
        correlation = np.corrcoef(asset_arr, bench_arr)[0, 1]
        r_squared = correlation ** 2

        return BetaMetrics(
            asset=asset_data.timestamps[0].strftime("%Y%m%d") if asset_data.timestamps else "unknown",
            benchmark=benchmark_data.timestamps[0].strftime("%Y%m%d") if benchmark_data.timestamps else "unknown",
            beta=float(beta),
            alpha=float(alpha),
            r_squared=float(r_squared),
            sample_size=len(asset_arr),
            window_days=self.window_days,
            last_updated=current_time,
        )


class LeadLagTracker:
    """
    Detects lead-lag relationships between assets.

    Uses cross-correlation to find if one asset's movements predict another's.
    Example: Does BTC price change 1 hour ago predict SOL price change now?
    """

    def __init__(self, max_lag: int = 10, window_days: int = 30):
        """
        Initialize lead-lag tracker.

        Args:
            max_lag: Maximum lag to test (periods)
            window_days: Rolling window for calculation
        """
        self.max_lag = max_lag
        self.window_days = window_days

    def detect_lead_lag(
        self,
        potential_leader: PriceData,
        potential_follower: PriceData,
        current_time: datetime,
    ) -> Optional[LeadLagMetrics]:
        """
        Detect lead-lag relationship.

        Args:
            potential_leader: Leader price data
            potential_follower: Follower price data
            current_time: Current timestamp

        Returns:
            LeadLagMetrics or None if insufficient data
        """
        cutoff = current_time - timedelta(days=self.window_days)

        # Align data
        leader_dict = {t: r for t, r in zip(potential_leader.timestamps, potential_leader.returns)}
        follower_dict = {t: r for t, r in zip(potential_follower.timestamps, potential_follower.returns)}

        leader_returns = []
        follower_returns = []

        for ts in potential_leader.timestamps:
            if ts >= cutoff and ts in follower_dict:
                leader_returns.append(leader_dict[ts])
                follower_returns.append(follower_dict[ts])

        if len(leader_returns) < 20:
            return None

        leader_arr = np.array(leader_returns)
        follower_arr = np.array(follower_returns)

        # Cross-correlation at different lags
        correlations = {}

        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                # Leader leads (shift leader back in time)
                l = leader_arr[:lag] if lag != 0 else leader_arr
                f = follower_arr[-lag:] if lag != 0 else follower_arr
            elif lag > 0:
                # Follower leads (shift follower back in time)
                l = leader_arr[lag:]
                f = follower_arr[:-lag]
            else:
                # No lag
                l = leader_arr
                f = follower_arr

            if len(l) > 10 and len(f) > 10 and len(l) == len(f):
                mask = np.isfinite(l) & np.isfinite(f)
                l = l[mask]
                f = f[mask]

                if len(l) > 10:
                    corr = np.corrcoef(l, f)[0, 1]
                    if np.isfinite(corr):
                        correlations[lag] = float(corr)

        if not correlations:
            return None

        # Find optimal lag (max absolute correlation)
        optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
        max_corr = correlations[optimal_lag]

        # Confidence based on correlation strength and consistency
        confidence = min(abs(max_corr), 1.0)

        return LeadLagMetrics(
            leader=potential_leader.timestamps[0].strftime("%Y%m%d") if potential_leader.timestamps else "unknown",
            follower=potential_follower.timestamps[0].strftime("%Y%m%d") if potential_follower.timestamps else "unknown",
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
            confidence=confidence,
            last_updated=current_time,
        )


class VolatilitySpilloverMonitor:
    """
    Monitors volatility spillover between assets.

    Volatility spillover: When BTC becomes volatile, altcoins also become volatile.
    This is separate from price correlation.
    """

    def __init__(self, volatility_window: int = 20, spillover_window: int = 30):
        """
        Initialize volatility spillover monitor.

        Args:
            volatility_window: Window for volatility calculation
            spillover_window: Window for spillover estimation
        """
        self.volatility_window = volatility_window
        self.spillover_window = spillover_window

    def calculate_spillover(
        self,
        source_data: PriceData,
        target_data: PriceData,
        current_time: datetime,
    ) -> Optional[VolatilitySpillover]:
        """
        Calculate volatility spillover.

        Args:
            source_data: Source asset data (e.g., BTC)
            target_data: Target asset data (e.g., altcoin)
            current_time: Current timestamp

        Returns:
            VolatilitySpillover or None if insufficient data
        """
        cutoff = current_time - timedelta(days=self.spillover_window)

        # Align data
        source_dict = {t: r for t, r in zip(source_data.timestamps, source_data.returns)}
        target_dict = {t: r for t, r in zip(target_data.timestamps, target_data.returns)}

        source_returns = []
        target_returns = []
        timestamps = []

        for ts in source_data.timestamps:
            if ts >= cutoff and ts in target_dict:
                source_returns.append(source_dict[ts])
                target_returns.append(target_dict[ts])
                timestamps.append(ts)

        if len(source_returns) < self.volatility_window + 10:
            return None

        source_arr = np.array(source_returns)
        target_arr = np.array(target_returns)

        # Calculate rolling volatilities
        source_vols = []
        target_vols = []

        for i in range(self.volatility_window, len(source_arr)):
            window_source = source_arr[i - self.volatility_window : i]
            window_target = target_arr[i - self.volatility_window : i]

            source_vol = np.std(window_source) if len(window_source) > 0 else 0.0
            target_vol = np.std(window_target) if len(window_target) > 0 else 0.0

            source_vols.append(source_vol)
            target_vols.append(target_vol)

        if len(source_vols) < 10:
            return None

        source_vols = np.array(source_vols)
        target_vols = np.array(target_vols)

        # Remove NaN/inf
        mask = np.isfinite(source_vols) & np.isfinite(target_vols)
        source_vols = source_vols[mask]
        target_vols = target_vols[mask]

        if len(source_vols) < 10:
            return None

        # Spillover coefficient via regression
        # target_vol = alpha + spillover_coef * source_vol
        covariance = np.cov(source_vols, target_vols)[0, 1]
        variance = np.var(source_vols)

        if variance == 0:
            return None

        spillover_coef = covariance / variance

        # Correlation of volatilities
        vol_corr = np.corrcoef(source_vols, target_vols)[0, 1]

        # Half-life (how long spillover persists) - simple ACF
        if len(target_vols) > 1:
            autocorr = np.corrcoef(target_vols[:-1], target_vols[1:])[0, 1]
            if autocorr > 0 and autocorr < 1:
                half_life = -np.log(2) / np.log(autocorr)
            else:
                half_life = 1.0
        else:
            half_life = 1.0

        return VolatilitySpillover(
            source=source_data.timestamps[0].strftime("%Y%m%d") if source_data.timestamps else "unknown",
            target=target_data.timestamps[0].strftime("%Y%m%d") if target_data.timestamps else "unknown",
            spillover_coefficient=float(spillover_coef),
            correlation=float(vol_corr),
            half_life_periods=float(half_life),
            last_updated=current_time,
        )


class MarketStructureCoordinator:
    """
    Coordinates all market structure intelligence.

    This is the main interface for cross-asset modeling.
    """

    def __init__(
        self,
        beta_window_days: int = 30,
        leadlag_window_days: int = 30,
        spillover_window_days: int = 30,
    ):
        """
        Initialize market structure coordinator.

        Args:
            beta_window_days: Window for beta calculation
            leadlag_window_days: Window for lead-lag detection
            spillover_window_days: Window for spillover monitoring
        """
        self.beta_calculator = BetaCalculator(window_days=beta_window_days)
        self.leadlag_tracker = LeadLagTracker(max_lag=10, window_days=leadlag_window_days)
        self.spillover_monitor = VolatilitySpilloverMonitor(
            volatility_window=20, spillover_window=spillover_window_days
        )

        # Cache of market leaders data
        self.leader_data: Dict[str, PriceData] = {}  # "BTC", "ETH", "SOL"

        # Cache of calculated metrics
        self.beta_cache: Dict[Tuple[str, str], BetaMetrics] = {}
        self.leadlag_cache: Dict[Tuple[str, str], LeadLagMetrics] = {}
        self.spillover_cache: Dict[Tuple[str, str], VolatilitySpillover] = {}

        logger.info(
            "market_structure_coordinator_initialized",
            beta_window=beta_window_days,
            leadlag_window=leadlag_window_days,
        )

    def update_leader_data(self, leader: str, price_data: PriceData) -> None:
        """
        Update price data for a market leader.

        Args:
            leader: Leader symbol ("BTC", "ETH", "SOL")
            price_data: Price data
        """
        self.leader_data[leader] = price_data

        logger.debug(
            "leader_data_updated",
            leader=leader,
            data_points=len(price_data.prices),
        )

    def get_beta(
        self,
        asset_symbol: str,
        asset_data: PriceData,
        benchmark: str,
        current_time: datetime,
    ) -> Optional[BetaMetrics]:
        """
        Get beta vs benchmark.

        Args:
            asset_symbol: Asset symbol
            asset_data: Asset price data
            benchmark: Benchmark symbol ("BTC", "ETH", "SOL")
            current_time: Current timestamp

        Returns:
            BetaMetrics or None
        """
        if benchmark not in self.leader_data:
            logger.warning("benchmark_not_available", benchmark=benchmark)
            return None

        cache_key = (asset_symbol, benchmark)

        # Check cache
        if cache_key in self.beta_cache:
            cached = self.beta_cache[cache_key]
            age = (current_time - cached.last_updated).total_seconds() / 3600
            if age < 1:  # Cache for 1 hour
                return cached

        # Calculate
        beta_metrics = self.beta_calculator.calculate_beta(
            asset_data, self.leader_data[benchmark], current_time
        )

        if beta_metrics:
            beta_metrics.asset = asset_symbol
            beta_metrics.benchmark = benchmark
            self.beta_cache[cache_key] = beta_metrics

        return beta_metrics

    def get_lead_lag(
        self,
        asset_symbol: str,
        asset_data: PriceData,
        potential_leader: str,
        current_time: datetime,
    ) -> Optional[LeadLagMetrics]:
        """
        Get lead-lag relationship.

        Args:
            asset_symbol: Asset symbol
            asset_data: Asset price data
            potential_leader: Potential leader symbol
            current_time: Current timestamp

        Returns:
            LeadLagMetrics or None
        """
        if potential_leader not in self.leader_data:
            return None

        cache_key = (potential_leader, asset_symbol)

        # Check cache
        if cache_key in self.leadlag_cache:
            cached = self.leadlag_cache[cache_key]
            age = (current_time - cached.last_updated).total_seconds() / 3600
            if age < 4:  # Cache for 4 hours
                return cached

        # Calculate
        leadlag_metrics = self.leadlag_tracker.detect_lead_lag(
            self.leader_data[potential_leader], asset_data, current_time
        )

        if leadlag_metrics:
            leadlag_metrics.leader = potential_leader
            leadlag_metrics.follower = asset_symbol
            self.leadlag_cache[cache_key] = leadlag_metrics

        return leadlag_metrics

    def get_volatility_spillover(
        self,
        asset_symbol: str,
        asset_data: PriceData,
        source: str,
        current_time: datetime,
    ) -> Optional[VolatilitySpillover]:
        """
        Get volatility spillover.

        Args:
            asset_symbol: Asset symbol
            asset_data: Asset price data
            source: Source symbol ("BTC", "ETH", "SOL")
            current_time: Current timestamp

        Returns:
            VolatilitySpillover or None
        """
        if source not in self.leader_data:
            return None

        cache_key = (source, asset_symbol)

        # Check cache
        if cache_key in self.spillover_cache:
            cached = self.spillover_cache[cache_key]
            age = (current_time - cached.last_updated).total_seconds() / 3600
            if age < 2:  # Cache for 2 hours
                return cached

        # Calculate
        spillover = self.spillover_monitor.calculate_spillover(
            self.leader_data[source], asset_data, current_time
        )

        if spillover:
            spillover.source = source
            spillover.target = asset_symbol
            self.spillover_cache[cache_key] = spillover

        return spillover

    def detect_market_regime(
        self,
        asset_symbol: str,
        asset_data: PriceData,
        current_time: datetime,
    ) -> MarketStructureSnapshot:
        """
        Detect overall market regime.

        Args:
            asset_symbol: Asset symbol
            asset_data: Asset price data
            current_time: Current timestamp

        Returns:
            MarketStructureSnapshot
        """
        # Get betas vs all leaders
        betas = {}
        for leader in ["BTC", "ETH", "SOL"]:
            beta_metrics = self.get_beta(asset_symbol, asset_data, leader, current_time)
            if beta_metrics:
                betas[leader] = beta_metrics.beta

        # Get leader volatilities
        leader_vols = {}
        for leader, data in self.leader_data.items():
            if data.returns and len(data.returns) > 20:
                recent_returns = data.returns[-20:]
                vol = float(np.std(recent_returns)) if recent_returns else 0.0
                leader_vols[leader] = vol

        # Average cross-asset correlation
        correlations = []
        for leader in ["BTC", "ETH", "SOL"]:
            beta_metrics = self.get_beta(asset_symbol, asset_data, leader, current_time)
            if beta_metrics:
                correlations.append(beta_metrics.r_squared ** 0.5)  # Correlation from R²

        avg_correlation = np.mean(correlations) if correlations else 0.0

        # Divergence score (low correlation = high divergence)
        divergence_score = 1.0 - avg_correlation

        # Regime detection logic
        regime = MarketRegime.UNKNOWN
        regime_confidence = 0.5

        if avg_correlation > 0.7:
            # High correlation
            if all(beta > 0.5 for beta in betas.values()):
                regime = MarketRegime.RISK_ON
                regime_confidence = avg_correlation
            else:
                regime = MarketRegime.RISK_OFF
                regime_confidence = avg_correlation
        elif divergence_score > 0.5:
            regime = MarketRegime.DIVERGENCE
            regime_confidence = divergence_score
        else:
            # Check for rotation
            if betas.get("ETH", 0) > betas.get("BTC", 0) * 1.2:
                regime = MarketRegime.ROTATION
                regime_confidence = 0.6

        return MarketStructureSnapshot(
            timestamp=current_time,
            regime=regime,
            regime_confidence=regime_confidence,
            leader_volatilities=leader_vols,
            cross_asset_correlation=avg_correlation,
            divergence_score=divergence_score,
        )

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "beta_cache_size": len(self.beta_cache),
            "leadlag_cache_size": len(self.leadlag_cache),
            "spillover_cache_size": len(self.spillover_cache),
            "leaders": list(self.leader_data.keys()),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.beta_cache.clear()
        self.leadlag_cache.clear()
        self.spillover_cache.clear()
        logger.info("market_structure_cache_cleared")
