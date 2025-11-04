"""
Granger Causality Analysis for Cross-Asset Relationships

Implements online Granger causality testing to determine if one asset's movements
predict another asset's movements.

Granger causality tests:
- Does X(t-1) help predict Y(t)?
- Compare: Y(t) ~ Y(t-1) vs Y(t) ~ Y(t-1) + X(t-1)
- If adding X improves prediction → X "Granger-causes" Y

This is more rigorous than simple correlation and helps identify true predictive
relationships for better entry timing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import structlog

from .market_structure import PriceData

logger = structlog.get_logger(__name__)


@dataclass
class GrangerCausalRelationship:
    """Granger causal relationship between two assets."""

    leader: str  # Asset that predicts (e.g., "BTC")
    follower: str  # Asset being predicted (e.g., "SOL")
    optimal_lag: int  # Best lag in periods
    f_statistic: float  # F-test statistic
    p_value: float  # Statistical significance
    strength: float  # 0-1, how much variance explained
    r_squared_improvement: float  # R² improvement by adding leader
    confidence: float  # 0-1, reliability of relationship
    sample_size: int
    last_updated: datetime


class GrangerCausalityDetector:
    """
    Online Granger causality detector.

    Tests if leader asset helps predict follower asset using incremental
    regression analysis.
    """

    def __init__(
        self,
        max_lag: int = 10,
        window_days: int = 30,
        min_periods: int = 50,
        significance_level: float = 0.05,
    ):
        """
        Initialize Granger causality detector.

        Args:
            max_lag: Maximum lag to test (periods)
            window_days: Rolling window for testing
            min_periods: Minimum data points needed
            significance_level: P-value threshold for significance
        """
        self.max_lag = max_lag
        self.window_days = window_days
        self.min_periods = min_periods
        self.significance_level = significance_level

        logger.info(
            "granger_causality_detector_initialized",
            max_lag=max_lag,
            window_days=window_days,
            min_periods=min_periods,
        )

    def test_causality(
        self,
        leader_data: PriceData,
        follower_data: PriceData,
        current_time: datetime,
    ) -> Optional[GrangerCausalRelationship]:
        """
        Test if leader Granger-causes follower.

        Args:
            leader_data: Leader asset price data
            follower_data: Follower asset price data
            current_time: Current timestamp

        Returns:
            GrangerCausalRelationship or None if insufficient data
        """
        # Get recent window
        cutoff = current_time - timedelta(days=self.window_days)

        # Align data
        leader_dict = {t: r for t, r in zip(leader_data.timestamps, leader_data.returns)}
        follower_dict = {t: r for t, r in zip(follower_data.timestamps, follower_data.returns)}

        leader_returns = []
        follower_returns = []
        timestamps = []

        for ts in leader_data.timestamps:
            if ts >= cutoff and ts in follower_dict:
                leader_returns.append(leader_dict[ts])
                follower_returns.append(follower_dict[ts])
                timestamps.append(ts)

        if len(leader_returns) < self.min_periods:
            return None

        # Convert to arrays
        leader_arr = np.array(leader_returns)
        follower_arr = np.array(follower_returns)

        # Remove NaN/Inf
        mask = np.isfinite(leader_arr) & np.isfinite(follower_arr)
        leader_arr = leader_arr[mask]
        follower_arr = follower_arr[mask]

        if len(leader_arr) < self.min_periods:
            return None

        # Test each lag and find optimal
        best_lag = 1
        best_f_stat = 0.0
        best_p_value = 1.0
        best_r2_improvement = 0.0

        for lag in range(1, min(self.max_lag + 1, len(follower_arr) // 4)):
            f_stat, p_value, r2_improvement = self._test_lag(
                leader_arr, follower_arr, lag
            )

            if p_value < best_p_value:
                best_lag = lag
                best_f_stat = f_stat
                best_p_value = p_value
                best_r2_improvement = r2_improvement

        # Calculate strength and confidence
        strength = min(best_r2_improvement * 10, 1.0)  # Scale R² improvement
        confidence = 1.0 - best_p_value  # Lower p-value = higher confidence

        # Only return if statistically significant
        if best_p_value > self.significance_level:
            strength = 0.0  # Not significant
            confidence = 0.0

        return GrangerCausalRelationship(
            leader=leader_data.timestamps[0].strftime("%Y%m%d") if leader_data.timestamps else "unknown",
            follower=follower_data.timestamps[0].strftime("%Y%m%d") if follower_data.timestamps else "unknown",
            optimal_lag=best_lag,
            f_statistic=float(best_f_stat),
            p_value=float(best_p_value),
            strength=float(strength),
            r_squared_improvement=float(best_r2_improvement),
            confidence=float(confidence),
            sample_size=len(follower_arr),
            last_updated=current_time,
        )

    def _test_lag(
        self,
        leader: np.ndarray,
        follower: np.ndarray,
        lag: int,
    ) -> Tuple[float, float, float]:
        """
        Test Granger causality at specific lag.

        Compares:
        - Restricted model: Y(t) = c + b1*Y(t-1) + ... + bn*Y(t-n)
        - Unrestricted model: Y(t) = c + b1*Y(t-1) + ... + bn*Y(t-n) + d1*X(t-1) + ... + dm*X(t-m)

        Args:
            leader: Leader returns
            follower: Follower returns
            lag: Number of lags

        Returns:
            (f_statistic, p_value, r2_improvement)
        """
        # Prepare data matrices
        n = len(follower) - lag
        if n < 20:
            return 0.0, 1.0, 0.0

        # Dependent variable: Y(t)
        y = follower[lag:]

        # Restricted model: only past Y values
        X_restricted = self._build_design_matrix(follower, lag, lag)

        # Unrestricted model: past Y + past X values
        X_unrestricted = self._build_design_matrix_with_leader(
            follower, leader, lag, lag
        )

        if X_restricted is None or X_unrestricted is None:
            return 0.0, 1.0, 0.0

        # Fit models
        try:
            # Restricted model
            beta_r, rss_r, r2_r = self._fit_ols(X_restricted, y)

            # Unrestricted model
            beta_u, rss_u, r2_u = self._fit_ols(X_unrestricted, y)

            # F-test
            # F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
            # q = number of restrictions (number of leader lags)
            # k = number of parameters in unrestricted model

            q = lag  # Number of leader lags added
            k = X_unrestricted.shape[1]
            n_samples = len(y)

            if rss_u < 1e-10 or n_samples <= k:
                return 0.0, 1.0, 0.0

            f_stat = ((rss_r - rss_u) / q) / (rss_u / (n_samples - k))

            # P-value from F-distribution
            p_value = 1.0 - stats.f.cdf(f_stat, q, n_samples - k)

            # R² improvement
            r2_improvement = r2_u - r2_r

            return float(f_stat), float(p_value), float(r2_improvement)

        except Exception as e:
            logger.warning("granger_test_failed", lag=lag, error=str(e))
            return 0.0, 1.0, 0.0

    def _build_design_matrix(
        self,
        series: np.ndarray,
        lag: int,
        n_lags: int,
    ) -> Optional[np.ndarray]:
        """
        Build design matrix with lagged values.

        Args:
            series: Time series
            lag: Starting point
            n_lags: Number of lags to include

        Returns:
            Design matrix or None
        """
        n = len(series) - lag
        if n < 10:
            return None

        X = np.ones((n, n_lags + 1))  # +1 for intercept

        for i in range(n_lags):
            X[:, i + 1] = series[lag - i - 1 : lag - i - 1 + n]

        return X

    def _build_design_matrix_with_leader(
        self,
        follower: np.ndarray,
        leader: np.ndarray,
        lag: int,
        n_lags: int,
    ) -> Optional[np.ndarray]:
        """
        Build design matrix with follower and leader lags.

        Args:
            follower: Follower series
            leader: Leader series
            lag: Starting point
            n_lags: Number of lags

        Returns:
            Design matrix or None
        """
        n = len(follower) - lag
        if n < 10:
            return None

        # Start with follower lags
        X = np.ones((n, 2 * n_lags + 1))  # +1 for intercept

        # Follower lags
        for i in range(n_lags):
            X[:, i + 1] = follower[lag - i - 1 : lag - i - 1 + n]

        # Leader lags
        for i in range(n_lags):
            X[:, n_lags + i + 1] = leader[lag - i - 1 : lag - i - 1 + n]

        return X

    def _fit_ols(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Fit OLS regression.

        Args:
            X: Design matrix
            y: Target variable

        Returns:
            (coefficients, residual_sum_squares, r_squared)
        """
        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y

            beta = np.linalg.solve(XtX, Xty)

            # Predictions
            y_pred = X @ beta

            # Residuals
            residuals = y - y_pred
            rss = np.sum(residuals ** 2)

            # R²
            tss = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - (rss / tss) if tss > 1e-10 else 0.0

            return beta, float(rss), float(r2)

        except np.linalg.LinAlgError:
            # Singular matrix
            return np.zeros(X.shape[1]), float(np.sum(y ** 2)), 0.0


class CausalGraphBuilder:
    """
    Builds a directed graph of Granger-causal relationships.

    Nodes = assets
    Edges = causal relationships (leader → follower)
    Edge weights = causality strength
    """

    def __init__(self, min_strength: float = 0.3, min_confidence: float = 0.7):
        """
        Initialize causal graph builder.

        Args:
            min_strength: Minimum strength to include edge
            min_confidence: Minimum confidence to include edge
        """
        self.min_strength = min_strength
        self.min_confidence = min_confidence
        self.relationships: Dict[Tuple[str, str], GrangerCausalRelationship] = {}

        logger.info(
            "causal_graph_builder_initialized",
            min_strength=min_strength,
            min_confidence=min_confidence,
        )

    def add_relationship(self, relationship: GrangerCausalRelationship) -> None:
        """
        Add causal relationship to graph.

        Args:
            relationship: Granger causal relationship
        """
        # Only add if meets thresholds
        if (
            relationship.strength >= self.min_strength
            and relationship.confidence >= self.min_confidence
        ):
            key = (relationship.leader, relationship.follower)
            self.relationships[key] = relationship

            logger.debug(
                "causal_relationship_added",
                leader=relationship.leader,
                follower=relationship.follower,
                strength=relationship.strength,
                lag=relationship.optimal_lag,
            )

    def get_leaders_for(self, follower: str) -> List[GrangerCausalRelationship]:
        """
        Get all leaders that cause this follower.

        Args:
            follower: Follower asset

        Returns:
            List of causal relationships
        """
        return [
            rel
            for (leader, fol), rel in self.relationships.items()
            if fol == follower
        ]

    def get_followers_for(self, leader: str) -> List[GrangerCausalRelationship]:
        """
        Get all followers caused by this leader.

        Args:
            leader: Leader asset

        Returns:
            List of causal relationships
        """
        return [
            rel
            for (lead, follower), rel in self.relationships.items()
            if lead == leader
        ]

    def get_optimal_entry_timing(
        self,
        leader: str,
        follower: str,
    ) -> Optional[Tuple[int, float]]:
        """
        Get optimal entry timing for follower after leader signal.

        Args:
            leader: Leader asset
            follower: Follower asset

        Returns:
            (optimal_lag, confidence) or None
        """
        key = (leader, follower)
        if key in self.relationships:
            rel = self.relationships[key]
            return (rel.optimal_lag, rel.confidence)

        return None

    def get_graph_stats(self) -> Dict:
        """
        Get statistics about causal graph.

        Returns:
            Dictionary of statistics
        """
        if not self.relationships:
            return {
                "num_relationships": 0,
                "avg_strength": 0.0,
                "avg_confidence": 0.0,
                "avg_lag": 0.0,
            }

        strengths = [rel.strength for rel in self.relationships.values()]
        confidences = [rel.confidence for rel in self.relationships.values()]
        lags = [rel.optimal_lag for rel in self.relationships.values()]

        return {
            "num_relationships": len(self.relationships),
            "avg_strength": np.mean(strengths),
            "avg_confidence": np.mean(confidences),
            "avg_lag": np.mean(lags),
            "max_strength": np.max(strengths),
            "max_confidence": np.max(confidences),
        }

    def clear(self) -> None:
        """Clear all relationships."""
        self.relationships.clear()
        logger.info("causal_graph_cleared")
