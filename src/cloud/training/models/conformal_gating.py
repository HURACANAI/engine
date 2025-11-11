"""
Conformal Prediction Gating - Distribution-Free Error Guarantees

Key Problem:
Traditional models give point predictions (e.g., "edge = +15 bps") with no reliability measure.
- We don't know if prediction is uncertain
- Can't guarantee error rate (e.g., "wrong <5% of time")
- No distribution assumptions needed

Solution: Conformal Prediction
- Uses past calibration data to build prediction intervals
- Guarantees coverage: "True value in interval with probability 1-α"
- Distribution-free: Works with ANY model
- Adaptive: Wider intervals when uncertain

Example:
    Traditional: "Predicted edge = +15 bps"
    → Take trade, hope it works

    Conformal: "Predicted edge = +15 bps, 90% interval: [+8, +22]"
    → Check if lower bound (+8 bps) beats costs (5 bps)
    → If yes, trade (even pessimistic case wins)
    → If no, skip (might lose money)

Benefits:
- Guaranteed error rates (e.g., wrong ≤5% of time)
- Uncertainty-aware decisions
- Works with existing models
- No assumptions about data distribution

Usage:
    conformal_gate = ConformalGate(
        significance_level=0.05,  # 95% coverage
        min_lower_bound_bps=5.0,   # Pessimistic case must beat 5 bps
    )

    # Calibrate on historical data
    conformal_gate.calibrate(
        predictions=[12.0, 15.5, 8.3, ...],
        actual_outcomes=[10.5, 18.2, 6.1, ...],
    )

    # Make prediction with interval
    result = conformal_gate.predict_with_interval(
        point_prediction=15.0,
        features={'confidence': 0.72, 'regime': 'TREND'},
    )

    if result.passes_gate:
        logger.info(f"Trade: {result.lower_bound:.1f} to {result.upper_bound:.1f} bps")
    else:
        logger.info(f"Skip: Pessimistic case ({result.lower_bound:.1f}) too low")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ConformalPrediction:
    """Prediction with conformal interval."""

    point_prediction: float  # Point estimate
    lower_bound: float  # Lower bound of interval
    upper_bound: float  # Upper bound of interval
    interval_width: float  # Width of interval (uncertainty)
    coverage_level: float  # e.g., 0.95 for 95% coverage

    passes_gate: bool
    reason: str


class ConformalGate:
    """
    Conformal prediction gate for distribution-free error guarantees.

    How It Works:
    1. Calibration: Collect errors on calibration set
    2. Quantile: Find error quantile at level (1-α)
    3. Prediction: New prediction ± quantile = interval
    4. Gate: Check if lower bound beats threshold

    Mathematical Guarantee:
    P(Y_true ∈ [pred - q, pred + q]) ≥ 1 - α

    where q = (1-α) quantile of |Y_true - Y_pred| on calibration set.

    Example:
        Calibration errors: [2, 5, 1, 8, 3, 4, 6, 2, 7, 3]
        α = 0.10 (90% coverage)
        q = 90th percentile = 7.0

        New prediction: 15 bps
        Interval: [15 - 7, 15 + 7] = [8, 22] bps

        Lower bound (8 bps) > threshold (5 bps) → PASS
    """

    def __init__(
        self,
        significance_level: float = 0.05,  # α (5% error rate)
        min_lower_bound_bps: float = 5.0,
        min_calibration_samples: int = 50,
    ):
        """
        Initialize conformal gate.

        Args:
            significance_level: α, desired error rate (0.05 = 95% coverage)
            min_lower_bound_bps: Minimum lower bound to pass gate
            min_calibration_samples: Min samples needed for calibration
        """
        self.alpha = significance_level
        self.min_lower_bound = min_lower_bound_bps
        self.min_samples = min_calibration_samples

        # Calibration data
        self.calibration_errors: List[float] = []
        self.is_calibrated = False
        self.error_quantile: Optional[float] = None

        logger.info(
            "conformal_gate_initialized",
            alpha=significance_level,
            coverage=1.0 - significance_level,
            min_lower_bound=min_lower_bound_bps,
        )

    def calibrate(
        self,
        predictions: List[float],
        actual_outcomes: List[float],
    ) -> None:
        """
        Calibrate conformal predictor using historical data.

        Args:
            predictions: Historical point predictions
            actual_outcomes: Actual realized outcomes
        """
        if len(predictions) != len(actual_outcomes):
            raise ValueError("Predictions and outcomes must have same length")

        if len(predictions) < self.min_samples:
            logger.warning(
                "insufficient_calibration_data",
                samples=len(predictions),
                required=self.min_samples,
            )
            return

        # Calculate absolute errors
        self.calibration_errors = [
            abs(pred - actual) for pred, actual in zip(predictions, actual_outcomes)
        ]

        # Calculate error quantile at level (1 - α)
        coverage_level = 1.0 - self.alpha
        self.error_quantile = np.percentile(self.calibration_errors, coverage_level * 100)

        self.is_calibrated = True

        logger.info(
            "conformal_gate_calibrated",
            samples=len(predictions),
            coverage_level=coverage_level,
            error_quantile=self.error_quantile,
            mean_error=np.mean(self.calibration_errors),
            max_error=np.max(self.calibration_errors),
        )

    def predict_with_interval(
        self,
        point_prediction: float,
        features: Optional[Dict[str, float]] = None,
    ) -> ConformalPrediction:
        """
        Make prediction with conformal interval.

        Args:
            point_prediction: Point estimate from model
            features: Optional features (for feature-conditional conformal)

        Returns:
            ConformalPrediction with interval and gate decision
        """
        if not self.is_calibrated:
            # Not calibrated - use wide default interval
            interval_width = 20.0  # Default: ±20 bps
            lower_bound = point_prediction - interval_width
            upper_bound = point_prediction + interval_width

            passes = lower_bound >= self.min_lower_bound
            reason = "Not calibrated - using default wide interval"

            logger.warning("conformal_prediction_uncalibrated")

        else:
            # Use calibrated quantile
            interval_width = self.error_quantile
            lower_bound = point_prediction - interval_width
            upper_bound = point_prediction + interval_width

            # Gate decision
            passes = lower_bound >= self.min_lower_bound

            if passes:
                reason = (
                    f"Lower bound ({lower_bound:.1f} bps) ≥ threshold ({self.min_lower_bound:.1f} bps), "
                    f"{(1-self.alpha)*100:.0f}% coverage"
                )
            else:
                reason = (
                    f"Lower bound ({lower_bound:.1f} bps) < threshold ({self.min_lower_bound:.1f} bps), "
                    f"even pessimistic case loses"
                )

        return ConformalPrediction(
            point_prediction=point_prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            interval_width=interval_width,
            coverage_level=1.0 - self.alpha,
            passes_gate=passes,
            reason=reason,
        )

    def add_calibration_sample(
        self,
        prediction: float,
        actual_outcome: float,
    ) -> None:
        """
        Add single sample to calibration set (online learning).

        Args:
            prediction: Predicted value
            actual_outcome: Actual realized value
        """
        error = abs(prediction - actual_outcome)
        self.calibration_errors.append(error)

        # Recalibrate if we have enough samples
        if len(self.calibration_errors) >= self.min_samples:
            coverage_level = 1.0 - self.alpha
            self.error_quantile = np.percentile(
                self.calibration_errors, coverage_level * 100
            )
            self.is_calibrated = True

    def get_statistics(self) -> Dict:
        """Get calibration statistics."""
        if not self.is_calibrated:
            return {
                'calibrated': False,
                'samples': len(self.calibration_errors),
                'required': self.min_samples,
            }

        return {
            'calibrated': True,
            'samples': len(self.calibration_errors),
            'coverage_level': 1.0 - self.alpha,
            'error_quantile': self.error_quantile,
            'mean_error': np.mean(self.calibration_errors),
            'median_error': np.median(self.calibration_errors),
            'max_error': np.max(self.calibration_errors),
            'min_error': np.min(self.calibration_errors),
        }


class AdaptiveConformalGate(ConformalGate):
    """
    Adaptive conformal predictor with feature-conditional intervals.

    Key Enhancement:
    Standard conformal uses same interval width for all predictions.
    Adaptive conformal adjusts interval based on difficulty:
    - Easy predictions (high confidence) → Narrow intervals
    - Hard predictions (low confidence) → Wide intervals

    How:
    1. Bin calibration samples by difficulty (e.g., by confidence)
    2. Calculate separate quantiles per bin
    3. For new prediction, use quantile from matching bin

    Example:
        High confidence (0.80-1.00): error quantile = 5 bps
        Med confidence (0.60-0.80): error quantile = 10 bps
        Low confidence (0.00-0.60): error quantile = 20 bps

        New prediction: +15 bps, confidence = 0.85 (high)
        → Use 5 bps quantile
        → Interval: [10, 20] bps (narrow!)

        New prediction: +15 bps, confidence = 0.55 (low)
        → Use 20 bps quantile
        → Interval: [-5, 35] bps (wide!)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_lower_bound_bps: float = 5.0,
        min_calibration_samples: int = 50,
        difficulty_feature: str = 'engine_conf',  # Feature to condition on
        n_bins: int = 5,  # Number of difficulty bins
    ):
        """
        Initialize adaptive conformal gate.

        Args:
            significance_level: α
            min_lower_bound_bps: Min lower bound
            min_calibration_samples: Min samples for calibration
            difficulty_feature: Feature to condition on (e.g., 'engine_conf')
            n_bins: Number of bins for adaptive intervals
        """
        super().__init__(significance_level, min_lower_bound_bps, min_calibration_samples)

        self.difficulty_feature = difficulty_feature
        self.n_bins = n_bins

        # Binned calibration data
        self.binned_errors: Dict[int, List[float]] = {i: [] for i in range(n_bins)}
        self.binned_quantiles: Dict[int, float] = {}

        logger.info(
            "adaptive_conformal_initialized",
            difficulty_feature=difficulty_feature,
            n_bins=n_bins,
        )

    def calibrate_adaptive(
        self,
        predictions: List[float],
        actual_outcomes: List[float],
        difficulty_values: List[float],  # e.g., confidence scores
    ) -> None:
        """
        Calibrate with adaptive binning.

        Args:
            predictions: Predictions
            actual_outcomes: Actual outcomes
            difficulty_values: Values of difficulty feature
        """
        if len(predictions) != len(actual_outcomes) != len(difficulty_values):
            raise ValueError("All inputs must have same length")

        # First do standard calibration
        self.calibrate(predictions, actual_outcomes)

        # Then bin by difficulty
        errors = [abs(p - a) for p, a in zip(predictions, actual_outcomes)]

        # Determine bin edges
        bin_edges = np.percentile(
            difficulty_values, np.linspace(0, 100, self.n_bins + 1)
        )

        # Assign samples to bins
        for error, diff_val in zip(errors, difficulty_values):
            bin_idx = min(
                np.searchsorted(bin_edges[1:], diff_val, side='right'),
                self.n_bins - 1,
            )
            self.binned_errors[bin_idx].append(error)

        # Calculate quantile per bin
        coverage_level = 1.0 - self.alpha

        for bin_idx, bin_errors in self.binned_errors.items():
            if len(bin_errors) >= 10:  # Need at least 10 samples per bin
                self.binned_quantiles[bin_idx] = np.percentile(
                    bin_errors, coverage_level * 100
                )
            else:
                # Fallback to global quantile
                self.binned_quantiles[bin_idx] = self.error_quantile

        logger.info(
            "adaptive_conformal_calibrated",
            bins=self.n_bins,
            quantiles={k: f"{v:.2f}" for k, v in self.binned_quantiles.items()},
        )

    def predict_with_interval(
        self,
        point_prediction: float,
        features: Optional[Dict[str, float]] = None,
    ) -> ConformalPrediction:
        """
        Make prediction with adaptive interval.

        Args:
            point_prediction: Point prediction
            features: Features (must contain difficulty_feature)

        Returns:
            ConformalPrediction with adaptive interval
        """
        # If no features or not calibrated adaptively, use standard
        if features is None or not self.binned_quantiles:
            return super().predict_with_interval(point_prediction, features)

        # Get difficulty value
        difficulty_val = features.get(self.difficulty_feature, 0.5)

        # Find appropriate bin
        # (Simplified: assume difficulty is in [0, 1])
        bin_idx = int(difficulty_val * self.n_bins)
        bin_idx = max(0, min(bin_idx, self.n_bins - 1))

        # Get quantile for this bin
        if bin_idx in self.binned_quantiles:
            interval_width = self.binned_quantiles[bin_idx]
        else:
            # Fallback to global
            interval_width = self.error_quantile

        lower_bound = point_prediction - interval_width
        upper_bound = point_prediction + interval_width

        passes = lower_bound >= self.min_lower_bound

        if passes:
            reason = (
                f"Adaptive: difficulty={difficulty_val:.2f}, bin={bin_idx}, "
                f"lower={lower_bound:.1f} ≥ {self.min_lower_bound:.1f}"
            )
        else:
            reason = (
                f"Adaptive: difficulty={difficulty_val:.2f}, bin={bin_idx}, "
                f"lower={lower_bound:.1f} < {self.min_lower_bound:.1f}"
            )

        return ConformalPrediction(
            point_prediction=point_prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            interval_width=interval_width,
            coverage_level=1.0 - self.alpha,
            passes_gate=passes,
            reason=reason,
        )


def run_conformal_example():
    """Example usage of conformal gating."""
    # Simulate calibration data
    np.random.seed(42)

    # True edge + noise
    n_samples = 100
    true_edges = np.random.normal(15, 5, n_samples)
    predictions = true_edges + np.random.normal(0, 3, n_samples)  # Model errors
    actual_outcomes = true_edges + np.random.normal(0, 2, n_samples)  # Execution noise

    # Initialize and calibrate
    gate = ConformalGate(
        significance_level=0.05,  # 95% coverage
        min_lower_bound_bps=5.0,
    )

    gate.calibrate(
        predictions=predictions.tolist(),
        actual_outcomes=actual_outcomes.tolist(),
    )

    # Make predictions
    test_predictions = [8.0, 12.0, 18.0, 25.0]

    logger.info("=" * 70)
    logger.info("CONFORMAL PREDICTION EXAMPLES")
    logger.info("=" * 70)

    for pred in test_predictions:
        result = gate.predict_with_interval(pred)

        logger.info(f"\nPoint Prediction: {result.point_prediction:.1f} bps")
        logger.info(f"Interval: [{result.lower_bound:.1f}, {result.upper_bound:.1f}] bps")
        logger.info(f"Width: {result.interval_width:.1f} bps")
        logger.info(f"Coverage: {result.coverage_level:.0%}")
        logger.info(f"Passes Gate: {result.passes_gate}")
        logger.info(f"Reason: {result.reason}")

    # Statistics
    logger.info(f"\n{'=' * 70}")
    logger.info("CALIBRATION STATISTICS")
    logger.info("=" * 70)
    stats = gate.get_statistics()
    logger.info(f"Calibrated: {stats['calibrated']}")
    logger.info(f"Samples: {stats['samples']}")
    logger.info(f"Coverage Level: {stats['coverage_level']:.0%}")
    logger.info(f"Error Quantile: {stats['error_quantile']:.2f} bps")
    logger.info(f"Mean Error: {stats['mean_error']:.2f} bps")


if __name__ == "__main__":
    run_conformal_example()
