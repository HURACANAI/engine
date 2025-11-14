"""
Confidence Calibrator

Calibrates model probability outputs to match actual frequencies.

Supports three calibration methods:
- Isotonic Regression: Non-parametric, preserves order
- Platt Scaling: Logistic regression
- Temperature Scaling: Single parameter scaling
"""

from typing import Literal, Optional

import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = structlog.get_logger(__name__)


CalibrationMethod = Literal["isotonic", "platt", "temperature"]


class ConfidenceCalibrator:
    """
    Confidence Calibrator

    Calibrates model confidence scores to match actual probabilities.

    Example:
        # Train calibrator on validation set
        calibrator = ConfidenceCalibrator(method="isotonic")
        calibrator.fit(
            probabilities=val_probs,
            actual_outcomes=val_labels
        )

        # Calibrate test predictions
        calibrated_probs = calibrator.calibrate(test_probs)

        # Check calibration improvement
        from .metrics import calculate_brier_score, calculate_ece

        before_brier = calculate_brier_score(test_probs, test_labels)
        after_brier = calculate_brier_score(calibrated_probs, test_labels)

        print(f"Brier score: {before_brier:.3f} → {after_brier:.3f}")
    """

    def __init__(
        self,
        method: CalibrationMethod = "isotonic",
        temperature_init: float = 1.0
    ):
        """
        Initialize calibrator

        Args:
            method: Calibration method (isotonic, platt, temperature)
            temperature_init: Initial temperature for temperature scaling
        """
        self.method = method
        self.temperature_init = temperature_init

        self.is_fitted = False
        self._calibrator = None
        self._temperature = temperature_init

    def fit(
        self,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> "ConfidenceCalibrator":
        """
        Fit calibrator on validation data

        Args:
            probabilities: Uncalibrated probabilities [0-1]
            actual_outcomes: Actual binary outcomes {0, 1}

        Returns:
            self (for chaining)

        Example:
            calibrator = ConfidenceCalibrator().fit(val_probs, val_labels)
        """
        probabilities = np.asarray(probabilities).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        if len(probabilities) != len(actual_outcomes):
            raise ValueError(
                f"Length mismatch: {len(probabilities)} probs vs "
                f"{len(actual_outcomes)} outcomes"
            )

        if len(probabilities) < 10:
            logger.warning(
                "insufficient_calibration_data",
                num_samples=len(probabilities),
                method=self.method
            )

        # Clip probabilities to avoid log(0) issues
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

        if self.method == "isotonic":
            self._fit_isotonic(probabilities, actual_outcomes)
        elif self.method == "platt":
            self._fit_platt(probabilities, actual_outcomes)
        elif self.method == "temperature":
            self._fit_temperature(probabilities, actual_outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

        logger.info(
            "calibrator_fitted",
            method=self.method,
            num_samples=len(probabilities),
            temperature=self._temperature if self.method == "temperature" else None
        )

        return self

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities

        Args:
            probabilities: Uncalibrated probabilities [0-1]

        Returns:
            Calibrated probabilities [0-1]

        Raises:
            RuntimeError: If calibrator not fitted

        Example:
            calibrated = calibrator.calibrate(test_probs)
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrating")

        probabilities = np.asarray(probabilities).flatten()
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

        if self.method == "isotonic":
            calibrated = self._calibrator.predict(probabilities)
        elif self.method == "platt":
            calibrated = self._calibrator.predict_proba(
                probabilities.reshape(-1, 1)
            )[:, 1]
        elif self.method == "temperature":
            # Apply temperature scaling
            logits = np.log(probabilities / (1 - probabilities))
            scaled_logits = logits / self._temperature
            calibrated = 1 / (1 + np.exp(-scaled_logits))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Ensure output is in [0, 1]
        calibrated = np.clip(calibrated, 0, 1)

        return calibrated

    def _fit_isotonic(
        self,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> None:
        """
        Fit isotonic regression calibrator

        Isotonic regression is non-parametric and preserves monotonicity.
        Works well when you have enough calibration data.
        """
        self._calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip"
        )
        self._calibrator.fit(probabilities, actual_outcomes)

    def _fit_platt(
        self,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> None:
        """
        Fit Platt scaling (logistic regression)

        Platt scaling fits a logistic regression on top of model outputs.
        Works well with limited calibration data.
        """
        # Convert probabilities to log-odds (logits)
        logits = np.log(probabilities / (1 - probabilities))

        self._calibrator = LogisticRegression(solver='lbfgs')
        self._calibrator.fit(logits.reshape(-1, 1), actual_outcomes)

    def _fit_temperature(
        self,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> None:
        """
        Fit temperature scaling

        Temperature scaling learns a single scalar T to divide logits.
        Very efficient, works well for neural networks.

        Method: Grid search over temperature values to minimize NLL
        """
        # Convert to logits
        logits = np.log(probabilities / (1 - probabilities))

        # Grid search for best temperature
        best_temp = self.temperature_init
        best_nll = float('inf')

        for temp in np.linspace(0.1, 5.0, 50):
            # Scale logits
            scaled_logits = logits / temp

            # Convert back to probabilities
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)

            # Calculate negative log likelihood
            nll = -np.mean(
                actual_outcomes * np.log(scaled_probs) +
                (1 - actual_outcomes) * np.log(1 - scaled_probs)
            )

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self._temperature = best_temp

    def get_params(self) -> dict:
        """
        Get calibrator parameters

        Returns:
            Dict with method and parameters
        """
        params = {
            "method": self.method,
            "is_fitted": self.is_fitted
        }

        if self.method == "temperature":
            params["temperature"] = self._temperature

        return params


class RegimeSpecificCalibrator:
    """
    Regime-Specific Calibrator

    Maintains separate calibrators for each market regime.

    Example:
        calibrator = RegimeSpecificCalibrator(method="isotonic")

        # Fit on validation data
        calibrator.fit(
            probabilities=val_probs,
            actual_outcomes=val_labels,
            regimes=val_regimes
        )

        # Calibrate test data
        calibrated = calibrator.calibrate(
            probabilities=test_probs,
            regimes=test_regimes
        )
    """

    def __init__(
        self,
        method: CalibrationMethod = "isotonic",
        min_samples_per_regime: int = 30
    ):
        """
        Initialize regime-specific calibrator

        Args:
            method: Calibration method
            min_samples_per_regime: Minimum samples needed to fit per regime
        """
        self.method = method
        self.min_samples_per_regime = min_samples_per_regime

        self.calibrators: dict[str, ConfidenceCalibrator] = {}
        self.fallback_calibrator: Optional[ConfidenceCalibrator] = None
        self.is_fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
        regimes: np.ndarray
    ) -> "RegimeSpecificCalibrator":
        """
        Fit calibrators for each regime

        Args:
            probabilities: Uncalibrated probabilities
            actual_outcomes: Actual outcomes
            regimes: Regime labels for each sample

        Returns:
            self
        """
        probabilities = np.asarray(probabilities)
        actual_outcomes = np.asarray(actual_outcomes)
        regimes = np.asarray(regimes)

        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_probs = probabilities[regime_mask]
            regime_outcomes = actual_outcomes[regime_mask]

            if len(regime_probs) >= self.min_samples_per_regime:
                # Fit regime-specific calibrator
                calibrator = ConfidenceCalibrator(method=self.method)
                calibrator.fit(regime_probs, regime_outcomes)
                self.calibrators[regime] = calibrator

                logger.info(
                    "regime_calibrator_fitted",
                    regime=regime,
                    num_samples=len(regime_probs)
                )
            else:
                logger.warning(
                    "insufficient_regime_samples",
                    regime=regime,
                    num_samples=len(regime_probs),
                    min_required=self.min_samples_per_regime
                )

        # Fit fallback calibrator on all data
        self.fallback_calibrator = ConfidenceCalibrator(method=self.method)
        self.fallback_calibrator.fit(probabilities, actual_outcomes)

        self.is_fitted = True

        logger.info(
            "regime_calibrator_complete",
            num_regimes=len(self.calibrators),
            total_samples=len(probabilities)
        )

        return self

    def calibrate(
        self,
        probabilities: np.ndarray,
        regimes: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate probabilities using regime-specific calibrators

        Args:
            probabilities: Uncalibrated probabilities
            regimes: Regime labels

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first")

        probabilities = np.asarray(probabilities)
        regimes = np.asarray(regimes)

        calibrated = np.zeros_like(probabilities)

        for regime in np.unique(regimes):
            regime_mask = regimes == regime

            if regime in self.calibrators:
                # Use regime-specific calibrator
                calibrator = self.calibrators[regime]
            else:
                # Use fallback
                calibrator = self.fallback_calibrator
                logger.debug(
                    "using_fallback_calibrator",
                    regime=regime,
                    num_samples=regime_mask.sum()
                )

            calibrated[regime_mask] = calibrator.calibrate(
                probabilities[regime_mask]
            )

        return calibrated

    def get_regimes(self) -> list[str]:
        """Get list of regimes with fitted calibrators"""
        return list(self.calibrators.keys())
