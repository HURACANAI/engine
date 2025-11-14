"""
Regime Early Warning System

Detects regime shifts before they fully materialize, allowing models to adapt.

Key Features:
- Statistical change detection (CUSUM, Page's test)
- Volatility regime shifts
- Volume anomalies
- Correlation breakdowns
- Lead time: 5-30 minutes before full regime shift

Usage:
    from models.regime_warning import RegimeWarningSystem

    warning_system = RegimeWarningSystem()

    # Check for regime shift warnings
    warning = warning_system.check_for_warnings(
        candles_df=recent_candles,
        current_regime="trending"
    )

    if warning.shift_likely:
        print(f"WARNING: Likely shift to {warning.predicted_regime}")
        print(f"Confidence: {warning.confidence:.0%}")
        print(f"Lead time: {warning.estimated_lead_minutes} min")

        # Take action (e.g., reduce position size)
        if warning.confidence > 0.7:
            reduce_exposure()
"""

from .detector import (
    RegimeWarningSystem,
    RegimeWarning,
    WarningSignal
)
from .indicators import (
    detect_volatility_shift,
    detect_volume_anomaly,
    detect_correlation_breakdown
)

__all__ = [
    # Warning system
    "RegimeWarningSystem",
    "RegimeWarning",
    "WarningSignal",

    # Indicators
    "detect_volatility_shift",
    "detect_volume_anomaly",
    "detect_correlation_breakdown",
]
