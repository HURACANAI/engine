"""
Reality Deviation Score (RDS) - Model Alignment Metric

Based on aligning models with reality. Compares Engine predictions vs actual
outcomes for last 100 trades. Computes rolling R² or correlation coefficient.
If correlation < 0.2, trigger diagnostic alert for Mechanic.

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PredictionRecord:
    """Record of a prediction and actual outcome."""
    timestamp: datetime
    symbol: str
    prediction: float
    actual: float
    confidence: float
    regime: str


@dataclass
class RDSReport:
    """Reality Deviation Score report."""
    timestamp: datetime
    symbol: str
    rds_score: float  # 0-1, higher = better alignment
    correlation: float  # Correlation coefficient
    r_squared: float  # R² score
    mae: float  # Mean absolute error
    mse: float  # Mean squared error
    sample_size: int
    alert_triggered: bool
    threshold: float


class RealityDeviationScore:
    """
    Calculates Reality Deviation Score to measure model alignment with truth.
    
    Usage:
        rds = RealityDeviationScore(alert_threshold=0.2)
        
        # Record predictions and outcomes
        rds.record_prediction(
            prediction=0.05,
            actual=0.03,
            symbol="BTC/USDT",
            confidence=0.8
        )
        
        # Calculate RDS
        report = rds.calculate_rds(symbol="BTC/USDT")
        
        if report.alert_triggered:
            # Trigger diagnostic alert for Mechanic
            trigger_diagnostic_alert(symbol)
    """
    
    def __init__(
        self,
        alert_threshold: float = 0.2,
        lookback_trades: int = 100
    ):
        """
        Initialize RDS calculator.
        
        Args:
            alert_threshold: Correlation threshold for alert (default: 0.2)
            lookback_trades: Number of recent trades to consider (default: 100)
        """
        self.alert_threshold = alert_threshold
        self.lookback_trades = lookback_trades
        self.predictions: Dict[str, List[PredictionRecord]] = {}  # symbol -> records
        
        logger.info(
            "reality_deviation_score_initialized",
            alert_threshold=alert_threshold,
            lookback_trades=lookback_trades
        )
    
    def record_prediction(
        self,
        prediction: float,
        actual: float,
        symbol: str,
        confidence: float = 1.0,
        regime: str = "unknown"
    ) -> None:
        """
        Record a prediction and its actual outcome.
        
        Args:
            prediction: Model prediction
            actual: Actual outcome
            symbol: Trading symbol
            confidence: Prediction confidence (0-1)
            regime: Market regime
        """
        if symbol not in self.predictions:
            self.predictions[symbol] = []
        
        record = PredictionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            regime=regime
        )
        
        self.predictions[symbol].append(record)
        
        # Keep only lookback_trades most recent
        if len(self.predictions[symbol]) > self.lookback_trades:
            self.predictions[symbol] = self.predictions[symbol][-self.lookback_trades:]
        
        logger.debug(
            "prediction_recorded",
            symbol=symbol,
            prediction=prediction,
            actual=actual
        )
    
    def calculate_rds(
        self,
        symbol: str,
        use_confidence_weighting: bool = True
    ) -> RDSReport:
        """
        Calculate Reality Deviation Score.
        
        Args:
            symbol: Trading symbol
            use_confidence_weighting: Whether to weight by confidence
        
        Returns:
            RDSReport with alignment metrics
        """
        if symbol not in self.predictions or len(self.predictions[symbol]) < 10:
            # Not enough data
            return RDSReport(
                timestamp=datetime.now(),
                symbol=symbol,
                rds_score=0.0,
                correlation=0.0,
                r_squared=0.0,
                mae=0.0,
                mse=0.0,
                sample_size=0,
                alert_triggered=False,
                threshold=self.alert_threshold
            )
        
        records = self.predictions[symbol]
        predictions = np.array([r.prediction for r in records])
        actuals = np.array([r.actual for r in records])
        confidences = np.array([r.confidence for r in records])
        
        # Calculate correlation
        if len(predictions) > 1 and np.std(predictions) > 0 and np.std(actuals) > 0:
            correlation = float(np.corrcoef(predictions, actuals)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Calculate R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0.0
        r_squared = float(max(0.0, min(1.0, r_squared)))  # Clamp to [0, 1]
        
        # Calculate errors
        mae = float(np.mean(np.abs(predictions - actuals)))
        mse = float(np.mean((predictions - actuals) ** 2))
        
        # RDS score: combination of correlation and R²
        # Higher correlation and R² = better alignment
        rds_score = (abs(correlation) + r_squared) / 2.0
        
        # Alert if correlation is too low
        alert_triggered = abs(correlation) < self.alert_threshold
        
        report = RDSReport(
            timestamp=datetime.now(),
            symbol=symbol,
            rds_score=rds_score,
            correlation=correlation,
            r_squared=r_squared,
            mae=mae,
            mse=mse,
            sample_size=len(records),
            alert_triggered=alert_triggered,
            threshold=self.alert_threshold
        )
        
        logger.info(
            "rds_calculated",
            symbol=symbol,
            rds_score=rds_score,
            correlation=correlation,
            r_squared=r_squared,
            alert_triggered=alert_triggered
        )
        
        return report
    
    def get_regime_rds(
        self,
        symbol: str,
        regime: str
    ) -> Optional[RDSReport]:
        """
        Calculate RDS for a specific regime.
        
        Args:
            symbol: Trading symbol
            regime: Market regime
        
        Returns:
            RDSReport for the regime, or None if insufficient data
        """
        if symbol not in self.predictions:
            return None
        
        regime_records = [r for r in self.predictions[symbol] if r.regime == regime]
        
        if len(regime_records) < 10:
            return None
        
        # Temporarily replace predictions for calculation
        original = self.predictions[symbol]
        self.predictions[symbol] = regime_records
        
        try:
            report = self.calculate_rds(symbol)
            return report
        finally:
            self.predictions[symbol] = original
    
    def get_all_symbols_rds(self) -> Dict[str, RDSReport]:
        """Get RDS for all symbols."""
        reports = {}
        for symbol in self.predictions.keys():
            reports[symbol] = self.calculate_rds(symbol)
        return reports
    
    def clear_predictions(self, symbol: Optional[str] = None) -> None:
        """Clear prediction history."""
        if symbol:
            if symbol in self.predictions:
                del self.predictions[symbol]
                logger.info("predictions_cleared", symbol=symbol)
        else:
            self.predictions.clear()
            logger.info("all_predictions_cleared")

