"""
Fee/Latency Calibration Function

Calibrates fees and latency for each exchange based on historical data.
Tracks actual execution costs and network latency to improve cost estimates.

Key Features:
- Fee calibration (maker/taker fees per exchange)
- Latency measurement and calibration
- Historical cost tracking
- Exchange-specific calibration
- Auto-update based on recent data

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeeCalibration:
    """Fee calibration for an exchange."""
    exchange: str
    maker_fee_bps: float
    taker_fee_bps: float
    maker_rebate_bps: float  # Negative fee (rebate)
    last_updated: datetime
    sample_count: int
    confidence: float  # 0-1, calibration confidence


@dataclass
class LatencyCalibration:
    """Latency calibration for an exchange."""
    exchange: str
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    last_updated: datetime
    sample_count: int
    reliability_score: float  # 0-1, based on latency consistency


@dataclass
class ExecutionRecord:
    """Record of an actual execution for calibration."""
    exchange: str
    symbol: str
    timestamp: datetime
    order_type: str  # "maker" or "taker"
    size_usd: float
    expected_fee_bps: float
    actual_fee_bps: float
    expected_latency_ms: float
    actual_latency_ms: float
    price: float


class FeeLatencyCalibrator:
    """
    Calibrates fees and latency for exchanges.
    
    Usage:
        calibrator = FeeLatencyCalibrator()
        
        # Record actual execution
        calibrator.record_execution(
            exchange="binance",
            order_type="taker",
            actual_fee_bps=5.2,
            actual_latency_ms=45.0
        )
        
        # Get calibrated fees
        fee_cal = calibrator.get_fee_calibration("binance")
        print(f"Taker fee: {fee_cal.taker_fee_bps} bps")
        
        # Get calibrated latency
        latency_cal = calibrator.get_latency_calibration("binance")
        print(f"Mean latency: {latency_cal.mean_latency_ms} ms")
    """
    
    def __init__(
        self,
        lookback_days: int = 30,
        min_samples: int = 10
    ):
        """
        Initialize fee/latency calibrator.
        
        Args:
            lookback_days: Days of history to use for calibration
            min_samples: Minimum samples required for calibration
        """
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        
        # Storage for execution records
        self.execution_records: List[ExecutionRecord] = []
        
        # Cached calibrations
        self.fee_calibrations: Dict[str, FeeCalibration] = {}
        self.latency_calibrations: Dict[str, LatencyCalibration] = {}
        
        logger.info(
            "fee_latency_calibrator_initialized",
            lookback_days=lookback_days,
            min_samples=min_samples
        )
    
    def record_execution(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        size_usd: float,
        actual_fee_bps: float,
        actual_latency_ms: float,
        price: float,
        expected_fee_bps: Optional[float] = None,
        expected_latency_ms: Optional[float] = None
    ) -> None:
        """
        Record an actual execution for calibration.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            order_type: "maker" or "taker"
            size_usd: Trade size in USD
            actual_fee_bps: Actual fee paid in basis points
            actual_latency_ms: Actual latency in milliseconds
            price: Execution price
            expected_fee_bps: Expected fee (for comparison)
            expected_latency_ms: Expected latency (for comparison)
        """
        record = ExecutionRecord(
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.now(),
            order_type=order_type.lower(),
            size_usd=size_usd,
            expected_fee_bps=expected_fee_bps or 0.0,
            actual_fee_bps=actual_fee_bps,
            expected_latency_ms=expected_latency_ms or 0.0,
            actual_latency_ms=actual_latency_ms,
            price=price
        )
        
        self.execution_records.append(record)
        
        # Clean old records
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        self.execution_records = [
            r for r in self.execution_records
            if r.timestamp >= cutoff_date
        ]
        
        # Recalibrate
        self._calibrate_fees(exchange)
        self._calibrate_latency(exchange)
        
        logger.debug(
            "execution_recorded",
            exchange=exchange,
            order_type=order_type,
            actual_fee_bps=actual_fee_bps,
            actual_latency_ms=actual_latency_ms
        )
    
    def _calibrate_fees(self, exchange: str) -> None:
        """Calibrate fees for an exchange."""
        # Filter records for this exchange
        exchange_records = [
            r for r in self.execution_records
            if r.exchange == exchange
        ]
        
        if len(exchange_records) < self.min_samples:
            return
        
        # Separate maker and taker
        maker_records = [r for r in exchange_records if r.order_type == "maker"]
        taker_records = [r for r in exchange_records if r.order_type == "taker"]
        
        # Calculate maker fee
        if maker_records:
            maker_fees = [r.actual_fee_bps for r in maker_records]
            maker_fee_bps = float(np.median(maker_fees))
            maker_rebate_bps = -maker_fee_bps if maker_fee_bps < 0 else 0.0
            maker_count = len(maker_records)
        else:
            maker_fee_bps = 2.0  # Default
            maker_rebate_bps = 0.0
            maker_count = 0
        
        # Calculate taker fee
        if taker_records:
            taker_fees = [r.actual_fee_bps for r in taker_records]
            taker_fee_bps = float(np.median(taker_fees))
            taker_count = len(taker_records)
        else:
            taker_fee_bps = 5.0  # Default
            taker_count = 0
        
        # Calculate confidence (based on sample size)
        total_samples = len(exchange_records)
        confidence = min(1.0, total_samples / (self.min_samples * 2))
        
        self.fee_calibrations[exchange] = FeeCalibration(
            exchange=exchange,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            maker_rebate_bps=maker_rebate_bps,
            last_updated=datetime.now(),
            sample_count=total_samples,
            confidence=confidence
        )
        
        logger.info(
            "fees_calibrated",
            exchange=exchange,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            samples=total_samples,
            confidence=confidence
        )
    
    def _calibrate_latency(self, exchange: str) -> None:
        """Calibrate latency for an exchange."""
        # Filter records for this exchange
        exchange_records = [
            r for r in self.execution_records
            if r.exchange == exchange
        ]
        
        if len(exchange_records) < self.min_samples:
            return
        
        latencies = [r.actual_latency_ms for r in exchange_records]
        
        mean_latency = float(np.mean(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        
        # Calculate reliability score (lower variance = higher reliability)
        latency_std = float(np.std(latencies))
        reliability_score = 1.0 / (1.0 + latency_std / mean_latency) if mean_latency > 0 else 0.5
        
        self.latency_calibrations[exchange] = LatencyCalibration(
            exchange=exchange,
            mean_latency_ms=mean_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            last_updated=datetime.now(),
            sample_count=len(exchange_records),
            reliability_score=reliability_score
        )
        
        logger.info(
            "latency_calibrated",
            exchange=exchange,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            reliability=reliability_score
        )
    
    def get_fee_calibration(self, exchange: str) -> Optional[FeeCalibration]:
        """Get fee calibration for an exchange."""
        return self.fee_calibrations.get(exchange)
    
    def get_latency_calibration(self, exchange: str) -> Optional[LatencyCalibration]:
        """Get latency calibration for an exchange."""
        return self.latency_calibrations.get(exchange)
    
    def get_estimated_fee(
        self,
        exchange: str,
        order_type: str,
        fallback_bps: float = 5.0
    ) -> float:
        """
        Get estimated fee for an order type.
        
        Args:
            exchange: Exchange name
            order_type: "maker" or "taker"
            fallback_bps: Fallback fee if not calibrated
        
        Returns:
            Estimated fee in basis points
        """
        calibration = self.fee_calibrations.get(exchange)
        
        if not calibration:
            return fallback_bps
        
        if order_type.lower() == "maker":
            return calibration.maker_fee_bps
        else:
            return calibration.taker_fee_bps
    
    def get_estimated_latency(
        self,
        exchange: str,
        percentile: str = "p50",
        fallback_ms: float = 50.0
    ) -> float:
        """
        Get estimated latency for an exchange.
        
        Args:
            exchange: Exchange name
            percentile: "mean", "p50", "p95", "p99"
            fallback_ms: Fallback latency if not calibrated
        
        Returns:
            Estimated latency in milliseconds
        """
        calibration = self.latency_calibrations.get(exchange)
        
        if not calibration:
            return fallback_ms
        
        if percentile == "mean":
            return calibration.mean_latency_ms
        elif percentile == "p50":
            return calibration.p50_latency_ms
        elif percentile == "p95":
            return calibration.p95_latency_ms
        elif percentile == "p99":
            return calibration.p99_latency_ms
        else:
            return calibration.mean_latency_ms
    
    def get_all_calibrations(self) -> Dict[str, Dict[str, any]]:
        """Get all calibrations."""
        return {
            "fees": {
                exchange: {
                    "maker_fee_bps": cal.maker_fee_bps,
                    "taker_fee_bps": cal.taker_fee_bps,
                    "maker_rebate_bps": cal.maker_rebate_bps,
                    "confidence": cal.confidence,
                    "samples": cal.sample_count
                }
                for exchange, cal in self.fee_calibrations.items()
            },
            "latency": {
                exchange: {
                    "mean_ms": cal.mean_latency_ms,
                    "p95_ms": cal.p95_latency_ms,
                    "reliability": cal.reliability_score,
                    "samples": cal.sample_count
                }
                for exchange, cal in self.latency_calibrations.items()
            }
        }

