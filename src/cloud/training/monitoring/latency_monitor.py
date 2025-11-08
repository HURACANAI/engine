"""
Latency Monitoring System

Tracks nanosecond-level metrics for every trade, module latency, and model inference duration.
Detects slowdowns and provides diagnostics.

Key Features:
- Nanosecond timestamps
- Tick-to-trade time tracking
- Module latency tracking
- Model inference duration
- Latency dashboards
- Slowdown detection

Author: Huracan Engine Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Deque
from collections import deque
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LatencyEvent:
    """Single latency event"""
    event_id: str
    module_name: str
    event_type: str  # "start", "end", "checkpoint"
    timestamp_ns: int
    duration_ns: Optional[int] = None  # Duration from start to this event
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class LatencyMeasurement:
    """Latency measurement for a module"""
    module_name: str
    duration_ns: int
    timestamp_ns: int
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class TickToTradeMetrics:
    """Tick-to-trade latency metrics"""
    symbol: str
    tick_timestamp_ns: int
    trade_timestamp_ns: int
    total_duration_ns: int
    module_breakdown: Dict[str, int] = field(default_factory=dict)
    stages: List[LatencyMeasurement] = field(default_factory=list)


class LatencyMonitor:
    """
    Latency monitoring system with nanosecond precision.
    
    Tracks:
    - Module latency
    - Tick-to-trade time
    - Model inference duration
    - Event throughput
    - Slowdown detection
    
    Usage:
        monitor = LatencyMonitor()
        
        with monitor.track("model_inference"):
            model.predict(data)
        
        metrics = monitor.get_metrics("model_inference")
    """
    
    def __init__(
        self,
        window_size: int = 1000,  # Number of measurements to keep
        alert_threshold_ns: Optional[int] = None  # Alert if latency exceeds this
    ):
        """
        Initialize latency monitor.
        
        Args:
            window_size: Number of measurements to keep in rolling window
            alert_threshold_ns: Alert threshold in nanoseconds
        """
        self.window_size = window_size
        self.alert_threshold_ns = alert_threshold_ns
        
        # Store latency measurements per module
        self.measurements: Dict[str, Deque[LatencyMeasurement]] = {}
        
        # Store tick-to-trade metrics
        self.tick_to_trade_metrics: List[TickToTradeMetrics] = deque(maxlen=window_size)
        
        # Store event timeline
        self.event_timeline: List[LatencyEvent] = deque(maxlen=window_size * 10)
        
        logger.info(
            "latency_monitor_initialized",
            window_size=window_size,
            alert_threshold_ns=alert_threshold_ns
        )
    
    def track(self, module_name: str, metadata: Optional[Dict[str, any]] = None):
        """Context manager for tracking latency"""
        return LatencyTracker(self, module_name, metadata or {})
    
    def record_measurement(
        self,
        module_name: str,
        duration_ns: int,
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """
        Record a latency measurement.
        
        Args:
            module_name: Name of the module
            duration_ns: Duration in nanoseconds
            metadata: Optional metadata
        """
        if module_name not in self.measurements:
            self.measurements[module_name] = deque(maxlen=self.window_size)
        
        measurement = LatencyMeasurement(
            module_name=module_name,
            duration_ns=duration_ns,
            timestamp_ns=time.perf_counter_ns(),
            metadata=metadata or {}
        )
        
        self.measurements[module_name].append(measurement)
        
        # Check for alerts
        if self.alert_threshold_ns and duration_ns > self.alert_threshold_ns:
            logger.warning(
                "latency_alert",
                module_name=module_name,
                duration_ns=duration_ns,
                threshold_ns=self.alert_threshold_ns,
                metadata=metadata
            )
    
    def record_tick_to_trade(
        self,
        symbol: str,
        tick_timestamp_ns: int,
        trade_timestamp_ns: int,
        module_breakdown: Optional[Dict[str, int]] = None,
        stages: Optional[List[LatencyMeasurement]] = None
    ) -> TickToTradeMetrics:
        """
        Record tick-to-trade latency.
        
        Args:
            symbol: Trading symbol
            tick_timestamp_ns: Timestamp when tick received (nanoseconds)
            trade_timestamp_ns: Timestamp when trade executed (nanoseconds)
            module_breakdown: Breakdown of latency by module
            stages: List of latency measurements for each stage
        
        Returns:
            TickToTradeMetrics
        """
        total_duration_ns = trade_timestamp_ns - tick_timestamp_ns
        
        metrics = TickToTradeMetrics(
            symbol=symbol,
            tick_timestamp_ns=tick_timestamp_ns,
            trade_timestamp_ns=trade_timestamp_ns,
            total_duration_ns=total_duration_ns,
            module_breakdown=module_breakdown or {},
            stages=stages or []
        )
        
        self.tick_to_trade_metrics.append(metrics)
        
        logger.debug(
            "tick_to_trade_recorded",
            symbol=symbol,
            total_duration_ns=total_duration_ns,
            total_duration_ms=total_duration_ns / 1_000_000
        )
        
        return metrics
    
    def get_metrics(self, module_name: str) -> Dict[str, float]:
        """
        Get latency metrics for a module.
        
        Args:
            module_name: Module name
        
        Returns:
            Dictionary with metrics (mean, median, p95, p99, max, min)
        """
        if module_name not in self.measurements:
            return {}
        
        measurements = list(self.measurements[module_name])
        if not measurements:
            return {}
        
        durations = [m.duration_ns for m in measurements]
        
        # Calculate statistics
        mean_ns = sum(durations) / len(durations)
        median_ns = sorted(durations)[len(durations) // 2]
        p95_ns = sorted(durations)[int(len(durations) * 0.95)]
        p99_ns = sorted(durations)[int(len(durations) * 0.99)]
        max_ns = max(durations)
        min_ns = min(durations)
        
        return {
            "mean_ns": mean_ns,
            "mean_ms": mean_ns / 1_000_000,
            "median_ns": median_ns,
            "median_ms": median_ns / 1_000_000,
            "p95_ns": p95_ns,
            "p95_ms": p95_ns / 1_000_000,
            "p99_ns": p99_ns,
            "p99_ms": p99_ns / 1_000_000,
            "max_ns": max_ns,
            "max_ms": max_ns / 1_000_000,
            "min_ns": min_ns,
            "min_ms": min_ns / 1_000_000,
            "count": len(durations)
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all modules"""
        return {
            module_name: self.get_metrics(module_name)
            for module_name in self.measurements.keys()
        }
    
    def get_tick_to_trade_stats(self) -> Dict[str, float]:
        """Get tick-to-trade statistics"""
        if not self.tick_to_trade_metrics:
            return {}
        
        durations = [m.total_duration_ns for m in self.tick_to_trade_metrics]
        
        mean_ns = sum(durations) / len(durations)
        median_ns = sorted(durations)[len(durations) // 2]
        p95_ns = sorted(durations)[int(len(durations) * 0.95)]
        p99_ns = sorted(durations)[int(len(durations) * 0.99)]
        max_ns = max(durations)
        min_ns = min(durations)
        
        return {
            "mean_ns": mean_ns,
            "mean_ms": mean_ns / 1_000_000,
            "median_ns": median_ns,
            "median_ms": median_ns / 1_000_000,
            "p95_ns": p95_ns,
            "p95_ms": p95_ns / 1_000_000,
            "p99_ns": p99_ns,
            "p99_ms": p99_ns / 1_000_000,
            "max_ns": max_ns,
            "max_ms": max_ns / 1_000_000,
            "min_ns": min_ns,
            "min_ms": min_ns / 1_000_000,
            "count": len(durations)
        }
    
    def detect_slowdown(self, module_name: str, threshold_multiplier: float = 2.0) -> bool:
        """
        Detect if a module is experiencing slowdown.
        
        Args:
            module_name: Module name
            threshold_multiplier: Multiplier for mean latency to trigger alert
        
        Returns:
            True if slowdown detected
        """
        metrics = self.get_metrics(module_name)
        if not metrics:
            return False
        
        mean_latency = metrics["mean_ns"]
        recent_measurements = list(self.measurements[module_name])[-10:]
        
        if len(recent_measurements) < 10:
            return False
        
        recent_mean = sum(m.duration_ns for m in recent_measurements) / len(recent_measurements)
        
        # Slowdown if recent mean is significantly higher than overall mean
        if recent_mean > mean_latency * threshold_multiplier:
            logger.warning(
                "slowdown_detected",
                module_name=module_name,
                overall_mean_ns=mean_latency,
                recent_mean_ns=recent_mean,
                threshold_multiplier=threshold_multiplier
            )
            return True
        
        return False
    
    def get_dashboard_data(self) -> Dict[str, any]:
        """Get data for latency dashboard"""
        return {
            "module_metrics": self.get_all_metrics(),
            "tick_to_trade_stats": self.get_tick_to_trade_stats(),
            "recent_events": list(self.event_timeline)[-100:],  # Last 100 events
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class LatencyTracker:
    """Context manager for tracking latency"""
    
    def __init__(
        self,
        monitor: LatencyMonitor,
        module_name: str,
        metadata: Dict[str, any]
    ):
        self.monitor = monitor
        self.module_name = module_name
        self.metadata = metadata
        self.start_ns: Optional[int] = None
    
    def __enter__(self) -> "LatencyTracker":
        self.start_ns = time.perf_counter_ns()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_ns is not None:
            end_ns = time.perf_counter_ns()
            duration_ns = end_ns - self.start_ns
            
            self.monitor.record_measurement(
                module_name=self.module_name,
                duration_ns=duration_ns,
                metadata=self.metadata
            )

