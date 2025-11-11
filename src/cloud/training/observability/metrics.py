"""
Observability System for Scalable Architecture

Provides Prometheus-style metrics, health checks, and performance monitoring.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, using in-memory metrics")


class MetricType(Enum):
    """Metric type."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Metric value."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Metrics collector for system monitoring.
    
    Provides Prometheus-style metrics:
    - Counters: Total trades, errors, etc.
    - Gauges: Active positions, exposure, etc.
    - Histograms: Latency, trade size, etc.
    - Summaries: Performance metrics
    """
    
    def __init__(self, use_prometheus: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            use_prometheus: Use Prometheus client if available
        """
        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        
        # In-memory metrics
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Prometheus metrics
        if self.use_prometheus:
            self._init_prometheus_metrics()
        
        logger.info(
            "metrics_collector_initialized",
            use_prometheus=self.use_prometheus,
        )
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Counters
        self.trades_total = Counter(
            "huracan_trades_total",
            "Total number of trades executed",
            ["coin", "direction", "exchange"],
        )
        self.errors_total = Counter(
            "huracan_errors_total",
            "Total number of errors",
            ["error_type", "component"],
        )
        self.messages_processed_total = Counter(
            "huracan_messages_processed_total",
            "Total number of messages processed",
            ["stream_type", "coin"],
        )
        
        # Gauges
        self.active_positions = Gauge(
            "huracan_active_positions",
            "Number of active positions",
            ["coin", "direction"],
        )
        self.exposure_pct = Gauge(
            "huracan_exposure_pct",
            "Portfolio exposure percentage",
            ["type", "category"],  # type: coin/sector/exchange/global, category: coin name/sector name/etc
        )
        self.active_coins = Gauge(
            "huracan_active_coins",
            "Number of active coins",
        )
        
        # Histograms
        self.latency_seconds = Histogram(
            "huracan_latency_seconds",
            "Latency in seconds",
            ["operation", "coin"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        )
        self.trade_size_usd = Histogram(
            "huracan_trade_size_usd",
            "Trade size in USD",
            ["coin"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
        )
        self.cost_bps = Histogram(
            "huracan_cost_bps",
            "Cost in basis points",
            ["coin", "cost_type"],  # cost_type: spread/fee/funding/total
            buckets=[1, 2, 5, 10, 20, 50, 100],
        )
        
        # Summaries
        self.hit_rate = Summary(
            "huracan_hit_rate",
            "Win rate",
            ["coin", "timeframe"],
        )
        self.sharpe_ratio = Summary(
            "huracan_sharpe_ratio",
            "Sharpe ratio",
            ["coin"],
        )
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        self.counters[name] += value
        
        if self.use_prometheus:
            if name == "trades_total":
                self.trades_total.labels(
                    coin=labels.get("coin", ""),
                    direction=labels.get("direction", ""),
                    exchange=labels.get("exchange", ""),
                ).inc(value)
            elif name == "errors_total":
                self.errors_total.labels(
                    error_type=labels.get("error_type", ""),
                    component=labels.get("component", ""),
                ).inc(value)
            elif name == "messages_processed_total":
                self.messages_processed_total.labels(
                    stream_type=labels.get("stream_type", ""),
                    coin=labels.get("coin", ""),
                ).inc(value)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        self.gauges[name] = value
        
        if self.use_prometheus:
            if name == "active_positions":
                self.active_positions.labels(
                    coin=labels.get("coin", ""),
                    direction=labels.get("direction", ""),
                ).set(value)
            elif name == "exposure_pct":
                self.exposure_pct.labels(
                    type=labels.get("type", ""),
                    category=labels.get("category", ""),
                ).set(value)
            elif name == "active_coins":
                self.active_coins.set(value)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        self.histograms[name].append(value)
        
        # Keep only last 1000 values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        if self.use_prometheus:
            if name == "latency_seconds":
                self.latency_seconds.labels(
                    operation=labels.get("operation", ""),
                    coin=labels.get("coin", ""),
                ).observe(value)
            elif name == "trade_size_usd":
                self.trade_size_usd.labels(
                    coin=labels.get("coin", ""),
                ).observe(value)
            elif name == "cost_bps":
                self.cost_bps.labels(
                    coin=labels.get("coin", ""),
                    cost_type=labels.get("cost_type", ""),
                ).observe(value)
    
    def observe_summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a summary value."""
        if name not in self.summaries:
            self.summaries[name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
            }
        
        summary = self.summaries[name]
        summary["count"] += 1
        summary["sum"] += value
        summary["min"] = min(summary["min"], value)
        summary["max"] = max(summary["max"], value)
        
        if self.use_prometheus:
            if name == "hit_rate":
                self.hit_rate.labels(
                    coin=labels.get("coin", ""),
                    timeframe=labels.get("timeframe", ""),
                ).observe(value)
            elif name == "sharpe_ratio":
                self.sharpe_ratio.labels(
                    coin=labels.get("coin", ""),
                ).observe(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                }
                for name, values in self.histograms.items()
            },
            "summaries": dict(self.summaries),
        }
    
    def get_metric_value(self, name: str, metric_type: MetricType) -> Optional[float]:
        """Get a specific metric value."""
        if metric_type == MetricType.COUNTER:
            return self.counters.get(name)
        elif metric_type == MetricType.GAUGE:
            return self.gauges.get(name)
        elif metric_type == MetricType.HISTOGRAM:
            values = self.histograms.get(name, [])
            return sum(values) / len(values) if values else None
        elif metric_type == MetricType.SUMMARY:
            summary = self.summaries.get(name)
            return summary["sum"] / summary["count"] if summary and summary["count"] > 0 else None
        return None


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Any] = {}
        self.last_check: Dict[str, float] = {}
        self.check_interval: Dict[str, float] = {}
        logger.info("health_checker_initialized")
    
    def register_check(
        self,
        name: str,
        check_func: callable,
        interval_seconds: float = 30.0,
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Async function that returns (healthy: bool, message: str)
            interval_seconds: Check interval in seconds
        """
        self.checks[name] = check_func
        self.check_interval[name] = interval_seconds
        self.last_check[name] = 0.0
        logger.info("health_check_registered", name=name, interval=interval_seconds)
    
    async def run_check(self, name: str) -> tuple[bool, str]:
        """
        Run a health check.
        
        Args:
            name: Check name
        
        Returns:
            Tuple of (healthy: bool, message: str)
        """
        if name not in self.checks:
            return False, f"Check '{name}' not found"
        
        try:
            check_func = self.checks[name]
            healthy, message = await check_func()
            self.last_check[name] = time.time()
            return healthy, message
        except Exception as e:
            logger.error("health_check_failed", name=name, error=str(e))
            return False, f"Check failed: {str(e)}"
    
    async def run_all_checks(self) -> Dict[str, tuple[bool, str]]:
        """Run all health checks."""
        results = {}
        for name in self.checks:
            results[name] = await self.run_check(name)
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all checks."""
        status = {}
        for name, check_func in self.checks.items():
            status[name] = {
                "last_check": self.last_check.get(name, 0.0),
                "interval": self.check_interval.get(name, 30.0),
                "age": time.time() - self.last_check.get(name, 0.0),
            }
        return status


class PerformanceMonitor:
    """Performance monitor for system operations."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_errors: Dict[str, int] = defaultdict(int)
        logger.info("performance_monitor_initialized", window_size=window_size)
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
    ) -> None:
        """
        Record an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation was successful
        """
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        if not success:
            self.operation_errors[operation] += 1
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        if operation not in self.operation_times:
            return None
        
        times = list(self.operation_times[operation])
        if not times:
            return None
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            "count": self.operation_counts[operation],
            "errors": self.operation_errors[operation],
            "error_rate": self.operation_errors[operation] / self.operation_counts[operation] if self.operation_counts[operation] > 0 else 0,
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "p50": sorted_times[n // 2],
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self.operation_times.keys()
        }

