"""
Prometheus Metrics Exporter

Exposes metrics for Prometheus scraping. Tracks PnL by engine, latency,
error rates, and system health.
"""

from __future__ import annotations

from typing import Dict, Optional
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)

# Try to import prometheus_client, but make it optional
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client_not_available", message="Prometheus metrics disabled")


class PrometheusMetrics:
    """
    Prometheus metrics exporter.
    
    Exposes metrics for:
    - PnL by engine, symbol, regime
    - Latency by engine and operation
    - Error rates by component
    - Consensus confidence
    - Cost model data
    - Training job status
    """
    
    def __init__(self, port: int = 9090) -> None:
        """
        Initialize Prometheus metrics.
        
        Args:
            port: Port to expose metrics on
        """
        self.port = port
        self.server_started = False
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_not_available", message="Metrics will not be exported")
            return
        
        # PnL metrics
        self.engine_pnl = Gauge(
            'engine_pnl_usd',
            'PnL by engine',
            ['engine', 'symbol', 'regime']
        )
        
        self.total_pnl = Gauge('total_pnl_usd', 'Total PnL')
        
        # Latency metrics
        self.engine_latency = Histogram(
            'engine_latency_ms',
            'Engine latency in milliseconds',
            ['engine', 'operation'],
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        
        # Error metrics
        self.engine_errors = Counter(
            'engine_errors_total',
            'Total errors by engine',
            ['engine', 'error_type']
        )
        
        # Consensus metrics
        self.consensus_confidence = Gauge(
            'consensus_confidence',
            'Consensus confidence',
            ['regime']
        )
        
        self.consensus_agreement = Gauge(
            'consensus_agreement_ratio',
            'Consensus agreement ratio',
            ['regime']
        )
        
        # Cost model metrics
        self.cost_spread = Gauge(
            'cost_model_spread_bps',
            'Spread in basis points',
            ['venue', 'symbol']
        )
        
        self.cost_total = Gauge(
            'cost_model_total_bps',
            'Total cost in basis points',
            ['venue', 'symbol']
        )
        
        # Training metrics
        self.training_job_duration = Histogram(
            'training_job_duration_seconds',
            'Training job duration',
            ['coin', 'regime'],
            buckets=[60, 300, 600, 1800, 3600, 7200]
        )
        
        self.training_jobs_active = Gauge(
            'training_jobs_active',
            'Number of active training jobs'
        )
        
        self.training_jobs_completed = Counter(
            'training_jobs_completed_total',
            'Total completed training jobs',
            ['status']  # 'success' or 'failed'
        )
        
        # System health metrics
        self.system_health = Gauge(
            'system_health',
            'System health status',
            ['component']
        )
        
        logger.info("prometheus_metrics_initialized", port=port)
    
    def start_server(self) -> None:
        """Start Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if self.server_started:
            return
        
        try:
            start_http_server(self.port)
            self.server_started = True
            logger.info("prometheus_server_started", port=self.port)
        except Exception as e:
            logger.error("prometheus_server_start_failed", error=str(e))
    
    def record_engine_pnl(
        self,
        engine: str,
        symbol: str,
        regime: str,
        pnl_usd: float,
    ) -> None:
        """Record engine PnL."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.engine_pnl.labels(engine=engine, symbol=symbol, regime=regime).set(pnl_usd)
    
    def record_total_pnl(self, pnl_usd: float) -> None:
        """Record total PnL."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.total_pnl.set(pnl_usd)
    
    def record_engine_latency(
        self,
        engine: str,
        operation: str,
        latency_ms: float,
    ) -> None:
        """Record engine latency."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.engine_latency.labels(engine=engine, operation=operation).observe(latency_ms)
    
    def record_engine_error(
        self,
        engine: str,
        error_type: str,
    ) -> None:
        """Record engine error."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.engine_errors.labels(engine=engine, error_type=error_type).inc()
    
    def record_consensus_confidence(
        self,
        regime: str,
        confidence: float,
    ) -> None:
        """Record consensus confidence."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.consensus_confidence.labels(regime=regime).set(confidence)
    
    def record_consensus_agreement(
        self,
        regime: str,
        agreement_ratio: float,
    ) -> None:
        """Record consensus agreement ratio."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.consensus_agreement.labels(regime=regime).set(agreement_ratio)
    
    def record_cost_spread(
        self,
        venue: str,
        symbol: str,
        spread_bps: float,
    ) -> None:
        """Record cost model spread."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.cost_spread.labels(venue=venue, symbol=symbol).set(spread_bps)
    
    def record_cost_total(
        self,
        venue: str,
        symbol: str,
        total_cost_bps: float,
    ) -> None:
        """Record total cost."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.cost_total.labels(venue=venue, symbol=symbol).set(total_cost_bps)
    
    def record_training_job_duration(
        self,
        coin: str,
        regime: str,
        duration_seconds: float,
    ) -> None:
        """Record training job duration."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.training_job_duration.labels(coin=coin, regime=regime).observe(duration_seconds)
    
    def set_training_jobs_active(self, count: int) -> None:
        """Set number of active training jobs."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.training_jobs_active.set(count)
    
    def record_training_job_completed(self, status: str) -> None:
        """Record completed training job."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.training_jobs_completed.labels(status=status).inc()
    
    def set_system_health(self, component: str, health: float) -> None:
        """
        Set system health status.
        
        Args:
            component: Component name
            health: Health score (0.0 = down, 1.0 = healthy)
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.system_health.labels(component=component).set(health)


# Global metrics instance
_metrics_instance: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance

