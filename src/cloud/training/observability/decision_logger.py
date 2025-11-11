"""
Enhanced DecisionEvent Logger

Comprehensive logging system for all model decisions with structured data.
Integrates with existing event_schema and adds additional functionality.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .event_schema import (
    TradingDecisionEvent,
    EngineVoteEvent,
    EventLogger,
    get_event_logger,
)

logger = structlog.get_logger(__name__)


class EnhancedDecisionLogger:
    """
    Enhanced decision logger with async file I/O and structured storage.
    
    Features:
    - Async file I/O for performance
    - Structured JSON storage
    - Queryable event store
    - Integration with Prometheus metrics
    - Automatic event batching
    """
    
    def __init__(
        self,
        storage_path: Path,
        batch_size: int = 100,
        flush_interval_seconds: float = 60.0,
    ) -> None:
        """
        Initialize enhanced decision logger.
        
        Args:
            storage_path: Path to store decision events
            batch_size: Number of events to batch before writing
            flush_interval_seconds: Interval to flush events to disk
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        
        # Event buffer
        self.event_buffer: List[Dict[str, Any]] = []
        self.buffer_lock = asyncio.Lock()
        
        # Background flush task
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Base event logger
        self.base_logger = get_event_logger()
        
        logger.info(
            "enhanced_decision_logger_initialized",
            storage_path=str(storage_path),
            batch_size=batch_size,
        )
    
    async def start(self) -> None:
        """Start background flush task."""
        if self.running:
            return
        
        self.running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        logger.info("enhanced_decision_logger_started")
    
    async def stop(self) -> None:
        """Stop logger and flush remaining events."""
        self.running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self._flush_buffer()
        
        logger.info("enhanced_decision_logger_stopped")
    
    async def log_decision(
        self,
        event: TradingDecisionEvent,
        prometheus_metrics: Optional[Any] = None,  # PrometheusMetrics type
    ) -> None:
        """
        Log a trading decision event.
        
        Args:
            event: Trading decision event
            prometheus_metrics: Optional Prometheus metrics instance
        """
        # Log to base logger (structured logging)
        self.base_logger.log_decision(event)
        
        # Add to buffer for async file write
        event_dict = event.to_dict()
        
        async with self.buffer_lock:
            self.event_buffer.append(event_dict)
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.batch_size:
                await self._flush_buffer()
        
        # Update Prometheus metrics if available
        if prometheus_metrics:
            try:
                prometheus_metrics.record_consensus_confidence(
                    regime=event.regime,
                    confidence=event.consensus_score,
                )
            except Exception as e:
                logger.warning("prometheus_metrics_update_failed", error=str(e))
    
    async def _flush_loop(self) -> None:
        """Background loop to flush events periodically."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("flush_loop_error", error=str(e))
    
    async def _flush_buffer(self) -> None:
        """Flush event buffer to disk."""
        async with self.buffer_lock:
            if not self.event_buffer:
                return
            
            # Get events to flush
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
        
        # Write to file asynchronously
        if events_to_flush:
            await self._write_events(events_to_flush)
    
    async def _write_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Write events to file asynchronously.
        
        Args:
            events: List of event dictionaries
        """
        # Generate filename based on date
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename = self.storage_path / f"decisions_{date_str}.jsonl"
        
        # Write events as JSONL (one JSON object per line)
        try:
            # Use asyncio to run file I/O in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_events_sync,
                filename,
                events,
            )
            
            logger.debug(
                "events_written",
                filename=str(filename),
                num_events=len(events),
            )
        except Exception as e:
            logger.error(
                "event_write_failed",
                filename=str(filename),
                error=str(e),
            )
            # Re-add events to buffer on failure
            async with self.buffer_lock:
                self.event_buffer.extend(events)
    
    def _write_events_sync(self, filename: Path, events: List[Dict[str, Any]]) -> None:
        """Synchronous file write (runs in thread pool)."""
        with open(filename, "a") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
    
    async def query_events(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query decision events.
        
        Args:
            symbol: Filter by symbol
            regime: Filter by regime
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of events to return
        
        Returns:
            List of matching events
        """
        # This would implement file-based querying
        # For now, returns empty list
        # In production, this would:
        # 1. Scan JSONL files in date range
        # 2. Filter by criteria
        # 3. Return matching events
        
        logger.debug(
            "querying_events",
            symbol=symbol,
            regime=regime,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
        )
        
        return []

