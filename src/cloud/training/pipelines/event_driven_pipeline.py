"""
Event-Driven Pipeline

Async market data ingestion with nanosecond timestamps.
Replaces sequential logic with event queues.

Key Features:
- Async market data ingestion
- Nanosecond timestamps
- Event queues
- Event-driven processing
- Latency tracking between steps
- Module performance diagnostics

Author: Huracan Engine Team
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timezone
from collections import deque

import structlog

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Event type"""
    MARKET_UPDATE = "market_update"
    TICK = "tick"
    ORDER_BOOK_UPDATE = "order_book_update"
    TRADE = "trade"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


@dataclass
class MarketEvent:
    """Market event"""
    event_id: str
    event_type: EventType
    symbol: str
    timestamp_ns: int
    data: Dict[str, any] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class EventProcessingStage:
    """Event processing stage"""
    stage_name: str
    start_ns: int
    end_ns: Optional[int] = None
    duration_ns: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class EventProcessingResult:
    """Event processing result"""
    event_id: str
    stages: List[EventProcessingStage]
    total_duration_ns: int
    success: bool
    output: Optional[any] = None


class EventQueue:
    """Event queue for async processing"""
    
    def __init__(self, maxsize: int = 1000):
        """Initialize event queue"""
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.processed_count = 0
        self.dropped_count = 0
    
    async def put(self, event: MarketEvent) -> bool:
        """Put event in queue"""
        try:
            await self.queue.put(event)
            return True
        except asyncio.QueueFull:
            self.dropped_count += 1
            logger.warning("event_queue_full", event_id=event.event_id)
            return False
    
    async def get(self) -> MarketEvent:
        """Get event from queue"""
        event = await self.queue.get()
        self.processed_count += 1
        return event
    
    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()


class EventDrivenPipeline:
    """
    Event-Driven Pipeline.
    
    Processes market events asynchronously with nanosecond timestamps.
    Tracks latency between processing stages.
    
    Usage:
        pipeline = EventDrivenPipeline()
        
        # Register handlers
        pipeline.register_handler(EventType.MARKET_UPDATE, market_update_handler)
        pipeline.register_handler(EventType.TICK, tick_handler)
        
        # Start pipeline
        asyncio.run(pipeline.start())
        
        # Send events
        await pipeline.send_event(MarketEvent(...))
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        num_workers: int = 4
    ):
        """
        Initialize event-driven pipeline.
        
        Args:
            max_queue_size: Maximum queue size
            num_workers: Number of worker tasks
        """
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Event queues
        self.input_queue = EventQueue(maxsize=max_queue_size)
        self.output_queue = EventQueue(maxsize=max_queue_size)
        
        # Event handlers
        self.handlers: Dict[EventType, List[Callable]] = {}
        
        # Processing tracking
        self.processing_results: deque = deque(maxlen=10000)
        self.running = False
        
        # Latency tracking
        self.stage_latencies: Dict[str, List[int]] = {}
        
        logger.info(
            "event_driven_pipeline_initialized",
            max_queue_size=max_queue_size,
            num_workers=num_workers
        )
    
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[MarketEvent], Any]
    ) -> None:
        """
        Register event handler.
        
        Args:
            event_type: Event type
            handler: Handler function
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        
        logger.info(
            "event_handler_registered",
            event_type=event_type.value,
            handler=handler.__name__
        )
    
    async def send_event(self, event: MarketEvent) -> bool:
        """
        Send event to pipeline.
        
        Args:
            event: Market event
        
        Returns:
            True if event was queued
        """
        return await self.input_queue.put(event)
    
    async def process_event(self, event: MarketEvent) -> EventProcessingResult:
        """
        Process event through pipeline.
        
        Args:
            event: Market event
        
        Returns:
            EventProcessingResult
        """
        stages = []
        start_ns = time.perf_counter_ns()
        success = True
        output = None
        
        # Get handlers for event type
        handlers = self.handlers.get(event.event_type, [])
        
        if not handlers:
            logger.warning(
                "no_handlers_for_event",
                event_type=event.event_type.value,
                event_id=event.event_id
            )
            return EventProcessingResult(
                event_id=event.event_id,
                stages=[],
                total_duration_ns=0,
                success=False
            )
        
        # Process through each handler
        for handler in handlers:
            stage_name = handler.__name__
            stage_start_ns = time.perf_counter_ns()
            
            try:
                # Execute handler
                result = await self._execute_handler(handler, event)
                output = result
                
                stage_end_ns = time.perf_counter_ns()
                duration_ns = stage_end_ns - stage_start_ns
                
                # Track latency
                if stage_name not in self.stage_latencies:
                    self.stage_latencies[stage_name] = []
                self.stage_latencies[stage_name].append(duration_ns)
                
                stages.append(EventProcessingStage(
                    stage_name=stage_name,
                    start_ns=stage_start_ns,
                    end_ns=stage_end_ns,
                    duration_ns=duration_ns,
                    success=True
                ))
                
            except Exception as e:
                stage_end_ns = time.perf_counter_ns()
                duration_ns = stage_end_ns - stage_start_ns
                
                stages.append(EventProcessingStage(
                    stage_name=stage_name,
                    start_ns=stage_start_ns,
                    end_ns=stage_end_ns,
                    duration_ns=duration_ns,
                    success=False,
                    error=str(e)
                ))
                
                success = False
                logger.error(
                    "event_handler_failed",
                    event_id=event.event_id,
                    stage_name=stage_name,
                    error=str(e)
                )
                break
        
        total_duration_ns = time.perf_counter_ns() - start_ns
        
        result = EventProcessingResult(
            event_id=event.event_id,
            stages=stages,
            total_duration_ns=total_duration_ns,
            success=success,
            output=output
        )
        
        self.processing_results.append(result)
        
        return result
    
    async def _execute_handler(self, handler: Callable, event: MarketEvent) -> Any:
        """Execute handler (supports both sync and async)"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(event)
        else:
            return handler(event)
    
    async def worker(self, worker_id: int) -> None:
        """Worker task for processing events"""
        logger.info("event_worker_started", worker_id=worker_id)
        
        while self.running:
            try:
                # Get event from queue
                event = await self.input_queue.get()
                
                # Process event
                result = await self.process_event(event)
                
                # Put result in output queue
                if result.success and result.output:
                    output_event = MarketEvent(
                        event_id=f"output_{result.event_id}",
                        event_type=EventType.SIGNAL,  # Default output type
                        symbol=event.symbol,
                        timestamp_ns=time.perf_counter_ns(),
                        data={"result": result.output},
                        metadata={"source_event_id": event.event_id}
                    )
                    await self.output_queue.put(output_event)
                
                # Mark task as done
                self.input_queue.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "event_worker_error",
                    worker_id=worker_id,
                    error=str(e)
                )
        
        logger.info("event_worker_stopped", worker_id=worker_id)
    
    async def start(self) -> None:
        """Start event-driven pipeline"""
        self.running = True
        
        # Start worker tasks
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
        
        logger.info(
            "event_driven_pipeline_started",
            num_workers=self.num_workers
        )
        
        # Wait for workers (they run until stopped)
        await asyncio.gather(*workers)
    
    async def stop(self) -> None:
        """Stop event-driven pipeline"""
        self.running = False
        
        # Cancel worker tasks
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("event_driven_pipeline_stopped")
    
    def get_stage_latencies(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics for each stage"""
        stats = {}
        
        for stage_name, latencies in self.stage_latencies.items():
            if latencies:
                stats[stage_name] = {
                    "mean_ns": sum(latencies) / len(latencies),
                    "mean_ms": (sum(latencies) / len(latencies)) / 1_000_000,
                    "p50_ns": sorted(latencies)[len(latencies) // 2],
                    "p95_ns": sorted(latencies)[int(len(latencies) * 0.95)],
                    "p99_ns": sorted(latencies)[int(len(latencies) * 0.99)],
                    "max_ns": max(latencies),
                    "min_ns": min(latencies),
                    "count": len(latencies)
                }
        
        return stats
    
    def get_pipeline_metrics(self) -> Dict[str, any]:
        """Get pipeline metrics"""
        return {
            "input_queue_size": self.input_queue.size(),
            "output_queue_size": self.output_queue.size(),
            "processed_count": self.input_queue.processed_count,
            "dropped_count": self.input_queue.dropped_count,
            "stage_latencies": self.get_stage_latencies(),
            "recent_results": len(self.processing_results)
        }

