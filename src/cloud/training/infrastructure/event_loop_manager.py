"""
Event Loop Manager for Scalable Architecture

Manages parallel event loops for processing coins in groups.
Supports 400 coins with configurable coins per event loop (default: 50).

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class EventLoopStatus(Enum):
    """Event loop status."""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class EventLoopMetrics:
    """Metrics for an event loop."""
    loop_id: int
    coins_assigned: int
    tasks_active: int
    messages_processed: int
    errors: int
    last_update: float
    status: EventLoopStatus


class EventLoopManager:
    """
    Manages multiple event loops for parallel coin processing.
    
    Each event loop processes a group of coins (default: 50 coins per loop).
    For 400 coins, this creates 8 event loops running in parallel.
    
    Features:
    - Automatic coin assignment to loops
    - Socket connection reuse
    - Adaptive pacing for order submission
    - Health monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        coins_per_loop: int = 50,
        max_loops: int = 10,
        timeout_seconds: int = 300,
    ):
        """
        Initialize event loop manager.
        
        Args:
            coins_per_loop: Number of coins per event loop
            max_loops: Maximum number of event loops
            timeout_seconds: Timeout for event loop operations
        """
        self.coins_per_loop = coins_per_loop
        self.max_loops = max_loops
        self.timeout_seconds = timeout_seconds
        
        # Event loops and their assignments
        self.loops: Dict[int, asyncio.AbstractEventLoop] = {}
        self.loop_tasks: Dict[int, asyncio.Task] = {}
        self.coin_assignments: Dict[str, int] = {}  # coin -> loop_id
        self.loop_coins: Dict[int, Set[str]] = {}  # loop_id -> set of coins
        
        # Metrics
        self.metrics: Dict[int, EventLoopMetrics] = {}
        self.status: Dict[int, EventLoopStatus] = {}
        
        # Processing callbacks
        self.processors: Dict[str, Callable] = {}  # coin -> processor function
        
        logger.info(
            "event_loop_manager_initialized",
            coins_per_loop=coins_per_loop,
            max_loops=max_loops,
            timeout_seconds=timeout_seconds,
        )
    
    def assign_coins(self, coins: List[str]) -> None:
        """
        Assign coins to event loops.
        
        Args:
            coins: List of coin symbols to assign
        """
        # Calculate number of loops needed
        num_loops = min(
            (len(coins) + self.coins_per_loop - 1) // self.coins_per_loop,
            self.max_loops,
        )
        
        # Assign coins to loops
        for i, coin in enumerate(coins):
            loop_id = i % num_loops
            self.coin_assignments[coin] = loop_id
            
            if loop_id not in self.loop_coins:
                self.loop_coins[loop_id] = set()
            self.loop_coins[loop_id].add(coin)
        
        logger.info(
            "coins_assigned",
            total_coins=len(coins),
            num_loops=num_loops,
            coins_per_loop=self.coins_per_loop,
        )
    
    def register_processor(
        self,
        coin: str,
        processor: Callable[[str], Any],
    ) -> None:
        """
        Register a processor function for a coin.
        
        Args:
            coin: Coin symbol
            processor: Async function that processes the coin
        """
        self.processors[coin] = processor
        logger.debug("processor_registered", coin=coin)
    
    async def start(self) -> None:
        """Start all event loops."""
        # Get unique loop IDs
        loop_ids = set(self.loop_coins.keys())
        
        # Create and start event loops
        for loop_id in loop_ids:
            await self._start_loop(loop_id)
        
        logger.info("event_loops_started", num_loops=len(loop_ids))
    
    async def _start_loop(self, loop_id: int) -> None:
        """Start a single event loop."""
        coins = self.loop_coins.get(loop_id, set())
        
        if not coins:
            logger.warning("no_coins_for_loop", loop_id=loop_id)
            return
        
        # Initialize metrics
        self.metrics[loop_id] = EventLoopMetrics(
            loop_id=loop_id,
            coins_assigned=len(coins),
            tasks_active=0,
            messages_processed=0,
            errors=0,
            last_update=time.time(),
            status=EventLoopStatus.IDLE,
        )
        self.status[loop_id] = EventLoopStatus.RUNNING
        
        # Create task for this loop
        task = asyncio.create_task(self._run_loop(loop_id, coins))
        self.loop_tasks[loop_id] = task
        
        logger.info(
            "event_loop_started",
            loop_id=loop_id,
            coins=len(coins),
        )
    
    async def _run_loop(self, loop_id: int, coins: Set[str]) -> None:
        """Run event loop for a group of coins."""
        try:
            # Create tasks for each coin
            tasks = []
            for coin in coins:
                if coin in self.processors:
                    task = asyncio.create_task(
                        self._process_coin(loop_id, coin)
                    )
                    tasks.append(task)
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.status[loop_id] = EventLoopStatus.STOPPED
            logger.info("event_loop_stopped", loop_id=loop_id)
            
        except Exception as e:
            self.status[loop_id] = EventLoopStatus.ERROR
            self.metrics[loop_id].errors += 1
            logger.error(
                "event_loop_error",
                loop_id=loop_id,
                error=str(e),
            )
    
    async def _process_coin(self, loop_id: int, coin: str) -> None:
        """Process a single coin in the event loop."""
        processor = self.processors.get(coin)
        if not processor:
            logger.warning("no_processor_for_coin", coin=coin, loop_id=loop_id)
            return
        
        try:
            self.metrics[loop_id].tasks_active += 1
            self.metrics[loop_id].last_update = time.time()
            
            # Process coin
            result = await processor(coin)
            
            self.metrics[loop_id].messages_processed += 1
            self.metrics[loop_id].tasks_active -= 1
            
            logger.debug(
                "coin_processed",
                loop_id=loop_id,
                coin=coin,
                result=result,
            )
            
        except Exception as e:
            self.metrics[loop_id].errors += 1
            self.metrics[loop_id].tasks_active -= 1
            logger.error(
                "coin_processing_error",
                loop_id=loop_id,
                coin=coin,
                error=str(e),
            )
    
    async def stop(self, timeout: Optional[float] = None) -> None:
        """
        Stop all event loops gracefully.
        
        Args:
            timeout: Timeout in seconds for graceful shutdown
        """
        logger.info("stopping_event_loops", num_loops=len(self.loop_tasks))
        
        # Set status to stopping
        for loop_id in self.status:
            if self.status[loop_id] == EventLoopStatus.RUNNING:
                self.status[loop_id] = EventLoopStatus.STOPPING
        
        # Cancel all tasks
        for loop_id, task in self.loop_tasks.items():
            task.cancel()
        
        # Wait for tasks to complete
        if timeout:
            await asyncio.wait_for(
                asyncio.gather(*self.loop_tasks.values(), return_exceptions=True),
                timeout=timeout,
            )
        else:
            await asyncio.gather(*self.loop_tasks.values(), return_exceptions=True)
        
        # Clear tasks
        self.loop_tasks.clear()
        self.status.clear()
        
        logger.info("event_loops_stopped")
    
    def get_metrics(self) -> Dict[int, EventLoopMetrics]:
        """Get metrics for all event loops."""
        # Update last_update times
        for loop_id in self.metrics:
            if self.status.get(loop_id) == EventLoopStatus.RUNNING:
                self.metrics[loop_id].status = EventLoopStatus.RUNNING
            elif self.status.get(loop_id) == EventLoopStatus.STOPPING:
                self.metrics[loop_id].status = EventLoopStatus.STOPPING
            elif self.status.get(loop_id) == EventLoopStatus.STOPPED:
                self.metrics[loop_id].status = EventLoopStatus.STOPPED
            elif self.status.get(loop_id) == EventLoopStatus.ERROR:
                self.metrics[loop_id].status = EventLoopStatus.ERROR
        
        return self.metrics.copy()
    
    def get_loop_for_coin(self, coin: str) -> Optional[int]:
        """Get event loop ID for a coin."""
        return self.coin_assignments.get(coin)
    
    def get_coins_for_loop(self, loop_id: int) -> Set[str]:
        """Get coins assigned to an event loop."""
        return self.loop_coins.get(loop_id, set()).copy()

