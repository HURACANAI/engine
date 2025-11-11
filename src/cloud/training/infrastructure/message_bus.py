"""
Message Bus Infrastructure for Scalable Architecture

Provides abstract interface and Redis Streams implementation for
decoupled, scalable data pipelines supporting 400 coins and 500 trades.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not available, message bus will use in-memory fallback")


class StreamType(Enum):
    """Message stream types."""
    MARKET_DATA = "market_data"
    FEATURES = "features"
    SIGNALS = "signals"
    ORDERS = "orders"
    EXECUTIONS = "executions"
    RISK_EVENTS = "risk_events"


@dataclass
class Message:
    """Message structure for message bus."""
    stream_type: StreamType
    coin: Optional[str]  # None for global streams (orders, executions)
    data: Dict[str, Any]
    timestamp: float
    message_id: Optional[str] = None


class MessageBusInterface(ABC):
    """Abstract interface for message bus implementations."""
    
    @abstractmethod
    async def publish(self, message: Message) -> str:
        """Publish a message to a stream."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
    ) -> AsyncIterator[Message]:
        """Subscribe to a stream and yield messages."""
        pass
    
    @abstractmethod
    async def create_consumer_group(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
    ) -> bool:
        """Create a consumer group for a stream."""
        pass
    
    @abstractmethod
    async def ack_message(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
        message_id: str = "*",
    ) -> int:
        """Acknowledge processed messages."""
        pass


class RedisStreamsMessageBus(MessageBusInterface):
    """
    Redis Streams implementation of message bus.
    
    Supports:
    - Multiple streams per coin
    - Consumer groups for horizontal scaling
    - Message persistence
    - Backpressure handling
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        stream_prefix: str = "huracan",
        max_consumers_per_group: int = 10,
        db: int = 0,
    ):
        """
        Initialize Redis Streams message bus.
        
        Args:
            host: Redis host
            port: Redis port
            stream_prefix: Prefix for stream names
            max_consumers_per_group: Maximum consumers per group
            db: Redis database number
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")
        
        self.host = host
        self.port = port
        self.stream_prefix = stream_prefix
        self.max_consumers = max_consumers_per_group
        self.db = db
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False
        
        logger.info(
            "redis_streams_bus_initialized",
            host=host,
            port=port,
            prefix=stream_prefix,
        )
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self.redis = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis.ping()
            self._connected = True
            logger.info("redis_connected", host=self.host, port=self.port)
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e), host=self.host, port=self.port)
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("redis_disconnected")
    
    def _get_stream_name(self, stream_type: StreamType, coin: Optional[str] = None) -> str:
        """Get stream name for stream type and coin."""
        if coin:
            return f"{self.stream_prefix}:{stream_type.value}:{coin}"
        return f"{self.stream_prefix}:{stream_type.value}:global"
    
    async def publish(self, message: Message) -> str:
        """Publish a message to a stream."""
        if not self._connected:
            await self.connect()
        
        stream_name = self._get_stream_name(message.stream_type, message.coin)
        
        # Prepare message data
        message_data = {
            "data": json.dumps(message.data),
            "timestamp": str(message.timestamp),
            "coin": message.coin or "",
        }
        
        try:
            # Add message to stream
            message_id = await self.redis.xadd(
                stream_name,
                message_data,
                maxlen=10000,  # Keep last 10k messages
            )
            
            logger.debug(
                "message_published",
                stream=stream_name,
                message_id=message_id,
                coin=message.coin,
            )
            
            return message_id
        except Exception as e:
            logger.error(
                "message_publish_failed",
                stream=stream_name,
                error=str(e),
            )
            raise
    
    async def subscribe(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        block_ms: int = 1000,
    ) -> AsyncIterator[Message]:
        """Subscribe to a stream and yield messages."""
        if not self._connected:
            await self.connect()
        
        stream_name = self._get_stream_name(stream_type, coin)
        
        # Create consumer group if provided
        if consumer_group:
            try:
                await self.create_consumer_group(stream_type, coin, consumer_group)
            except Exception:
                pass  # Group may already exist
        
        # Generate consumer name if not provided
        if not consumer_name:
            consumer_name = f"consumer_{int(time.time() * 1000)}"
        
        logger.info(
            "subscribed_to_stream",
            stream=stream_name,
            group=consumer_group,
            consumer=consumer_name,
        )
        
        # Read messages in a loop
        while True:
            try:
                # Read from stream
                if consumer_group:
                    # Read from consumer group
                    messages = await self.redis.xreadgroup(
                        consumer_group,
                        consumer_name,
                        {stream_name: ">"},  # Read pending and new messages
                        count=10,  # Read up to 10 messages at a time
                        block=block_ms,
                    )
                else:
                    # Read directly from stream
                    messages = await self.redis.xread(
                        {stream_name: "$"},  # Read from end
                        count=10,
                        block=block_ms,
                    )
                
                # Process messages
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        try:
                            # Parse message
                            data = json.loads(msg_data["data"])
                            timestamp = float(msg_data.get("timestamp", time.time()))
                            coin = msg_data.get("coin") or None
                            
                            message = Message(
                                stream_type=stream_type,
                                coin=coin,
                                data=data,
                                timestamp=timestamp,
                                message_id=msg_id,
                            )
                            
                            yield message
                        except Exception as e:
                            logger.error(
                                "message_parse_failed",
                                stream=stream_name,
                                message_id=msg_id,
                                error=str(e),
                            )
                            continue
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                logger.info("subscription_cancelled", stream=stream_name)
                break
            except Exception as e:
                logger.error(
                    "subscription_error",
                    stream=stream_name,
                    error=str(e),
                )
                await asyncio.sleep(1)  # Wait before retry
    
    async def create_consumer_group(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
    ) -> bool:
        """Create a consumer group for a stream."""
        if not self._connected:
            await self.connect()
        
        stream_name = self._get_stream_name(stream_type, coin)
        
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id="0",  # Start from beginning
                mkstream=True,  # Create stream if it doesn't exist
            )
            logger.info(
                "consumer_group_created",
                stream=stream_name,
                group=group_name,
            )
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                logger.debug(
                    "consumer_group_exists",
                    stream=stream_name,
                    group=group_name,
                )
                return True
            logger.error(
                "consumer_group_creation_failed",
                stream=stream_name,
                group=group_name,
                error=str(e),
            )
            return False
    
    async def ack_message(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
        message_id: str = "*",
    ) -> int:
        """Acknowledge processed messages."""
        if not self._connected:
            await self.connect()
        
        stream_name = self._get_stream_name(stream_type, coin)
        
        try:
            acked = await self.redis.xack(stream_name, group_name, message_id)
            logger.debug(
                "message_acked",
                stream=stream_name,
                group=group_name,
                message_id=message_id,
                count=acked,
            )
            return acked
        except Exception as e:
            logger.error(
                "message_ack_failed",
                stream=stream_name,
                group=group_name,
                error=str(e),
            )
            return 0


class InMemoryMessageBus(MessageBusInterface):
    """
    In-memory message bus for testing and development.
    
    Not suitable for production use.
    """
    
    def __init__(self):
        """Initialize in-memory message bus."""
        self.streams: Dict[str, List[Message]] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        logger.info("in_memory_bus_initialized")
    
    def _get_stream_name(self, stream_type: StreamType, coin: Optional[str] = None) -> str:
        """Get stream name for stream type and coin."""
        if coin:
            return f"{stream_type.value}:{coin}"
        return f"{stream_type.value}:global"
    
    async def publish(self, message: Message) -> str:
        """Publish a message to a stream."""
        stream_name = self._get_stream_name(message.stream_type, message.coin)
        
        # Generate message ID
        message_id = f"{int(time.time() * 1000000)}-0"
        message.message_id = message_id
        
        # Store message
        if stream_name not in self.streams:
            self.streams[stream_name] = []
        self.streams[stream_name].append(message)
        
        # Keep only last 1000 messages
        if len(self.streams[stream_name]) > 1000:
            self.streams[stream_name] = self.streams[stream_name][-1000:]
        
        # Notify subscribers
        if stream_name in self.subscribers:
            for queue in self.subscribers[stream_name]:
                try:
                    await queue.put(message)
                except Exception:
                    pass  # Subscriber may have disconnected
        
        logger.debug("message_published", stream=stream_name, message_id=message_id)
        return message_id
    
    async def subscribe(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
    ) -> AsyncIterator[Message]:
        """Subscribe to a stream and yield messages."""
        stream_name = self._get_stream_name(stream_type, coin)
        
        # Create queue for this subscriber
        queue: asyncio.Queue = asyncio.Queue()
        if stream_name not in self.subscribers:
            self.subscribers[stream_name] = []
        self.subscribers[stream_name].append(queue)
        
        logger.info("subscribed_to_stream", stream=stream_name)
        
        # Yield messages from queue
        try:
            while True:
                message = await queue.get()
                yield message
        except asyncio.CancelledError:
            logger.info("subscription_cancelled", stream=stream_name)
            if stream_name in self.subscribers:
                self.subscribers[stream_name].remove(queue)
    
    async def create_consumer_group(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
    ) -> bool:
        """Create a consumer group (no-op for in-memory)."""
        return True
    
    async def ack_message(
        self,
        stream_type: StreamType,
        coin: Optional[str] = None,
        group_name: str = "default",
        message_id: str = "*",
    ) -> int:
        """Acknowledge message (no-op for in-memory)."""
        return 1


def create_message_bus(config: Dict[str, Any]) -> MessageBusInterface:
    """
    Create message bus from configuration.
    
    Args:
        config: Configuration dictionary with message_bus settings
    
    Returns:
        MessageBusInterface instance
    """
    bus_config = config.get("message_bus", {})
    bus_type = bus_config.get("type", "redis_streams")
    
    if bus_type == "redis_streams":
        if not REDIS_AVAILABLE:
            logger.warning("redis not available, falling back to in-memory bus")
            return InMemoryMessageBus()
        
        return RedisStreamsMessageBus(
            host=bus_config.get("host", "localhost"),
            port=bus_config.get("port", 6379),
            stream_prefix=bus_config.get("stream_prefix", "huracan"),
            max_consumers_per_group=bus_config.get("max_consumers_per_group", 10),
        )
    elif bus_type == "in_memory":
        return InMemoryMessageBus()
    else:
        raise ValueError(f"Unknown message bus type: {bus_type}")

