"""
Infrastructure module for scalable architecture.

Provides message bus, event loop management, and scalable engine integration.
"""

from .message_bus import (
    MessageBusInterface,
    RedisStreamsMessageBus,
    InMemoryMessageBus,
    create_message_bus,
    Message,
    StreamType,
)
from .event_loop_manager import (
    EventLoopManager,
    EventLoopStatus,
    EventLoopMetrics,
)
from .scalable_engine import (
    ScalableEngine,
    ScalableEngineConfig,
    create_scalable_engine_from_config,
)

__all__ = [
    "MessageBusInterface",
    "RedisStreamsMessageBus",
    "InMemoryMessageBus",
    "create_message_bus",
    "Message",
    "StreamType",
    "EventLoopManager",
    "EventLoopStatus",
    "EventLoopMetrics",
    "ScalableEngine",
    "ScalableEngineConfig",
    "create_scalable_engine_from_config",
]

