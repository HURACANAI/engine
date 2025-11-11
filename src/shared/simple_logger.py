"""
Simple Human-Readable Logger

Provides clean, human-readable log output.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Optional

import structlog  # type: ignore[import-untyped]


class SimpleLogger:
    """Simple human-readable logger."""
    
    def __init__(self, name: str = "huracan"):
        """Initialize simple logger.
        
        Args:
            name: Logger name
        """
        self.logger = structlog.get_logger(name)
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging for human-readable output."""
        structlog.configure(
            processors=[
                self._timestamp_processor,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer(),  # Human-readable output
            ],
            wrapper_class=structlog.make_filtering_bound_logger("INFO"),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _timestamp_processor(self, logger, method_name, event_dict):
        """Add timestamp in human-readable format."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        event_dict["timestamp"] = timestamp
        return event_dict
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self.logger.info(f"✅ {message}", **kwargs)
    
    def skip(self, message: str, **kwargs):
        """Log skip message."""
        self.logger.warning(f"⚠️ {message}", **kwargs)
    
    def fail(self, message: str, **kwargs):
        """Log failure message."""
        self.logger.error(f"❌ {message}", **kwargs)


# Global logger instance
logger = SimpleLogger()

