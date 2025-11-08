"""
Trade Feedback System

Automates feedback capture for every fill, slippage, or rejection to update
model training datasets.

Key Features:
- Fill feedback capture
- Slippage tracking
- Rejection tracking
- Order lifecycle tracking
- Training dataset updates
- Integration with model retraining

Author: Huracan Engine Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class FeedbackType(Enum):
    """Feedback type"""
    FILL = "fill"
    SLIPPAGE = "slippage"
    REJECTION = "rejection"
    CANCEL = "cancel"
    TIMEOUT = "timeout"


@dataclass
class TradeFeedback:
    """Trade feedback record"""
    feedback_id: str
    order_id: str
    symbol: str
    direction: str  # "buy" or "sell"
    feedback_type: FeedbackType
    timestamp: datetime
    
    # Order details
    requested_price: float
    requested_size: float
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    
    # Slippage
    slippage_bps: Optional[float] = None
    expected_slippage_bps: Optional[float] = None
    slippage_error_bps: Optional[float] = None
    
    # Rejection details
    rejection_reason: Optional[str] = None
    rejection_code: Optional[str] = None
    
    # Market conditions at time of order
    market_price: float
    spread_bps: float
    liquidity_score: Optional[float] = None
    volatility_bps: Optional[float] = None
    
    # Signal details
    signal_confidence: Optional[float] = None
    signal_source: Optional[str] = None
    model_id: Optional[str] = None
    
    # Latency
    tick_to_trade_ns: Optional[int] = None
    order_submit_latency_ns: Optional[int] = None
    fill_latency_ns: Optional[int] = None
    
    # Outcome
    outcome_bps: Optional[float] = None  # Realized P&L in bps
    outcome_usd: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class FeedbackBatch:
    """Batch of feedback records for training"""
    batch_id: str
    feedbacks: List[TradeFeedback]
    created_at: datetime
    processed: bool = False
    processed_at: Optional[datetime] = None


class TradeFeedbackCollector:
    """
    Collects trade feedback for model training.
    
    Captures:
    - Fills and slippage
    - Rejections and reasons
    - Market conditions
    - Signal details
    - Outcomes
    
    Usage:
        collector = TradeFeedbackCollector()
        
        collector.record_fill(
            order_id="order_123",
            symbol="BTCUSDT",
            requested_price=50000.0,
            filled_price=50010.0,
            ...
        )
        
        # Get feedback for training
        feedback_batch = collector.get_feedback_batch()
    """
    
    def __init__(
        self,
        batch_size: int = 100,  # Number of feedbacks per batch
        storage_path: Optional[str] = None
    ):
        """
        Initialize trade feedback collector.
        
        Args:
            batch_size: Number of feedbacks per batch
            storage_path: Path to store feedback data
        """
        self.batch_size = batch_size
        self.storage_path = storage_path
        
        # Store feedbacks
        self.feedbacks: List[TradeFeedback] = []
        self.batches: List[FeedbackBatch] = []
        
        logger.info(
            "trade_feedback_collector_initialized",
            batch_size=batch_size,
            storage_path=storage_path
        )
    
    def record_fill(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        requested_price: float,
        requested_size: float,
        filled_price: float,
        filled_size: float,
        market_price: float,
        spread_bps: float,
        signal_confidence: Optional[float] = None,
        signal_source: Optional[str] = None,
        model_id: Optional[str] = None,
        liquidity_score: Optional[float] = None,
        volatility_bps: Optional[float] = None,
        tick_to_trade_ns: Optional[int] = None,
        order_submit_latency_ns: Optional[int] = None,
        fill_latency_ns: Optional[int] = None,
        outcome_bps: Optional[float] = None,
        outcome_usd: Optional[float] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> TradeFeedback:
        """
        Record a filled order.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            direction: Trade direction
            requested_price: Requested price
            requested_size: Requested size
            filled_price: Filled price
            filled_size: Filled size
            market_price: Market price at time of order
            spread_bps: Spread in bps
            signal_confidence: Signal confidence
            signal_source: Signal source
            model_id: Model ID
            liquidity_score: Liquidity score
            volatility_bps: Volatility in bps
            tick_to_trade_ns: Tick-to-trade latency
            order_submit_latency_ns: Order submit latency
            fill_latency_ns: Fill latency
            outcome_bps: Outcome in bps
            outcome_usd: Outcome in USD
            metadata: Optional metadata
        
        Returns:
            TradeFeedback
        """
        # Calculate slippage
        if direction == "buy":
            slippage_bps = ((filled_price - requested_price) / requested_price) * 10000
        else:
            slippage_bps = ((requested_price - filled_price) / requested_price) * 10000
        
        feedback = TradeFeedback(
            feedback_id=f"feedback_{int(time.time() * 1e9)}",
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            feedback_type=FeedbackType.FILL,
            timestamp=datetime.now(timezone.utc),
            requested_price=requested_price,
            requested_size=requested_size,
            filled_price=filled_price,
            filled_size=filled_size,
            slippage_bps=slippage_bps,
            market_price=market_price,
            spread_bps=spread_bps,
            signal_confidence=signal_confidence,
            signal_source=signal_source,
            model_id=model_id,
            liquidity_score=liquidity_score,
            volatility_bps=volatility_bps,
            tick_to_trade_ns=tick_to_trade_ns,
            order_submit_latency_ns=order_submit_latency_ns,
            fill_latency_ns=fill_latency_ns,
            outcome_bps=outcome_bps,
            outcome_usd=outcome_usd,
            metadata=metadata or {}
        )
        
        self.feedbacks.append(feedback)
        
        logger.info(
            "trade_feedback_fill_recorded",
            order_id=order_id,
            symbol=symbol,
            slippage_bps=slippage_bps,
            filled_size=filled_size
        )
        
        # Check if batch is ready
        if len(self.feedbacks) >= self.batch_size:
            self._create_batch()
        
        return feedback
    
    def record_rejection(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        requested_price: float,
        requested_size: float,
        rejection_reason: str,
        rejection_code: Optional[str] = None,
        market_price: float = 0.0,
        spread_bps: float = 0.0,
        signal_confidence: Optional[float] = None,
        signal_source: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> TradeFeedback:
        """
        Record a rejected order.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            direction: Trade direction
            requested_price: Requested price
            requested_size: Requested size
            rejection_reason: Rejection reason
            rejection_code: Rejection code
            market_price: Market price
            spread_bps: Spread in bps
            signal_confidence: Signal confidence
            signal_source: Signal source
            model_id: Model ID
            metadata: Optional metadata
        
        Returns:
            TradeFeedback
        """
        feedback = TradeFeedback(
            feedback_id=f"feedback_{int(time.time() * 1e9)}",
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            feedback_type=FeedbackType.REJECTION,
            timestamp=datetime.now(timezone.utc),
            requested_price=requested_price,
            requested_size=requested_size,
            rejection_reason=rejection_reason,
            rejection_code=rejection_code,
            market_price=market_price,
            spread_bps=spread_bps,
            signal_confidence=signal_confidence,
            signal_source=signal_source,
            model_id=model_id,
            metadata=metadata or {}
        )
        
        self.feedbacks.append(feedback)
        
        logger.info(
            "trade_feedback_rejection_recorded",
            order_id=order_id,
            symbol=symbol,
            rejection_reason=rejection_reason
        )
        
        # Check if batch is ready
        if len(self.feedbacks) >= self.batch_size:
            self._create_batch()
        
        return feedback
    
    def record_slippage(
        self,
        order_id: str,
        symbol: str,
        actual_slippage_bps: float,
        expected_slippage_bps: float,
        metadata: Optional[Dict[str, any]] = None
    ) -> None:
        """
        Record slippage feedback.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            actual_slippage_bps: Actual slippage in bps
            expected_slippage_bps: Expected slippage in bps
            metadata: Optional metadata
        """
        # Find the feedback for this order
        feedback = None
        for f in reversed(self.feedbacks):
            if f.order_id == order_id:
                feedback = f
                break
        
        if feedback:
            feedback.slippage_bps = actual_slippage_bps
            feedback.expected_slippage_bps = expected_slippage_bps
            feedback.slippage_error_bps = actual_slippage_bps - expected_slippage_bps
            
            logger.debug(
                "trade_feedback_slippage_recorded",
                order_id=order_id,
                symbol=symbol,
                actual_slippage_bps=actual_slippage_bps,
                expected_slippage_bps=expected_slippage_bps,
                slippage_error_bps=feedback.slippage_error_bps
            )
    
    def _create_batch(self) -> FeedbackBatch:
        """Create a feedback batch"""
        if len(self.feedbacks) < self.batch_size:
            return None
        
        # Take batch_size feedbacks
        batch_feedbacks = self.feedbacks[:self.batch_size]
        self.feedbacks = self.feedbacks[self.batch_size:]
        
        batch = FeedbackBatch(
            batch_id=f"batch_{int(time.time() * 1e9)}",
            feedbacks=batch_feedbacks,
            created_at=datetime.now(timezone.utc)
        )
        
        self.batches.append(batch)
        
        logger.info(
            "feedback_batch_created",
            batch_id=batch.batch_id,
            num_feedbacks=len(batch_feedbacks)
        )
        
        return batch
    
    def get_feedback_batch(self, processed: bool = False) -> Optional[FeedbackBatch]:
        """
        Get next unprocessed feedback batch.
        
        Args:
            processed: If True, return processed batches
        
        Returns:
            FeedbackBatch or None
        """
        for batch in self.batches:
            if batch.processed == processed:
                return batch
        
        return None
    
    def mark_batch_processed(self, batch_id: str) -> None:
        """Mark a batch as processed"""
        for batch in self.batches:
            if batch.batch_id == batch_id:
                batch.processed = True
                batch.processed_at = datetime.now(timezone.utc)
                logger.info("feedback_batch_processed", batch_id=batch_id)
                break
    
    def get_feedback_for_training(self) -> List[Dict[str, any]]:
        """
        Get feedback data formatted for training.
        
        Returns:
            List of feedback records as dictionaries
        """
        training_data = []
        
        for feedback in self.feedbacks:
            # Only include filled orders with outcomes
            if feedback.feedback_type == FeedbackType.FILL and feedback.outcome_bps is not None:
                training_data.append({
                    "symbol": feedback.symbol,
                    "direction": feedback.direction,
                    "signal_confidence": feedback.signal_confidence,
                    "market_price": feedback.market_price,
                    "spread_bps": feedback.spread_bps,
                    "liquidity_score": feedback.liquidity_score,
                    "volatility_bps": feedback.volatility_bps,
                    "slippage_bps": feedback.slippage_bps,
                    "expected_slippage_bps": feedback.expected_slippage_bps,
                    "outcome_bps": feedback.outcome_bps,
                    "outcome_usd": feedback.outcome_usd,
                    "model_id": feedback.model_id,
                    "timestamp": feedback.timestamp.isoformat()
                })
        
        return training_data

