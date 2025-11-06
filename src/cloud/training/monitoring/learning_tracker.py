"""Comprehensive learning tracker that categorizes what the engine learns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class LearningCategory(str, Enum):
    """Categories of what the engine learns."""
    
    # Price Action & Market Microstructure
    PRICE_ACTION = "price_action"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    VOLATILITY_PATTERNS = "volatility_patterns"
    VOLUME_PATTERNS = "volume_patterns"
    ORDER_FLOW = "order_flow"
    
    # Cross-Asset & Relative Value
    CORRELATION = "correlation"
    RELATIVE_VALUE = "relative_value"
    LEAD_LAG = "lead_lag"
    SPREAD_PATTERNS = "spread_patterns"
    
    # Learning & Meta
    META_LEARNING = "meta_learning"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"
    ENSEMBLE_COMBINATION = "ensemble_combination"
    PATTERN_RECOGNITION = "pattern_recognition"
    
    # Exotic & Research
    ANOMALY_DETECTION = "anomaly_detection"
    REGIME_DETECTION = "regime_detection"
    CONCEPT_DRIFT = "concept_drift"
    
    # Model Creation
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_PUBLICATION = "model_publication"
    HAMILTON_MODEL = "hamilton_model"  # Model for Hamilton to use
    
    # Errors & Debugging
    ERROR_PATTERNS = "error_patterns"
    FAILURE_MODES = "failure_modes"
    RECOVERY_STRATEGIES = "recovery_strategies"
    
    # Performance
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RISK_MANAGEMENT = "risk_management"
    POSITION_SIZING = "position_sizing"


@dataclass
class EngineLearning:
    """What a single alpha engine is learning."""
    engine_name: str
    category: LearningCategory
    learning_type: str  # e.g., "pattern", "weight", "threshold"
    description: str
    confidence: float  # 0.0 to 1.0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelCreation:
    """Tracking model creation for Hamilton."""
    symbol: str
    model_type: str  # e.g., "baseline", "incremental", "ensemble"
    training_samples: int
    validation_metrics: Dict[str, float]
    publication_status: str  # "published", "rejected", "pending"
    rejection_reason: Optional[str] = None
    hamilton_ready: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ErrorLearning:
    """What the engine learns from errors."""
    error_type: str
    error_message: str
    context: Dict[str, Any]
    recovery_action: Optional[str] = None
    learned_pattern: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LearningSnapshot:
    """Complete snapshot of what the engine is learning."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    engine_learnings: List[EngineLearning] = field(default_factory=list)
    model_creations: List[ModelCreation] = field(default_factory=list)
    error_learnings: List[ErrorLearning] = field(default_factory=list)
    overall_confidence: float = 0.0
    categories_learned: Dict[LearningCategory, float] = field(default_factory=dict)
    top_insights: List[str] = field(default_factory=list)


class LearningTracker:
    """Tracks and categorizes what the engine learns."""
    
    def __init__(self, output_dir: Path = Path("logs/learning")):
        """Initialize learning tracker.
        
        Args:
            output_dir: Directory to save learning snapshots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._engine_learnings: List[EngineLearning] = []
        self._model_creations: List[ModelCreation] = []
        self._error_learnings: List[ErrorLearning] = []
        
        logger.info("learning_tracker_initialized", output_dir=str(self.output_dir))
    
    def track_engine_learning(
        self,
        engine_name: str,
        category: LearningCategory,
        learning_type: str,
        description: str,
        confidence: float,
        examples: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Track what a single engine is learning.
        
        Args:
            engine_name: Name of the alpha engine
            category: Category of learning
            learning_type: Type of learning (e.g., "pattern", "weight")
            description: Description of what was learned
            confidence: Confidence in the learning (0.0 to 1.0)
            examples: Examples of the learning
            metrics: Metrics related to the learning
        """
        learning = EngineLearning(
            engine_name=engine_name,
            category=category,
            learning_type=learning_type,
            description=description,
            confidence=confidence,
            examples=examples or [],
            metrics=metrics or {},
        )
        
        self._engine_learnings.append(learning)
        
        logger.info(
            "engine_learning_tracked",
            engine=engine_name,
            category=category.value,
            learning_type=learning_type,
            confidence=confidence,
        )
    
    def track_model_creation(
        self,
        symbol: str,
        model_type: str,
        training_samples: int,
        validation_metrics: Dict[str, float],
        publication_status: str,
        rejection_reason: Optional[str] = None,
        hamilton_ready: bool = False,
    ) -> None:
        """Track model creation for Hamilton.
        
        Args:
            symbol: Trading symbol
            model_type: Type of model (e.g., "baseline", "incremental")
            training_samples: Number of training samples
            validation_metrics: Validation metrics
            publication_status: Publication status
            rejection_reason: Reason for rejection if rejected
            hamilton_ready: Whether model is ready for Hamilton
        """
        creation = ModelCreation(
            symbol=symbol,
            model_type=model_type,
            training_samples=training_samples,
            validation_metrics=validation_metrics,
            publication_status=publication_status,
            rejection_reason=rejection_reason,
            hamilton_ready=hamilton_ready,
        )
        
        self._model_creations.append(creation)
        
        logger.info(
            "model_creation_tracked",
            symbol=symbol,
            model_type=model_type,
            publication_status=publication_status,
            hamilton_ready=hamilton_ready,
        )
    
    def track_error_learning(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        recovery_action: Optional[str] = None,
        learned_pattern: Optional[str] = None,
    ) -> None:
        """Track what the engine learns from errors.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Context where error occurred
            recovery_action: Action taken to recover
            learned_pattern: Pattern learned from the error
        """
        learning = ErrorLearning(
            error_type=error_type,
            error_message=error_message,
            context=context,
            recovery_action=recovery_action,
            learned_pattern=learned_pattern,
        )
        
        self._error_learnings.append(learning)
        
        logger.info(
            "error_learning_tracked",
            error_type=error_type,
            recovery_action=recovery_action,
            learned_pattern=learned_pattern,
        )
    
    def generate_snapshot(self) -> LearningSnapshot:
        """Generate a snapshot of current learning state.
        
        Returns:
            Learning snapshot
        """
        # Calculate overall confidence
        if self._engine_learnings:
            overall_confidence = sum(l.confidence for l in self._engine_learnings) / len(self._engine_learnings)
        else:
            overall_confidence = 0.0
        
        # Categorize learnings
        categories_learned: Dict[LearningCategory, float] = {}
        for learning in self._engine_learnings:
            if learning.category not in categories_learned:
                categories_learned[learning.category] = 0.0
            categories_learned[learning.category] += learning.confidence
        
        # Normalize category confidences
        for category in categories_learned:
            category_count = sum(1 for l in self._engine_learnings if l.category == category)
            if category_count > 0:
                categories_learned[category] /= category_count
        
        # Extract top insights
        top_learnings = sorted(
            self._engine_learnings,
            key=lambda x: x.confidence,
            reverse=True,
        )[:5]
        top_insights = [
            f"{l.engine_name}: {l.description} (confidence: {l.confidence:.2f})"
            for l in top_learnings
        ]
        
        snapshot = LearningSnapshot(
            engine_learnings=self._engine_learnings.copy(),
            model_creations=self._model_creations.copy(),
            error_learnings=self._error_learnings.copy(),
            overall_confidence=overall_confidence,
            categories_learned=categories_learned,
            top_insights=top_insights,
        )
        
        return snapshot
    
    def save_snapshot(self, snapshot: Optional[LearningSnapshot] = None) -> Path:
        """Save learning snapshot to file.
        
        Args:
            snapshot: Snapshot to save (generates new one if None)
            
        Returns:
            Path to saved snapshot file
        """
        if snapshot is None:
            snapshot = self.generate_snapshot()
        
        timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.output_dir / f"learning_snapshot_{timestamp_str}.json"
        
        # Convert to dict for JSON serialization
        snapshot_dict = {
            "timestamp": snapshot.timestamp.isoformat(),
            "overall_confidence": snapshot.overall_confidence,
            "categories_learned": {k.value: v for k, v in snapshot.categories_learned.items()},
            "top_insights": snapshot.top_insights,
            "engine_learnings": [asdict(l) for l in snapshot.engine_learnings],
            "model_creations": [asdict(m) for m in snapshot.model_creations],
            "error_learnings": [asdict(e) for e in snapshot.error_learnings],
        }
        
        # Convert datetime fields
        for learning in snapshot_dict["engine_learnings"]:
            learning["timestamp"] = learning["timestamp"].isoformat() if isinstance(learning["timestamp"], datetime) else learning["timestamp"]
        for creation in snapshot_dict["model_creations"]:
            creation["timestamp"] = creation["timestamp"].isoformat() if isinstance(creation["timestamp"], datetime) else creation["timestamp"]
        for error in snapshot_dict["error_learnings"]:
            error["timestamp"] = error["timestamp"].isoformat() if isinstance(error["timestamp"], datetime) else error["timestamp"]
        
        with open(snapshot_path, "w") as f:
            json.dump(snapshot_dict, f, indent=2)
        
        logger.info("learning_snapshot_saved", path=str(snapshot_path))
        return snapshot_path
    
    def get_latest_snapshot(self) -> Optional[LearningSnapshot]:
        """Get the latest learning snapshot.
        
        Returns:
            Latest snapshot or None
        """
        return self.generate_snapshot()
    
    def clear_old_learnings(self, keep_recent: int = 100) -> None:
        """Clear old learnings, keeping only recent ones.
        
        Args:
            keep_recent: Number of recent learnings to keep
        """
        if len(self._engine_learnings) > keep_recent:
            self._engine_learnings = self._engine_learnings[-keep_recent:]
        if len(self._model_creations) > keep_recent:
            self._model_creations = self._model_creations[-keep_recent:]
        if len(self._error_learnings) > keep_recent:
            self._error_learnings = self._error_learnings[-keep_recent:]

