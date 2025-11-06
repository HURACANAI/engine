"""
Training models package.

Includes:
- Multi-model trainer (ensemble learning)
- Incremental trainer (daily updates)
- Reliable pattern detector (cross-coin patterns)
- Correlation analyzer (enhanced with reliable patterns)
- Meta-label trainer
- Alpha engines
- And more...
"""

# Incremental training
from .incremental_trainer import IncrementalModelTrainer, IncrementalTrainingResult

# Reliable pattern detection
from .reliable_pattern_detector import (
    ReliablePatternDetector,
    PatternReliability,
    PatternObservation,
)

# Correlation analysis (enhanced with reliable patterns)
from .correlation_analyzer import (
    CorrelationAnalyzer,
    CorrelationMetrics,
    PortfolioCorrelationRisk,
    MarketWideEvent,
)

__all__ = [
    # Incremental training
    "IncrementalModelTrainer",
    "IncrementalTrainingResult",
    # Reliable pattern detection
    "ReliablePatternDetector",
    "PatternReliability",
    "PatternObservation",
    # Correlation analysis
    "CorrelationAnalyzer",
    "CorrelationMetrics",
    "PortfolioCorrelationRisk",
    "MarketWideEvent",
]

