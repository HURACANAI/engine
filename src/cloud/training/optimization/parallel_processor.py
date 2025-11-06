"""
Parallel Signal Processing with Ray

Processes alpha engine signals in parallel for better performance.

Usage:
    processor = ParallelSignalProcessor(num_workers=6)
    signals = processor.process_all_engines(
        features=features,
        regime=regime,
        engines=all_engines,
    )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import ray
import structlog

from ..models.alpha_engines import AlphaEngineCoordinator, AlphaSignal, TradingTechnique

logger = structlog.get_logger(__name__)


@ray.remote
def process_engine_remote(engine, features: Dict, regime: str) -> AlphaSignal:
    """Remote function to process a single engine."""
    return engine.generate_signal(features, regime)


class ParallelSignalProcessor:
    """
    Parallel signal processing using Ray.

    Processes all alpha engines in parallel for better performance.

    Usage:
        processor = ParallelSignalProcessor(num_workers=6)
        signals = processor.process_all_engines(
            features=features,
            regime=regime,
            engines=all_engines,
        )
    """

    def __init__(self, num_workers: int = 6, use_ray: bool = True):
        """
        Initialize parallel signal processor.

        Args:
            num_workers: Number of parallel workers
            use_ray: Whether to use Ray for parallel processing
        """
        self.num_workers = num_workers
        self.use_ray = use_ray and ray.is_initialized()

        if self.use_ray:
            logger.info("parallel_signal_processor_initialized_ray", num_workers=num_workers)
        else:
            logger.info("parallel_signal_processor_initialized_sequential", num_workers=num_workers)

    def process_all_engines(
        self,
        features: Dict,
        regime: str,
        engines: Dict[TradingTechnique, any],
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """
        Process all engines in parallel.

        Args:
            features: Market features
            regime: Current market regime
            engines: Dictionary of engines by technique

        Returns:
            Dictionary of signals by technique
        """
        if self.use_ray:
            return self._process_parallel_ray(features, regime, engines)
        else:
            return self._process_sequential(features, regime, engines)

    def _process_parallel_ray(
        self,
        features: Dict,
        regime: str,
        engines: Dict[TradingTechnique, any],
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """Process engines in parallel using Ray."""
        # Create remote tasks
        futures = []
        technique_order = []

        for technique, engine in engines.items():
            # Put engine in Ray object store
            engine_ref = ray.put(engine)
            future = process_engine_remote.remote(engine_ref, features, regime)
            futures.append(future)
            technique_order.append(technique)

        # Wait for all results
        results = ray.get(futures)

        # Map results back to techniques
        signals = {}
        for technique, signal in zip(technique_order, results):
            signals[technique] = signal

        logger.debug(
            "parallel_processing_complete",
            num_engines=len(engines),
            num_signals=len(signals),
        )

        return signals

    def _process_sequential(
        self,
        features: Dict,
        regime: str,
        engines: Dict[TradingTechnique, any],
    ) -> Dict[TradingTechnique, AlphaSignal]:
        """Process engines sequentially (fallback)."""
        signals = {}

        for technique, engine in engines.items():
            signal = engine.generate_signal(features, regime)
            signals[technique] = signal

        logger.debug(
            "sequential_processing_complete",
            num_engines=len(engines),
            num_signals=len(signals),
        )

        return signals

    def get_statistics(self) -> dict:
        """Get processor statistics."""
        return {
            'num_workers': self.num_workers,
            'use_ray': self.use_ray,
            'ray_initialized': ray.is_initialized() if self.use_ray else False,
        }

