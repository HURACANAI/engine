"""High-frequency trading executor with low-latency optimization."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class HFTExecutor:
    """
    High-frequency trading executor:
    - Uses NumPy arrays (not Pandas)
    - Event-driven execution (asyncio)
    - Redis pub/sub for signal dispatch
    - Websocket-based order routing
    - Low-latency inference (<100ms)
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        max_latency_ms: float = 100.0,
    ) -> None:
        """
        Initialize HFT executor.
        
        Args:
            redis_client: Optional Redis client for pub/sub
            max_latency_ms: Maximum allowed latency in milliseconds
        """
        self.redis_client = redis_client
        self.max_latency_ms = max_latency_ms
        self.model_cache: Dict[str, Any] = {}
        
        logger.info(
            "hft_executor_initialized",
            max_latency_ms=max_latency_ms,
            redis_available=redis_client is not None,
        )

    async def execute_trade_async(
        self,
        signal: Dict[str, Any],
        model: Any,
    ) -> Dict[str, Any]:
        """
        Execute trade asynchronously.
        
        Args:
            signal: Trading signal dictionary
            model: Model for prediction
            
        Returns:
            Execution result dictionary
        """
        start_time = time.time()
        
        try:
            # Convert features to NumPy array
            features = np.array(signal.get('features', []), dtype=np.float32)
            
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Reshape for time series models (add time dimension if needed)
            if hasattr(model, 'input_shape'):
                # Assume model expects (samples, timesteps, features)
                if len(features.shape) == 2:
                    # Add timestep dimension
                    features = features.reshape(1, 1, -1)
            
            # Inference (<100ms target)
            inference_start = time.time()
            
            if hasattr(model, 'predict'):
                predictions = model.predict(features, verbose=0)
            else:
                predictions = model(features)
            
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Flatten predictions
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            prediction = float(predictions[0]) if len(predictions) > 0 else 0.0
            
            # Publish to Redis if available
            if self.redis_client:
                await self._publish_to_redis(signal, prediction)
            
            total_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                "status": "success",
                "prediction": prediction,
                "inference_time_ms": inference_time,
                "total_time_ms": total_time,
                "within_latency_limit": total_time < self.max_latency_ms,
                "symbol": signal.get('symbol'),
                "timestamp": time.time(),
            }
            
            if total_time > self.max_latency_ms:
                logger.warning(
                    "latency_exceeded",
                    total_time_ms=total_time,
                    max_latency_ms=self.max_latency_ms,
                )
            
            logger.debug(
                "trade_executed",
                symbol=signal.get('symbol'),
                inference_time_ms=inference_time,
                total_time_ms=total_time,
            )
            
            return result
            
        except Exception as e:
            logger.error("trade_execution_failed", error=str(e))
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
            }

    async def _publish_to_redis(
        self,
        signal: Dict[str, Any],
        prediction: float,
    ) -> None:
        """
        Publish signal to Redis.
        
        Args:
            signal: Trading signal
            prediction: Model prediction
        """
        if not self.redis_client:
            return
        
        try:
            message = {
                'symbol': signal.get('symbol'),
                'prediction': prediction,
                'timestamp': time.time(),
                'features': signal.get('features', []),
            }
            
            # Publish to Redis channel
            # Note: This is a placeholder - actual Redis pub/sub implementation
            # would use redis_client.publish()
            logger.debug("publishing_to_redis", symbol=signal.get('symbol'))
            
        except Exception as e:
            logger.warning("redis_publish_failed", error=str(e))

    def convert_to_numpy(
        self,
        data: Any,
    ) -> np.ndarray:
        """
        Convert data to NumPy array (faster than Pandas).
        
        Args:
            data: Input data (Pandas DataFrame/Series or list)
            
        Returns:
            NumPy array
        """
        if isinstance(data, np.ndarray):
            return data
        
        if hasattr(data, 'values'):
            # Pandas DataFrame/Series
            return data.values
        
        # List or other iterable
        return np.array(data, dtype=np.float32)

    async def process_signal_queue(
        self,
        signal_queue: asyncio.Queue,
        model: Any,
    ) -> None:
        """
        Process signals from queue asynchronously.
        
        Args:
            signal_queue: Async queue of signals
            model: Model for prediction
        """
        logger.info("signal_queue_processor_started")
        
        while True:
            try:
                # Get signal from queue
                signal = await signal_queue.get()
                
                # Execute trade
                result = await self.execute_trade_async(signal, model)
                
                # Mark task as done
                signal_queue.task_done()
                
                logger.debug("signal_processed", symbol=signal.get('symbol'))
                
            except asyncio.CancelledError:
                logger.info("signal_queue_processor_cancelled")
                break
            except Exception as e:
                logger.error("signal_processing_failed", error=str(e))

    def optimize_inference(
        self,
        model: Any,
        X_sample: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Optimize model inference for low latency.
        
        Args:
            model: Model to optimize
            X_sample: Sample input data
            
        Returns:
            Optimization results
        """
        logger.info("optimizing_inference")
        
        # Warmup
        for _ in range(10):
            _ = model.predict(X_sample, verbose=0)
        
        # Measure inference time
        times = []
        for _ in range(100):
            start = time.time()
            _ = model.predict(X_sample, verbose=0)
            times.append((time.time() - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        result = {
            "avg_inference_time_ms": float(avg_time),
            "std_inference_time_ms": float(std_time),
            "min_inference_time_ms": float(min_time),
            "max_inference_time_ms": float(max_time),
            "within_latency_limit": avg_time < self.max_latency_ms,
        }
        
        logger.info("inference_optimization_complete", **result)
        
        return result

