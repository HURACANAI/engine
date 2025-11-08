"""
Compiled Inference Layer

ONNX Runtime and TorchScript for fast ML model inference.
Pre-loads models and avoids Python loops for hardware-level speed.

Key Features:
- ONNX Runtime inference
- TorchScript JIT compilation
- Model pre-loading and warming
- Batch inference optimization
- Latency monitoring
- Hardware-aware scheduling

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class InferenceBackend(Enum):
    """Inference backend"""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    PYTORCH = "pytorch"  # Fallback


@dataclass
class InferenceResult:
    """Inference result"""
    prediction: np.ndarray
    latency_ms: float
    backend: InferenceBackend
    batch_size: int


class CompiledInferenceLayer:
    """
    Compiled Inference Layer.
    
    Fast ML model inference using ONNX Runtime or TorchScript.
    
    Usage:
        inference = CompiledInferenceLayer(backend=InferenceBackend.ONNX)
        
        # Load model
        inference.load_model("model.onnx")
        
        # Warm up
        inference.warm_up()
        
        # Inference
        result = inference.predict(features)
    """
    
    def __init__(
        self,
        backend: InferenceBackend = InferenceBackend.ONNX,
        device: Optional[str] = None
    ):
        """
        Initialize compiled inference layer.
        
        Args:
            backend: Inference backend (ONNX, TORCHSCRIPT, PYTORCH)
            device: Device ("cpu", "cuda", or None for auto)
        """
        self.backend = backend
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Model storage
        self.onnx_session: Optional[Any] = None
        self.torchscript_model: Optional[Any] = None
        self.pytorch_model: Optional[Any] = None
        
        # Model metadata
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        
        # Latency tracking
        self.latency_history: List[float] = []
        self.warmed_up = False
        
        logger.info(
            "compiled_inference_layer_initialized",
            backend=backend.value,
            device=self.device
        )
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if self.backend == InferenceBackend.ONNX:
            self._load_onnx_model(model_path)
        elif self.backend == InferenceBackend.TORCHSCRIPT:
            self._load_torchscript_model(model_path)
        else:
            self._load_pytorch_model(model_path)
        
        logger.info("model_loaded", path=str(model_path), backend=self.backend.value)
    
    def _load_onnx_model(self, model_path: Path) -> None:
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install onnxruntime.")
        
        # Create inference session
        providers = ['CPUExecutionProvider']
        if self.device == "cuda" and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        
        self.onnx_session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers
        )
        
        # Get input/output names and shapes
        self.input_names = [input.name for input in self.onnx_session.get_inputs()]
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        for input_info in self.onnx_session.get_inputs():
            shape = input_info.shape
            # Replace dynamic dimensions with None
            shape = tuple(None if isinstance(d, str) or d == -1 else d for d in shape)
            self.input_shapes[input_info.name] = shape
    
    def _load_torchscript_model(self, model_path: Path) -> None:
        """Load TorchScript model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install torch.")
        
        self.torchscript_model = torch.jit.load(str(model_path), map_location=self.device)
        self.torchscript_model.eval()
        
        # Get input/output shapes (simplified)
        # In production, would inspect the model graph
        logger.warning("torchscript_shape_inference_not_implemented")
    
    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model (fallback)"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install torch.")
        
        # This would load a PyTorch model state dict
        # For now, just log a warning
        logger.warning("pytorch_model_loading_not_fully_implemented", path=str(model_path))
    
    def warm_up(self, num_iterations: int = 10, batch_size: int = 1) -> None:
        """
        Warm up the model with dummy inputs.
        
        Args:
            num_iterations: Number of warm-up iterations
            batch_size: Batch size for warm-up
        """
        if not self.input_names:
            logger.warning("cannot_warm_up_no_input_info")
            return
        
        logger.info("model_warm_up_start", iterations=num_iterations, batch_size=batch_size)
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs(batch_size)
        
        # Run warm-up iterations
        for i in range(num_iterations):
            try:
                self._predict_internal(dummy_inputs)
            except Exception as e:
                logger.warning("warm_up_iteration_failed", iteration=i, error=str(e))
        
        self.warmed_up = True
        logger.info("model_warm_up_complete")
    
    def _create_dummy_inputs(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Create dummy inputs for warm-up"""
        dummy_inputs = {}
        
        for input_name, shape in self.input_shapes.items():
            # Replace None/batch dimension with batch_size
            shape_list = list(shape)
            if shape_list and (shape_list[0] is None or shape_list[0] == -1):
                shape_list[0] = batch_size
            
            # Create dummy input
            dummy_inputs[input_name] = np.random.randn(*shape_list).astype(np.float32)
        
        return dummy_inputs
    
    def predict(
        self,
        features: np.ndarray | Dict[str, np.ndarray],
        batch_size: Optional[int] = None
    ) -> InferenceResult:
        """
        Run inference.
        
        Args:
            features: Input features (numpy array or dict)
            batch_size: Batch size (optional, for batching)
        
        Returns:
            InferenceResult
        """
        start_time = time.perf_counter()
        
        # Prepare inputs
        if isinstance(features, np.ndarray):
            # Single input
            if len(self.input_names) != 1:
                raise ValueError(f"Expected {len(self.input_names)} inputs, got 1 array")
            inputs = {self.input_names[0]: features}
        else:
            # Multiple inputs
            inputs = features
        
        # Ensure inputs are in correct format
        inputs = self._prepare_inputs(inputs)
        
        # Run inference
        outputs = self._predict_internal(inputs)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_history.append(latency_ms)
        
        # Get prediction (assume single output for now)
        if isinstance(outputs, dict):
            prediction = list(outputs.values())[0]
        else:
            prediction = outputs
        
        # Get batch size
        actual_batch_size = prediction.shape[0] if len(prediction.shape) > 0 else 1
        
        return InferenceResult(
            prediction=prediction,
            latency_ms=latency_ms,
            backend=self.backend,
            batch_size=actual_batch_size
        )
    
    def _prepare_inputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare inputs for inference"""
        prepared = {}
        
        for input_name, input_data in inputs.items():
            # Ensure correct dtype
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # Ensure correct shape (handle batch dimension)
            if input_name in self.input_shapes:
                expected_shape = self.input_shapes[input_name]
                # Handle dynamic batch dimension
                if expected_shape[0] is None or expected_shape[0] == -1:
                    # Dynamic batch, keep as is
                    pass
                elif len(input_data.shape) < len(expected_shape):
                    # Add missing dimensions
                    input_data = np.expand_dims(input_data, axis=0)
            
            prepared[input_name] = input_data
        
        return prepared
    
    def _predict_internal(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray] | np.ndarray:
        """Internal prediction method"""
        if self.backend == InferenceBackend.ONNX:
            return self._predict_onnx(inputs)
        elif self.backend == InferenceBackend.TORCHSCRIPT:
            return self._predict_torchscript(inputs)
        else:
            return self._predict_pytorch(inputs)
    
    def _predict_onnx(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ONNX inference"""
        if not self.onnx_session:
            raise RuntimeError("ONNX model not loaded")
        
        outputs = self.onnx_session.run(self.output_names, inputs)
        
        # Convert to dict
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def _predict_torchscript(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """TorchScript inference"""
        if not self.torchscript_model:
            raise RuntimeError("TorchScript model not loaded")
        
        # Convert inputs to tensors
        if len(inputs) == 1:
            input_tensor = torch.from_numpy(list(inputs.values())[0]).to(self.device)
        else:
            input_tensor = {k: torch.from_numpy(v).to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            output = self.torchscript_model(input_tensor)
        
        # Convert back to numpy
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        else:
            return {k: v.cpu().numpy() for k, v in output.items()}
    
    def _predict_pytorch(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """PyTorch inference (fallback)"""
        raise NotImplementedError("PyTorch inference not fully implemented")
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latency_history:
            return {}
        
        latencies = np.array(self.latency_history)
        
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "count": len(latencies)
        }
    
    def batch_predict(
        self,
        features_list: List[np.ndarray | Dict[str, np.ndarray]],
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """
        Run batch inference.
        
        Args:
            features_list: List of input features
            batch_size: Batch size
        
        Returns:
            List of InferenceResult
        """
        results = []
        
        for i in range(0, len(features_list), batch_size):
            batch = features_list[i:i + batch_size]
            
            # Stack into batch
            if isinstance(batch[0], np.ndarray):
                batch_array = np.stack(batch, axis=0)
                batch_result = self.predict(batch_array, batch_size=len(batch))
                results.append(batch_result)
            else:
                # Multiple inputs - would need to handle differently
                # For now, predict individually
                for features in batch:
                    result = self.predict(features)
                    results.append(result)
        
        return results

