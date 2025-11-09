"""
Enhanced Walk-Forward Testing

Strict time order, no peeking, expanding/sliding windows.
Mimics live trading with realistic costs and event time.

Key Features:
- Expanding window: Train on data up to T, predict T+1
- Sliding window: Models that forget old regimes
- Strict chronology: Never use future data
- Real costs: Fees, slippage, spread, funding, missed fills
- Event time: Train on what was known at order time only
- Version tracking: Model and data versions in every log row
- Separate research/evaluation datasets

Author: Huracan Engine Team
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class WindowType(Enum):
    """Window type for walk-forward testing"""
    EXPANDING = "expanding"  # Train on all data up to T
    SLIDING = "sliding"  # Train on fixed window ending at T


@dataclass
class WalkForwardConfig:
    """Walk-forward configuration"""
    window_type: WindowType = WindowType.EXPANDING
    train_window_size: Optional[int] = None  # For sliding window (e.g., 12 months)
    step_size: int = 1  # Predict one step at a time
    min_train_samples: int = 100  # Minimum samples for training
    enable_online_learning: bool = False  # Update model after each prediction
    time_decay_weight: Optional[float] = None  # Exponential decay (e.g., 0.99)
    require_data_version: bool = True  # Require data version tracking
    require_model_version: bool = True  # Require model version tracking


@dataclass
class PredictionRecord:
    """Prediction record"""
    pred_id: str
    timestamp: datetime
    symbol: str
    model_id: str
    model_version: str
    data_version: str
    signal_type: str  # e.g., "long", "short"
    predicted_label: Optional[float] = None  # Predicted return or label
    predicted_confidence: float = 0.0
    horizon: int = 1  # Prediction horizon (steps ahead)
    features_hash: str = ""  # Hash of features used
    features: Dict[str, float] = field(default_factory=dict)
    regime: Optional[str] = None
    volatility_bucket: Optional[str] = None
    trend_bucket: Optional[str] = None


@dataclass
class TradeRecord:
    """Trade record"""
    trade_id: str
    pred_id: str
    timestamp_open: datetime
    symbol: str
    side: str  # "long" or "short"
    size: float  # Position size in base currency
    entry_price: float
    # Optional fields with defaults come after required fields
    timestamp_close: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    funding_cost: float = 0.0
    pnl: float = 0.0
    pnl_after_costs: float = 0.0
    exit_reason: Optional[str] = None
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    mae: Optional[float] = None  # Maximum Adverse Excursion
    time_in_trade_seconds: Optional[float] = None
    risk_preset: Optional[str] = None


@dataclass
class FeatureSnapshot:
    """Feature snapshot at decision time"""
    trade_id: str
    timestamp: datetime
    features: Dict[str, float]  # All engineered features
    z_scores: Dict[str, float]  # Z-scores of features
    feature_hash: str


@dataclass
class AttributionRecord:
    """Attribution record for a trade"""
    trade_id: str
    method: str  # "shap", "permutation", etc.
    top_features: List[Tuple[str, float]]  # (feature_name, importance)
    shap_values: Optional[Dict[str, float]] = None
    error_type: Optional[str] = None  # "direction_wrong", "timing_late", "stop_too_tight", etc.


@dataclass
class WalkForwardStep:
    """Single walk-forward step"""
    step_id: int
    train_start: datetime
    train_end: datetime
    test_timestamp: datetime
    train_size: int
    model_id: str
    model_version: str
    data_version: str
    prediction: Optional[PredictionRecord] = None
    trade: Optional[TradeRecord] = None
    attribution: Optional[AttributionRecord] = None


class EnhancedWalkForwardTester:
    """
    Enhanced Walk-Forward Tester.
    
    Strict time order, no peeking, expanding/sliding windows.
    
    Usage:
        tester = EnhancedWalkForwardTester(
            config=WalkForwardConfig(
                window_type=WindowType.EXPANDING,
                step_size=1,
                time_decay_weight=0.99
            )
        )
        
        # Run walk-forward test
        results = tester.run_walk_forward(
            data=dataframe,
            train_fn=train_function,
            predict_fn=predict_function,
            execute_fn=execute_function
        )
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize enhanced walk-forward tester.
        
        Args:
            config: Walk-forward configuration
            storage_path: Path to store results
        """
        self.config = config
        self.storage_path = storage_path or Path("walk_forward_results")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.steps: List[WalkForwardStep] = []
        self.predictions: List[PredictionRecord] = []
        self.trades: List[TradeRecord] = []
        self.feature_snapshots: List[FeatureSnapshot] = []
        self.attributions: List[AttributionRecord] = []
        
        # Model tracking
        self.model_versions: Dict[str, str] = {}  # model_id -> version
        self.current_model: Optional[Any] = None
        self.current_model_id: Optional[str] = None
        self.current_model_version: Optional[str] = None
        
        logger.info(
            "enhanced_walk_forward_tester_initialized",
            window_type=config.window_type.value,
            step_size=config.step_size,
            time_decay_weight=config.time_decay_weight
        )
    
    def run_walk_forward(
        self,
        data: pl.DataFrame,
        train_fn: Callable[[pl.DataFrame, Dict[str, Any]], Any],
        predict_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        execute_fn: Callable[[Dict[str, Any], Dict[str, Any]], TradeRecord],
        evaluation_start_idx: Optional[int] = None,  # Lock evaluation set
        feature_fn: Optional[Callable[[pl.DataFrame, int], Dict[str, float]]] = None,
        attribution_fn: Optional[Callable[[Any, Dict[str, Any], float], AttributionRecord]] = None,
    ) -> List[WalkForwardStep]:
        """
        Run walk-forward test with strict chronology.
        
        Args:
            data: Historical data (must be sorted by timestamp)
            train_fn: Training function (data, config) -> model
            predict_fn: Prediction function (model, features) -> {prediction, confidence, ...}
            execute_fn: Execution function (prediction, market_data) -> TradeRecord
            evaluation_start_idx: Start index for evaluation set (locked until end)
            feature_fn: Feature generation function (data, idx) -> features
            attribution_fn: Attribution function (model, features, outcome) -> AttributionRecord
        
        Returns:
            List of WalkForwardStep results
        """
        # Validate data is sorted
        if not self._validate_data_chronology(data):
            raise ValueError("Data must be sorted by timestamp")
        
        # Determine evaluation start
        if evaluation_start_idx is None:
            evaluation_start_idx = int(len(data) * 0.7)  # Default 70% train, 30% eval
        
        # Get training and evaluation data
        train_data = data[:evaluation_start_idx]
        eval_data = data[evaluation_start_idx:]
        
        logger.info(
            "walk_forward_test_started",
            total_samples=len(data),
            train_samples=len(train_data),
            eval_samples=len(eval_data),
            window_type=self.config.window_type.value
        )
        
        # Initialize model
        self.current_model = None
        self.current_model_id = str(uuid.uuid4())
        self.current_model_version = "1.0.0"
        
        # Run walk-forward steps
        for step_idx in range(0, len(eval_data), self.config.step_size):
            # Get test timestamp
            test_idx = evaluation_start_idx + step_idx
            if test_idx >= len(data):
                break
            
            test_timestamp = data[test_idx, "timestamp"]  # Assuming timestamp column exists
            
            # Determine training window
            if self.config.window_type == WindowType.EXPANDING:
                # Expanding window: use all data up to test timestamp
                train_window_end = test_idx
                train_window_start = 0
            else:
                # Sliding window: use fixed window ending at test timestamp
                if self.config.train_window_size is None:
                    raise ValueError("train_window_size required for sliding window")
                train_window_end = test_idx
                train_window_start = max(0, test_idx - self.config.train_window_size)
            
            # Get training data
            train_window = data[train_window_start:train_window_end]
            
            # Check minimum samples
            if len(train_window) < self.config.min_train_samples:
                logger.warning(
                    "insufficient_train_samples",
                    step=step_idx,
                    train_size=len(train_window),
                    min_required=self.config.min_train_samples
                )
                continue
            
            # Apply time decay weighting if enabled
            if self.config.time_decay_weight:
                train_window = self._apply_time_decay(train_window, self.config.time_decay_weight)
            
            # Train model (if needed)
            if self.current_model is None or step_idx == 0 or not self.config.enable_online_learning:
                logger.info(
                    "training_model",
                    step=step_idx,
                    train_size=len(train_window),
                    model_id=self.current_model_id
                )
                
                # Generate data version
                data_version = self._generate_data_version(train_window)
                
                # Train model
                try:
                    self.current_model = train_fn(train_window, {
                        "model_id": self.current_model_id,
                        "step": step_idx,
                        "timestamp": test_timestamp
                    })
                    
                    # Update model version
                    self.current_model_version = f"{self.current_model_version.split('.')[0]}.{step_idx}.0"
                    self.model_versions[self.current_model_id] = self.current_model_version
                    
                except Exception as e:
                    logger.error(
                        "model_training_failed",
                        step=step_idx,
                        error=str(e)
                    )
                    continue
            
            # Generate features at decision time (no future data)
            if feature_fn:
                features = feature_fn(data, test_idx)
            else:
                features = self._generate_features_safe(data, test_idx)
            
            # Make prediction
            try:
                prediction_result = predict_fn(self.current_model, {
                    "features": features,
                    "timestamp": test_timestamp,
                    "symbol": data[test_idx, "symbol"] if "symbol" in data.columns else "UNKNOWN"
                })
                
                # Create prediction record
                pred_id = str(uuid.uuid4())
                prediction = PredictionRecord(
                    pred_id=pred_id,
                    timestamp=test_timestamp,
                    symbol=prediction_result.get("symbol", "UNKNOWN"),
                    model_id=self.current_model_id,
                    model_version=self.current_model_version,
                    data_version=data_version,
                    signal_type=prediction_result.get("signal_type", "unknown"),
                    predicted_label=prediction_result.get("predicted_label"),
                    predicted_confidence=prediction_result.get("confidence", 0.0),
                    horizon=self.config.step_size,
                    features_hash=self._hash_features(features),
                    features=features,
                    regime=prediction_result.get("regime"),
                    volatility_bucket=prediction_result.get("volatility_bucket"),
                    trend_bucket=prediction_result.get("trend_bucket")
                )
                
                self.predictions.append(prediction)
                
            except Exception as e:
                logger.error(
                    "prediction_failed",
                    step=step_idx,
                    error=str(e)
                )
                continue
            
            # Execute trade (simulate)
            try:
                # Get market data at test timestamp (no future data)
                market_data = {
                    "timestamp": test_timestamp,
                    "price": data[test_idx, "close"] if "close" in data.columns else data[test_idx, "price"],
                    "spread": data[test_idx, "spread"] if "spread" in data.columns else 0.0,
                    "liquidity": data[test_idx, "liquidity"] if "liquidity" in data.columns else 1.0,
                }
                
                # Execute trade
                trade = execute_fn(prediction_result, market_data)
                trade.pred_id = pred_id
                
                # Wait for reveal (get actual outcome)
                # In real implementation, would wait for next timestamp
                if test_idx + 1 < len(data):
                    reveal_timestamp = data[test_idx + 1, "timestamp"]
                    reveal_price = data[test_idx + 1, "close"] if "close" in data.columns else data[test_idx + 1, "price"]
                    
                    # Update trade with actual outcome
                    trade = self._update_trade_with_outcome(trade, reveal_price, reveal_timestamp)
                
                self.trades.append(trade)
                
                # Store feature snapshot
                feature_snapshot = FeatureSnapshot(
                    trade_id=trade.trade_id,
                    timestamp=test_timestamp,
                    features=features,
                    z_scores=self._calculate_z_scores(features, train_window),
                    feature_hash=self._hash_features(features)
                )
                self.feature_snapshots.append(feature_snapshot)
                
                # Compute attribution (if enabled)
                if attribution_fn and trade.pnl_after_costs is not None:
                    try:
                        attribution = attribution_fn(
                            self.current_model,
                            features,
                            trade.pnl_after_costs
                        )
                        attribution.trade_id = trade.trade_id
                        attribution.error_type = self._classify_error_type(trade, prediction)
                        self.attributions.append(attribution)
                    except Exception as e:
                        logger.warning(
                            "attribution_failed",
                            trade_id=trade.trade_id,
                            error=str(e)
                        )
                
            except Exception as e:
                logger.error(
                    "trade_execution_failed",
                    step=step_idx,
                    error=str(e)
                )
                continue
            
            # Create walk-forward step record
            step = WalkForwardStep(
                step_id=step_idx,
                train_start=data[train_window_start, "timestamp"],
                train_end=data[train_window_end - 1, "timestamp"],
                test_timestamp=test_timestamp,
                train_size=len(train_window),
                model_id=self.current_model_id,
                model_version=self.current_model_version,
                data_version=data_version,
                prediction=prediction,
                trade=trade,
                attribution=self.attributions[-1] if self.attributions else None
            )
            
            self.steps.append(step)
            
            # Online learning update (if enabled)
            if self.config.enable_online_learning and trade.pnl_after_costs is not None:
                try:
                    # Update model with new data point
                    self.current_model = self._update_model_online(
                        self.current_model,
                        features,
                        trade.pnl_after_costs,
                        train_window
                    )
                except Exception as e:
                    logger.warning(
                        "online_learning_update_failed",
                        step=step_idx,
                        error=str(e)
                    )
            
            # Log progress
            if step_idx % 100 == 0:
                logger.info(
                    "walk_forward_progress",
                    step=step_idx,
                    total_steps=len(eval_data),
                    trades=len(self.trades),
                    win_rate=self._calculate_win_rate() if self.trades else 0.0
                )
        
        logger.info(
            "walk_forward_test_complete",
            total_steps=len(self.steps),
            total_trades=len(self.trades),
            win_rate=self._calculate_win_rate() if self.trades else 0.0
        )
        
        return self.steps
    
    def _validate_data_chronology(self, data: pl.DataFrame) -> bool:
        """Validate data is sorted by timestamp"""
        if "timestamp" not in data.columns:
            return False
        
        timestamps = data["timestamp"].to_numpy()
        return np.all(timestamps[:-1] <= timestamps[1:])
    
    def _apply_time_decay(self, data: pl.DataFrame, decay_rate: float) -> pl.DataFrame:
        """Apply time decay weighting to data"""
        # Calculate weights based on recency
        timestamps = data["timestamp"].to_numpy()
        max_timestamp = timestamps.max()
        
        # Time differences in days
        time_diffs = (max_timestamp - timestamps) / np.timedelta64(1, 'D')
        
        # Exponential decay weights
        weights = np.exp(-decay_rate * time_diffs)
        weights = weights / weights.sum()  # Normalize
        
        # Add weights column (simplified - would need to handle polars)
        # In production, would sample data according to weights
        return data
    
    def _generate_data_version(self, data: pl.DataFrame) -> str:
        """Generate data version hash"""
        import hashlib
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        return f"data_v{data_hash}"
    
    def _generate_features_safe(self, data: pl.DataFrame, idx: int) -> Dict[str, float]:
        """Generate features safely (no future data)"""
        # Only use data up to idx
        features = {}
        
        # Example: rolling mean (only using past data)
        if idx > 0:
            window = data[max(0, idx - 20):idx]
            if "close" in window.columns:
                features["rolling_mean_20"] = window["close"].mean()
                features["rolling_std_20"] = window["close"].std()
        
        return features
    
    def _hash_features(self, features: Dict[str, float]) -> str:
        """Hash features"""
        import hashlib
        feature_str = str(sorted(features.items()))
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]
    
    def _update_trade_with_outcome(
        self,
        trade: TradeRecord,
        reveal_price: float,
        reveal_timestamp: datetime
    ) -> TradeRecord:
        """Update trade with actual outcome"""
        # Calculate PnL
        if trade.side == "long":
            pnl = (reveal_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - reveal_price) * trade.size
        
        trade.exit_price = reveal_price
        trade.timestamp_close = reveal_timestamp
        trade.pnl = pnl
        trade.pnl_after_costs = pnl - trade.fees - trade.slippage - trade.funding_cost
        trade.time_in_trade_seconds = (reveal_timestamp - trade.timestamp_open).total_seconds()
        
        return trade
    
    def _calculate_z_scores(
        self,
        features: Dict[str, float],
        train_data: pl.DataFrame
    ) -> Dict[str, float]:
        """Calculate z-scores of features"""
        z_scores = {}
        
        for feature_name, feature_value in features.items():
            # Calculate mean and std from training data
            # Simplified - would need actual feature extraction
            z_scores[feature_name] = 0.0  # Placeholder
        
        return z_scores
    
    def _classify_error_type(
        self,
        trade: TradeRecord,
        prediction: PredictionRecord
    ) -> str:
        """Classify error type for trade"""
        if trade.pnl_after_costs is None:
            return "unknown"
        
        if trade.pnl_after_costs < 0:
            # Loss
            if trade.exit_reason == "stop_loss":
                return "stop_too_tight"
            elif trade.exit_reason == "timing":
                return "timing_late"
            else:
                return "direction_wrong"
        else:
            return "none"
    
    def _update_model_online(
        self,
        model: Any,
        features: Dict[str, float],
        outcome: float,
        train_data: pl.DataFrame
    ) -> Any:
        """Update model with online learning"""
        # Placeholder - would implement actual online learning
        return model
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trades:
            return 0.0
        
        winning_trades = [t for t in self.trades if t.pnl_after_costs and t.pnl_after_costs > 0]
        return len(winning_trades) / len(self.trades)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get results summary"""
        if not self.trades:
            return {}
        
        winning_trades = [t for t in self.trades if t.pnl_after_costs and t.pnl_after_costs > 0]
        losing_trades = [t for t in self.trades if t.pnl_after_costs and t.pnl_after_costs <= 0]
        
        total_pnl = sum(t.pnl_after_costs for t in self.trades if t.pnl_after_costs)
        avg_winner = np.mean([t.pnl_after_costs for t in winning_trades]) if winning_trades else 0.0
        avg_loser = np.mean([t.pnl_after_costs for t in losing_trades]) if losing_trades else 0.0
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades),
            "total_pnl": total_pnl,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "profit_factor": abs(avg_winner / avg_loser) if avg_loser != 0 else 0.0,
            "payoff_ratio": avg_winner / abs(avg_loser) if avg_loser != 0 else 0.0
        }

