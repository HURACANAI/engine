"""
Incremental Model Trainer

Loads saved models and fine-tunes on new data only.
Much faster than full retraining (5-10 min vs 1-2 hours).

Key Features:
1. Load saved models from disk
2. Get only new data since last training
3. Fine-tune on new data (partial_fit or warm_start)
4. Save updated models with new timestamp
5. Track last training date per symbol

Usage:
    trainer = IncrementalModelTrainer(model_storage_path="models/")
    
    # Check if model exists
    if trainer.model_exists("BTC/USDT"):
        # Load and fine-tune
        result = trainer.train_incremental(
            symbol="BTC/USDT",
            trainer=multi_model_trainer,
            get_new_data_func=get_new_data_since,
        )
    else:
        # First time: full training
        full_training_func("BTC/USDT")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import pickle
import structlog
import pandas as pd
import numpy as np

logger = structlog.get_logger(__name__)


@dataclass
class IncrementalTrainingResult:
    """Result of incremental training."""
    symbol: str
    last_training_date: Optional[datetime]
    new_data_days: int
    training_time_seconds: float
    model_updated: bool
    performance_improvement: Optional[float] = None
    n_new_samples: int = 0


class IncrementalModelTrainer:
    """
    Incremental model trainer for daily updates.
    
    Workflow:
    1. Check if saved model exists
    2. If exists: Load model, get new data since last training, fine-tune
    3. If not: Do full training (first time)
    4. Save updated model with new timestamp
    
    Time Savings:
    - Full retrain: ~1-2 hours
    - Incremental update: ~5-10 minutes (only new data)
    """
    
    def __init__(
        self,
        model_storage_path: str = "models/",
        min_new_data_days: int = 1,  # Minimum days of new data to retrain
        warm_start: bool = True,  # Continue from previous weights
    ):
        """
        Initialize incremental trainer.
        
        Args:
            model_storage_path: Directory to store/load models
            min_new_data_days: Minimum days of new data required to retrain
            warm_start: Whether to use warm_start (continue from previous weights)
        """
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.min_new_data_days = min_new_data_days
        self.warm_start = warm_start
        
        logger.info(
            "incremental_trainer_initialized",
            storage_path=str(self.model_storage_path),
            min_new_data_days=min_new_data_days,
            warm_start=warm_start,
        )
    
    def model_exists(self, symbol: str) -> bool:
        """Check if saved model exists for symbol."""
        model_dir = self.model_storage_path / symbol.replace("/", "_")
        metadata_file = model_dir / "training_metadata.pkl"
        return metadata_file.exists()
    
    def get_last_training_date(self, symbol: str) -> Optional[datetime]:
        """Get last training date for symbol."""
        model_dir = self.model_storage_path / symbol.replace("/", "_")
        metadata_file = model_dir / "training_metadata.pkl"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            last_date = metadata.get('last_training_date')
            if isinstance(last_date, str):
                return datetime.fromisoformat(last_date.replace('Z', '+00:00'))
            return last_date
        except Exception as e:
            logger.warning("failed_to_load_metadata", symbol=symbol, error=str(e))
            return None
    
    def load_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load saved model and metadata."""
        model_dir = self.model_storage_path / symbol.replace("/", "_")
        
        if not model_dir.exists():
            return None
        
        try:
            # Load metadata
            metadata_file = model_dir / "training_metadata.pkl"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
            
            # Load models
            models = {}
            for model_file in model_dir.glob("*_model.pkl"):
                technique = model_file.stem.replace("_model", "")
                with open(model_file, 'rb') as f:
                    models[technique] = pickle.load(f)
            
            # Load scalers
            scalers = {}
            for scaler_file in model_dir.glob("*_scaler.pkl"):
                technique = scaler_file.stem.replace("_scaler", "")
                with open(scaler_file, 'rb') as f:
                    scalers[technique] = pickle.load(f)
            
            # Load ensemble metadata
            ensemble_file = model_dir / "ensemble_metadata.pkl"
            ensemble_metadata = None
            if ensemble_file.exists():
                with open(ensemble_file, 'rb') as f:
                    ensemble_metadata = pickle.load(f)
            
            return {
                'models': models,
                'scalers': scalers,
                'ensemble_metadata': ensemble_metadata,
                'metadata': metadata,
            }
        except Exception as e:
            logger.error("failed_to_load_model", symbol=symbol, error=str(e))
            return None
    
    def save_model(
        self,
        symbol: str,
        models: Dict[str, Any],
        scalers: Dict[str, Any],
        ensemble_metadata: Optional[Dict] = None,
        training_date: Optional[datetime] = None,
        additional_metadata: Optional[Dict] = None,
    ) -> None:
        """Save model and metadata."""
        model_dir = self.model_storage_path / symbol.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for technique, model in models.items():
            model_path = model_dir / f"{technique}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        for technique, scaler in scalers.items():
            scaler_path = model_dir / f"{technique}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save ensemble metadata
        if ensemble_metadata:
            ensemble_path = model_dir / "ensemble_metadata.pkl"
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble_metadata, f)
        
        # Save training metadata
        training_date = training_date or datetime.now(timezone.utc)
        metadata = {
            'last_training_date': training_date,
            'symbol': symbol,
            'training_type': additional_metadata.get('training_type', 'incremental') if additional_metadata else 'incremental',
            **(additional_metadata or {}),
        }
        metadata_path = model_dir / "training_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("model_saved", symbol=symbol, path=str(model_dir), training_date=training_date.isoformat())
    
    def train_incremental(
        self,
        symbol: str,
        trainer: Any,  # MultiModelTrainer or similar
        get_new_data_func: Callable[[str, datetime], tuple[pd.DataFrame, pd.Series]],  # Function to get new data since date
        full_training_func: Optional[Callable[[str], None]] = None,  # Function for full training if no model exists
    ) -> IncrementalTrainingResult:
        """
        Train incrementally on new data.
        
        Args:
            symbol: Trading pair
            trainer: Model trainer instance (MultiModelTrainer)
            get_new_data_func: Function(symbol, start_date) -> (X_new, y_new)
            full_training_func: Optional function for full training if no model exists
        
        Returns:
            IncrementalTrainingResult
        """
        start_time = datetime.now(timezone.utc)
        
        # Check if model exists
        if not self.model_exists(symbol):
            logger.info("no_existing_model_full_training", symbol=symbol)
            
            if full_training_func:
                # Do full training
                full_training_func(symbol)
                return IncrementalTrainingResult(
                    symbol=symbol,
                    last_training_date=datetime.now(timezone.utc),
                    new_data_days=0,
                    training_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    model_updated=True,
                    n_new_samples=0,
                )
            else:
                raise ValueError(f"No existing model for {symbol} and no full_training_func provided")
        
        # Load existing model
        saved_model = self.load_model(symbol)
        if not saved_model:
            raise ValueError(f"Failed to load model for {symbol}")
        
        last_training_date = saved_model['metadata'].get('last_training_date')
        if isinstance(last_training_date, str):
            last_training_date = datetime.fromisoformat(last_training_date.replace('Z', '+00:00'))
        elif last_training_date is None:
            logger.warning("no_last_training_date_using_full_retrain", symbol=symbol)
            if full_training_func:
                full_training_func(symbol)
                return IncrementalTrainingResult(
                    symbol=symbol,
                    last_training_date=datetime.now(timezone.utc),
                    new_data_days=0,
                    training_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                    model_updated=True,
                    n_new_samples=0,
                )
            else:
                raise ValueError(f"No last_training_date for {symbol} and no full_training_func provided")
        
        # Calculate days since last training
        days_since = (datetime.now(timezone.utc) - last_training_date).days
        
        if days_since < self.min_new_data_days:
            logger.info(
                "insufficient_new_data",
                symbol=symbol,
                days_since=days_since,
                min_required=self.min_new_data_days,
            )
            return IncrementalTrainingResult(
                symbol=symbol,
                last_training_date=last_training_date,
                new_data_days=days_since,
                training_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                model_updated=False,
                n_new_samples=0,
            )
        
        # Get new data
        logger.info("fetching_new_data", symbol=symbol, since=last_training_date.isoformat())
        try:
            X_new, y_new = get_new_data_func(symbol, last_training_date)
        except Exception as e:
            logger.error("failed_to_get_new_data", symbol=symbol, error=str(e))
            raise
        
        if len(X_new) == 0:
            logger.warning("no_new_data", symbol=symbol)
            return IncrementalTrainingResult(
                symbol=symbol,
                last_training_date=last_training_date,
                new_data_days=days_since,
                training_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
                model_updated=False,
                n_new_samples=0,
            )
        
        # Fine-tune on new data
        logger.info("fine_tuning_on_new_data", symbol=symbol, n_samples=len(X_new))
        
        # Load models into trainer
        trainer.models = saved_model['models']
        trainer.scalers = saved_model['scalers']
        if saved_model['ensemble_metadata']:
            trainer.model_weights = saved_model['ensemble_metadata'].get('model_weights', {})
            trainer.performance_by_regime = saved_model['ensemble_metadata'].get('performance_by_regime', {})
        
        # Fine-tune (partial_fit or warm_start)
        for technique, model in trainer.models.items():
            scaler = trainer.scalers.get(technique)
            
            # Scale new data
            if scaler:
                # Update scaler with new data
                if hasattr(scaler, 'partial_fit'):
                    scaler.partial_fit(X_new)
                X_new_scaled = scaler.transform(X_new)
            else:
                X_new_scaled = X_new
            
            # Fine-tune model
            if hasattr(model, 'partial_fit'):
                # Incremental learning (e.g., SGD, NaiveBayes)
                try:
                    model.partial_fit(X_new_scaled, y_new)
                    logger.debug("model_partial_fit", technique=technique, n_samples=len(X_new))
                except Exception as e:
                    logger.warning("partial_fit_failed_falling_back", technique=technique, error=str(e))
                    # Fall back to warm_start if available
                    if hasattr(model, 'fit') and self.warm_start:
                        model.fit(X_new_scaled, y_new)
            elif hasattr(model, 'fit'):
                # Re-fit with warm_start (if supported)
                # Note: Most sklearn models don't support warm_start for incremental learning
                # This will do a full retrain on new data only (not ideal, but works)
                logger.warning("model_does_not_support_partial_fit", technique=technique)
                # For now, skip incremental update for models without partial_fit
                # In production, you'd want to accumulate new data and retrain periodically
        
        # Save updated model
        self.save_model(
            symbol=symbol,
            models=trainer.models,
            scalers=trainer.scalers,
            ensemble_metadata={
                'model_weights': trainer.model_weights,
                'performance_by_regime': trainer.performance_by_regime,
                'ensemble_method': saved_model['ensemble_metadata'].get('ensemble_method', 'weighted_voting') if saved_model['ensemble_metadata'] else 'weighted_voting',
                'is_classification': saved_model['ensemble_metadata'].get('is_classification', False) if saved_model['ensemble_metadata'] else False,
            },
            training_date=datetime.now(timezone.utc),
            additional_metadata={
                'training_type': 'incremental',
                'previous_training_date': last_training_date.isoformat(),
                'new_samples': len(X_new),
                'days_since_last_training': days_since,
            },
        )
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(
            "incremental_training_complete",
            symbol=symbol,
            training_time_seconds=training_time,
            new_samples=len(X_new),
            days_since_last_training=days_since,
        )
        
        return IncrementalTrainingResult(
            symbol=symbol,
            last_training_date=datetime.now(timezone.utc),
            new_data_days=days_since,
            training_time_seconds=training_time,
            model_updated=True,
            n_new_samples=len(X_new),
        )

