"""
Distributed Model Registry for Scalable Architecture

Tracks models per coin with metadata (version, Sharpe, regime, retrain date).
Supports distributed retraining with Ray/Dask on RunPod.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not available, model registry will use file-based storage")


class ModelStatus(Enum):
    """Model status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Model metadata."""
    coin: str
    version: int
    sharpe_ratio: float
    win_rate: float
    regime: str
    last_retrain_date: datetime
    is_active: bool
    status: ModelStatus
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ModelRegistry:
    """
    Distributed model registry for tracking models per coin.
    
    Features:
    - Version tracking
    - Performance metrics (Sharpe, win rate, etc.)
    - Regime-specific models
    - Best-N models active per regime
    - Distributed retraining support
    """
    
    def __init__(
        self,
        storage_path: str = "/models/trained",
        metadata_db: Optional[str] = None,
    ):
        """
        Initialize model registry.
        
        Args:
            storage_path: Path to model storage
            metadata_db: PostgreSQL connection string (optional)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_db = metadata_db
        self.use_postgres = metadata_db is not None and POSTGRES_AVAILABLE
        
        # In-memory cache
        self.metadata_cache: Dict[str, Dict[int, ModelMetadata]] = {}  # coin -> version -> metadata
        
        # Initialize database if using Postgres
        if self.use_postgres:
            self._init_database()
        
        logger.info(
            "model_registry_initialized",
            storage_path=str(self.storage_path),
            use_postgres=self.use_postgres,
        )
    
    def _init_database(self) -> None:
        """Initialize PostgreSQL database schema."""
        if not self.use_postgres:
            return
        
        try:
            conn = psycopg2.connect(self.metadata_db)
            cur = conn.cursor()
            
            # Create table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    coin VARCHAR(20) NOT NULL,
                    version INTEGER NOT NULL,
                    sharpe_ratio FLOAT,
                    win_rate FLOAT,
                    regime VARCHAR(20),
                    last_retrain_date TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE,
                    status VARCHAR(20) DEFAULT 'inactive',
                    performance_metrics JSONB,
                    model_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (coin, version)
                );
            """)
            
            # Create index on coin and is_active
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_registry_coin_active
                ON model_registry(coin, is_active);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info("model_registry_database_initialized")
        except Exception as e:
            logger.error("model_registry_database_init_failed", error=str(e))
            self.use_postgres = False
    
    def register_model(
        self,
        coin: str,
        version: int,
        sharpe_ratio: float,
        win_rate: float,
        regime: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        is_active: bool = False,
    ) -> ModelMetadata:
        """
        Register a model in the registry.
        
        Args:
            coin: Coin symbol
            version: Model version
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate
            regime: Market regime
            performance_metrics: Additional performance metrics
            model_path: Path to model file
            is_active: Whether model is active
        
        Returns:
            ModelMetadata instance
        """
        # Create model metadata
        metadata = ModelMetadata(
            coin=coin,
            version=version,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            regime=regime,
            last_retrain_date=datetime.now(),
            is_active=is_active,
            status=ModelStatus.ACTIVE if is_active else ModelStatus.INACTIVE,
            performance_metrics=performance_metrics or {},
            model_path=model_path,
        )
        
        # Store in cache
        if coin not in self.metadata_cache:
            self.metadata_cache[coin] = {}
        self.metadata_cache[coin][version] = metadata
        
        # Store in database if using Postgres
        if self.use_postgres:
            self._store_metadata_db(metadata)
        else:
            # Store in file
            self._store_metadata_file(metadata)
        
        logger.info(
            "model_registered",
            coin=coin,
            version=version,
            sharpe=sharpe_ratio,
            win_rate=win_rate,
            regime=regime,
            is_active=is_active,
        )
        
        return metadata
    
    def _store_metadata_db(self, metadata: ModelMetadata) -> None:
        """Store metadata in PostgreSQL database."""
        if not self.use_postgres:
            return
        
        try:
            conn = psycopg2.connect(self.metadata_db)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO model_registry (
                    coin, version, sharpe_ratio, win_rate, regime,
                    last_retrain_date, is_active, status, performance_metrics,
                    model_path, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (coin, version) DO UPDATE SET
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    win_rate = EXCLUDED.win_rate,
                    regime = EXCLUDED.regime,
                    last_retrain_date = EXCLUDED.last_retrain_date,
                    is_active = EXCLUDED.is_active,
                    status = EXCLUDED.status,
                    performance_metrics = EXCLUDED.performance_metrics,
                    model_path = EXCLUDED.model_path,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                metadata.coin,
                metadata.version,
                metadata.sharpe_ratio,
                metadata.win_rate,
                metadata.regime,
                metadata.last_retrain_date,
                metadata.is_active,
                metadata.status.value,
                json.dumps(metadata.performance_metrics),
                metadata.model_path,
                metadata.created_at,
                metadata.updated_at,
            ))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error("model_metadata_store_failed", error=str(e))
    
    def _store_metadata_file(self, metadata: ModelMetadata) -> None:
        """Store metadata in file."""
        coin_dir = self.storage_path / metadata.coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = coin_dir / f"metadata_v{metadata.version}.json"
        
        with open(metadata_file, "w") as f:
            json.dump({
                "coin": metadata.coin,
                "version": metadata.version,
                "sharpe_ratio": metadata.sharpe_ratio,
                "win_rate": metadata.win_rate,
                "regime": metadata.regime,
                "last_retrain_date": metadata.last_retrain_date.isoformat(),
                "is_active": metadata.is_active,
                "status": metadata.status.value,
                "performance_metrics": metadata.performance_metrics,
                "model_path": metadata.model_path,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
            }, f, indent=2)
    
    def get_model_metadata(
        self,
        coin: str,
        version: Optional[int] = None,
    ) -> Optional[ModelMetadata]:
        """
        Get model metadata.
        
        Args:
            coin: Coin symbol
            version: Model version (None for latest)
        
        Returns:
            ModelMetadata instance or None
        """
        # Check cache first
        if coin in self.metadata_cache:
            if version is not None:
                return self.metadata_cache[coin].get(version)
            else:
                # Get latest version
                if self.metadata_cache[coin]:
                    return max(
                        self.metadata_cache[coin].values(),
                        key=lambda m: m.version
                    )
        
        # Load from database or file
        if self.use_postgres:
            return self._load_metadata_db(coin, version)
        else:
            return self._load_metadata_file(coin, version)
    
    def _load_metadata_db(
        self,
        coin: str,
        version: Optional[int] = None,
    ) -> Optional[ModelMetadata]:
        """Load metadata from PostgreSQL database."""
        if not self.use_postgres:
            return None
        
        try:
            conn = psycopg2.connect(self.metadata_db)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if version is not None:
                cur.execute("""
                    SELECT * FROM model_registry
                    WHERE coin = %s AND version = %s
                    ORDER BY version DESC
                    LIMIT 1;
                """, (coin, version))
            else:
                cur.execute("""
                    SELECT * FROM model_registry
                    WHERE coin = %s
                    ORDER BY version DESC
                    LIMIT 1;
                """, (coin,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                return self._row_to_metadata(dict(row))
        except Exception as e:
            logger.error("model_metadata_load_failed", error=str(e))
        
        return None
    
    def _load_metadata_file(
        self,
        coin: str,
        version: Optional[int] = None,
    ) -> Optional[ModelMetadata]:
        """Load metadata from file."""
        coin_dir = self.storage_path / coin
        
        if not coin_dir.exists():
            return None
        
        # Find metadata files
        metadata_files = list(coin_dir.glob("metadata_v*.json"))
        
        if not metadata_files:
            return None
        
        # Get latest version if not specified
        if version is None:
            metadata_files.sort(key=lambda f: int(f.stem.split("_v")[1]))
            metadata_file = metadata_files[-1]
        else:
            metadata_file = coin_dir / f"metadata_v{version}.json"
            if not metadata_file.exists():
                return None
        
        # Load metadata
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        return self._dict_to_metadata(data)
    
    def _row_to_metadata(self, row: Dict[str, Any]) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            coin=row["coin"],
            version=row["version"],
            sharpe_ratio=row["sharpe_ratio"],
            win_rate=row["win_rate"],
            regime=row["regime"],
            last_retrain_date=row["last_retrain_date"],
            is_active=row["is_active"],
            status=ModelStatus(row["status"]),
            performance_metrics=row.get("performance_metrics", {}),
            model_path=row.get("model_path"),
            created_at=row.get("created_at", datetime.now()),
            updated_at=row.get("updated_at", datetime.now()),
        )
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> ModelMetadata:
        """Convert dictionary to ModelMetadata."""
        return ModelMetadata(
            coin=data["coin"],
            version=data["version"],
            sharpe_ratio=data["sharpe_ratio"],
            win_rate=data["win_rate"],
            regime=data["regime"],
            last_retrain_date=datetime.fromisoformat(data["last_retrain_date"]),
            is_active=data["is_active"],
            status=ModelStatus(data["status"]),
            performance_metrics=data.get("performance_metrics", {}),
            model_path=data.get("model_path"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
        )
    
    def get_active_models(
        self,
        coin: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 10,
    ) -> List[ModelMetadata]:
        """
        Get active models.
        
        Args:
            coin: Coin symbol (None for all coins)
            regime: Market regime (None for all regimes)
            limit: Maximum number of models to return
        
        Returns:
            List of ModelMetadata instances
        """
        active_models = []
        
        # Get from cache
        for coin_symbol, versions in self.metadata_cache.items():
            if coin and coin_symbol != coin:
                continue
            
            for version, metadata in versions.items():
                if not metadata.is_active:
                    continue
                if regime and metadata.regime != regime:
                    continue
                active_models.append(metadata)
        
        # Sort by Sharpe ratio (descending)
        active_models.sort(key=lambda m: m.sharpe_ratio, reverse=True)
        
        return active_models[:limit]
    
    def set_model_active(
        self,
        coin: str,
        version: int,
        is_active: bool = True,
    ) -> bool:
        """
        Set model active status.
        
        Args:
            coin: Coin symbol
            version: Model version
            is_active: Whether model is active
        
        Returns:
            True if successful
        """
        metadata = self.get_model_metadata(coin, version)
        if not metadata:
            return False
        
        metadata.is_active = is_active
        metadata.status = ModelStatus.ACTIVE if is_active else ModelStatus.INACTIVE
        metadata.updated_at = datetime.now()
        
        # Update cache
        if coin not in self.metadata_cache:
            self.metadata_cache[coin] = {}
        self.metadata_cache[coin][version] = metadata
        
        # Update database or file
        if self.use_postgres:
            self._store_metadata_db(metadata)
        else:
            self._store_metadata_file(metadata)
        
        logger.info(
            "model_status_updated",
            coin=coin,
            version=version,
            is_active=is_active,
        )
        
        return True
    
    def get_model_path(self, coin: str, version: Optional[int] = None) -> Optional[str]:
        """
        Get model file path.
        
        Args:
            coin: Coin symbol
            version: Model version (None for latest)
        
        Returns:
            Model file path or None
        """
        metadata = self.get_model_metadata(coin, version)
        if not metadata:
            return None
        
        if metadata.model_path:
            return metadata.model_path
        
        # Generate path from storage path
        if version is None:
            version = metadata.version
        
        model_file = self.storage_path / coin / f"model_v{version}.pkl"
        if model_file.exists():
            return str(model_file)
        
        return None

