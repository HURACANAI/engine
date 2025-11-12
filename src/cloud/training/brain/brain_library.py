"""Brain Library - Centralized storage for all Huracan data and metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

from ..database.pool import DatabaseConnectionPool

logger = structlog.get_logger(__name__)


class BrainLibrary:
    """
    Centralized storage for:
    - Liquidation data
    - Funding rates
    - Open interest
    - Sentiment scores
    - Feature importance rankings
    - Model comparisons
    - Model metrics
    - Model registry
    - Data quality logs
    """

    def __init__(
        self,
        dsn: str,
        use_pool: bool = True,
        pool_minconn: int = 2,
        pool_maxconn: int = 10,
    ) -> None:
        """
        Initialize Brain Library.
        
        Args:
            dsn: Database connection string
            use_pool: Whether to use connection pooling
            pool_minconn: Minimum pool connections
            pool_maxconn: Maximum pool connections
        """
        self._dsn = dsn
        self._use_pool = use_pool
        
        if use_pool:
            self._pool = DatabaseConnectionPool(
                dsn=dsn,
                minconn=pool_minconn,
                maxconn=pool_maxconn,
            )
        else:
            self._conn = None
        
        self._initialize_schema()
        logger.info("brain_library_initialized", use_pool=use_pool)

    def _initialize_schema(self) -> None:
        """Create all Brain Library tables if they don't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Liquidation data table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS liquidations (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        size_usd DECIMAL(20, 8) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        liquidation_type VARCHAR(20),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(timestamp, exchange, symbol, side, price)
                    )
                """)
                
                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_liquidations_timestamp 
                    ON liquidations(timestamp)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_liquidations_symbol 
                    ON liquidations(symbol, timestamp)
                """)
                
                # Funding rates table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS funding_rates (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        funding_rate DECIMAL(10, 8) NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(timestamp, exchange, symbol)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol 
                    ON funding_rates(symbol, timestamp)
                """)
                
                # Open interest table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS open_interest (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        exchange VARCHAR(20) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        open_interest DECIMAL(20, 8) NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(timestamp, exchange, symbol)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_open_interest_symbol 
                    ON open_interest(symbol, timestamp)
                """)
                
                # Sentiment scores table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_scores (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        sentiment_score DECIMAL(5, 4) NOT NULL,
                        source VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(timestamp, symbol, source)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_symbol 
                    ON sentiment_scores(symbol, timestamp)
                """)
                
                # Feature importance table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feature_importance (
                        id SERIAL PRIMARY KEY,
                        analysis_date DATE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        feature_name VARCHAR(100) NOT NULL,
                        importance_score DECIMAL(10, 6) NOT NULL,
                        rank INTEGER NOT NULL,
                        method VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(analysis_date, symbol, feature_name, method)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_feature_importance_symbol 
                    ON feature_importance(symbol, analysis_date DESC)
                """)
                
                # Model comparisons table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_comparisons (
                        id SERIAL PRIMARY KEY,
                        comparison_date DATE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        model_type VARCHAR(50) NOT NULL,
                        accuracy DECIMAL(10, 6),
                        sharpe_ratio DECIMAL(10, 6),
                        sortino_ratio DECIMAL(10, 6),
                        max_drawdown DECIMAL(10, 6),
                        profit_factor DECIMAL(10, 6),
                        composite_score DECIMAL(10, 6),
                        model_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(comparison_date, symbol, model_type)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_comparisons_symbol 
                    ON model_comparisons(symbol, comparison_date DESC)
                """)
                
                # Model registry table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_registry (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) UNIQUE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        model_type VARCHAR(50) NOT NULL,
                        version INTEGER NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        selected_date DATE NOT NULL,
                        composite_score DECIMAL(10, 6),
                        hyperparameters JSONB,
                        dataset_id VARCHAR(255),
                        feature_set JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_registry_symbol 
                    ON model_registry(symbol, is_active)
                """)
                
                # Model metrics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) NOT NULL,
                        evaluation_date DATE NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        sharpe_ratio DECIMAL(10, 6),
                        sortino_ratio DECIMAL(10, 6),
                        hit_ratio DECIMAL(10, 6),
                        profit_factor DECIMAL(10, 6),
                        max_drawdown DECIMAL(10, 6),
                        calmar_ratio DECIMAL(10, 6),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(model_id, evaluation_date, symbol)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_metrics_model 
                    ON model_metrics(model_id, evaluation_date DESC)
                """)
                
                # Data quality logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20),
                        issue_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        details TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_quality_timestamp 
                    ON data_quality_logs(timestamp DESC)
                """)
                
                # Model manifests table (for versioning)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_manifests (
                        id SERIAL PRIMARY KEY,
                        model_id VARCHAR(255) UNIQUE NOT NULL,
                        version INTEGER NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        hyperparameters JSONB NOT NULL,
                        dataset_id VARCHAR(255) NOT NULL,
                        feature_set JSONB NOT NULL,
                        training_metrics JSONB,
                        validation_metrics JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_manifests_symbol 
                    ON model_manifests(symbol, version DESC)
                """)
                
                # Rollback logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rollback_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        previous_version INTEGER NOT NULL,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Returns table (for storing raw and log returns)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS returns (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        raw_returns DECIMAL(20, 10),
                        log_returns DECIMAL(20, 10),
                        price DECIMAL(20, 8),
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(timestamp, symbol)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_returns_symbol 
                    ON returns(symbol, timestamp DESC)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_returns_timestamp 
                    ON returns(timestamp DESC)
                """)
                
                conn.commit()
                logger.info("brain_library_schema_initialized")

    def _get_connection(self):
        """Get database connection."""
        if self._use_pool:
            return self._pool.get_connection()
        else:
            import psycopg2
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self._dsn)
            return self._conn

    # Liquidation data methods
    def store_liquidation(
        self,
        timestamp: datetime,
        exchange: str,
        symbol: str,
        side: str,
        size_usd: float,
        price: float,
        liquidation_type: Optional[str] = None,
    ) -> int:
        """Store a liquidation event."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO liquidations 
                    (timestamp, exchange, symbol, side, size_usd, price, liquidation_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, exchange, symbol, side, price) DO NOTHING
                    RETURNING id
                """, (timestamp, exchange, symbol, side, size_usd, price, liquidation_type))
                result = cur.fetchone()
                conn.commit()
                if result:
                    return result[0]
                return 0

    def get_liquidations(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get liquidations for a symbol in a time range."""
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, exchange, symbol, side, size_usd, price, liquidation_type
                FROM liquidations
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            df = pl.read_database(query, conn, parameters=(symbol, start_time, end_time))
            return df

    # Funding rates methods
    def store_funding_rate(
        self,
        timestamp: datetime,
        exchange: str,
        symbol: str,
        funding_rate: float,
    ) -> int:
        """Store a funding rate."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO funding_rates (timestamp, exchange, symbol, funding_rate)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (timestamp, exchange, symbol) 
                    DO UPDATE SET funding_rate = EXCLUDED.funding_rate
                    RETURNING id
                """, (timestamp, exchange, symbol, funding_rate))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_funding_rates(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get funding rates for a symbol."""
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, exchange, symbol, funding_rate
                FROM funding_rates
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            df = pl.read_database(query, conn, parameters=(symbol, start_time, end_time))
            return df

    # Open interest methods
    def store_open_interest(
        self,
        timestamp: datetime,
        exchange: str,
        symbol: str,
        open_interest: float,
    ) -> int:
        """Store open interest data."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO open_interest (timestamp, exchange, symbol, open_interest)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (timestamp, exchange, symbol) 
                    DO UPDATE SET open_interest = EXCLUDED.open_interest
                    RETURNING id
                """, (timestamp, exchange, symbol, open_interest))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_open_interest(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get open interest for a symbol."""
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, exchange, symbol, open_interest
                FROM open_interest
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            df = pl.read_database(query, conn, parameters=(symbol, start_time, end_time))
            return df

    # Sentiment methods
    def store_sentiment(
        self,
        timestamp: datetime,
        symbol: str,
        sentiment_score: float,
        source: Optional[str] = None,
    ) -> int:
        """Store sentiment score."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sentiment_scores (timestamp, symbol, sentiment_score, source)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol, source) 
                    DO UPDATE SET sentiment_score = EXCLUDED.sentiment_score
                    RETURNING id
                """, (timestamp, symbol, sentiment_score, source))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_sentiment(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get sentiment scores for a symbol."""
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, symbol, sentiment_score, source
                FROM sentiment_scores
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            df = pl.read_database(query, conn, parameters=(symbol, start_time, end_time))
            return df

    # Feature importance methods
    def store_feature_importance(
        self,
        analysis_date: datetime,
        symbol: str,
        feature_rankings: List[Dict[str, Any]],
        method: str = "shap",
    ) -> None:
        """Store feature importance rankings."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for rank, feature_data in enumerate(feature_rankings, 1):
                    cur.execute("""
                        INSERT INTO feature_importance 
                        (analysis_date, symbol, feature_name, importance_score, rank, method)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (analysis_date, symbol, feature_name, method)
                        DO UPDATE SET 
                            importance_score = EXCLUDED.importance_score,
                            rank = EXCLUDED.rank
                    """, (
                        analysis_date.date(),
                        symbol,
                        feature_data["feature_name"],
                        feature_data["importance_score"],
                        rank,
                        method,
                    ))
                conn.commit()
                logger.info(
                    "feature_importance_stored",
                    symbol=symbol,
                    date=analysis_date.date(),
                    num_features=len(feature_rankings),
                )

    def get_top_features(
        self,
        symbol: str,
        top_n: int = 20,
        method: str = "shap",
    ) -> List[str]:
        """Get top N features for a symbol."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feature_name
                    FROM feature_importance
                    WHERE symbol = %s AND method = %s
                    ORDER BY analysis_date DESC, rank ASC
                    LIMIT %s
                """, (symbol, method, top_n))
                results = cur.fetchall()
                return [row[0] for row in results]

    # Model comparison methods
    def store_model_comparison(
        self,
        comparison_date: datetime,
        symbol: str,
        model_type: str,
        metrics: Dict[str, float],
        model_id: Optional[str] = None,
    ) -> int:
        """Store model comparison results."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Calculate composite score
                composite = (
                    0.4 * metrics.get("sharpe_ratio", 0.0) +
                    0.3 * metrics.get("profit_factor", 0.0) +
                    0.2 * (1.0 - abs(metrics.get("max_drawdown", 0.0))) +
                    0.1 * metrics.get("accuracy", 0.0)
                )
                
                cur.execute("""
                    INSERT INTO model_comparisons 
                    (comparison_date, symbol, model_type, accuracy, sharpe_ratio, 
                     sortino_ratio, max_drawdown, profit_factor, composite_score, model_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (comparison_date, symbol, model_type)
                    DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        profit_factor = EXCLUDED.profit_factor,
                        composite_score = EXCLUDED.composite_score,
                        model_id = EXCLUDED.model_id
                    RETURNING id
                """, (
                    comparison_date.date(),
                    symbol,
                    model_type,
                    metrics.get("accuracy"),
                    metrics.get("sharpe_ratio"),
                    metrics.get("sortino_ratio"),
                    metrics.get("max_drawdown"),
                    metrics.get("profit_factor"),
                    composite,
                    model_id,
                ))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_best_model(
        self,
        symbol: str,
        comparison_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get best model for a symbol based on composite score."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if comparison_date:
                    cur.execute("""
                        SELECT model_type, composite_score, model_id
                        FROM model_comparisons
                        WHERE symbol = %s AND comparison_date = %s
                        ORDER BY composite_score DESC
                        LIMIT 1
                    """, (symbol, comparison_date.date()))
                else:
                    cur.execute("""
                        SELECT model_type, composite_score, model_id
                        FROM model_comparisons
                        WHERE symbol = %s
                        ORDER BY comparison_date DESC, composite_score DESC
                        LIMIT 1
                    """, (symbol,))
                
                result = cur.fetchone()
                if result:
                    return {
                        "model_type": result[0],
                        "composite_score": float(result[1]),
                        "model_id": result[2],
                    }
                return None

    # Model registry methods
    def register_model(
        self,
        model_id: str,
        symbol: str,
        model_type: str,
        version: int,
        composite_score: float,
        hyperparameters: Dict[str, Any],
        dataset_id: str,
        feature_set: List[str],
    ) -> None:
        """Register a model in the registry."""
        import json
        import numpy as np
        
        def clean_for_json(obj: Any) -> Any:
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj if not (isinstance(item, float) and (np.isnan(item) or np.isinf(item)))]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                val = float(obj)
                # Replace NaN and Inf with None
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [clean_for_json(item) for item in obj.tolist()]
            elif obj is None:
                return None
            else:
                # Try to serialize, if it fails return string representation
                try:
                    json.dumps(obj, allow_nan=False)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Clean hyperparameters before JSON serialization
        cleaned_hyperparameters = clean_for_json(hyperparameters)
        cleaned_feature_set = clean_for_json(feature_set)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Deactivate previous versions
                cur.execute("""
                    UPDATE model_registry
                    SET is_active = FALSE
                    WHERE symbol = %s AND model_type = %s
                """, (symbol, model_type))
                
                # Register new model
                cur.execute("""
                    INSERT INTO model_registry 
                    (model_id, symbol, model_type, version, composite_score, 
                     hyperparameters, dataset_id, feature_set, selected_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id)
                    DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        model_type = EXCLUDED.model_type,
                        version = EXCLUDED.version,
                        is_active = EXCLUDED.is_active,
                        composite_score = EXCLUDED.composite_score,
                        hyperparameters = EXCLUDED.hyperparameters,
                        dataset_id = EXCLUDED.dataset_id,
                        feature_set = EXCLUDED.feature_set
                    RETURNING id
                """, (
                    model_id,
                    symbol,
                    model_type,
                    version,
                    composite_score,
                    json.dumps(cleaned_hyperparameters, allow_nan=False),
                    dataset_id,
                    json.dumps(cleaned_feature_set, allow_nan=False),
                    datetime.now().date(),
                ))
                conn.commit()
                logger.info("model_registered", model_id=model_id, symbol=symbol, model_type=model_type)

    def get_active_model(
        self,
        symbol: str,
        model_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get active model for a symbol."""
        import json
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if model_type:
                    cur.execute("""
                        SELECT model_id, model_type, version, composite_score, 
                               hyperparameters, dataset_id, feature_set
                        FROM model_registry
                        WHERE symbol = %s AND model_type = %s AND is_active = TRUE
                        ORDER BY selected_date DESC
                        LIMIT 1
                    """, (symbol, model_type))
                else:
                    cur.execute("""
                        SELECT model_id, model_type, version, composite_score, 
                               hyperparameters, dataset_id, feature_set
                        FROM model_registry
                        WHERE symbol = %s AND is_active = TRUE
                        ORDER BY composite_score DESC, selected_date DESC
                        LIMIT 1
                    """, (symbol,))
                
                result = cur.fetchone()
                if result:
                    return {
                        "model_id": result[0],
                        "model_type": result[1],
                        "version": result[2],
                        "composite_score": float(result[3]),
                        "hyperparameters": json.loads(result[4]) if result[4] else {},
                        "dataset_id": result[5],
                        "feature_set": json.loads(result[6]) if result[6] else [],
                    }
                return None

    # Model metrics methods
    def store_model_metrics(
        self,
        model_id: str,
        evaluation_date: datetime,
        symbol: str,
        metrics: Dict[str, float],
    ) -> int:
        """Store model evaluation metrics."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_metrics 
                    (model_id, evaluation_date, symbol, sharpe_ratio, sortino_ratio, 
                     hit_ratio, profit_factor, max_drawdown, calmar_ratio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id, evaluation_date, symbol)
                    DO UPDATE SET
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        hit_ratio = EXCLUDED.hit_ratio,
                        profit_factor = EXCLUDED.profit_factor,
                        max_drawdown = EXCLUDED.max_drawdown,
                        calmar_ratio = EXCLUDED.calmar_ratio
                    RETURNING id
                """, (
                    model_id,
                    evaluation_date.date(),
                    symbol,
                    metrics.get("sharpe_ratio"),
                    metrics.get("sortino_ratio"),
                    metrics.get("hit_ratio"),
                    metrics.get("profit_factor"),
                    metrics.get("max_drawdown"),
                    metrics.get("calmar_ratio"),
                ))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_model_metrics(
        self,
        model_id: str,
        symbol: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get metrics for a model."""
        with self._get_connection() as conn:
            if symbol:
                query = """
                    SELECT evaluation_date, symbol, sharpe_ratio, sortino_ratio, 
                           hit_ratio, profit_factor, max_drawdown, calmar_ratio
                    FROM model_metrics
                    WHERE model_id = %s AND symbol = %s
                    ORDER BY evaluation_date DESC
                """
                df = pl.read_database(query, conn, parameters=(model_id, symbol))
            else:
                query = """
                    SELECT evaluation_date, symbol, sharpe_ratio, sortino_ratio, 
                           hit_ratio, profit_factor, max_drawdown, calmar_ratio
                    FROM model_metrics
                    WHERE model_id = %s
                    ORDER BY evaluation_date DESC
                """
                df = pl.read_database(query, conn, parameters=(model_id,))
            return df

    # Data quality methods
    def log_data_quality_issue(
        self,
        timestamp: datetime,
        symbol: Optional[str],
        issue_type: str,
        severity: str,
        details: str,
    ) -> int:
        """Log a data quality issue."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_quality_logs 
                    (timestamp, symbol, issue_type, severity, details)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (timestamp, symbol, issue_type, severity, details))
                result = cur.fetchone()
                conn.commit()
                logger.warning(
                    "data_quality_issue_logged",
                    symbol=symbol,
                    issue_type=issue_type,
                    severity=severity,
                )
                return result[0] if result else 0

    def get_data_quality_summary(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get data quality summary for a time period."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_issues,
                        COUNT(DISTINCT symbol) as affected_symbols,
                        COUNT(*) FILTER (WHERE severity = 'critical') as critical_issues,
                        COUNT(*) FILTER (WHERE severity = 'warning') as warnings,
                        COUNT(*) FILTER (WHERE resolved = TRUE) as resolved
                    FROM data_quality_logs
                    WHERE timestamp >= %s AND timestamp <= %s
                """, (start_time, end_time))
                result = cur.fetchone()
                return {
                    "total_issues": result[0] or 0,
                    "affected_symbols": result[1] or 0,
                    "critical_issues": result[2] or 0,
                    "warnings": result[3] or 0,
                    "resolved": result[4] or 0,
                }

    # Model manifest methods (for versioning)
    def store_model_manifest(
        self,
        model_id: str,
        version: int,
        symbol: str,
        hyperparameters: Dict[str, Any],
        dataset_id: str,
        feature_set: List[str],
        training_metrics: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store model manifest for versioning."""
        import json
        import numpy as np
        
        def clean_for_json(obj: Any) -> Any:
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj if not (isinstance(item, float) and (np.isnan(item) or np.isinf(item)))]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                val = float(obj)
                # Replace NaN and Inf with None
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [clean_for_json(item) for item in obj.tolist()]
            elif obj is None:
                return None
            else:
                # Try to serialize, if it fails return string representation
                try:
                    json.dumps(obj, allow_nan=False)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Clean all data before JSON serialization
        cleaned_hyperparameters = clean_for_json(hyperparameters)
        cleaned_feature_set = clean_for_json(feature_set)
        cleaned_training_metrics = clean_for_json(training_metrics) if training_metrics else None
        cleaned_validation_metrics = clean_for_json(validation_metrics) if validation_metrics else None
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO model_manifests 
                    (model_id, version, symbol, hyperparameters, dataset_id, 
                     feature_set, training_metrics, validation_metrics)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id)
                    DO UPDATE SET
                        version = EXCLUDED.version,
                        hyperparameters = EXCLUDED.hyperparameters,
                        dataset_id = EXCLUDED.dataset_id,
                        feature_set = EXCLUDED.feature_set,
                        training_metrics = EXCLUDED.training_metrics,
                        validation_metrics = EXCLUDED.validation_metrics
                    RETURNING id
                """, (
                    model_id,
                    version,
                    symbol,
                    json.dumps(cleaned_hyperparameters, allow_nan=False),
                    dataset_id,
                    json.dumps(cleaned_feature_set, allow_nan=False),
                    json.dumps(cleaned_training_metrics, allow_nan=False) if cleaned_training_metrics else None,
                    json.dumps(cleaned_validation_metrics, allow_nan=False) if cleaned_validation_metrics else None,
                ))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else 0

    def get_model_manifest(
        self,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get model manifest."""
        import json
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT version, symbol, hyperparameters, dataset_id, 
                           feature_set, training_metrics, validation_metrics
                    FROM model_manifests
                    WHERE model_id = %s
                """, (model_id,))
                result = cur.fetchone()
                if result:
                    return {
                        "version": result[0],
                        "symbol": result[1],
                        "hyperparameters": json.loads(result[2]) if result[2] else {},
                        "dataset_id": result[3],
                        "feature_set": json.loads(result[4]) if result[4] else [],
                        "training_metrics": json.loads(result[5]) if result[5] else {},
                        "validation_metrics": json.loads(result[6]) if result[6] else {},
                    }
                return None

    def log_rollback(
        self,
        model_id: str,
        previous_version: int,
        reason: str,
    ) -> int:
        """Log a model rollback event."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rollback_logs 
                    (timestamp, model_id, previous_version, reason)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (datetime.now(), model_id, previous_version, reason))
                result = cur.fetchone()
                conn.commit()
                logger.warning(
                    "model_rollback_logged",
                    model_id=model_id,
                    previous_version=previous_version,
                    reason=reason,
                )
                return result[0] if result else 0

    # Returns methods
    def store_returns(
        self,
        symbol: str,
        timestamps: list[datetime] | pd.Series,
        raw_returns: list[float] | pd.Series | np.ndarray,
        log_returns: list[float] | pd.Series | np.ndarray,
        prices: Optional[list[float] | pd.Series | np.ndarray] = None,
    ) -> int:
        """
        Store returns data in Brain Library.
        
        Args:
            symbol: Trading symbol
            timestamps: List of timestamps
            raw_returns: Raw returns (percent change)
            log_returns: Log returns
            prices: Optional price values
            
        Returns:
            Number of rows inserted
        """
        if len(timestamps) != len(raw_returns) or len(timestamps) != len(log_returns):
            raise ValueError("timestamps, raw_returns, and log_returns must have same length")
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                inserted = 0
                for i, ts in enumerate(timestamps):
                    raw_ret = float(raw_returns[i]) if not pd.isna(raw_returns[i]) else None
                    log_ret = float(log_returns[i]) if not pd.isna(log_returns[i]) else None
                    price = float(prices[i]) if prices is not None and i < len(prices) and not pd.isna(prices[i]) else None
                    
                    cur.execute("""
                        INSERT INTO returns 
                        (timestamp, symbol, raw_returns, log_returns, price)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, symbol) 
                        DO UPDATE SET 
                            raw_returns = EXCLUDED.raw_returns,
                            log_returns = EXCLUDED.log_returns,
                            price = EXCLUDED.price
                    """, (ts, symbol, raw_ret, log_ret, price))
                    inserted += 1
                
                conn.commit()
                logger.info(
                    "returns_stored",
                    symbol=symbol,
                    rows=inserted
                )
                return inserted

    def get_returns(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """
        Get returns data for a symbol in a time range.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with returns data
        """
        with self._get_connection() as conn:
            query = """
                SELECT timestamp, symbol, raw_returns, log_returns, price
                FROM returns
                WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
                ORDER BY timestamp
            """
            df = pl.read_database(query, conn, parameters=(symbol, start_time, end_time))
            return df

