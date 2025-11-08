"""
Comprehensive Health Check System

Validates all critical components before engine startup.
If any critical check fails, the engine shuts down gracefully.

Author: Huracan Engine Team
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class CheckStatus(Enum):
    """Status of a health check"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    critical: bool = True
    duration_ms: float = 0.0


@dataclass
class HealthCheckReport:
    """Complete health check report"""
    checks: List[HealthCheckResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    critical_failures: int = 0
    total_duration_ms: float = 0.0
    overall_status: CheckStatus = CheckStatus.PASSED

    def add_check(self, result: HealthCheckResult) -> None:
        """Add a check result to the report"""
        self.checks.append(result)
        self.total_checks += 1
        
        if result.status == CheckStatus.PASSED:
            self.passed_checks += 1
        elif result.status == CheckStatus.FAILED:
            self.failed_checks += 1
            if result.critical:
                self.critical_failures += 1
        elif result.status == CheckStatus.WARNING:
            self.warning_checks += 1
        elif result.status == CheckStatus.SKIPPED:
            self.skipped_checks += 1
        
        # Update overall status
        if self.critical_failures > 0:
            self.overall_status = CheckStatus.FAILED
        elif self.failed_checks > 0 and self.overall_status == CheckStatus.PASSED:
            self.overall_status = CheckStatus.WARNING
        elif self.warning_checks > 0 and self.overall_status == CheckStatus.PASSED:
            self.overall_status = CheckStatus.WARNING

    def is_healthy(self) -> bool:
        """Check if system is healthy (no critical failures)"""
        return self.critical_failures == 0

    def get_summary(self) -> str:
        """Get human-readable summary"""
        status_emoji = {
            CheckStatus.PASSED: "✅",
            CheckStatus.FAILED: "❌",
            CheckStatus.WARNING: "⚠️",
            CheckStatus.SKIPPED: "⏭️"
        }
        
        emoji = status_emoji.get(self.overall_status, "❓")
        return (
            f"{emoji} Health Check Summary: "
            f"{self.passed_checks}/{self.total_checks} passed, "
            f"{self.failed_checks} failed ({self.critical_failures} critical), "
            f"{self.warning_checks} warnings, "
            f"{self.skipped_checks} skipped"
        )


class HealthChecker:
    """
    Comprehensive health checker for all engine components.
    
    Validates:
    - Configuration and settings
    - Database connection
    - Exchange client
    - Dropbox connection
    - Data availability (coin data)
    - Alpha engines
    - AI Council
    - Model Factory
    - API keys
    - File system permissions
    - And more...
    """

    def __init__(self, settings: Optional[Any] = None):
        """
        Initialize health checker.
        
        Args:
            settings: EngineSettings instance (optional, will load if not provided)
        """
        self.settings = settings
        self.report = HealthCheckReport()
        self.start_time = time.time()

    def run_all_checks(self) -> HealthCheckReport:
        """
        Run all health checks.
        
        Returns:
            HealthCheckReport with all check results
        """
        logger.info("health_check_starting")
        
        # Load settings if not provided
        if self.settings is None:
            try:
                from ..config.settings import EngineSettings
                self.settings = EngineSettings.load()
            except Exception as e:
                self.report.add_check(HealthCheckResult(
                    name="settings_load",
                    status=CheckStatus.FAILED,
                    message=f"Failed to load settings: {e}",
                    critical=True
                ))
                return self.report

        # Run all checks
        self._check_settings()
        self._check_file_system()
        self._check_database()
        self._check_api_keys()
        self._check_exchange_client()
        self._check_dropbox()
        self._check_data_availability()
        self._check_alpha_engines()
        self._check_model_factory()
        self._check_ai_council()
        self._check_feature_recipe()
        self._check_ray_cluster()
        self._check_notifications()

        # Calculate total duration
        self.report.total_duration_ms = (time.time() - self.start_time) * 1000
        
        logger.info(
            "health_check_complete",
            total_checks=self.report.total_checks,
            passed=self.report.passed_checks,
            failed=self.report.failed_checks,
            critical_failures=self.report.critical_failures,
            duration_ms=self.report.total_duration_ms,
            healthy=self.report.is_healthy()
        )

        return self.report

    def _check_settings(self) -> None:
        """Check if settings are loaded correctly"""
        start = time.time()
        try:
            if self.settings is None:
                raise ValueError("Settings not loaded")
            
            # Check critical settings exist (database might be optional)
            if not hasattr(self.settings, 'training'):
                raise ValueError("Training settings missing")
            if not hasattr(self.settings, 'exchange'):
                raise ValueError("Exchange settings missing")
            # Database is optional - don't fail if it doesn't exist
            
            self.report.add_check(HealthCheckResult(
                name="settings",
                status=CheckStatus.PASSED,
                message="Settings loaded successfully",
                details={
                    "environment": getattr(self.settings, 'environment', 'unknown'),
                    "mode": getattr(self.settings, 'mode', 'unknown')
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="settings",
                status=CheckStatus.FAILED,
                message=f"Settings check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_file_system(self) -> None:
        """Check file system permissions and required directories"""
        start = time.time()
        try:
            required_dirs = [
                Path("logs"),
                Path("models"),
                Path("data/candles"),
                Path("data/cache"),
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        missing_dirs.append(f"{dir_path}: {e}")
            
            if missing_dirs:
                raise ValueError(f"Cannot create required directories: {missing_dirs}")
            
            # Check write permissions
            test_file = Path("logs/.health_check_test")
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                raise ValueError(f"Cannot write to logs directory: {e}")
            
            self.report.add_check(HealthCheckResult(
                name="file_system",
                status=CheckStatus.PASSED,
                message="File system check passed",
                details={"checked_dirs": len(required_dirs)},
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="file_system",
                status=CheckStatus.FAILED,
                message=f"File system check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_database(self) -> None:
        """Check database connection"""
        start = time.time()
        try:
            # Check if database settings exist
            if not hasattr(self.settings, 'database'):
                self.report.add_check(HealthCheckResult(
                    name="database",
                    status=CheckStatus.SKIPPED,
                    message="Database settings not configured",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            if not self.settings.database.enabled:
                self.report.add_check(HealthCheckResult(
                    name="database",
                    status=CheckStatus.SKIPPED,
                    message="Database is disabled in settings",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            from sqlalchemy import create_engine, text
            from ..config.settings import EngineSettings
            
            dsn = self.settings.database.dsn
            if not dsn:
                raise ValueError("Database DSN not configured")
            
            engine = create_engine(dsn)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.report.add_check(HealthCheckResult(
                name="database",
                status=CheckStatus.PASSED,
                message="Database connection successful",
                details={"dsn": dsn.split("@")[1] if "@" in dsn else "configured"},  # Hide credentials
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="database",
                status=CheckStatus.FAILED,
                message=f"Database connection failed: {e}",
                critical=True,  # Database is critical for production
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_api_keys(self) -> None:
        """Check if required API keys are present"""
        start = time.time()
        try:
            import os
            from ..config.settings import EngineSettings
            
            required_keys = {
                "exchange": ["BINANCE_API_KEY", "BINANCE_SECRET_KEY"],  # Example
            }
            
            missing_keys = []
            available_keys = []
            
            # Check exchange API keys
            exchange_creds = getattr(self.settings.exchange, 'credentials', None)
            if not exchange_creds:
                missing_keys.append("Exchange credentials not configured")
            else:
                # Handle both dict and object-like credentials
                if isinstance(exchange_creds, dict):
                    creds_dict = exchange_creds
                else:
                    # Convert to dict if it's an object
                    creds_dict = exchange_creds.__dict__ if hasattr(exchange_creds, '__dict__') else {}
                
                if not creds_dict:
                    missing_keys.append("Exchange credentials empty")
                else:
                    for exchange_id, creds in creds_dict.items():
                        # Handle both dict and object-like creds
                        if isinstance(creds, dict):
                            api_key = creds.get('api_key')
                            secret = creds.get('secret')
                        else:
                            api_key = getattr(creds, 'api_key', None)
                            secret = getattr(creds, 'secret', None)
                        
                        if not api_key or not secret:
                            missing_keys.append(f"{exchange_id} credentials incomplete")
                        else:
                            available_keys.append(exchange_id)
            
            # Check optional API keys (for AI features)
            optional_keys = {
                "ANTHROPIC_KEY": "AI Council / Strategy Translator",
                "OPENAI_KEY": "AI Council / Strategy Translator",
                "DROPBOX_ACCESS_TOKEN": "Dropbox sync",
            }
            
            optional_missing = []
            for key_name, purpose in optional_keys.items():
                env_value = os.getenv(key_name)
                if key_name == "DROPBOX_ACCESS_TOKEN":
                    # Check both env var and settings
                    dropbox_token = env_value or (getattr(self.settings, 'dropbox', None) and getattr(self.settings.dropbox, 'access_token', None))
                    if not dropbox_token:
                        optional_missing.append(f"{key_name} ({purpose})")
                elif not env_value:
                    optional_missing.append(f"{key_name} ({purpose})")
            
            if missing_keys:
                self.report.add_check(HealthCheckResult(
                    name="api_keys",
                    status=CheckStatus.FAILED,
                    message=f"Missing critical API keys: {', '.join(missing_keys)}",
                    details={
                        "missing": missing_keys,
                        "available_exchanges": available_keys
                    },
                    critical=True,
                    duration_ms=(time.time() - start) * 1000
                ))
            else:
                message = f"All critical API keys present ({len(available_keys)} exchanges)"
                if optional_missing:
                    message += f". Optional keys missing: {', '.join(optional_missing)}"
                
                self.report.add_check(HealthCheckResult(
                    name="api_keys",
                    status=CheckStatus.PASSED if not optional_missing else CheckStatus.WARNING,
                    message=message,
                    details={
                        "available_exchanges": available_keys,
                        "optional_missing": optional_missing
                    },
                    duration_ms=(time.time() - start) * 1000
                ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="api_keys",
                status=CheckStatus.FAILED,
                message=f"API keys check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_exchange_client(self) -> None:
        """Check exchange client connectivity"""
        start = time.time()
        try:
            from .exchange import ExchangeClient
            from ..config.settings import EngineSettings
            
            exchange_id = "binance"  # Default exchange
            exchange_creds = getattr(self.settings.exchange, 'credentials', {}).get(exchange_id, {})
            
            if not exchange_creds.get('api_key'):
                self.report.add_check(HealthCheckResult(
                    name="exchange_client",
                    status=CheckStatus.SKIPPED,
                    message=f"Exchange credentials not configured for {exchange_id}",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            client = ExchangeClient(
                exchange_id=exchange_id,
                api_key=exchange_creds.get('api_key'),
                api_secret=exchange_creds.get('secret'),
                testnet=getattr(self.settings.exchange, 'testnet', False)
            )
            
            # Test connection by fetching ticker
            ticker = client.get_ticker("BTC/USDT")
            if not ticker:
                raise ValueError("Failed to fetch ticker from exchange")
            
            self.report.add_check(HealthCheckResult(
                name="exchange_client",
                status=CheckStatus.PASSED,
                message=f"Exchange client connected to {exchange_id}",
                details={
                    "exchange": exchange_id,
                    "test_ticker": "BTC/USDT"
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="exchange_client",
                status=CheckStatus.FAILED,
                message=f"Exchange client check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_dropbox(self) -> None:
        """Check Dropbox connection"""
        start = time.time()
        try:
            # Check if dropbox settings exist
            if not hasattr(self.settings, 'dropbox'):
                self.report.add_check(HealthCheckResult(
                    name="dropbox",
                    status=CheckStatus.SKIPPED,
                    message="Dropbox settings not configured",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            if not getattr(self.settings.dropbox, 'enabled', False):
                self.report.add_check(HealthCheckResult(
                    name="dropbox",
                    status=CheckStatus.SKIPPED,
                    message="Dropbox is disabled in settings",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            from ..integrations.dropbox_sync import DropboxSync
            
            access_token = getattr(self.settings.dropbox, 'access_token', None)
            if not access_token:
                self.report.add_check(HealthCheckResult(
                    name="dropbox",
                    status=CheckStatus.WARNING,
                    message="Dropbox enabled but access token not configured",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Initialize and test connection
            dropbox_sync = DropboxSync(
                access_token=access_token,
                app_folder=getattr(self.settings.dropbox, 'app_folder', 'Runpodhuracan'),
                enabled=True
            )
            
            # Test by getting account info
            account_info = dropbox_sync._dbx.users_get_current_account()
            
            self.report.add_check(HealthCheckResult(
                name="dropbox",
                status=CheckStatus.PASSED,
                message="Dropbox connection successful",
                details={
                    "account_email": account_info.email,
                    "app_folder": getattr(self.settings.dropbox, 'app_folder', 'Runpodhuracan')
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="dropbox",
                status=CheckStatus.FAILED,
                message=f"Dropbox connection failed: {e}",
                critical=False,  # Dropbox is optional
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_data_availability(self) -> None:
        """Check if coin data is available"""
        start = time.time()
        try:
            import polars as pl
            
            # Check if data directory exists and has files
            data_dir = Path("data/candles")
            if not data_dir.exists():
                self.report.add_check(HealthCheckResult(
                    name="data_availability",
                    status=CheckStatus.WARNING,
                    message="Data directory does not exist",
                    details={"data_dir": str(data_dir)},
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Count parquet files
            parquet_files = list(data_dir.glob("*.parquet"))
            if not parquet_files:
                self.report.add_check(HealthCheckResult(
                    name="data_availability",
                    status=CheckStatus.WARNING,
                    message="No coin data files found",
                    details={"data_dir": str(data_dir), "files_found": 0},
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Try to load a sample file directly with polars
            import polars as pl
            sample_file = parquet_files[0]
            
            # Extract symbol from filename (e.g., "BTCUSDT_1m.parquet" -> "BTCUSDT")
            symbol = sample_file.stem.split("_")[0]
            
            try:
                # Read parquet file directly
                data = pl.read_parquet(sample_file)
                if data.is_empty():
                    raise ValueError(f"Data file {sample_file.name} is empty")
                
                row_count = len(data)
                
                self.report.add_check(HealthCheckResult(
                    name="data_availability",
                    status=CheckStatus.PASSED,
                    message=f"Coin data available ({len(parquet_files)} files)",
                    details={
                        "files_found": len(parquet_files),
                        "sample_symbol": symbol,
                        "sample_rows": row_count,
                        "data_dir": str(data_dir)
                    },
                    duration_ms=(time.time() - start) * 1000
                ))
            except Exception as e:
                self.report.add_check(HealthCheckResult(
                    name="data_availability",
                    status=CheckStatus.WARNING,
                    message=f"Data files found but cannot load: {e}",
                    details={
                        "files_found": len(parquet_files),
                        "error": str(e)
                    },
                    duration_ms=(time.time() - start) * 1000
                ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="data_availability",
                status=CheckStatus.WARNING,
                message=f"Data availability check failed: {e}",
                critical=False,  # Data can be downloaded if missing
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_alpha_engines(self) -> None:
        """Check if alpha engines can be initialized"""
        start = time.time()
        try:
            from ..models.alpha_engines import AlphaEngineCoordinator
            
            # Initialize coordinator
            coordinator = AlphaEngineCoordinator(
                use_bandit=False,  # Skip bandit for health check
                use_parallel=False,  # Skip parallel for health check
                use_adaptive_weighting=False  # Skip adaptive for health check
            )
            
            if len(coordinator.engines) == 0:
                raise ValueError("No alpha engines initialized")
            
            # Test signal generation with dummy data
            test_features = {
                "close": 100.0,
                "volume": 1000.0,
                "rsi_14": 50.0,
                "trend_strength": 0.5
            }
            
            signals = coordinator.generate_all_signals(
                features=test_features,
                current_regime="RANGE"
            )
            
            if len(signals) == 0:
                raise ValueError("No signals generated from engines")
            
            self.report.add_check(HealthCheckResult(
                name="alpha_engines",
                status=CheckStatus.PASSED,
                message=f"Alpha engines initialized successfully",
                details={
                    "engine_count": len(coordinator.engines),
                    "ai_engine_count": len(coordinator.ai_engines),
                    "signals_generated": len(signals)
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="alpha_engines",
                status=CheckStatus.FAILED,
                message=f"Alpha engines check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_model_factory(self) -> None:
        """Check Model Factory availability"""
        start = time.time()
        try:
            from pathlib import Path
            import sys
            
            strategy_research = Path("strategy-research")
            if not strategy_research.exists():
                self.report.add_check(HealthCheckResult(
                    name="model_factory",
                    status=CheckStatus.SKIPPED,
                    message="Strategy-research directory not found",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Try to import Model Factory
            if str(strategy_research) not in sys.path:
                sys.path.insert(0, str(strategy_research))
            
            try:
                from models.model_factory import ModelFactory
                factory = ModelFactory()
                
                available_models = len(factory._models)
                
                self.report.add_check(HealthCheckResult(
                    name="model_factory",
                    status=CheckStatus.PASSED,
                    message="Model Factory initialized",
                    details={
                        "available_models": available_models,
                        "providers": list(factory.MODEL_IMPLEMENTATIONS.keys())
                    },
                    duration_ms=(time.time() - start) * 1000
                ))
            except ImportError as e:
                self.report.add_check(HealthCheckResult(
                    name="model_factory",
                    status=CheckStatus.WARNING,
                    message=f"Model Factory not available: {e}",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="model_factory",
                status=CheckStatus.WARNING,
                message=f"Model Factory check failed: {e}",
                critical=False,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_ai_council(self) -> None:
        """Check AI Council availability"""
        start = time.time()
        try:
            from observability.ai_council.model_factory_adapter import get_adapter
            
            adapter = get_adapter(use_factory=True)
            
            if not adapter.use_factory:
                self.report.add_check(HealthCheckResult(
                    name="ai_council",
                    status=CheckStatus.WARNING,
                    message="AI Council Model Factory not available",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            available_analysts = adapter.get_available_analysts()
            
            self.report.add_check(HealthCheckResult(
                name="ai_council",
                status=CheckStatus.PASSED if available_analysts else CheckStatus.WARNING,
                message=f"AI Council initialized ({len(available_analysts)} analysts available)",
                details={
                    "available_analysts": available_analysts,
                    "factory_available": adapter.use_factory
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="ai_council",
                status=CheckStatus.WARNING,
                message=f"AI Council check failed: {e}",
                critical=False,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_feature_recipe(self) -> None:
        """Check Feature Recipe can generate features"""
        start = time.time()
        try:
            from src.shared.features.recipe import FeatureRecipe
            import polars as pl
            
            # Create minimal test data with 'ts' column (required by FeatureRecipe)
            from datetime import datetime, timezone
            test_data = pl.DataFrame({
                "ts": [
                    datetime.fromtimestamp(1000000000, tz=timezone.utc),
                    datetime.fromtimestamp(1000000060, tz=timezone.utc),
                    datetime.fromtimestamp(1000000120, tz=timezone.utc)
                ],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0]
            })
            
            recipe = FeatureRecipe()
            features = recipe.build(test_data)
            
            if features.is_empty():
                raise ValueError("Feature recipe produced empty output")
            
            self.report.add_check(HealthCheckResult(
                name="feature_recipe",
                status=CheckStatus.PASSED,
                message="Feature Recipe working",
                details={
                    "input_rows": len(test_data),
                    "output_rows": len(features),
                    "feature_count": len(features.columns)
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="feature_recipe",
                status=CheckStatus.FAILED,
                message=f"Feature Recipe check failed: {e}",
                critical=True,
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_ray_cluster(self) -> None:
        """Check Ray cluster availability"""
        start = time.time()
        try:
            import ray
            
            if not ray.is_initialized():
                # Ray is not required, just warn
                self.report.add_check(HealthCheckResult(
                    name="ray_cluster",
                    status=CheckStatus.WARNING,
                    message="Ray cluster not initialized (will initialize on demand)",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Check cluster status
            cluster_resources = ray.cluster_resources()
            
            self.report.add_check(HealthCheckResult(
                name="ray_cluster",
                status=CheckStatus.PASSED,
                message="Ray cluster available",
                details={
                    "nodes": cluster_resources.get("node:__internal__", 1),
                    "cpus": cluster_resources.get("CPU", 0)
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="ray_cluster",
                status=CheckStatus.WARNING,
                message=f"Ray cluster check failed: {e}",
                critical=False,  # Ray is optional
                duration_ms=(time.time() - start) * 1000
            ))

    def _check_notifications(self) -> None:
        """Check notification systems"""
        start = time.time()
        try:
            # Check if notifications settings exist
            if not hasattr(self.settings, 'notifications'):
                self.report.add_check(HealthCheckResult(
                    name="notifications",
                    status=CheckStatus.SKIPPED,
                    message="Notification settings not configured",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            if not getattr(self.settings.notifications, 'telegram_enabled', False):
                self.report.add_check(HealthCheckResult(
                    name="notifications",
                    status=CheckStatus.SKIPPED,
                    message="Telegram notifications disabled",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            from .notifications import NotificationClient
            
            telegram_token = getattr(self.settings.notifications, 'telegram_bot_token', None)
            if not telegram_token:
                self.report.add_check(HealthCheckResult(
                    name="notifications",
                    status=CheckStatus.WARNING,
                    message="Telegram enabled but token not configured",
                    critical=False,
                    duration_ms=(time.time() - start) * 1000
                ))
                return
            
            # Initialize client (don't actually send message)
            client = NotificationClient(
                telegram_bot_token=telegram_token,
                telegram_chat_id=getattr(self.settings.notifications, 'telegram_chat_id', None)
            )
            
            self.report.add_check(HealthCheckResult(
                name="notifications",
                status=CheckStatus.PASSED,
                message="Notification client initialized",
                details={
                    "telegram_enabled": True,
                    "chat_id_configured": getattr(self.settings.notifications, 'telegram_chat_id', None) is not None
                },
                duration_ms=(time.time() - start) * 1000
            ))
        except Exception as e:
            self.report.add_check(HealthCheckResult(
                name="notifications",
                status=CheckStatus.WARNING,
                message=f"Notifications check failed: {e}",
                critical=False,
                duration_ms=(time.time() - start) * 1000
            ))


def _get_fix_suggestion(check_name: str) -> List[str]:
    """Get fix suggestions for a failed check"""
    suggestions = {
        "settings": [
            "Check config files in config/ directory",
            "Verify YAML files are valid",
            "Check environment variables",
            "Run: python -c 'from cloud.training.config.settings import EngineSettings; EngineSettings.load()'"
        ],
        "file_system": [
            "Check directory permissions",
            "Ensure you have write access to logs/, models/, data/ directories",
            "Run: mkdir -p logs models data/candles data/cache",
            "Check disk space: df -h"
        ],
        "database": [
            "Ensure PostgreSQL is running: pg_isready",
            "Check database credentials in settings",
            "Verify database exists: psql -l",
            "Test connection: psql -h localhost -U your_user -d huracan",
            "Check database DSN format: postgresql://user:pass@host:port/dbname"
        ],
        "api_keys": [
            "Configure exchange API keys in settings",
            "Check config/exchange.yaml or environment variables",
            "Verify API keys are valid and have correct permissions",
            "For Binance: Get API key from https://www.binance.com/en/my/settings/api-management"
        ],
        "exchange_client": [
            "Verify exchange API keys are correct",
            "Check network connectivity to exchange",
            "Verify API keys have read permissions",
            "Test with: python -c 'from cloud.training.services.exchange import ExchangeClient; client = ExchangeClient(...); print(client.get_ticker(\"BTC/USDT\"))'",
            "Check if exchange is in maintenance mode"
        ],
        "dropbox": [
            "Get Dropbox access token from https://www.dropbox.com/developers/apps",
            "Add token to .env file: DROPBOX_ACCESS_TOKEN=your_token",
            "Or add to config files",
            "Verify token has correct permissions (files.content.write, files.content.read)"
        ],
        "data_availability": [
            "Download coin data: python scripts/download_and_upload_candles.py",
            "Or: python scripts/auto_sync_candles.py",
            "Check data/candles/ directory exists",
            "Verify parquet files are present: ls -la data/candles/",
            "Data will be downloaded automatically on first run if exchange client works"
        ],
        "alpha_engines": [
            "Check engine initialization logs",
            "Verify all engine dependencies are installed",
            "Check for import errors in engine modules",
            "Run: python -c 'from cloud.training.models.alpha_engines import AlphaEngineCoordinator; AlphaEngineCoordinator()'",
            "Check engine files in src/cloud/training/models/"
        ],
        "model_factory": [
            "Install strategy-research dependencies: cd strategy-research && pip install -r requirements.txt",
            "Check strategy-research directory exists",
            "Verify Model Factory can be imported",
            "This is optional - engine works without it"
        ],
        "ai_council": [
            "Configure API keys for AI models (ANTHROPIC_KEY, OPENAI_KEY, etc.)",
            "Install strategy-research component",
            "This is optional - engine works without it"
        ],
        "feature_recipe": [
            "Check FeatureRecipe implementation",
            "Verify polars is installed: pip install polars",
            "Check feature recipe file: src/shared/features/recipe.py",
            "This is critical - engine cannot generate features without this"
        ],
        "ray_cluster": [
            "Ray is optional - engine works without it",
            "To enable: ray start --head",
            "Or let engine initialize Ray automatically"
        ],
        "notifications": [
            "Configure Telegram bot token in settings",
            "Get token from @BotFather on Telegram",
            "Add to .env: TELEGRAM_BOT_TOKEN=your_token",
            "This is optional - engine works without notifications"
        ]
    }
    return suggestions.get(check_name, ["Check logs for more details", "Review error message above"])


def run_health_check(settings: Optional[Any] = None) -> HealthCheckReport:
    """
    Run comprehensive health check and return report.
    
    Args:
        settings: EngineSettings instance (optional)
    
    Returns:
        HealthCheckReport
    """
    checker = HealthChecker(settings=settings)
    return checker.run_all_checks()


def validate_health_and_exit(settings: Optional[Any] = None, exit_on_failure: bool = True) -> bool:
    """
    Run health check and exit if critical failures are found.
    
    Args:
        settings: EngineSettings instance (optional)
        exit_on_failure: If True, exit with code 1 on critical failure
    
    Returns:
        True if healthy, False if unhealthy
    """
    report = run_health_check(settings)
    
    # Print report header
    print("\n" + "=" * 80)
    print("🏥 HEALTH CHECK REPORT")
    print("=" * 80)
    print(report.get_summary())
    print()
    
    # Separate checks by status
    passed_checks = [c for c in report.checks if c.status == CheckStatus.PASSED]
    failed_checks = [c for c in report.checks if c.status == CheckStatus.FAILED]
    warning_checks = [c for c in report.checks if c.status == CheckStatus.WARNING]
    skipped_checks = [c for c in report.checks if c.status == CheckStatus.SKIPPED]
    
    # Print passed checks (brief)
    if passed_checks:
        print("✅ PASSED CHECKS:")
        for check in passed_checks:
            print(f"   ✅ {check.name}: {check.message}")
        print()
    
    # Print failed checks (detailed)
    if failed_checks:
        print("❌ FAILED CHECKS:")
        print("=" * 80)
        critical_failures = [c for c in failed_checks if c.critical]
        non_critical_failures = [c for c in failed_checks if not c.critical]
        
        if critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(critical_failures)}):")
            print("-" * 80)
            for i, check in enumerate(critical_failures, 1):
                print(f"\n{i}. ❌ {check.name.upper()} [CRITICAL]")
                print(f"   Error: {check.message}")
                if check.details:
                    print("   Details:")
                    for key, value in check.details.items():
                        print(f"      • {key}: {value}")
                print(f"   Duration: {check.duration_ms:.2f}ms")
                
                # Add fix suggestions
                fix_suggestion = _get_fix_suggestion(check.name)
                if fix_suggestion:
                    print(f"   💡 How to fix:")
                    for suggestion in fix_suggestion:
                        print(f"      • {suggestion}")
        
        if non_critical_failures:
            print(f"\n⚠️  NON-CRITICAL FAILURES ({len(non_critical_failures)}):")
            print("-" * 80)
            for i, check in enumerate(non_critical_failures, 1):
                print(f"\n{i}. ❌ {check.name.upper()}")
                print(f"   Error: {check.message}")
                if check.details:
                    print("   Details:")
                    for key, value in check.details.items():
                        print(f"      • {key}: {value}")
                print(f"   Duration: {check.duration_ms:.2f}ms")
        
        print()
    
    # Print warnings
    if warning_checks:
        print("⚠️  WARNINGS:")
        print("-" * 80)
        for check in warning_checks:
            print(f"   ⚠️  {check.name}: {check.message}")
            if check.details:
                for key, value in check.details.items():
                    print(f"      • {key}: {value}")
        print()
    
    # Print skipped checks (brief)
    if skipped_checks:
        print("⏭️  SKIPPED CHECKS:")
        for check in skipped_checks:
            print(f"   ⏭️  {check.name}: {check.message}")
        print()
    
    print(f"⏱️  Total duration: {report.total_duration_ms:.2f}ms")
    print("=" * 80)
    print()
    
    # Final verdict
    if not report.is_healthy():
        print("❌" * 40)
        print("❌ HEALTH CHECK FAILED - CRITICAL ERRORS DETECTED")
        print("❌" * 40)
        print()
        print("🚨 SUMMARY OF CRITICAL FAILURES:")
        print("-" * 80)
        for i, check in enumerate([c for c in failed_checks if c.critical], 1):
            print(f"{i}. {check.name.upper()}: {check.message}")
        print("-" * 80)
        print()
        print("🛑 ENGINE WILL NOT START")
        print("   Please fix the critical errors above before starting the engine.")
        print()
        print("💡 QUICK FIX GUIDE:")
        print("   1. Review each critical failure above")
        print("   2. Follow the 'How to fix' suggestions for each failure")
        print("   3. Re-run the health check: python scripts/health_check.py")
        print("   4. Once all critical checks pass, the engine will start")
        print()
        
        if exit_on_failure:
            sys.exit(1)
        return False
    
    if report.overall_status == CheckStatus.WARNING:
        print("⚠️  HEALTH CHECK PASSED WITH WARNINGS")
        print("   Engine will start but some features may not work correctly")
        if warning_checks:
            print("   Warnings:")
            for check in warning_checks:
                print(f"      • {check.name}: {check.message}")
        print()
    
    print("✅ HEALTH CHECK PASSED - Engine is ready to start")
    print()
    
    return True


def _get_fix_suggestion(check_name: str) -> List[str]:
    """Get fix suggestions for a failed check"""
    suggestions = {
        "settings": [
            "Check config files in config/ directory",
            "Verify YAML files are valid",
            "Check environment variables",
            "Run: python -c 'from cloud.training.config.settings import EngineSettings; EngineSettings.load()'"
        ],
        "file_system": [
            "Check directory permissions",
            "Ensure you have write access to logs/, models/, data/ directories",
            "Run: mkdir -p logs models data/candles data/cache",
            "Check disk space: df -h"
        ],
        "database": [
            "Ensure PostgreSQL is running: pg_isready",
            "Check database credentials in settings",
            "Verify database exists: psql -l",
            "Test connection: psql -h localhost -U your_user -d huracan",
            "Check database DSN format: postgresql://user:pass@host:port/dbname"
        ],
        "api_keys": [
            "Configure exchange API keys in settings",
            "Check config/exchange.yaml or environment variables",
            "Verify API keys are valid and have correct permissions",
            "For Binance: Get API key from https://www.binance.com/en/my/settings/api-management"
        ],
        "exchange_client": [
            "Verify exchange API keys are correct",
            "Check network connectivity to exchange",
            "Verify API keys have read permissions",
            "Test with: python -c 'from cloud.training.services.exchange import ExchangeClient; client = ExchangeClient(...); print(client.get_ticker(\"BTC/USDT\"))'",
            "Check if exchange is in maintenance mode"
        ],
        "dropbox": [
            "Get Dropbox access token from https://www.dropbox.com/developers/apps",
            "Add token to .env file: DROPBOX_ACCESS_TOKEN=your_token",
            "Or add to config files",
            "Verify token has correct permissions (files.content.write, files.content.read)"
        ],
        "data_availability": [
            "Download coin data: python scripts/download_and_upload_candles.py",
            "Or: python scripts/auto_sync_candles.py",
            "Check data/candles/ directory exists",
            "Verify parquet files are present: ls -la data/candles/",
            "Data will be downloaded automatically on first run if exchange client works"
        ],
        "alpha_engines": [
            "Check engine initialization logs",
            "Verify all engine dependencies are installed",
            "Check for import errors in engine modules",
            "Run: python -c 'from cloud.training.models.alpha_engines import AlphaEngineCoordinator; AlphaEngineCoordinator()'",
            "Check engine files in src/cloud/training/models/"
        ],
        "model_factory": [
            "Install strategy-research dependencies: cd strategy-research && pip install -r requirements.txt",
            "Check strategy-research directory exists",
            "Verify Model Factory can be imported",
            "This is optional - engine works without it"
        ],
        "ai_council": [
            "Configure API keys for AI models (ANTHROPIC_KEY, OPENAI_KEY, etc.)",
            "Install strategy-research component",
            "This is optional - engine works without it"
        ],
        "feature_recipe": [
            "Check FeatureRecipe implementation",
            "Verify polars is installed: pip install polars",
            "Check feature recipe file: src/shared/features/recipe.py",
            "This is critical - engine cannot generate features without this"
        ],
        "ray_cluster": [
            "Ray is optional - engine works without it",
            "To enable: ray start --head",
            "Or let engine initialize Ray automatically"
        ],
        "notifications": [
            "Configure Telegram bot token in settings",
            "Get token from @BotFather on Telegram",
            "Add to .env: TELEGRAM_BOT_TOKEN=your_token",
            "This is optional - engine works without notifications"
        ]
    }
    return suggestions.get(check_name, ["Check logs for more details", "Review error message above"])

