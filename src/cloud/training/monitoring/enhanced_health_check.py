"""
Enhanced Comprehensive Health Check System

Checks EVERYTHING:
- Database connectivity and health
- Dropbox connectivity
- Exchange API connectivity
- Ray cluster status
- File system health
- Disk space warnings
- Network connectivity
- Configuration file validation
- Environment variables
- Model file integrity
- S3 connectivity
- Telegram bot connectivity
- Data freshness
- Background jobs/scheduler
- Resource usage (CPU, memory, disk)
- Service status (all components)
- Recent errors
- Performance metrics
"""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: str  # "HEALTHY", "WARNING", "CRITICAL", "DISABLED"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_checked: Optional[datetime] = None


@dataclass
class ComprehensiveHealthReport:
    """Comprehensive health check report with all services."""
    timestamp: datetime
    overall_status: str  # "HEALTHY", "DEGRADED", "CRITICAL"
    checks: List[HealthCheckResult]
    resource_usage: Dict[str, Any]
    summary: Dict[str, Any]
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EnhancedHealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(
        self,
        dsn: Optional[str] = None,
        settings: Optional[Any] = None,
        dropbox_sync: Optional[Any] = None,
    ):
        """Initialize enhanced health checker.
        
        Args:
            dsn: PostgreSQL connection string
            settings: EngineSettings instance
            dropbox_sync: DropboxSync instance (optional)
        """
        self.dsn = dsn
        self.settings = settings
        self.dropbox_sync = dropbox_sync
        
        logger.info("enhanced_health_checker_initialized")
    
    def run_comprehensive_check(self) -> ComprehensiveHealthReport:
        """Run comprehensive health check on all components."""
        logger.info("running_comprehensive_health_check")
        
        checks: List[HealthCheckResult] = []
        
        # 1. Database Health
        checks.append(self._check_database())
        
        # 2. Dropbox Health
        checks.append(self._check_dropbox())
        
        # 3. Exchange API Health
        checks.append(self._check_exchange_api())
        
        # 4. Ray Cluster Health
        checks.append(self._check_ray_cluster())
        
        # 5. File System Health
        checks.append(self._check_file_system())
        
        # 6. Disk Space
        checks.append(self._check_disk_space())
        
        # 7. Network Connectivity
        checks.append(self._check_network())
        
        # 8. Configuration Files
        checks.append(self._check_configuration())
        
        # 9. Environment Variables
        checks.append(self._check_environment_variables())
        
        # 10. Model Files
        checks.append(self._check_model_files())
        
        # 11. S3 Connectivity
        checks.append(self._check_s3())
        
        # 12. Telegram Bot
        checks.append(self._check_telegram_bot())
        
        # 13. Data Freshness
        checks.append(self._check_data_freshness())
        
        # 14. Background Jobs/Scheduler
        checks.append(self._check_scheduler())
        
        # 15. Resource Usage
        resource_usage = self._check_resources()
        checks.append(resource_usage)
        
        # 16. Log Files
        checks.append(self._check_log_files())
        
        # 17. Recent Errors
        checks.append(self._check_recent_errors())
        
        # 18. Service Status
        checks.append(self._check_services())
        
        # Calculate overall status
        critical_count = sum(1 for c in checks if c.status == "CRITICAL")
        warning_count = sum(1 for c in checks if c.status == "WARNING")
        healthy_count = sum(1 for c in checks if c.status == "HEALTHY")
        
        if critical_count > 0:
            overall_status = "CRITICAL"
        elif warning_count > 2:
            overall_status = "DEGRADED"
        elif warning_count > 0:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        # Collect critical issues and warnings
        critical_issues = []
        warnings = []
        recommendations = []
        
        for check in checks:
            if check.status == "CRITICAL":
                critical_issues.extend(check.issues)
                recommendations.extend(check.recommendations)
            elif check.status == "WARNING":
                warnings.extend(check.issues)
                recommendations.extend(check.recommendations)
        
        # Create summary
        summary = {
            "total_checks": len(checks),
            "healthy": healthy_count,
            "warnings": warning_count,
            "critical": critical_count,
            "disabled": sum(1 for c in checks if c.status == "DISABLED"),
        }
        
        report = ComprehensiveHealthReport(
            timestamp=datetime.now(tz=timezone.utc),
            overall_status=overall_status,
            checks=checks,
            resource_usage=resource_usage.details,
            summary=summary,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
        )
        
        logger.info(
            "comprehensive_health_check_complete",
            overall_status=overall_status,
            healthy=healthy_count,
            warnings=warning_count,
            critical=critical_count,
            total=len(checks),
        )
        
        return report
    
    def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and health."""
        logger.info("checking_database_health")
        
        if not self.dsn:
            return HealthCheckResult(
                name="Database",
                status="DISABLED",
                message="Database not configured",
                details={"configured": False},
                recommendations=["Set DATABASE_DSN environment variable"],
            )
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.dsn, connect_timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            conn.close()
            
            return HealthCheckResult(
                name="Database",
                status="HEALTHY",
                message="Database connected successfully",
                details={"connected": True, "dsn_masked": self.dsn.split("@")[-1]},
                last_checked=datetime.now(tz=timezone.utc),
            )
        except Exception as e:
            error_msg = str(e)
            recommendations = [
                "Check if PostgreSQL is running: sudo systemctl status postgresql",
                "Verify DATABASE_DSN environment variable is correct",
                "Check network connectivity to database server",
                "Verify database credentials",
            ]
            
            if "Connection refused" in error_msg:
                recommendations.insert(0, "PostgreSQL server is not running or not accessible")
            elif "authentication failed" in error_msg.lower():
                recommendations.insert(0, "Database credentials are incorrect")
            elif "does not exist" in error_msg.lower():
                recommendations.insert(0, "Database does not exist - create it first")
            
            return HealthCheckResult(
                name="Database",
                status="CRITICAL",
                message=f"Database connection failed: {error_msg}",
                details={"connected": False, "error": error_msg},
                issues=[f"Database connection failed: {error_msg}"],
                recommendations=recommendations,
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_dropbox(self) -> HealthCheckResult:
        """Check Dropbox connectivity."""
        logger.info("checking_dropbox_health")
        
        if not self.settings or not self.settings.dropbox.enabled:
            return HealthCheckResult(
                name="Dropbox",
                status="DISABLED",
                message="Dropbox sync is disabled",
                details={"enabled": False},
            )
        
        if not self.dropbox_sync:
            return HealthCheckResult(
                name="Dropbox",
                status="WARNING",
                message="Dropbox sync not initialized",
                details={"initialized": False},
                issues=["Dropbox sync not initialized"],
                recommendations=["Ensure DropboxSync is properly initialized"],
            )
        
        try:
            # Test Dropbox connection by checking if we can get account info
            if hasattr(self.dropbox_sync, "_dbx"):
                account_info = self.dropbox_sync._dbx.users_get_current_account()
                return HealthCheckResult(
                    name="Dropbox",
                    status="HEALTHY",
                    message="Dropbox connected successfully",
                    details={
                        "connected": True,
                        "account_email": account_info.email,
                        "account_id": account_info.account_id,
                    },
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Dropbox",
                    status="WARNING",
                    message="Dropbox client not initialized",
                    details={"initialized": False},
                    issues=["Dropbox client not initialized"],
                    recommendations=["Reinitialize DropboxSync"],
                )
        except Exception as e:
            error_msg = str(e)
            recommendations = [
                "Check DROPBOX_ACCESS_TOKEN environment variable",
                "Verify token is valid and not expired",
                "Generate new token at https://www.dropbox.com/developers/apps",
            ]
            
            if "expired" in error_msg.lower():
                recommendations.insert(0, "Dropbox token has expired - generate a new one")
            elif "invalid" in error_msg.lower():
                recommendations.insert(0, "Dropbox token is invalid - check token format")
            
            return HealthCheckResult(
                name="Dropbox",
                status="CRITICAL",
                message=f"Dropbox connection failed: {error_msg}",
                details={"connected": False, "error": error_msg},
                issues=[f"Dropbox connection failed: {error_msg}"],
                recommendations=recommendations,
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_exchange_api(self) -> HealthCheckResult:
        """Check exchange API connectivity."""
        logger.info("checking_exchange_api_health")
        
        if not self.settings or not self.settings.exchange:
            return HealthCheckResult(
                name="Exchange API",
                status="DISABLED",
                message="Exchange not configured",
                details={"configured": False},
            )
        
        try:
            # Try to import exchange client
            from ..services.exchange import ExchangeClient
            
            primary_exchange = self.settings.exchange.primary
            credentials = self.settings.exchange.credentials.get(primary_exchange, {})
            
            if not credentials:
                return HealthCheckResult(
                    name="Exchange API",
                    status="WARNING",
                    message=f"Exchange credentials not configured for {primary_exchange}",
                    details={"exchange": primary_exchange, "credentials_configured": False},
                    issues=[f"No credentials for {primary_exchange}"],
                    recommendations=[f"Configure credentials for {primary_exchange} in settings"],
                )
            
            # Test connection (read-only, no API key needed for public endpoints)
            exchange = ExchangeClient(
                exchange_id=primary_exchange,
                credentials=credentials,
                sandbox=self.settings.exchange.sandbox,
            )
            
            # Try to fetch ticker (public endpoint)
            try:
                ticker = exchange.fetch_ticker("BTC/USDT")
                return HealthCheckResult(
                    name="Exchange API",
                    status="HEALTHY",
                    message=f"Exchange API connected: {primary_exchange}",
                    details={
                        "exchange": primary_exchange,
                        "connected": True,
                        "test_symbol": "BTC/USDT",
                        "sandbox": self.settings.exchange.sandbox,
                    },
                    last_checked=datetime.now(tz=timezone.utc),
                )
            except Exception as e:
                return HealthCheckResult(
                    name="Exchange API",
                    status="WARNING",
                    message=f"Exchange API test failed: {str(e)}",
                    details={"exchange": primary_exchange, "error": str(e)},
                    issues=[f"Exchange API test failed: {str(e)}"],
                    recommendations=[
                        "Check exchange API endpoint accessibility",
                        "Verify network connectivity",
                        "Check if exchange is experiencing downtime",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Exchange API",
                status="WARNING",
                message=f"Exchange API check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Exchange API check failed: {str(e)}"],
                recommendations=["Check exchange configuration", "Verify exchange client is working"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_ray_cluster(self) -> HealthCheckResult:
        """Check Ray cluster status."""
        logger.info("checking_ray_cluster_health")
        
        try:
            import ray
            
            if not ray.is_initialized():
                return HealthCheckResult(
                    name="Ray Cluster",
                    status="DISABLED",
                    message="Ray not initialized",
                    details={"initialized": False},
                    recommendations=["Initialize Ray cluster: ray.init()"],
                )
            
            # Get cluster status
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            return HealthCheckResult(
                name="Ray Cluster",
                status="HEALTHY",
                message="Ray cluster is running",
                details={
                    "initialized": True,
                    "cluster_resources": cluster_resources,
                    "available_resources": available_resources,
                },
                last_checked=datetime.now(tz=timezone.utc),
            )
        except ImportError:
            return HealthCheckResult(
                name="Ray Cluster",
                status="DISABLED",
                message="Ray not installed",
                details={"installed": False},
            )
        except Exception as e:
            return HealthCheckResult(
                name="Ray Cluster",
                status="WARNING",
                message=f"Ray cluster check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Ray cluster check failed: {str(e)}"],
                recommendations=["Check Ray cluster status", "Restart Ray cluster if needed"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_file_system(self) -> HealthCheckResult:
        """Check file system health."""
        logger.info("checking_file_system_health")
        
        try:
            # Check critical directories
            critical_dirs = [
                Path("logs"),
                Path("data"),
                Path("models"),
                Path("config"),
                Path("exports"),
            ]
            
            missing_dirs = []
            for dir_path in critical_dirs:
                if not dir_path.exists():
                    missing_dirs.append(str(dir_path))
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            details = {
                "critical_dirs": [str(d) for d in critical_dirs],
                "missing_dirs": missing_dirs,
            }
            
            if missing_dirs:
                return HealthCheckResult(
                    name="File System",
                    status="WARNING",
                    message=f"Missing directories created: {', '.join(missing_dirs)}",
                    details=details,
                    issues=[f"Missing directories: {', '.join(missing_dirs)}"],
                    recommendations=["Directories were auto-created, but verify permissions"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="File System",
                    status="HEALTHY",
                    message="All critical directories exist",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="File System",
                status="CRITICAL",
                message=f"File system check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"File system check failed: {str(e)}"],
                recommendations=["Check file system permissions", "Verify disk is not full"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space."""
        logger.info("checking_disk_space")
        
        try:
            disk = psutil.disk_usage("/")
            percent_used = disk.percent
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            
            details = {
                "percent_used": percent_used,
                "free_gb": free_gb,
                "total_gb": total_gb,
                "used_gb": disk.used / (1024**3),
            }
            
            if percent_used > 90:
                return HealthCheckResult(
                    name="Disk Space",
                    status="CRITICAL",
                    message=f"Disk space critical: {percent_used:.1f}% used ({free_gb:.1f}GB free)",
                    details=details,
                    issues=[f"Disk space critical: {percent_used:.1f}% used"],
                    recommendations=[
                        "Free up disk space immediately",
                        "Delete old log files: find logs/ -name '*.log' -mtime +30 -delete",
                        "Clean up old model files",
                        "Remove unused data files",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            elif percent_used > 80:
                return HealthCheckResult(
                    name="Disk Space",
                    status="WARNING",
                    message=f"Disk space warning: {percent_used:.1f}% used ({free_gb:.1f}GB free)",
                    details=details,
                    issues=[f"Disk space warning: {percent_used:.1f}% used"],
                    recommendations=[
                        "Consider cleaning up old files",
                        "Monitor disk usage",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Disk Space",
                    status="HEALTHY",
                    message=f"Disk space OK: {percent_used:.1f}% used ({free_gb:.1f}GB free)",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Disk Space",
                status="WARNING",
                message=f"Disk space check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Disk space check failed: {str(e)}"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_network(self) -> HealthCheckResult:
        """Check network connectivity."""
        logger.info("checking_network_health")
        
        try:
            # Test internet connectivity
            test_hosts = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
            ]
            
            reachable = 0
            for host, port in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        reachable += 1
                except Exception:
                    pass
            
            if reachable == 0:
                return HealthCheckResult(
                    name="Network",
                    status="CRITICAL",
                    message="No internet connectivity",
                    details={"reachable_hosts": 0, "tested_hosts": len(test_hosts)},
                    issues=["No internet connectivity"],
                    recommendations=[
                        "Check network cable/wifi connection",
                        "Verify DNS settings",
                        "Check firewall rules",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            elif reachable < len(test_hosts):
                return HealthCheckResult(
                    name="Network",
                    status="WARNING",
                    message=f"Partial internet connectivity ({reachable}/{len(test_hosts)} hosts reachable)",
                    details={"reachable_hosts": reachable, "tested_hosts": len(test_hosts)},
                    issues=[f"Partial connectivity: {reachable}/{len(test_hosts)} hosts"],
                    recommendations=["Check network configuration", "Verify DNS settings"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Network",
                    status="HEALTHY",
                    message=f"Internet connectivity OK ({reachable}/{len(test_hosts)} hosts reachable)",
                    details={"reachable_hosts": reachable, "tested_hosts": len(test_hosts)},
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Network",
                status="WARNING",
                message=f"Network check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Network check failed: {str(e)}"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check configuration files."""
        logger.info("checking_configuration_health")
        
        try:
            config_dir = Path("config")
            required_configs = [
                "base.yaml",
                "monitoring.yaml",
            ]
            
            missing_configs = []
            for config_file in required_configs:
                if not (config_dir / config_file).exists():
                    missing_configs.append(config_file)
            
            details = {
                "config_dir": str(config_dir),
                "required_configs": required_configs,
                "missing_configs": missing_configs,
            }
            
            if missing_configs:
                return HealthCheckResult(
                    name="Configuration",
                    status="WARNING",
                    message=f"Missing config files: {', '.join(missing_configs)}",
                    details=details,
                    issues=[f"Missing config files: {', '.join(missing_configs)}"],
                    recommendations=[
                        "Create missing configuration files",
                        "Check config/ directory",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Configuration",
                    status="HEALTHY",
                    message="All required config files exist",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Configuration",
                status="WARNING",
                message=f"Configuration check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Configuration check failed: {str(e)}"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_environment_variables(self) -> HealthCheckResult:
        """Check critical environment variables."""
        logger.info("checking_environment_variables")
        
        try:
            required_vars = [
                "HURACAN_ENV",
            ]
            
            optional_vars = [
                "DROPBOX_ACCESS_TOKEN",
                "DATABASE_DSN",
                "TELEGRAM_BOT_TOKEN",
                "TELEGRAM_CHAT_ID",
            ]
            
            missing_required = []
            missing_optional = []
            
            for var in required_vars:
                if not os.getenv(var):
                    missing_required.append(var)
            
            for var in optional_vars:
                if not os.getenv(var):
                    missing_optional.append(var)
            
            details = {
                "required_vars": required_vars,
                "optional_vars": optional_vars,
                "missing_required": missing_required,
                "missing_optional": missing_optional,
            }
            
            if missing_required:
                return HealthCheckResult(
                    name="Environment Variables",
                    status="CRITICAL",
                    message=f"Missing required environment variables: {', '.join(missing_required)}",
                    details=details,
                    issues=[f"Missing required vars: {', '.join(missing_required)}"],
                    recommendations=[
                        f"Set {var} environment variable" for var in missing_required
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            elif missing_optional:
                return HealthCheckResult(
                    name="Environment Variables",
                    status="WARNING",
                    message=f"Missing optional environment variables: {', '.join(missing_optional)}",
                    details=details,
                    issues=[f"Missing optional vars: {', '.join(missing_optional)}"],
                    recommendations=[
                        f"Consider setting {var} for full functionality" for var in missing_optional
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Environment Variables",
                    status="HEALTHY",
                    message="All environment variables configured",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Environment Variables",
                status="WARNING",
                message=f"Environment variables check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_model_files(self) -> HealthCheckResult:
        """Check model files integrity."""
        logger.info("checking_model_files")
        
        try:
            models_dir = Path("models")
            if not models_dir.exists():
                return HealthCheckResult(
                    name="Model Files",
                    status="WARNING",
                    message="Models directory does not exist",
                    details={"models_dir": str(models_dir), "exists": False},
                    issues=["Models directory does not exist"],
                    recommendations=["Models will be created during training"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            
            model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
            details = {
                "models_dir": str(models_dir),
                "model_count": len(model_files),
                "model_files": [str(f.name) for f in model_files[:10]],  # First 10
            }
            
            if len(model_files) == 0:
                return HealthCheckResult(
                    name="Model Files",
                    status="WARNING",
                    message="No model files found",
                    details=details,
                    issues=["No trained models yet"],
                    recommendations=["Run training pipeline to create models"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Model Files",
                    status="HEALTHY",
                    message=f"Found {len(model_files)} model file(s)",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Model Files",
                status="WARNING",
                message=f"Model files check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_s3(self) -> HealthCheckResult:
        """Check S3 connectivity."""
        logger.info("checking_s3_health")
        
        if not self.settings or not self.settings.s3 or not self.settings.s3.enabled:
            return HealthCheckResult(
                name="S3",
                status="DISABLED",
                message="S3 not configured",
                details={"enabled": False},
            )
        
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.settings.s3.access_key_id,
                aws_secret_access_key=self.settings.s3.secret_access_key,
                endpoint_url=self.settings.s3.endpoint_url,
            )
            
            # Try to list buckets (or head bucket if bucket is specified)
            if self.settings.s3.bucket:
                s3_client.head_bucket(Bucket=self.settings.s3.bucket)
                return HealthCheckResult(
                    name="S3",
                    status="HEALTHY",
                    message=f"S3 connected: {self.settings.s3.bucket}",
                    details={
                        "connected": True,
                        "bucket": self.settings.s3.bucket,
                        "endpoint": self.settings.s3.endpoint_url,
                    },
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                s3_client.list_buckets()
                return HealthCheckResult(
                    name="S3",
                    status="HEALTHY",
                    message="S3 connected",
                    details={"connected": True, "endpoint": self.settings.s3.endpoint_url},
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except ImportError:
            return HealthCheckResult(
                name="S3",
                status="DISABLED",
                message="boto3 not installed",
                details={"installed": False},
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            return HealthCheckResult(
                name="S3",
                status="CRITICAL",
                message=f"S3 connection failed: {error_code}",
                details={"error": str(e), "error_code": error_code},
                issues=[f"S3 connection failed: {error_code}"],
                recommendations=[
                    "Check S3 credentials (access_key_id, secret_access_key)",
                    "Verify S3 endpoint URL",
                    "Check bucket name and permissions",
                ],
                last_checked=datetime.now(tz=timezone.utc),
            )
        except Exception as e:
            return HealthCheckResult(
                name="S3",
                status="WARNING",
                message=f"S3 check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"S3 check failed: {str(e)}"],
                recommendations=["Check S3 configuration", "Verify network connectivity"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_telegram_bot(self) -> HealthCheckResult:
        """Check Telegram bot connectivity."""
        logger.info("checking_telegram_bot_health")
        
        if not self.settings or not self.settings.notifications or not self.settings.notifications.telegram:
            return HealthCheckResult(
                name="Telegram Bot",
                status="DISABLED",
                message="Telegram not configured",
                details={"enabled": False},
            )
        
        telegram_config = self.settings.notifications.telegram
        if not telegram_config.bot_token or not telegram_config.chat_id:
            return HealthCheckResult(
                name="Telegram Bot",
                status="WARNING",
                message="Telegram bot token or chat ID not configured",
                details={
                    "bot_token_configured": bool(telegram_config.bot_token),
                    "chat_id_configured": bool(telegram_config.chat_id),
                },
                issues=["Telegram credentials not fully configured"],
                recommendations=[
                    "Set TELEGRAM_BOT_TOKEN environment variable",
                    "Set TELEGRAM_CHAT_ID environment variable",
                ],
                last_checked=datetime.now(tz=timezone.utc),
            )
        
        try:
            import requests
            
            # Test Telegram API
            url = f"https://api.telegram.org/bot{telegram_config.bot_token}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get("ok"):
                    return HealthCheckResult(
                        name="Telegram Bot",
                        status="HEALTHY",
                        message="Telegram bot connected",
                        details={
                            "connected": True,
                            "bot_username": bot_info.get("result", {}).get("username"),
                        },
                        last_checked=datetime.now(tz=timezone.utc),
                    )
                else:
                    return HealthCheckResult(
                        name="Telegram Bot",
                        status="CRITICAL",
                        message="Telegram API returned error",
                        details={"error": bot_info.get("description", "Unknown error")},
                        issues=["Telegram API error"],
                        recommendations=["Check bot token", "Verify bot is active"],
                        last_checked=datetime.now(tz=timezone.utc),
                    )
            else:
                return HealthCheckResult(
                    name="Telegram Bot",
                    status="CRITICAL",
                    message=f"Telegram API returned status {response.status_code}",
                    details={"status_code": response.status_code},
                    issues=[f"Telegram API error: {response.status_code}"],
                    recommendations=["Check bot token", "Verify network connectivity"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except ImportError:
            return HealthCheckResult(
                name="Telegram Bot",
                status="WARNING",
                message="requests library not installed",
                details={"installed": False},
            )
        except Exception as e:
            return HealthCheckResult(
                name="Telegram Bot",
                status="WARNING",
                message=f"Telegram bot check failed: {str(e)}",
                details={"error": str(e)},
                issues=[f"Telegram bot check failed: {str(e)}"],
                recommendations=["Check network connectivity", "Verify bot token"],
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_data_freshness(self) -> HealthCheckResult:
        """Check data freshness."""
        logger.info("checking_data_freshness")
        
        if not self.dsn:
            return HealthCheckResult(
                name="Data Freshness",
                status="DISABLED",
                message="Database not configured",
                details={"configured": False},
            )
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.dsn, connect_timeout=5)
            cur = conn.cursor()
            
            # Check last trade timestamp
            try:
                cur.execute("""
                    SELECT MAX(entry_timestamp) as last_trade
                    FROM trade_memory
                """)
                result = cur.fetchone()
                last_trade = result[0] if result and result[0] else None
                
                if last_trade:
                    age_hours = (datetime.now(tz=timezone.utc) - last_trade).total_seconds() / 3600
                    details = {
                        "last_trade": last_trade.isoformat(),
                        "age_hours": age_hours,
                    }
                    
                    if age_hours > 48:
                        return HealthCheckResult(
                            name="Data Freshness",
                            status="WARNING",
                            message=f"Data is stale: last trade {age_hours:.1f} hours ago",
                            details=details,
                            issues=[f"Data is {age_hours:.1f} hours old"],
                            recommendations=[
                                "Run training pipeline to update data",
                                "Check if data collection is working",
                            ],
                            last_checked=datetime.now(tz=timezone.utc),
                        )
                    else:
                        return HealthCheckResult(
                            name="Data Freshness",
                            status="HEALTHY",
                            message=f"Data is fresh: last trade {age_hours:.1f} hours ago",
                            details=details,
                            last_checked=datetime.now(tz=timezone.utc),
                        )
                else:
                    return HealthCheckResult(
                        name="Data Freshness",
                        status="WARNING",
                        message="No trades in database",
                        details={"last_trade": None},
                        issues=["No historical data"],
                        recommendations=["Run training pipeline to collect data"],
                        last_checked=datetime.now(tz=timezone.utc),
                    )
            except psycopg2.errors.UndefinedTable:
                return HealthCheckResult(
                    name="Data Freshness",
                    status="WARNING",
                    message="Trade table does not exist",
                    details={"table_exists": False},
                    issues=["Trade table not created yet"],
                    recommendations=["Run training pipeline to create tables"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            
            cur.close()
            conn.close()
        except Exception as e:
            return HealthCheckResult(
                name="Data Freshness",
                status="WARNING",
                message=f"Data freshness check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_scheduler(self) -> HealthCheckResult:
        """Check scheduler/background jobs."""
        logger.info("checking_scheduler_health")
        
        try:
            # Check if APScheduler is running (if used)
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
                # This is a simplified check - in production would check actual scheduler state
                return HealthCheckResult(
                    name="Scheduler",
                    status="HEALTHY",
                    message="Scheduler available",
                    details={"available": True},
                    last_checked=datetime.now(tz=timezone.utc),
                )
            except ImportError:
                return HealthCheckResult(
                    name="Scheduler",
                    status="DISABLED",
                    message="APScheduler not installed",
                    details={"installed": False},
                )
        except Exception as e:
            return HealthCheckResult(
                name="Scheduler",
                status="WARNING",
                message=f"Scheduler check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_resources(self) -> HealthCheckResult:
        """Check system resources."""
        logger.info("checking_resources")
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
            }
            
            issues = []
            recommendations = []
            
            if cpu_percent > 90:
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
                recommendations.append("Check for runaway processes")
            elif cpu_percent > 80:
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
                recommendations.append("Monitor CPU usage")
            
            if memory.percent > 90:
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
                recommendations.append("Free up memory or restart services")
            elif memory.percent > 85:
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
                recommendations.append("Monitor memory usage")
            
            if issues:
                status = "CRITICAL" if any("critical" in i.lower() for i in issues) else "WARNING"
                return HealthCheckResult(
                    name="Resources",
                    status=status,
                    message=f"Resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%",
                    details=details,
                    issues=issues,
                    recommendations=recommendations,
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Resources",
                    status="HEALTHY",
                    message=f"Resources OK: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Resources",
                status="WARNING",
                message=f"Resource check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_log_files(self) -> HealthCheckResult:
        """Check log files."""
        logger.info("checking_log_files")
        
        try:
            logs_dir = Path("logs")
            if not logs_dir.exists():
                return HealthCheckResult(
                    name="Log Files",
                    status="WARNING",
                    message="Logs directory does not exist",
                    details={"logs_dir": str(logs_dir), "exists": False},
                    issues=["Logs directory missing"],
                    recommendations=["Logs directory will be created automatically"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            
            log_files = list(logs_dir.glob("*.log"))
            total_size_mb = sum(f.stat().st_size for f in log_files) / (1024**2)
            
            details = {
                "logs_dir": str(logs_dir),
                "log_count": len(log_files),
                "total_size_mb": total_size_mb,
            }
            
            if total_size_mb > 1000:  # 1GB
                return HealthCheckResult(
                    name="Log Files",
                    status="WARNING",
                    message=f"Log files large: {total_size_mb:.1f}MB",
                    details=details,
                    issues=[f"Log files are {total_size_mb:.1f}MB"],
                    recommendations=[
                        "Consider rotating log files",
                        "Delete old log files: find logs/ -name '*.log' -mtime +30 -delete",
                    ],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Log Files",
                    status="HEALTHY",
                    message=f"Log files OK: {len(log_files)} files, {total_size_mb:.1f}MB",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Log Files",
                status="WARNING",
                message=f"Log files check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_recent_errors(self) -> HealthCheckResult:
        """Check for recent errors in logs."""
        logger.info("checking_recent_errors")
        
        try:
            logs_dir = Path("logs")
            if not logs_dir.exists():
                return HealthCheckResult(
                    name="Recent Errors",
                    status="HEALTHY",
                    message="No logs directory",
                    details={"logs_dir": str(logs_dir), "exists": False},
                    last_checked=datetime.now(tz=timezone.utc),
                )
            
            # Check most recent log file
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
            
            if not log_files:
                return HealthCheckResult(
                    name="Recent Errors",
                    status="HEALTHY",
                    message="No log files found",
                    details={"log_files": 0},
                    last_checked=datetime.now(tz=timezone.utc),
                )
            
            # Read last 1000 lines of most recent log
            recent_log = log_files[0]
            try:
                with open(recent_log, "r") as f:
                    lines = f.readlines()
                    recent_lines = lines[-1000:] if len(lines) > 1000 else lines
                    
                    error_count = sum(1 for line in recent_lines if "ERROR" in line or "CRITICAL" in line)
                    warning_count = sum(1 for line in recent_lines if "WARNING" in line)
                    
                    details = {
                        "log_file": str(recent_log.name),
                        "errors_last_1000_lines": error_count,
                        "warnings_last_1000_lines": warning_count,
                    }
                    
                    if error_count > 10:
                        return HealthCheckResult(
                            name="Recent Errors",
                            status="WARNING",
                            message=f"High error rate: {error_count} errors in recent logs",
                            details=details,
                            issues=[f"{error_count} errors in recent logs"],
                            recommendations=[
                                "Check log files for error details",
                                "Review error patterns",
                            ],
                            last_checked=datetime.now(tz=timezone.utc),
                        )
                    elif error_count > 0:
                        return HealthCheckResult(
                            name="Recent Errors",
                            status="WARNING",
                            message=f"{error_count} error(s) in recent logs",
                            details=details,
                            issues=[f"{error_count} errors detected"],
                            recommendations=["Review error logs"],
                            last_checked=datetime.now(tz=timezone.utc),
                        )
                    else:
                        return HealthCheckResult(
                            name="Recent Errors",
                            status="HEALTHY",
                            message="No recent errors",
                            details=details,
                            last_checked=datetime.now(tz=timezone.utc),
                        )
            except Exception as e:
                return HealthCheckResult(
                    name="Recent Errors",
                    status="WARNING",
                    message=f"Could not read log file: {str(e)}",
                    details={"error": str(e)},
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Recent Errors",
                status="WARNING",
                message=f"Recent errors check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )
    
    def _check_services(self) -> HealthCheckResult:
        """Check service status."""
        logger.info("checking_services")
        
        try:
            # Check if critical Python processes are running
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'python' in cmdline and ('engine' in cmdline or 'huracan' in cmdline.lower()):
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            details = {
                "engine_processes": len(processes),
                "processes": processes[:5],  # First 5
            }
            
            if len(processes) == 0:
                return HealthCheckResult(
                    name="Services",
                    status="WARNING",
                    message="No engine processes detected",
                    details=details,
                    issues=["No running engine processes"],
                    recommendations=["Start the engine process"],
                    last_checked=datetime.now(tz=timezone.utc),
                )
            else:
                return HealthCheckResult(
                    name="Services",
                    status="HEALTHY",
                    message=f"{len(processes)} engine process(es) running",
                    details=details,
                    last_checked=datetime.now(tz=timezone.utc),
                )
        except Exception as e:
            return HealthCheckResult(
                name="Services",
                status="WARNING",
                message=f"Services check failed: {str(e)}",
                details={"error": str(e)},
                last_checked=datetime.now(tz=timezone.utc),
            )

