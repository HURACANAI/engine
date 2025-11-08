# Health Check System

## Overview

The Huracan Engine includes a comprehensive health check system that validates all critical components before startup. If any critical check fails, the engine shuts down gracefully to prevent issues.

## Features

- **Comprehensive Testing**: Tests all critical components before startup
- **Automatic Shutdown**: Shuts down on critical failures
- **Detailed Reporting**: Provides clear error messages and diagnostics
- **Fast Execution**: Runs all checks in parallel where possible
- **Configurable**: Can mark components as critical or optional

## What Gets Checked

### Critical Checks (Engine Shuts Down on Failure)

1. **Settings**: Configuration loaded correctly
2. **File System**: Required directories exist and are writable
3. **Database**: Connection to PostgreSQL (if enabled)
4. **API Keys**: Exchange credentials configured
5. **Exchange Client**: Can connect to exchange and fetch data
6. **Alpha Engines**: All engines can initialize and generate signals
7. **Feature Recipe**: Can generate features from data

### Optional Checks (Warnings Only)

1. **Dropbox**: Connection to Dropbox (optional for sync)
2. **Data Availability**: Coin data files present (can be downloaded)
3. **Model Factory**: AI model factory available (optional for AI features)
4. **AI Council**: AI Council initialized (optional for AI analysis)
5. **Ray Cluster**: Ray cluster available (optional for parallel processing)
6. **Notifications**: Telegram notifications configured (optional)

## Usage

### Automatic (Recommended)

The health check runs automatically before the engine starts:

```bash
python -m src.cloud.training.pipelines.daily_retrain
```

If any critical checks fail, the engine will exit with code 1.

### Manual (Standalone)

Run the health check manually:

```bash
python scripts/health_check.py
```

Or as a module:

```python
from cloud.training.services.health_check import validate_health_and_exit
from cloud.training.config.settings import EngineSettings

settings = EngineSettings.load()
is_healthy = validate_health_and_exit(settings=settings, exit_on_failure=True)
```

### Programmatic

```python
from cloud.training.services.health_check import HealthChecker, run_health_check
from cloud.training.config.settings import EngineSettings

settings = EngineSettings.load()
report = run_health_check(settings=settings)

if not report.is_healthy():
    print("Critical failures detected!")
    for check in report.checks:
        if check.status == CheckStatus.FAILED and check.critical:
            print(f"  ❌ {check.name}: {check.message}")
```

## Health Check Report

The health check generates a detailed report with:

- **Status**: Passed, Failed, Warning, or Skipped
- **Message**: Human-readable description
- **Details**: Additional information (e.g., file counts, connection info)
- **Duration**: Time taken for each check
- **Criticality**: Whether failure is critical

### Example Report

```
================================================================================
🏥 HEALTH CHECK REPORT
================================================================================
✅ Health Check Summary: 12/15 passed, 0 failed (0 critical), 3 warnings, 0 skipped

✅ settings: Settings loaded successfully
   environment: local
   mode: shadow

✅ file_system: File system check passed
   checked_dirs: 4

✅ database: Database connection successful
   dsn: localhost:5432/huracan

✅ api_keys: All critical API keys present (1 exchanges)
   available_exchanges: ['binance']

✅ exchange_client: Exchange client connected to binance
   exchange: binance
   test_ticker: BTC/USDT

⚠️ dropbox: Dropbox enabled but access token not configured

✅ alpha_engines: Alpha engines initialized successfully
   engine_count: 17
   ai_engine_count: 1
   signals_generated: 9

⚠️ data_availability: No coin data files found
   data_dir: data/candles
   files_found: 0

⏱️  Total duration: 2345.67ms
================================================================================
```

## Exit Codes

- **0**: All checks passed - Engine is ready to start
- **1**: Critical failures detected - Engine shuts down

## Configuration

### Marking Components as Critical

By default, the following are critical:
- Settings
- File System
- Database (if enabled)
- API Keys
- Exchange Client
- Alpha Engines
- Feature Recipe

To change criticality, modify the `critical` parameter in `HealthCheckResult`:

```python
self.report.add_check(HealthCheckResult(
    name="component_name",
    status=CheckStatus.FAILED,
    message="Component failed",
    critical=False  # Change to False to make it optional
))
```

### Skipping Checks

Checks can be skipped if a component is disabled:

```python
if not settings.component.enabled:
    self.report.add_check(HealthCheckResult(
        name="component",
        status=CheckStatus.SKIPPED,
        message="Component is disabled",
        critical=False
    ))
    return
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify DSN in settings
   - Check credentials

2. **Exchange Client Failed**
   - Verify API keys are configured
   - Check exchange is accessible
   - Verify credentials are correct

3. **Alpha Engines Failed**
   - Check engine initialization
   - Verify dependencies are installed
   - Check for import errors

4. **Data Availability Warning**
   - Download coin data: `python scripts/download_and_upload_candles.py`
   - Check data directory exists
   - Verify parquet files are present

5. **Dropbox Connection Failed**
   - Verify access token is configured
   - Check token is valid
   - Verify network connectivity

## Integration

The health check is integrated into:

1. **Daily Retrain Pipeline**: Runs before training starts
2. **Standalone Script**: Can be run manually
3. **API Endpoint**: Can be exposed as an HTTP endpoint (future)

## Best Practices

1. **Run Before Production**: Always run health check before deploying
2. **Monitor Warnings**: Address warnings even if not critical
3. **Check Logs**: Review detailed logs for failures
4. **Fix Issues**: Don't ignore critical failures
5. **Test Regularly**: Run health check regularly to catch issues early

## Future Enhancements

- [ ] Add HTTP endpoint for health checks
- [ ] Add health check metrics to monitoring
- [ ] Add automated recovery for some failures
- [ ] Add health check scheduling
- [ ] Add health check history
- [ ] Add health check alerts

