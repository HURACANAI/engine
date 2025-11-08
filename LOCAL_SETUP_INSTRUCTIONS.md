# Local Setup Instructions for Candle Download Script

## Quick Install

Install the minimal dependencies needed for the download script:

```bash
pip install pydantic pydantic-settings polars pyarrow ccxt structlog dropbox numpy
```

Or use the requirements file:

```bash
pip install -r scripts/install_minimal_requirements.txt
```

## Full Install (All Engine Dependencies)

If you want to install all engine dependencies:

```bash
pip install polars pyarrow pandas duckdb lightgbm xgboost "ray[default]" apscheduler sqlalchemy alembic boto3 s3fs pydantic pydantic-settings structlog prometheus-client great-expectations tenacity ccxt requests python-telegram-bot numpy scipy matplotlib psycopg2-binary psutil dropbox opentelemetry-api opentelemetry-sdk
```

## Verify Installation

Test that dependencies are installed:

```bash
python3 -c "import pydantic_settings; import polars; import ccxt; import dropbox; print('✅ All dependencies installed')"
```

## Run the Script

Once dependencies are installed:

```bash
python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
```

## Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r scripts/install_minimal_requirements.txt

# Run script
python scripts/download_and_upload_candles.py --symbols BTC/USDT ETH/USDT --days 150
```


