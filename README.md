# Huracan Engine

Huracan Engine is the nightly cloud training box that produces the Baseline Brain for the Huracan trading stack. This repository contains the scaffolding for data ingestion, feature engineering, modeling, validation, and artifact publishing as described in the full technical specification.

## Getting Started

1. Install [Poetry](https://python-poetry.org/).
2. Create a virtual environment using Python 3.11 (or rely on `.python-version`).
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Run linters and tests:
   ```bash
   make lint
   make test
   ```

## Project Layout

- `src/cloud/training`: engine-specific orchestration, pipelines, and services.
- `src/shared/features`: shared feature recipe consumed by Engine, Pilot, and Mechanic.
- `src/shared/contracts`: schemas for daily JSON contracts and metrics payloads.
- `config/`: environment-specific configuration profiles.
- `scripts/`: operational helper scripts.
- `tests/`: unit and integration tests.

## Next Steps

- Flesh out each placeholder class with real implementations respecting the design doc.
- Wire the Ray-based parallel execution within the `daily_retrain` pipeline.
- Implement Postgres migrations with Alembic and connect to Cloudflare R2 for artifact storage.
- Add telemetry (structlog JSON, Prometheus metrics) and Telegram notifications for the nightly run.
