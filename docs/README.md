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

## Document Library

- [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) – Single-page overview of accomplishments and open items.
- [COMPLETE_SYSTEM_OVERVIEW.md](COMPLETE_SYSTEM_OVERVIEW.md) – Architecture walkthrough originally prepared for stakeholders.
- [FINAL_STATUS.md](FINAL_STATUS.md) – Detailed report on current capabilities and gaps.
- [README_COMPLETE.md](README_COMPLETE.md) – End-to-end walkthrough of the system architecture.
- [SESSION_COMPLETE.md](SESSION_COMPLETE.md) – Session timeline and outcomes recap.
- [QUICKSTART.md](QUICKSTART.md) – Step-by-step guide for running the engine.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) – Remaining work mapped against requirements.
- [IMPROVEMENTS_IN_PROGRESS.md](IMPROVEMENTS_IN_PROGRESS.md) – Active enhancements and next tasks.
- [SETUP_GUIDE.md](SETUP_GUIDE.md) – Environment preparation and dependency setup instructions.
- [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) – Summary of components wired into the pipeline.
- [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) – Deployment considerations and verification steps.
- [RL_TRAINING_GUIDE.md](RL_TRAINING_GUIDE.md) – Explanation of the PPO training pipeline.
- [HEALTH_MONITORING_GUIDE.md](HEALTH_MONITORING_GUIDE.md) – Monitoring, alerting, and remediation playbook.
- [FEATURE_COMPARISON.md](FEATURE_COMPARISON.md) – Huracan versus legacy systems capability matrix.
- [REVUELTO_ANALYSIS.md](REVUELTO_ANALYSIS.md) – Full deep dive into Revuelto tactical features.
- [REVUELTO_INTEGRATION_SUMMARY.md](REVUELTO_INTEGRATION_SUMMARY.md) – Actionable summary of Revuelto feature adoption plan.
- [SYSTEM_OPERATIONAL.md](SYSTEM_OPERATIONAL.md) – Final verification log confirming end-to-end readiness.

## Next Steps

- Flesh out each placeholder class with real implementations respecting the design doc.
- Wire the Ray-based parallel execution within the `daily_retrain` pipeline.
- Implement Postgres migrations with Alembic and connect to Cloudflare R2 for artifact storage.
- Add telemetry (structlog JSON, Prometheus metrics) and Telegram notifications for the nightly run.
