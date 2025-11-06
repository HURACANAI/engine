# Huracan Engine v5.1

Huracan Engine is the **Cloud Training Box** that produces the nightly "Baseline Brain" for the Huracan trading stack. The project now ships a fully wired PPO agent, walk-forward shadow trading simulator, post-trade analytics backed by PostgreSQL + pgvector, health-monitoring control loop, **and comprehensive observability system** with AI-powered insights and interactive UIs.

## 🎯 Engine Scope

**IMPORTANT:** This codebase is **ONLY for building the Engine** (Cloud Training Box).

### What the Engine IS ✅
- **Daily baseline training** - Runs once per day at 02:00 UTC
- **Full model retraining** - Trains on 3-6 months of historical data
- **Shadow trading for learning** - Paper trades to train models (NO real money)
- **Model export** - Saves trained models to S3/Postgres for other components

### What the Engine is NOT ❌
- **Hourly incremental updates** - That's Mechanic (future component)
- **Live trading execution** - That's Pilot (future component)
- **Real-time order management** - That's Pilot (future component)

**See [ENGINE_SCOPE.md](./ENGINE_SCOPE.md) for complete scope documentation.**

**Main Entry Point:** `src/cloud/training/pipelines/daily_retrain.py`

---

## Highlights

- **Reinforcement learning agent** using PPO with contextual action biasing, running statistics, and gradient clipping ([src/cloud/training/agents/rl_agent.py](../src/cloud/training/agents/rl_agent.py)).
- **Shadow trading simulator** that walks historical data without lookahead, captures rich trade context, and feeds rewards back into the agent ([src/cloud/training/backtesting/shadow_trader.py](../src/cloud/training/backtesting/shadow_trader.py)).
- **Post-trade analytics suite** covering wins, losses, pattern stats, and post-exit tracking, all persisted via `MemoryStore` into PostgreSQL/pgvector ([src/cloud/training/analyzers](../src/cloud/training/analyzers), [src/cloud/training/memory](../src/cloud/training/memory)).
- **Confidence, regime, and feature importance models** packaged under [src/cloud/training/models](../src/cloud/training/models) to score trades before entry and explain outcomes.
- **Intelligence gates** (14 systems) that filter unprofitable trades through cost analysis, adverse selection detection, and meta-labeling.
- **Daily orchestration and monitoring** with Ray integration, health checks, alerting, and artifact publishing ([src/cloud/training/pipelines/daily_retrain.py](../src/cloud/training/pipelines/daily_retrain.py), [src/cloud/training/monitoring](../src/cloud/training/monitoring)).
- **Observability system** (33 modules) with event logging (113k events/sec), learning analytics, AI Council (7 models + judge), and 4 interactive UIs ([observability/](../observability/)).

## Repository Layout

- `src/cloud/training/agents` – PPO agent and supporting utilities.
- `src/cloud/training/backtesting` – Shadow trading engine plus backtest configuration.
- `src/cloud/training/analyzers` – Win/loss insight generators, pattern matcher, post-exit tracker.
- `src/cloud/training/models` – Regime detector, confidence scorer, feature importance learner, ensemble helpers, intelligence gates.
- `src/cloud/training/memory` – PostgreSQL schema and vector-backed memory store.
- `src/cloud/training/pipelines` – Daily retrain entrypoint and RL training pipeline.
- `src/cloud/training/services` – Orchestrator, costs, exchange client, artifact publishing, notifications.
- `src/cloud/training/monitoring` – Health monitor orchestrator, anomaly detection, auto-remediation.
- **`observability/`** – Event logging, learning analytics, AI Council, interactive UIs (33 modules, v5.1).
  - `observability/core/` – Event logger, hybrid storage, model registry, queue monitor.
  - `observability/analytics/` – Learning tracker, shadow trade journal, gate explainer, decision tracer, metrics computer.
  - `observability/ai_council/` – 7 analyst models + judge for AI-powered insights (planned).
  - `observability/ui/` – Live dashboard, trade viewer, gate inspector, model tracker (4 interactive UIs).
- `config/` – Base, local, and monitoring YAML profiles used by `EngineSettings`.
- `scripts/` – Setup helpers (`setup_database.sh`, `setup_rl_training.sh`), runner scripts (`run_daily_retrain.sh`, `run_health_monitor.py`).
- `tests/` – Unit and integration coverage (confidence scoring, regime detection, RL system checks, observability, etc.).
- `docs/` – Comprehensive documentation (50+ files including phase completion, guides, architecture).

## Getting Started

### 1. Prepare your environment

- Python **3.11** (pyenv or `python -m venv .venv` are both fine).
- PostgreSQL **14+** with access to install the `pgvector` extension.
- Optional: Ray cluster access if you plan to distribute daily retraining.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Poetry is the supported workflow (the project file lives under `infrastructure/pyproject.toml`):

```bash
cd infrastructure
poetry install
poetry shell  # optional: spawn a shell with the environment active
```

If you prefer plain `pip`, mirror the versions from `infrastructure/pyproject.toml` (the quick setup script installs the critical ones: torch, polars, psycopg2-binary, scipy).

### 3. Configure PostgreSQL

1. Ensure `DATABASE_URL` is exported, e.g.
   ```bash
   export DATABASE_URL='postgresql://user:password@localhost:5432/huracan'
   ```
2. Provision schema and pgvector tables:
   ```bash
   ./scripts/setup_database.sh
   ```
   The script validates connectivity, installs `vector`, and creates `trade_memory`, `post_exit_tracking`, `win_analysis`, `loss_analysis`, `pattern_library`, and `model_performance`.

### 4. Update configuration

- Edit `config/base.yaml` (or the environment-specific profile you are using) to supply the PostgreSQL DSN, exchange credentials, S3/artifact settings, and any overrides for the RL, shadow trading, memory, or monitoring blocks.
- `EngineSettings` can also read overrides from environment variables such as `HURACAN_ENV`, `HURACAN_MODE`, and `HURACAN_CONFIG_DIR`.

### 5. (Optional) Run the RL setup helper

`./scripts/setup_rl_training.sh` installs core Python dependencies (via Poetry), checks PostgreSQL connectivity, and ensures the docs/models directories exist.

## Running the Engine

- **Daily retrain pipeline** (loads data, runs shadow trading, updates the agent, publishes artifacts, executes health checks):
  ```bash
  ./scripts/run_daily_retrain.sh
  ```
  This script maps to `cloud.training.pipelines.daily_retrain:run_daily_retrain`.

- **Ad-hoc single-symbol training** with the RL pipeline:
  ```bash
  poetry run python - <<'PY'
from cloud.training.config.settings import EngineSettings
from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline
from cloud.training.services.exchange import ExchangeClient

settings = EngineSettings.load()
if not settings.postgres:
    raise RuntimeError("Postgres DSN must be configured")

pipeline = RLTrainingPipeline(settings=settings, dsn=settings.postgres.dsn)
client = ExchangeClient(settings.exchange.primary, sandbox=settings.exchange.sandbox)
metrics = pipeline.train_on_symbol(symbol="BTCUSDT", exchange_client=client, lookback_days=365)
print(metrics)
PY
```

- **Continuous health monitoring** with structured logging and (optional) Telegram alerts:
  ```bash
  poetry run ./scripts/run_health_monitor.py
  ```

## Testing & Quality Gates

- Unit/integration tests: `poetry run pytest`
- Linting and formatting: `poetry run ruff check src tests` and `poetry run ruff format src tests`
- Static typing: `poetry run mypy src`
- System smoke test: `poetry run python tests/verify_system.py`

## Additional Documentation

The `docs/` directory contains deep dives and runbooks:

### Core Guides
- [`SETUP_GUIDE.md`](SETUP_GUIDE.md) – full environment walkthrough.
- [`RL_TRAINING_GUIDE.md`](RL_TRAINING_GUIDE.md) – detailed PPO + shadow trading explanation.
- [`HEALTH_MONITORING_GUIDE.md`](HEALTH_MONITORING_GUIDE.md) – alerting, anomaly detection, and auto-remediation details.
- [`INTEGRATION_COMPLETE.md`](INTEGRATION_COMPLETE.md) & [`DEPLOYMENT_COMPLETE.md`](DEPLOYMENT_COMPLETE.md) – integration status and deployment checklist.
- [`README_COMPLETE.md`](README_COMPLETE.md) – comprehensive architecture summary.

### Complete System Documentation
- [`../COMPLETE_SYSTEM_DOCUMENTATION_V5.md`](../COMPLETE_SYSTEM_DOCUMENTATION_V5.md) – **v5.1 master reference** with complete A-Z coverage of all systems, strategies, and methods.

### Observability Documentation (v5.1 NEW!)
- [`../observability/FINAL_SUMMARY.md`](../observability/FINAL_SUMMARY.md) – Observability system overview (33 modules).
- [`../observability/AI_COUNCIL_ARCHITECTURE.md`](../observability/AI_COUNCIL_ARCHITECTURE.md) – AI Council design (7 analysts + judge).
- [`../observability/ENGINE_ARCHITECTURE.md`](../observability/ENGINE_ARCHITECTURE.md) – Engine vs Hamilton separation.
- [`../observability/INTEGRATION_GUIDE.md`](../observability/INTEGRATION_GUIDE.md) – 3-line integration guide.

### Phase Completion Docs
- Phase 1-4 completion documents ([`PHASE1_COMPLETE.md`](PHASE1_COMPLETE.md) through [`PHASE_5_COMPLETE.md`](PHASE_5_COMPLETE.md))
- Engine phase completions ([`ENGINE_PHASE1_COMPLETE.md`](ENGINE_PHASE1_COMPLETE.md) through [`ENGINE_PHASE4_WAVE3_COMPLETE.md`](ENGINE_PHASE4_WAVE3_COMPLETE.md))
- [`INTELLIGENCE_GATES_COMPLETE.md`](INTELLIGENCE_GATES_COMPLETE.md) – All 14 intelligence gates

These references stay in sync with the code paths listed above, so you can trace each subsystem from documentation to implementation without outdated pointers.

## Interactive Tools (v5.1 NEW!)

The observability system includes 4 interactive terminal UIs:

```bash
# Live dashboard - real-time learning metrics
python -m observability.ui.live_dashboard

# Shadow trade viewer - explore paper trades
python -m observability.ui.trade_viewer

# Gate inspector - analyze gate decisions
python -m observability.ui.gate_inspector

# Model tracker - track model evolution
python -m observability.ui.model_tracker_ui
```
