# Engine

**Purpose:** Train models nightly from historical data.

**Input:** Candle data from exchanges

**Output:** Models saved to `Dropbox/Huracan/models/baselines/DATE/SYMBOL/`

**Files:**
- `model.bin` - Trained model
- `metrics.json` - Performance metrics
- `feature_recipe.json` - Feature configuration

**Run:** Called automatically by `run_daily.py` or manually via `python -m engine.run`

**Configuration:** See `config.yaml` under `engine:` section

