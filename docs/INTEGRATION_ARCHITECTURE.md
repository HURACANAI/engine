# Moon-Dev AI Agents + Huracan Engine Integration Architecture

**Version:** 1.0
**Date:** 2025-11-08
**Status:** Initial Implementation

---

## 🎯 Executive Summary

This document describes the integration of **moon-dev-ai-agents** (automated strategy research) with the **Huracan Engine** (production ML trading system).

### Key Principle
**Keep systems separate, use adapters to connect them.**

- **moon-dev-ai-agents** → Strategy discovery and backtesting
- **Huracan Engine** → Production training and execution
- **Strategy Translator** → Bridge between the two

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 STRATEGY RESEARCH LAYER                 │
│                 (strategy-research/)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │  RBI Agent (Research-Backtest-Implement)│           │
│  ├─────────────────────────────────────────┤           │
│  │  Input: Strategy ideas (YouTube,        │           │
│  │         PDFs, text descriptions)        │           │
│  │                                         │           │
│  │  Process:                               │           │
│  │  1. AI analyzes strategy concept        │           │
│  │  2. Generates backtest code             │           │
│  │  3. Tests on 20+ data sources           │           │
│  │  4. Validates performance (>5% return)  │           │
│  │                                         │           │
│  │  Output: Validated backtests + metrics  │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
│  ┌─────────────────▼───────────────────────┐           │
│  │  Model Factory (Unified LLM Interface)  │           │
│  ├─────────────────────────────────────────┤           │
│  │  - Anthropic Claude                     │           │
│  │  - OpenAI GPT-4/5                       │           │
│  │  - DeepSeek (cheap!)                    │           │
│  │  - Google Gemini 2.5                    │           │
│  │  - Groq, Grok, Ollama, OpenRouter       │           │
│  └─────────────────────────────────────────┘           │
│                                                         │
└───────────────────┬─────────────────────────────────────┘
                    │ CSV/JSON Exports
                    │ (backtest_stats.csv, Python files)
                    │
┌───────────────────▼─────────────────────────────────────┐
│                  ADAPTER LAYER                          │
│        (engine/src/cloud/training/adapters/)            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │  Strategy Translator                    │           │
│  ├─────────────────────────────────────────┤           │
│  │  Input: Backtest Python code            │           │
│  │                                         │           │
│  │  Process:                               │           │
│  │  1. Parse backtest code (AI-powered)    │           │
│  │  2. Extract signal logic:               │           │
│  │     - Entry conditions                  │           │
│  │     - Exit conditions                   │           │
│  │     - Indicators needed                 │           │
│  │     - Confidence factors                │           │
│  │  3. Generate AlphaEngine subclass       │           │
│  │  4. Validate code structure             │           │
│  │                                         │           │
│  │  Output: AlphaEngine Python file        │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
└───────────────────┬─────────────────────────────────────┘
                    │ Generated AlphaEngine classes
                    │
┌───────────────────▼─────────────────────────────────────┐
│                  HURACAN ENGINE                         │
│              (engine/src/cloud/training/)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │  Alpha Engines (23 + AI-Generated)      │           │
│  ├─────────────────────────────────────────┤           │
│  │  - 23 hand-crafted engines              │           │
│  │  - N AI-generated engines               │           │
│  │    (from strategy-research pipeline)    │           │
│  │                                         │           │
│  │  All engines return:                    │           │
│  │  AlphaSignal(direction, confidence)     │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
│  ┌─────────────────▼───────────────────────┐           │
│  │  Phase 2: Portfolio Intelligence        │           │
│  │  - Pattern detection                    │           │
│  │  - Risk management                      │           │
│  │  - Position sizing                      │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
│  ┌─────────────────▼───────────────────────┐           │
│  │  Phase 3: Consensus & Calibration       │           │
│  │  - Engine consensus                     │           │
│  │  - Confidence calibration               │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
│  ┌─────────────────▼───────────────────────┐           │
│  │  Phase 4: Meta-Learning                 │           │
│  │  - Adaptive hyperparameters             │           │
│  │  - Self-diagnostic health checks        │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
│  ┌─────────────────▼───────────────────────┐           │
│  │  RL Agent (PPO)                         │           │
│  │  - Final trading decision               │           │
│  │  - Position management                  │           │
│  └─────────────────┬───────────────────────┘           │
│                    │                                   │
└───────────────────┬─────────────────────────────────────┘
                    │ MasterDecision + Trained Models
                    │
┌───────────────────▼─────────────────────────────────────┐
│             DOWNSTREAM SYSTEMS                          │
├─────────────────────────────────────────────────────────┤
│  - Hamilton (Live Trading Execution)                    │
│  - Logbook (Observability & Monitoring)                 │
│  - Postgres (Trade Memory & Analytics)                  │
│  - Dropbox (Backup & Sync)                              │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Daily Workflow

```
01:00 UTC - Strategy Research Phase
├─ RBI Agent processes ideas.txt
├─ Generates/validates backtests
├─ Saves passing strategies (>5% return)
└─ Outputs: backtest_stats.csv + Python files

01:30 UTC - Translation Phase
├─ Strategy Translator reads new backtests
├─ Extracts signal logic using AI
├─ Generates AlphaEngine code
├─ Saves to ai_generated_engines/
└─ Outputs: Ready-to-test engines

02:00 UTC - Engine Training Phase
├─ Huracan Engine daily retrain starts
├─ Loads all alpha engines (23 + new)
├─ Shadow trading validation
├─ Trains RL agent
└─ Exports baseline model to S3/Postgres

02:30 UTC - Deployment Phase
├─ Baseline model ready for Hamilton
├─ AI-generated engines marked for paper trading
└─ Performance monitoring begins
```

### Manual Validation (Before Production)

```
New AI-Generated Engine
    ↓
[1] Code Review (manual)
    ↓
[2] Walk-Forward Validation (automated)
    ↓
[3] Paper Trading (2-4 weeks)
    ↓
[4] Regime Testing (TREND/RANGE/PANIC)
    ↓
[5] Risk Metrics Check (Sharpe >1.5, DD <20%)
    ↓
[6] Approval (manual decision)
    ↓
PRODUCTION DEPLOYMENT
```

---

## 📂 Directory Structure

```
/Users/haq/Engine (HF1)/
│
├── strategy-research/              # NEW: Isolated research pipeline
│   ├── agents/
│   │   └── simple_rbi_agent.py     # Strategy discovery agent
│   ├── models/
│   │   ├── model_factory.py        # Unified LLM interface
│   │   ├── claude_model.py
│   │   ├── openai_model.py
│   │   ├── deepseek_model.py
│   │   └── ...                     # Other LLM providers
│   ├── data/
│   │   └── rbi/
│   │       ├── ideas.txt           # Strategy ideas (input)
│   │       ├── backtest_stats.csv  # Results (output)
│   │       └── MM_DD_YYYY/         # Date-stamped runs
│   │           ├── research/       # AI analysis
│   │           ├── backtests/      # Generated code
│   │           └── strategies/     # Validated strategies
│   ├── .env.example
│   └── README.md
│
├── engine/                         # EXISTING: Huracan Engine
│   └── src/
│       └── cloud/
│           └── training/
│               ├── adapters/       # NEW: Integration adapters
│               │   ├── __init__.py
│               │   └── strategy_translator.py  # Backtest→Engine
│               │
│               ├── models/
│               │   ├── alpha_engines.py        # Existing 23 engines
│               │   └── ai_generated_engines/   # NEW: AI-generated
│               │       ├── __init__.py         # Dynamic loader
│               │       ├── README.md
│               │       └── aigeneratedengine_*.py  # Generated files
│               │
│               ├── orchestrator/
│               │   ├── phase2_orchestrator.py
│               │   └── master_orchestrator.py
│               │
│               └── pipelines/
│                   └── daily_retrain.py
│
└── moon-dev-ai-agents/             # ORIGINAL: Reference only
    └── (full moon-dev codebase)
```

---

## 🔧 Component Details

### 1. Strategy Research Pipeline

**Location:** `/Users/haq/Engine (HF1)/strategy-research/`

**Purpose:** Automated strategy discovery using AI

**Components:**
- **RBI Agent**: Researches and backtests strategies
- **Model Factory**: Unified interface for LLMs (Claude, GPT, DeepSeek, etc.)
- **Data Storage**: Organized by date, includes backtest code and metrics

**Configuration:**
- `.env`: API keys for LLMs and data sources
- `ideas.txt`: Strategy ideas to process (YouTube URLs, PDFs, text)

**Output:**
- `backtest_stats.csv`: Performance metrics for all strategies
- Python files: Backtest code for passing strategies

### 2. Strategy Translator

**Location:** `/Users/haq/Engine (HF1)/engine/src/cloud/training/adapters/strategy_translator.py`

**Purpose:** Converts backtests into Huracan AlphaEngines

**Process:**
1. **Parse**: Read backtest Python code
2. **Extract**: Use AI to identify signal logic
3. **Generate**: Create AlphaEngine subclass
4. **Validate**: Ensure code structure is correct

**Key Methods:**
- `extract_strategy_logic()`: Parses backtest, returns ExtractedStrategy
- `generate_alpha_engine()`: Creates AlphaEngine code
- `translate_backtest()`: Full pipeline
- `batch_translate()`: Process multiple backtests

**Configuration:**
- LLM provider: Anthropic (default), OpenAI, DeepSeek
- Output directory: `engine/src/cloud/training/models/ai_generated_engines/`

### 3. AI-Generated Engines

**Location:** `/Users/haq/Engine (HF1)/engine/src/cloud/training/models/ai_generated_engines/`

**Purpose:** Storage for dynamically generated AlphaEngines

**Structure:**
Each engine is a Python file with:
- AlphaEngine subclass
- METADATA dict (performance, status, dates)
- `generate_signal()` method (core logic)

**Loading:**
- **Manual**: Import and add to `alpha_engines.py`
- **Dynamic**: Use `load_ai_engines()` from `__init__.py`

**Statuses:**
- `pending`: Just generated, not validated
- `testing`: In paper trading
- `approved`: Ready for production
- `deprecated`: Removed from production

---

## 🚀 Usage Guide

### Running the Research Pipeline

```bash
# 1. Set up strategy-research
cd /Users/haq/Engine\ \(HF1\)/strategy-research

# 2. Configure environment
cp .env.example .env
# Edit .env: Add API keys (at least one LLM provider)

# 3. Add strategy ideas
cat > data/rbi/ideas.txt << EOF
Buy when RSI < 30, sell when RSI > 70
Moving average crossover with volume spike
Breakout above resistance with high volume
EOF

# 4. Run RBI agent
python agents/simple_rbi_agent.py

# 5. Check results
cat data/rbi/backtest_stats.csv
```

### Translating Backtests to Engines

```bash
# From Engine directory
cd /Users/haq/Engine\ \(HF1\)/engine

# Run strategy translator
python -m cloud.training.adapters.strategy_translator

# Or programmatically:
python3 << EOF
from pathlib import Path
from cloud.training.adapters.strategy_translator import StrategyTranslator

translator = StrategyTranslator(llm_provider="anthropic")

# Translate all backtests from recent run
backtest_dir = Path("../strategy-research/data/rbi/11_08_2025/backtests")
engines = translator.batch_translate(backtest_dir)

print(f"Generated {len(engines)} engines")
for engine in engines:
    print(f"  - {engine.engine_name}: {engine.file_path.name}")
EOF
```

### Integrating into Engine

**Option A: Manual (Recommended for first few engines)**

```python
# Edit: engine/src/cloud/training/models/alpha_engines.py

# Add import
from .ai_generated_engines.aigeneratedengine_rsi_reversal_20251108_143022 import AIGeneratedEngine_RSI_Reversal

# Add to get_all_engines()
def get_all_engines():
    return [
        # ... existing 23 engines ...

        # AI-Generated Engines (reviewed and approved)
        AIGeneratedEngine_RSI_Reversal(),
    ]
```

**Option B: Dynamic (For mature pipeline)**

```python
# Edit: engine/src/cloud/training/models/alpha_engines.py

from .ai_generated_engines import load_ai_engines

def get_all_engines():
    base_engines = [
        # ... existing 23 engines ...
    ]

    # Load approved AI-generated engines
    ai_engines = load_ai_engines(status_filter="approved")

    return base_engines + ai_engines
```

---

## ⚙️ Configuration

### Strategy Research Configuration

**File:** `strategy-research/.env`

```bash
# Primary LLM (for research and code generation)
ANTHROPIC_KEY=your_key_here
OPENAI_KEY=your_key_here
DEEPSEEK_KEY=your_key_here

# Performance thresholds
MIN_RETURN_PCT=5.0        # Save strategies with >5% return
TARGET_RETURN_PCT=50.0    # AI tries to optimize to this

# Data sources
BACKTEST_DATA_SOURCES=BTC-USD,ETH-USD,SOL-USD
TIMEFRAME=15m
DAYS_BACK=90
```

### Strategy Translator Configuration

**File:** `engine/src/cloud/training/adapters/strategy_translator.py`

```python
# Default LLM for translation
DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_LLM_MODEL = "claude-3-5-sonnet-latest"

# Output directory
ENGINE_OUTPUT_DIR = "models/ai_generated_engines/"

# Validation requirements
MIN_CONFIDENCE = 0.5
REQUIRE_STOP_LOSS = True
REQUIRE_TAKE_PROFIT = False
```

### Engine Integration Configuration

**File:** `engine/src/cloud/training/models/ai_generated_engines/__init__.py`

```python
# Loading behavior
DEFAULT_STATUS_FILTER = "approved"  # Only load approved engines
AUTO_LOAD = False  # Set to True for automatic loading

# Validation settings
REQUIRE_METADATA = True  # Engines must have METADATA dict
VALIDATE_BEFORE_LOAD = True  # Check code structure before loading
```

---

## 📊 Performance Monitoring

### Tracking AI-Generated Engines

Each AI-generated engine includes metadata:

```python
class AIGeneratedEngine_Example(AlphaEngine):
    METADATA = {
        "source": "rbi_agent",
        "generation_date": "2025-11-08",
        "backtest_return": 8.5,
        "backtest_sharpe": 1.8,
        "symbols_tested": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "validation_status": "passed",
        "paper_trading_start": "2025-11-10",
        "status": "testing",  # pending/testing/approved/deprecated
        "strategy_type": "reversal",
        "description": "RSI-based reversal with volume confirmation"
    }
```

### Comparison Metrics

Track AI-generated engines vs. hand-crafted:

```python
# In observability system
metrics = {
    "engine_type": "ai_generated",  # or "hand_crafted"
    "win_rate": 0.72,
    "sharpe_ratio": 1.9,
    "max_drawdown": 0.15,
    "avg_confidence": 0.68,
    "signals_per_day": 12,
    "regime_performance": {
        "TREND": 0.75,
        "RANGE": 0.68,
        "PANIC": 0.55
    }
}
```

---

## 🛡️ Safety & Validation

### Pre-Deployment Checklist

Before moving an AI-generated engine to production:

- [ ] **Code Review**: Manual inspection for logic errors
- [ ] **Unit Tests**: Test edge cases (missing data, extreme values)
- [ ] **Backtest Validation**: Confirm >5% return on multiple assets
- [ ] **Walk-Forward**: Pass Engine's validation pipeline
- [ ] **Paper Trading**: 2-4 weeks shadow trading with monitoring
- [ ] **Regime Testing**: Verify performance in TREND, RANGE, PANIC
- [ ] **Risk Metrics**: Sharpe >1.5, Max DD <20%, Win rate >60%
- [ ] **Correlation Check**: Not too similar to existing engines
- [ ] **Approval**: Manual decision by system operator

### Failure Modes & Handling

**Issue:** Engine generates too many signals
**Action:** Increase confidence threshold or add filtering

**Issue:** Engine underperforms in specific regime
**Action:** Add regime-aware confidence adjustment

**Issue:** Code errors during runtime
**Action:** Add error handling, mark as deprecated, roll back

**Issue:** Overfitting to backtest data
**Action:** Longer paper trading period, stricter walk-forward validation

---

## 🔮 Future Enhancements

### Phase 1 (Weeks 1-2) ✅
- ✅ Set up strategy-research pipeline
- ✅ Build Strategy Translator
- ✅ Create AI-generated engines directory
- ✅ Document integration architecture

### Phase 2 (Weeks 3-4)
- [ ] Test with 5-10 real strategies
- [ ] Validate translation accuracy
- [ ] Deploy first AI-generated engine to paper trading
- [ ] Set up automated monitoring

### Phase 3 (Months 2-3)
- [ ] Integrate Model Factory into Engine's AI Council
- [ ] Add market intelligence agents (sentiment, funding, liquidations)
- [ ] Implement automatic approval pipeline (ML-based)
- [ ] Create feedback loop (live performance → RBI agent ideas)

### Phase 4 (Months 4+)
- [ ] Multi-timeframe strategy generation
- [ ] Cross-asset arbitrage strategies
- [ ] Ensemble strategies (combine multiple AI-generated)
- [ ] Self-improving RBI agent (learns from Engine's performance)

---

## 📚 References

- **Moon-Dev Repository**: https://github.com/moondevonyt/moon-dev-ai-agents
- **Huracan Engine Docs**: `engine/docs/README.md`
- **Huracan v5.6 System Docs**: `engine/COMPLETE_SYSTEM_DOCUMENTATION_V5.md`
- **Strategy Research README**: `strategy-research/README.md`
- **Strategy Translator Source**: `engine/src/cloud/training/adapters/strategy_translator.py`

---

## ✍️ Changelog

**2025-11-08**: Initial integration architecture
- Created strategy-research pipeline
- Built Strategy Translator adapter
- Set up AI-generated engines directory
- Documented complete integration flow

---

**Last Updated:** 2025-11-08
**Version:** 1.0
**Status:** Initial Implementation
