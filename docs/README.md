# Huracan AI Trading System - Integration Documentation

Complete documentation for the Huracan AI Trading System integration with moon-dev AI agents for automated strategy research and generation.

---

## 📚 Documentation

This repository contains comprehensive documentation for integrating AI-powered strategy research into the Huracan trading engine.

### Quick Start
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 15 minutes

### Technical Documentation
- **[INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md)** - Complete technical architecture
- **[INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)** - Current status and health metrics
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** - Executive overview

### System Guides
- **[HURACAN_ENGINE_COMPLETE_GUIDE.md](HURACAN_ENGINE_COMPLETE_GUIDE.md)** - A-Z engine explanation (25,000 words)
- **[INTEGRATION_MANIFEST.md](INTEGRATION_MANIFEST.md)** - Complete file inventory
- **[TESTING_RESULTS.md](TESTING_RESULTS.md)** - Test results and validation

---

## 🏗️ System Architecture

The Huracan AI Trading System consists of three independent repositories:

### 1. Strategy Research Pipeline
**Repository:** `huracan-strategy-research` (separate repo)
**Purpose:** AI-powered strategy generation

- RBI Agent (Research-Backtest-Implement)
- Model Factory (8 LLM providers)
- Backtest generation
- Strategy validation

**Setup:**
```bash
git clone https://github.com/YOUR_USERNAME/huracan-strategy-research.git
cd huracan-strategy-research
bash RUN_ME_FIRST.sh
```

### 2. Trading Engine
**Repository:** `engine` (https://github.com/HURACANAI/engine)
**Purpose:** Trading execution and model training

- 23 Alpha Engines
- Daily retraining pipeline
- Strategy Translator (backtest → AlphaEngine)
- Meta-learning system

**Setup:**
```bash
git clone https://github.com/HURACANAI/engine.git
cd engine
# Follow engine setup instructions
```

### 3. Integration Documentation
**Repository:** This repo
**Purpose:** Documentation and architecture specs

---

## 🚀 Complete Integration Flow

```
Strategy Idea
    ↓
huracan-strategy-research
    ├── RBI Agent (AI research)
    ├── Backtest generation
    └── Strategy validation
    ↓
engine (Strategy Translator)
    ├── Translate backtest → AlphaEngine
    ├── Add to ai_generated_engines/
    └── Integrate with 23 existing engines
    ↓
engine (Daily Retrain)
    ├── Train all engines
    ├── Meta-learning optimization
    └── Generate baseline model
    ↓
Hamilton (Live Trading)
    └── Execute trades
```

---

## ✅ Integration Status

**All Systems Operational:**
- ✅ Strategy Research Pipeline (huracan-strategy-research)
- ✅ Trading Engine (engine)
- ✅ Strategy Translator (engine/adapters)
- ✅ AI-Generated Engines (engine/ai_generated_engines)
- ✅ All tests passing (13/13)

---

## 📦 Repository Links

| Component | Repository | Status |
|-----------|------------|--------|
| **Strategy Research** | [huracan-strategy-research](https://github.com/YOUR_USERNAME/huracan-strategy-research) | ✅ Independent |
| **Trading Engine** | [engine](https://github.com/HURACANAI/engine) | ✅ Independent |
| **Documentation** | This repo | ✅ Complete |

---

## 🎯 Key Features

### Automated Strategy Pipeline
- Idea → Research → Backtest → AlphaEngine → Production
- Runs daily at 01:00 UTC (configurable)
- Hands-free strategy generation

### Cost Optimized
- Start with DeepSeek ($1/1M tokens)
- Upgrade to Claude for quality
- Mix providers for best value

### Production Ready
- All tests passing
- Error handling and validation
- Comprehensive logging
- Status-based filtering

---

## 📝 Quick Setup Guide

### 1. Clone Repositories
```bash
# Strategy Research
git clone https://github.com/YOUR_USERNAME/huracan-strategy-research.git

# Trading Engine
git clone https://github.com/HURACANAI/engine.git
```

### 2. Setup Strategy Research
```bash
cd huracan-strategy-research
bash RUN_ME_FIRST.sh
cp .env.example .env
nano .env  # Add API keys
python test_integration.py
```

### 3. Setup Engine
```bash
cd ../engine
# Follow engine-specific setup instructions
python test_strategy_translator.py
```

### 4. Run Your First Strategy
```bash
# Generate strategy
cd huracan-strategy-research
python agents/simple_rbi_agent.py

# Translate to AlphaEngine
cd ../engine
python -m cloud.training.adapters.strategy_translator
```

---

## 🔧 Configuration

### API Keys (Strategy Research)
Add to `huracan-strategy-research/.env`:
```bash
DEEPSEEK_KEY=sk-...        # Recommended: $1/1M tokens
ANTHROPIC_KEY=sk-ant-...   # Optional: Best quality
OPENAI_KEY=sk-...          # Optional: Good balance
```

### Strategy Ideas
Add to `huracan-strategy-research/data/rbi/ideas.txt`:
```
RSI oversold/overbought reversal on 15m timeframe
MACD crossover with volume confirmation
Breakout from consolidation with ATR filter
```

---

## 📊 Testing

All integration tests passing:

**Strategy Research:** 6/6 ✅
- Directory structure verified
- Model Factory working (8 providers)
- Configuration valid
- Ready for API keys

**Engine Integration:** 7/7 ✅
- Backtest translation working
- AlphaEngine generation working
- Signal generation validated
- Import and execution successful

---

## 🎓 Learning Resources

1. **Start here:** [QUICKSTART.md](QUICKSTART.md)
2. **Understand the system:** [HURACAN_ENGINE_COMPLETE_GUIDE.md](HURACAN_ENGINE_COMPLETE_GUIDE.md)
3. **Technical deep-dive:** [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md)
4. **Current status:** [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)

---

## 🤝 Support

- **Issues:** Open an issue in the relevant repository
- **Questions:** See documentation in this repo
- **Updates:** Check [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md)

---

## 📄 License

See individual repository licenses:
- huracan-strategy-research: (your license)
- engine: (engine repo license)

---

**Status:** ✅ All systems operational and ready for use

*Last Updated: 2025-11-08*
