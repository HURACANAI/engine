# Quick Start Guide: Moon-Dev Integration with Huracan Engine

Get up and running with AI-powered strategy research in 15 minutes.

---

## 🎯 What You'll Build

A pipeline that:
1. **Discovers** trading strategies using AI
2. **Backtests** them automatically
3. **Converts** them to Huracan AlphaEngines
4. **Deploys** to production (after validation)

---

## ⚡ Quick Setup (15 minutes)

### Step 1: Set Up Strategy Research (5 min)

```bash
# Navigate to strategy-research
cd /Users/haq/Engine\ \(HF1\)/strategy-research

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Edit .env and add AT LEAST ONE API key:
nano .env
```

**Minimum Configuration (.env):**
```bash
# Add just ONE of these (DeepSeek is cheapest!)
DEEPSEEK_KEY=your_key_here        # ~$0.027 per strategy
# OR
ANTHROPIC_KEY=your_key_here       # ~$0.10 per strategy
# OR
OPENAI_KEY=your_key_here          # ~$0.20 per strategy
```

### Step 2: Create Your First Strategy (3 min)

```bash
# Create strategy ideas file
cat > data/rbi/ideas.txt << 'EOF'
# Trading Strategy Ideas
# Lines starting with # are ignored

Buy when RSI drops below 30, sell when it rises above 70
Moving average crossover: Buy when 20 EMA crosses above 50 EMA
Breakout strategy: Buy when price breaks above 20-day high with volume spike
EOF
```

### Step 3: Run the RBI Agent (5 min)

```bash
# Process your strategy ideas
python agents/simple_rbi_agent.py

# Expected output:
# 🔍 Researching strategy: Buy when RSI drops below 30...
# ✅ Research complete
# 💻 Generating backtest code...
# ✅ Backtest code generated
# 📁 Files saved to: data/rbi/11_08_2025
```

### Step 4: Translate to AlphaEngine (2 min)

```bash
# Navigate to Engine
cd ../engine

# Run Strategy Translator
python3 << 'EOF'
from pathlib import Path
from cloud.training.adapters.strategy_translator import StrategyTranslator

# Initialize translator (uses Claude by default)
translator = StrategyTranslator(llm_provider="anthropic")

# Find your generated backtests
backtest_dir = Path("../strategy-research/data/rbi")
latest_run = sorted(backtest_dir.glob("*_*_*"))[-1]  # Most recent
backtests = latest_run / "backtests"

# Translate all backtests
engines = translator.batch_translate(backtests)

print(f"\n✅ Generated {len(engines)} AlphaEngines!")
for engine in engines:
    print(f"   📄 {engine.file_path.name}")
EOF
```

**That's it!** You now have AI-generated AlphaEngines ready for testing.

---

## 📊 Verify Everything Worked

### Check Generated Backtests

```bash
cd /Users/haq/Engine\ \(HF1\)/strategy-research

# View backtest files
ls -lh data/rbi/*/backtests/

# Should see Python files like:
# 143022_backtest.py
# 143045_backtest.py
```

### Check Generated Engines

```bash
cd /Users/haq/Engine\ \(HF1\)/engine

# View generated engines
ls -lh src/cloud/training/models/ai_generated_engines/

# Should see files like:
# aigeneratedengine_rsi_reversal_20251108_143530.py
```

### Inspect an Engine

```bash
# View the code
cat src/cloud/training/models/ai_generated_engines/aigeneratedengine_rsi_reversal*.py

# Should see proper AlphaEngine structure:
# class AIGeneratedEngine_RSI_Reversal(AlphaEngine):
#     def generate_signal(self, symbol, df, regime, meta):
#         ...
```

---

## 🧪 Test Your First Engine

### Manual Testing (Recommended First Time)

```python
# Create test script: test_ai_engine.py
cd /Users/haq/Engine\ \(HF1\)/engine

cat > test_ai_engine.py << 'EOF'
from pathlib import Path
from cloud.training.models.alpha_engines import AlphaSignal
import pandas as pd
import numpy as np

# Import your generated engine (adjust filename)
import sys
sys.path.insert(0, 'src/cloud/training/models/ai_generated_engines')
from aigeneratedengine_rsi_reversal_20251108_143530 import AIGeneratedEngine_RSI_Reversal

# Create test data
df = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100,
    'rsi_14': np.random.rand(100) * 100
})

# Initialize engine
engine = AIGeneratedEngine_RSI_Reversal()

# Generate signal
signal = engine.generate_signal(
    symbol='BTCUSDT',
    df=df,
    regime='TREND',
    meta={}
)

print(f"✅ Engine test successful!")
print(f"   Signal direction: {signal.direction}")
print(f"   Confidence: {signal.confidence}")
EOF

# Run test
python test_ai_engine.py
```

**Expected output:**
```
✅ Engine test successful!
   Signal direction: 1  # (or 0, -1)
   Confidence: 0.65     # (0.0 to 1.0)
```

---

## 🚀 Deploy to Production (After Validation)

### Option 1: Manual Integration (Recommended for First Engine)

```python
# Edit: src/cloud/training/models/alpha_engines.py

# 1. Add import at top
from .ai_generated_engines.aigeneratedengine_rsi_reversal_20251108_143530 import AIGeneratedEngine_RSI_Reversal

# 2. Add to get_all_engines() function
def get_all_engines():
    return [
        # ... existing 23 engines ...

        # AI-Generated Engines (reviewed and approved)
        AIGeneratedEngine_RSI_Reversal(),
    ]
```

### Option 2: Dynamic Loading (For Mature Pipeline)

```python
# Edit: src/cloud/training/models/alpha_engines.py

from .ai_generated_engines import load_ai_engines

def get_all_engines():
    base_engines = [
        # ... existing 23 engines ...
    ]

    # Load only approved AI engines
    ai_engines = load_ai_engines(status_filter="approved")

    return base_engines + ai_engines
```

### Mark Engine as Approved

```python
# Edit your generated engine file and add METADATA:

class AIGeneratedEngine_RSI_Reversal(AlphaEngine):
    METADATA = {
        "source": "rbi_agent",
        "generation_date": "2025-11-08",
        "backtest_return": 8.5,  # From RBI agent output
        "status": "approved",    # pending → testing → approved
        "strategy_type": "reversal",
        "description": "RSI-based reversal strategy"
    }

    def __init__(self):
        super().__init__(name="rsi_reversal")

    # ... rest of code ...
```

---

## 📈 Next Steps

### 1. Run Daily Automation

```bash
# Add to crontab (or use systemd timer)
crontab -e

# Add these lines:
# Daily at 01:00 - Generate new strategies
0 1 * * * cd /Users/haq/Engine\ \(HF1\)/strategy-research && .venv/bin/python agents/simple_rbi_agent.py

# Daily at 01:30 - Translate to engines
30 1 * * * cd /Users/haq/Engine\ \(HF1\)/engine && python3 -m cloud.training.adapters.strategy_translator

# Daily at 02:00 - Engine training (already configured)
0 2 * * * cd /Users/haq/Engine\ \(HF1\)/engine && ./scripts/run_daily_retrain.sh
```

### 2. Monitor Performance

```bash
# Check RBI agent output
tail -f /Users/haq/Engine\ \(HF1\)/strategy-research/data/rbi/backtest_stats.csv

# Check Engine training logs
tail -f /Users/haq/Engine\ \(HF1\)/engine/logs/daily_retrain.log

# View AI-generated engine performance
# (Use Engine's observability dashboard)
```

### 3. Add More Strategies

```bash
# Edit ideas.txt
nano /Users/haq/Engine\ \(HF1\)/strategy-research/data/rbi/ideas.txt

# Add lines like:
# MACD crossover with divergence confirmation
# Bollinger Band squeeze breakout
# Volume-weighted moving average trend following
```

### 4. Integrate Market Intelligence (Optional)

See: [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md#phase-4-market-intelligence-agents-optional-5-7-days)

---

## 🛠️ Troubleshooting

### Issue: "Model Factory not available"

**Solution:**
```bash
# Check that strategy-research is set up correctly
ls /Users/haq/Engine\ \(HF1\)/strategy-research/models/model_factory.py

# If missing, copy from moon-dev again
cp -r moon-dev-ai-agents/src/models/* strategy-research/models/
```

### Issue: "API key not found"

**Solution:**
```bash
# Check .env file exists
cat /Users/haq/Engine\ \(HF1\)/strategy-research/.env

# Verify at least one key is set (should see key value, not empty)
grep -E "ANTHROPIC_KEY|OPENAI_KEY|DEEPSEEK_KEY" /Users/haq/Engine\ \(HF1\)/strategy-research/.env
```

### Issue: "No backtests generated"

**Solution:**
```bash
# Check ideas.txt has valid ideas (not all comments)
cat /Users/haq/Engine\ \(HF1\)/strategy-research/data/rbi/ideas.txt

# Verify RBI agent ran without errors
python /Users/haq/Engine\ \(HF1\)/strategy-research/agents/simple_rbi_agent.py 2>&1 | tee rbi_debug.log
```

### Issue: "Strategy Translator fails"

**Solution:**
```bash
# Verify anthropic key is set (or use different provider)
cd /Users/haq/Engine\ \(HF1\)/engine

# Try with different LLM provider
python3 << 'EOF'
from cloud.training.adapters.strategy_translator import StrategyTranslator

# Use DeepSeek (cheaper) instead of Claude
translator = StrategyTranslator(llm_provider="deepseek")

# ... rest of translation code ...
EOF
```

---

## 💰 Cost Estimates

### Per Strategy (using different providers)

| Provider | Cost/Strategy | Speed | Quality |
|----------|---------------|-------|---------|
| DeepSeek | $0.027 | 6 min | Good |
| Anthropic Claude | $0.10 | 4 min | Excellent |
| OpenAI GPT-4 | $0.20 | 3 min | Excellent |

### Monthly Costs (10 strategies/day)

- **DeepSeek**: ~$8/month
- **Claude**: ~$30/month
- **GPT-4**: ~$60/month

**Recommendation**: Start with DeepSeek, upgrade to Claude/GPT-4 for production.

---

## 📚 Learn More

- **Full Architecture**: [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md)
- **Strategy Research Details**: [strategy-research/README.md](strategy-research/README.md)
- **Huracan Engine Docs**: [engine/docs/README.md](engine/docs/README.md)
- **Moon-Dev Original**: [moon-dev-ai-agents/README.md](moon-dev-ai-agents/README.md)

---

## ✅ You're Done!

You now have:
- ✅ AI-powered strategy research pipeline
- ✅ Automatic backtest generation
- ✅ Strategy → AlphaEngine translation
- ✅ Integration with Huracan Engine

**Next**: Run this daily, monitor performance, deploy winners to production!

---

**Questions?** Check [INTEGRATION_ARCHITECTURE.md](INTEGRATION_ARCHITECTURE.md) or review the code.
