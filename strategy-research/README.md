# Strategy Research Pipeline

## Overview

This component uses AI agents from the moon-dev-ai-agents repository to automatically discover, research, and backtest trading strategies. It feeds validated strategies into the Huracan Engine as new Alpha Engines.

## Architecture

```
Strategy Research Pipeline (This Component)
    ↓
Strategy Translator (Adapter)
    ↓
Huracan Engine (23+ Alpha Engines)
    ↓
Hamilton (Live Trading)
```

## Components

### 1. RBI Agent (Research-Backtest-Implement)
- **Purpose**: Automatically generates trading strategies from research sources
- **Input**: YouTube videos, PDFs, text descriptions of trading ideas
- **Process**:
  1. AI analyzes the trading strategy concept
  2. Generates backtest code using backtesting.py library
  3. Tests across 20+ market data sources
  4. Only saves strategies with >1% return threshold
- **Output**: Validated backtest code + performance metrics

### 2. Model Factory
- **Purpose**: Unified interface for all LLM providers
- **Supported Models**:
  - Anthropic Claude (Haiku, Sonnet, Opus)
  - OpenAI GPT-4/GPT-5
  - DeepSeek (Chat, Reasoner)
  - Google Gemini 2.5
  - Groq (fast inference)
  - xAI Grok
  - Ollama (local models)
  - OpenRouter (200+ models)

### 3. Strategy Translator (Coming Soon)
- **Purpose**: Converts backtests into Huracan AlphaEngines
- **Process**:
  1. Parse backtest code
  2. Extract signal logic (entry/exit conditions)
  3. Generate AlphaEngine subclass
  4. Validate on Engine's data pipeline

## Setup

### 1. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Required API keys (add at least ONE):
ANTHROPIC_KEY=your_key_here         # Claude models
OPENAI_KEY=your_key_here            # GPT models
DEEPSEEK_KEY=your_key_here          # DeepSeek (very cheap!)
GEMINI_KEY=your_key_here            # Google Gemini
GROQ_API_KEY=your_key_here          # Groq (fast)
XAI_API_KEY=your_key_here           # Grok models
OPENROUTER_API_KEY=your_key_here    # 200+ models

# Market data APIs:
BIRDEYE_API_KEY=your_key_here       # Solana token data
COINGECKO_API_KEY=your_key_here     # Crypto market data
```

### 2. Install Dependencies

```bash
cd strategy-research
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create Strategy Ideas File

```bash
# Create ideas.txt with trading strategies to research
cat > data/rbi/ideas.txt << EOF
# Trading strategy ideas (one per line)
Buy when RSI < 30 and sell when RSI > 70
Moving average crossover with volume confirmation
Breakout strategy with ATR-based stops
# Lines starting with # are ignored
EOF
```

## Usage

### Running the RBI Agent

```bash
# Activate environment
source .venv/bin/activate

# Run strategy research (processes all ideas in ideas.txt)
python agents/rbi_agent.py

# Or run the parallel version (18 threads, faster)
python agents/rbi_agent_pp_multi.py
```

### Output Structure

```
data/rbi/
├── ideas.txt                       # Input: Strategy ideas
├── backtest_stats.csv              # Output: All passing strategies
└── MM_DD_YYYY/                     # Date-stamped folders
    ├── research/                   # Strategy analysis
    ├── backtests/                  # Generated backtest code
    └── backtests_final/            # Debugged, working code
```

### Reviewing Results

```bash
# View all successful strategies
cat data/rbi/backtest_stats.csv

# Open web dashboard (if using parallel version)
cd data/rbi_pp_multi
python app.py
# Navigate to: http://localhost:8001
```

## Integration with Huracan Engine

### Workflow

1. **Daily at 01:00 UTC**: RBI agent runs, generates new strategies
2. **01:30 UTC**: Strategy Translator converts top performers
3. **02:00 UTC**: Huracan Engine trains (with new engines)
4. **02:30 UTC**: Engine exports baseline to Hamilton

### Strategy Validation Criteria

Before a strategy becomes an AlphaEngine:

1. **Performance Threshold**: Must achieve >5% return in backtest (adjustable)
2. **Data Diversity**: Must work on multiple assets (BTC, ETH, SOL, etc.)
3. **Regime Testing**: Must perform in TREND, RANGE, and PANIC regimes
4. **Walk-Forward Validation**: Must pass Engine's validation pipeline
5. **Risk Metrics**: Sharpe ratio >1.5, max drawdown <20%

## Configuration

### Model Selection

Edit `agents/rbi_agent.py`:

```python
# Research model (analyzes strategy concepts)
RESEARCH_CONFIG = {
    "type": "deepseek",      # cheap, good for research
    "name": "deepseek-chat"
}

# Backtest model (generates code)
BACKTEST_CONFIG = {
    "type": "deepseek",
    "name": "deepseek-reasoner"  # reasoning model for code gen
}

# Debug model (fixes errors)
DEBUG_CONFIG = {
    "type": "openai",
    "name": "gpt-5"  # best for debugging
}
```

### Performance Thresholds

Edit `agents/rbi_agent_pp_multi.py`:

```python
TARGET_RETURN = 50           # AI tries to optimize to this %
SAVE_IF_OVER_RETURN = 5.0    # Save if return > 5%
```

## Cost Estimation

Using DeepSeek (cheapest option):
- Cost per backtest: ~$0.027
- Time per backtest: ~6 minutes
- Processing 10 strategies/day: ~$0.27/day
- Monthly cost: ~$8

Using GPT-4:
- Cost per backtest: ~$0.15-0.30
- Monthly cost (10/day): ~$45-90

## Examples

### Example 1: RSI Reversal Strategy

**Input (ideas.txt):**
```
Buy when RSI < 30 on 15-minute chart, sell when RSI > 70
```

**Output (backtest_stats.csv):**
```
strategy_name,symbol,return_pct,sharpe,max_drawdown,num_trades
rsi_reversal,BTC-USD,8.5,1.8,12.3,145
rsi_reversal,ETH-USD,12.1,2.1,10.5,178
```

**Generated Engine:**
```python
class AIGeneratedEngine_RSI_Reversal(AlphaEngine):
    def generate_signal(self, symbol, df, regime, meta):
        rsi = df['rsi_14'].iloc[-1]
        if rsi < 30:
            return AlphaSignal(direction=1, confidence=0.65)
        elif rsi > 70:
            return AlphaSignal(direction=-1, confidence=0.65)
        return AlphaSignal(direction=0, confidence=0.0)
```

## Monitoring

### Health Checks

```bash
# Check if RBI agent is running
ps aux | grep rbi_agent

# View recent backtests
tail -f data/rbi/backtest_stats.csv

# Check for errors
tail -f logs/rbi_agent.log
```

### Performance Tracking

- **Success Rate**: % of strategies passing threshold
- **Average Return**: Mean return of passing strategies
- **Diversity**: Number of different strategy types discovered
- **Deployment Rate**: % of strategies promoted to Engine

## Troubleshooting

### Issue: No strategies passing threshold

**Solution**: Lower `SAVE_IF_OVER_RETURN` or improve strategy ideas

### Issue: API rate limits

**Solution**: Add delays between requests or use multiple API keys

### Issue: Backtest errors

**Solution**: Check `DEBUG_CONFIG` model quality, ensure data sources are accessible

## Next Steps

1. ✅ Set up strategy-research pipeline
2. ⏳ Build Strategy Translator
3. ⏳ Integrate with Engine's alpha_engines.py
4. ⏳ Deploy first AI-generated engine
5. ⏳ Add market intelligence agents (sentiment, funding, etc.)

## References

- **Moon Dev Repository**: https://github.com/moondevonyt/moon-dev-ai-agents
- **Huracan Engine Docs**: ../engine/docs/README.md
- **Integration Architecture**: ./docs/INTEGRATION_ARCHITECTURE.md
