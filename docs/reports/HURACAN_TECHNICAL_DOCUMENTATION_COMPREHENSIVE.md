# HURACAN TRADING ENGINE - COMPLETE TECHNICAL DOCUMENTATION

## Executive Summary

The Huracan Trading Engine is a sophisticated, multi-component AI-driven cryptocurrency trading system that combines 23 specialized alpha engines, reinforcement learning, meta-learning, portfolio optimization, and comprehensive risk management. It runs daily to train baseline models on 3-6 months of historical data, validates strategies with walk-forward testing, and generates trading signals for deployment.

---

## TABLE OF CONTENTS

1. [Architecture Overview](#architecture-overview)
2. [Core Pipeline Components](#core-pipeline-components)
3. [Machine Learning System](#machine-learning-system)
4. [Alpha Engines (23-Engine System)](#alpha-engines)
5. [Decision-Making Process](#decision-making-process)
6. [Configuration & Settings](#configuration--settings)
7. [Data Flow & Processing](#data-flow--processing)
8. [Deployment & Infrastructure](#deployment--infrastructure)

---

## ARCHITECTURE OVERVIEW

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    HURACAN ENGINE MAIN LOOP                     │
│                    (Runs daily at 02:00 UTC)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌──────────────┐ ┌─────────────┐
        │ Initialize  │ │   Dropbox    │ │  Telegram   │
        │ Exchange    │ │   Sync       │ │ Monitoring  │
        │ Client      │ │              │ │             │
        └─────────────┘ └──────────────┘ └─────────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Load Historical Data │
                    │ (3-6 months)         │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Feature Engineering │
                    │  (Feature Recipe)    │
                    └──────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ Regime     │  │ Shadow     │  │ RL Agent   │
        │ Detection  │  │ Trading    │  │ Training   │
        │            │  │            │  │            │
        └────────────┘  └────────────┘  └────────────┘
                │              │              │
                └──────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  23 Alpha Engines    │
                    │  Generate Signals    │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Engine Consensus     │
                    │ Voting System        │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Confidence Scoring   │
                    │ & Calibration        │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Portfolio Optimizer  │
                    │ & Position Sizer     │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Risk Manager         │
                    │ & Circuit Breaker    │
                    └──────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ Save       │  │ Register   │  │ Dropbox    │
        │ Model      │  │ in DB      │  │ Export     │
        │            │  │            │  │            │
        └────────────┘  └────────────┘  └────────────┘
```

### Directory Structure

```
/Users/haq/Engine (HF1)/engine/
├── src/
│   ├── cloud/
│   │   ├── config/          # Configuration files
│   │   ├── engine/          # Data processing, labeling, walk-forward
│   │   └── training/        # Main training logic
│   │       ├── agents/      # RL agent (PPO) implementation
│   │       ├── models/      # 23 Alpha engines
│   │       ├── pipelines/   # Daily retrain, RL training, feature workflows
│   │       ├── portfolio/   # Portfolio optimization, risk management
│   │       ├── services/    # Orchestration, exchange, data collectors
│   │       ├── datasets/    # Data loaders, quality checks
│   │       ├── monitoring/  # Health checks, Telegram monitoring
│   │       ├── ml_framework/# Neural networks, optimization, validation
│   │       └── integration/ # Dropbox sync, data export
│   └── shared/
│       ├── features/        # Shared feature recipe (FeatureRecipe)
│       └── contracts/       # Data contracts (Mechanic, Pilot)
├── observability/           # Monitoring, UI, AI Council
├── config/                  # Configuration files
├── tests/                   # Test suite
└── scripts/                 # Utility scripts
```

### Key Components Overview

| Component | Purpose | Location |
|-----------|---------|----------|
| **Daily Retrain Pipeline** | Main entry point, runs daily at 02:00 UTC | `pipelines/daily_retrain.py` |
| **RL Training Pipeline** | Shadow trading, pattern learning | `pipelines/rl_training_pipeline.py` |
| **Feature Recipe** | 50+ engineered features from price/volume | `shared/features/recipe.py` |
| **Regime Detector** | Classifies market into TREND/RANGE/PANIC | `models/regime_detector.py` |
| **23 Alpha Engines** | Specialized trading strategies | `models/alpha_engines.py` + 10 individual engines |
| **PPO Agent** | Reinforcement learning via PyTorch | `agents/rl_agent.py` |
| **Adaptive Meta Engine** | Tracks & re-weights engines dynamically | `models/adaptive_meta_engine.py` |
| **Engine Consensus** | Voting system across all engines | `models/engine_consensus.py` |
| **Confidence Scorer** | Regime-aware confidence calibration | `models/confidence_scorer.py` |
| **Portfolio Optimizer** | Allocates capital across symbols | `portfolio/optimizer.py` |
| **Risk Manager** | Position sizing, drawdown control | `portfolio/risk_manager.py` |

---

## CORE PIPELINE COMPONENTS

### 1. Daily Retrain Pipeline (`daily_retrain.py`)

**Entry Point:** `python -m cloud.training.pipelines.daily_retrain` or `cloud-training-daily-retrain`

**Execution Schedule:** Daily at 02:00 UTC (configured via APScheduler)

**Responsibilities:**
1. Initialize all services (Exchange, Dropbox, Telegram, Ray)
2. Load configuration from environment/files
3. Create dated Dropbox folder for today's outputs
4. Start continuous Dropbox sync in background
5. Run health checks (pre-training, post-training, emergency)
6. Execute training orchestrator
7. Export comprehensive data to Dropbox
8. Send Telegram notifications

**Key Configuration Parameters:**
```python
settings.dropbox.enabled = True                    # Dropbox sync
settings.notifications.telegram_enabled = True     # Telegram alerts
settings.training.monitoring.enabled = True        # Health monitoring
settings.training.window_days = 150                # Lookback period
settings.training.shadow_trading.position_size_gbp = 1000.0
```

**Feature Manager:** Gracefully handles non-critical feature failures
- Critical features: Exchange client (stops engine if failed)
- Non-critical features: Dropbox, Telegram, Health Monitor (engine continues)

### 2. Training Orchestrator (`services/orchestration.py`)

**Purpose:** Coordinates per-coin training jobs with retry logic

**Workflow:**
```python
TrainingOrchestrator(settings, exchange_client, ...).run()
├── For each symbol:
│   ├── Load historical data (3-6 months)
│   ├── Run RL training pipeline
│   ├── Validate walk-forward performance
│   ├── Apply mandatory out-of-sample gates
│   ├── Generate metrics and artifacts
│   ├── Publish to S3/Postgres
│   ├── Register model in database
│   └── Send notifications
└── Return results for all symbols
```

**Core Class: `TrainingOrchestrator`**
- `run()`: Main orchestration method
- `_train_symbol(symbol)`: Single symbol training
- `_validate_walk_forward()`: Walk-forward testing
- `_publish_artifacts()`: Save models and metrics

### 3. RL Training Pipeline (`pipelines/rl_training_pipeline.py`)

**Core Class: `RLTrainingPipeline`**

**Workflow:**
```python
RLTrainingPipeline(settings, dsn).train_on_symbol(symbol)
├── 1. Load historical data (365 days lookback)
├── 2. Seed replay buffer from memory store
├── 3. Run shadow trading
│   ├── Simulate every possible trade
│   ├── Track wins/losses
│   ├── Analyze patterns
│   └── Extract key learnings
├── 4. Train RL agent
│   ├── PPO policy gradient updates
│   ├── Value function training
│   └── Curriculum learning from regimes
├── 5. Validate strategy
│   ├── Out-of-sample testing
│   ├── Stress testing
│   └── Regime performance tracking
└── 6. Return metrics and trained model
```

**Key Components:**
- `ShadowTrader`: Backtests every possible trade
- `WinAnalyzer`: Analyzes profitable patterns
- `LossAnalyzer`: Understands failure modes
- `PatternMatcher`: Similarity-based pattern retrieval
- `PostExitTracker`: Watches what happens after trades
- `MemoryStore`: Persistent pattern database

### 4. Feature Engineering (`shared/features/recipe.py`)

**Purpose:** Standardized feature engineering across all components

**Feature Categories:**

#### Technical Indicators (30+ features)
```python
- RSI (14): Relative Strength Index
- ADX: Average Directional Index (trend strength)
- Bollinger Bands: Upper, lower, width
- Volatility regime: Short-term / long-term vol ratio
- Moving averages: SMA50, SMA200, EMA slopes
- MACD: Momentum divergence
- Volume indicators: OBV, CMF, volume rate of change
- ATR: Average True Range (volatility)
```

#### Microstructure Features
```python
- Uptick ratio: Bullish vs bearish ticks
- Spread BPS: Bid-ask spread in basis points
- Micro score: Overall microstructure health
- Volume jump Z-score: Abnormal volume
- Price position in range: 0-1 (bottom to top)
- Compression score: Range tightness
```

#### Pattern Features
```python
- Ignition score: Breakout initiation strength
- Breakout quality: High-quality breakout detection
- Pullback depth: Retracement amount
- Mean revert bias: Regression to mean
- Kurtosis: Fat tails (tail risk)
- Skewness: Asymmetry
```

#### Multi-Timeframe Features
```python
- Trend strength: Short EMA - Long EMA normalized
- Trend flags (5m, 1h): -1/0/+1 directional bias
- HTF bias: Higher timeframe alignment
- Leader/laggard scores: Relative strength
```

**Feature Calculation Example (ADX):**
```python
def _adx(high, low, close, period=14):
    # Calculate +DM and -DM (directional movement)
    plus_dm = max(high - prev_high, 0)
    minus_dm = max(prev_low - low, 0)
    
    # Calculate True Range
    tr = max(high - low, high - prev_close, low - prev_close)
    
    # Directional Indicators
    plus_di = EMA(plus_dm, period) / EMA(tr, period) * 100
    minus_di = EMA(minus_dm, period) / EMA(tr, period) * 100
    
    # ADX
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = EMA(dx, period)
    
    return adx  # 0-100 scale, >25 = strong trend
```

### 5. Regime Detection (`models/regime_detector.py`)

**Purpose:** Classify market conditions for regime-specific strategies

**Regime Classes:**
```python
class MarketRegime(Enum):
    TREND = "trend"      # ADX > 25, strong directional movement
    RANGE = "range"      # Low ATR%, high compression, mean reversion
    PANIC = "panic"      # High volatility spike, fat tails, stress
    UNKNOWN = "unknown"  # Insufficient data
```

**Detection Algorithm:**

```python
def detect_regime(features_df):
    # Extract regime features
    regime_features = RegimeFeatures(
        atr_pct=atr / close * 100,                    # Volatility %
        volatility_ratio=short_vol / long_vol,        # Vol acceleration
        volatility_percentile=percentile(vol_history),
        adx=adx,                                      # Trend strength
        trend_strength=trend,                         # -1 to +1
        ema_slope=ema_slope,
        momentum_slope=momentum_slope,
        kurtosis=kurtosis,                           # Fat tails
        skewness=skewness,                           # Asymmetry
        compression_score=compression,                # Range tightness
        bb_width_pct=bb_width
    )
    
    # Calculate regime scores using meta-weights
    trend_score = (
        -0.15                                  # Intercept
        + 1.8 * trend_strength
        + 0.035 * adx
        + 1.1 * ema_slope
        + 0.9 * momentum_slope
    )
    
    range_score = (
        0.1
        + 1.2 * compression_score
        - 0.8 * bb_width_pct
        - 0.9 * trend_strength
    )
    
    panic_score = (
        -1.0
        + 1.4 * volatility_ratio
        + 1.6 * volatility_percentile
        + 0.05 * kurtosis
        - 0.03 * skewness
    )
    
    # Determine regime
    if trend_score > 0.6:
        return MarketRegime.TREND
    elif range_score > 0.6:
        return MarketRegime.RANGE
    elif panic_score > 0.7:
        return MarketRegime.PANIC
    else:
        return MarketRegime.UNKNOWN
```

**Regime-Specific Thresholds:**
```python
confidence_thresholds = {
    "trend": 0.50,    # Most aggressive (strong moves are profitable)
    "range": 0.55,    # Moderate (mean reversion requires precision)
    "panic": 0.65,    # Conservative (high volatility = high risk)
    "unknown": 0.60,  # Default fallback
}
```

---

## ALPHA ENGINES (23-Engine System)

### Engine Architecture

**Base Engine Structure:**
```python
class AlphaEngine:
    def __init__(self):
        self.feature_weights = {}      # Features this engine trusts
        self.min_confidence = 0.55     # Minimum to generate signal
        
    def generate_signal(features, current_regime) -> AlphaSignal:
        # Calculate regime affinity (0-1, how well regime matches)
        # Calculate score from features
        # Determine direction (buy/sell/hold)
        # Return AlphaSignal with confidence
```

**Signal Structure:**
```python
@dataclass
class AlphaSignal:
    technique: TradingTechnique       # Which engine
    direction: str                    # "buy", "sell", "hold"
    confidence: float                 # 0-1
    reasoning: str                    # Human-readable explanation
    key_features: Dict[str, float]   # Feature values that triggered
    regime_affinity: float            # How well regime matches technique
```

### The 23 Engines

#### GROUP 1: Price-Action / Market-Microstructure (7 engines)

**1. Trend Engine**
- Best in: TREND regime
- Key features: trend_strength, ema_slope, momentum_slope, htf_bias, adx
- Feature weights: {trend_strength: 0.25, ema_slope: 0.20, momentum_slope: 0.20, htf_bias: 0.20, adx: 0.15}
- Strategy: Golden Cross (SMA50 > SMA200) / Death Cross (SMA50 < SMA200)
- Regime affinity: 1.0 (TREND), 0.3 (other)
- Minimum requirements: trend_strength > 0.6, adx > 25

**2. Range Engine**
- Best in: RANGE regime
- Key features: mean_revert_bias, bb_width, compression, volatility_regime, price_position
- Strategy: Buy oversold (price < 0.3 in range), sell overbought (price > 0.7)
- Regime affinity: 1.0 (RANGE), 0.4 (other)
- Fade extremes approach using Bollinger Bands

**3. Breakout Engine**
- Best in: TREND (transition from RANGE to TREND)
- Key features: ignition_score, breakout_quality, breakout_thrust, nr7_density
- Feature weights: {ignition_score: 0.35, breakout_quality: 0.30, breakout_thrust: 0.20, nr7_density: 0.15}
- Strategy: Enter on high-quality breakouts with volume confirmation
- Regime affinity: 1.0 (TREND), 0.7 (transition)

**4. Tape Engine (Microstructure)**
- Best in: All regimes (microstructure always matters)
- Key features: micro_score, uptick_ratio, spread_bps, vol_jump_z
- Strategy: Ride short-term order flow imbalances
- Regime affinity: 0.8 (all regimes)
- Reads: Strong buying (uptick > 0.6) vs selling (uptick < 0.4)

**5. Leader Engine (Relative Strength)**
- Best in: TREND regime
- Key features: rs_score, leader_bias, momentum_slope
- Strategy: Buy leaders (RS > 70), avoid laggards (RS < 30)
- Regime affinity: 1.0 (TREND), 0.5 (other)
- Captures momentum in best-performing coins

**6. Sweep Engine (Liquidity)**
- Best in: All regimes
- Key features: vol_jump_z, pullback_depth, kurtosis, price_position
- Strategy: Detect fake-outs and trap reversals (sweeps of lows/highs)
- Regime affinity: 0.7 (all regimes)

**7. Scalper Engine (Latency Arbitrage)**
- Best in: High-liquidity, low-latency conditions
- Strategy: Micro-arbitrage between exchanges
- Operates with ultra-tight stops

#### GROUP 2: Cross-Asset & Relative-Value (4 engines)

**8. Funding Carry Engine**
- Best in: Perpetual futures markets
- Key features: funding_rate, basis, implied_vol
- Strategy: Trade funding rate carry and arbitrage
- Regime affinity: 1.0 (high-vol regimes)

**9. Correlation / Cluster Engine**
- Best in: All regimes
- Key features: correlation_score, cluster_separation
- Strategy: Pair spread trading, correlation mean reversion
- Regime affinity: 0.8 (all regimes)

**10. Arbitrage Engine**
- Best in: Multi-exchange opportunities
- Strategy: Spot-futures arbitrage, cross-exchange spreads
- Key features: price_differential, liquidity_depth

**11. Volatility Expansion Engine**
- Best in: Transition from low to high volatility
- Strategy: Volatility expansion plays
- Key features: volatility_regime, vol_percentile

#### GROUP 3: Learning / Meta (3 engines)

**12. Adaptive Meta Engine**
- Tracks each engine's performance
- Re-weights engines dynamically based on profitability
- Learns which engines work in which conditions
- Implements curriculum learning

**13. Evolutionary Discovery Engine**
- Auto-discovers new profitable patterns
- Tests and validates novel strategies
- Adapts to changing market conditions

**14. Risk Engine**
- Volatility targeting and drawdown control
- Position sizing based on realized volatility
- Kelly-inspired leverage adjustment

#### GROUP 4: Exotic / Research Lab (5 engines)

**15. Flow Prediction Engine**
- Deep RL-based order flow prediction
- LSTM networks trained on tick data
- Predicts next candle direction

**16. Market Maker Inventory Engine**
- Inventory management
- Spread capture strategies
- Both-sided quoting

**17. Cross-Venue Latency Engine**
- Cross-exchange latency exploitation
- Statistical arbitrage on timing

**18. Anomaly Detection Engine**
- Detects market manipulation
- Identifies unusual order patterns
- Red flag signals

**19. Regime Transition Predictor**
- Anticipates regime changes
- BOCD (Bayesian Online Changepoint Detection)
- Gets ahead of regime transitions

#### GROUP 5: Additional Strategies (4 engines)

**20. Momentum Reversal Engine**
- Exhaustion/reversal trades
- Momentum decay detection
- Counter-trend entries

**21. Divergence Engine**
- Price/indicator divergences
- RSI/MACD divergence signals
- Potential reversal points

**22. Support/Resistance Bounce Engine**
- Key level identification
- Bounce and break trading
- Level profitability tracking

**23. Custom/AI-Generated Engines**
- Dynamically loaded AI-generated strategies
- Adapter framework for new engines
- Community-contributed strategies

### Engine Coordination System

**AlphaEngineCoordinator** (`models/alpha_engines.py`)

```python
class AlphaEngineCoordinator:
    def __init__(self, use_bandit=True, use_parallel=True, 
                 use_adaptive_weighting=True):
        self.engines = {}          # All 23 engines
        self.ai_engines = {}       # Dynamically loaded engines
        self.engine_performance = {} # Tracks each engine's results
        self.bandit = AlphaEngineBandit() if use_bandit
        
    def get_signals(self, features, regime) -> Dict[TradingTechnique, AlphaSignal]:
        # Run all engines in parallel (ThreadPoolExecutor)
        signals = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for technique, engine in self.engines.items():
                future = executor.submit(
                    engine.generate_signal, features, regime
                )
                futures[technique] = future
            
            for technique, future in futures.items():
                signals[technique] = future.result()
        
        return signals
    
    def combine_signals(self, signals, current_regime) -> AlphaSignal:
        # Use Engine Consensus voting
        # Apply adaptive weighting from meta-engine
        # Return combined signal with adjusted confidence
```

**Parallel Execution:**
- Uses ThreadPoolExecutor or Ray for multi-engine execution
- Typical execution time: 50-150ms for all 23 engines
- Supports 3-8 parallel workers

**Engine Performance Tracking:**
```python
self.engine_performance = {
    TradingTechnique.TREND: [0.75, 0.68, 0.72, ...],  # Recent confidences
    TradingTechnique.RANGE: [0.55, 0.60, ...],
    # ... all 23 engines
}
```

---

## DECISION-MAKING PROCESS

### Step 1: Signal Generation

**Input:** Current features (80-100 floats), market regime

**Process:**
1. Run all 23 alpha engines in parallel
2. Each engine generates AlphaSignal with direction & confidence
3. Collect all signals

**Output:** Dictionary of 23 signals (one per engine)

### Step 2: Engine Consensus Voting

**Purpose:** Prevent overconfidence, require agreement

**Class:** `EngineConsensus` (`models/engine_consensus.py`)

```python
def analyze_consensus(
    primary_engine,       # Engine with highest confidence
    primary_direction,    # "buy" or "sell"
    primary_confidence,
    all_opinions,         # Signals from all 23 engines
    current_regime
) -> ConsensusResult:
    
    # Count agreement/disagreement
    agreeing = [e for e in all_opinions if e.direction == primary_direction]
    disagreeing = [e for e in all_opinions if e.direction != primary_direction]
    
    # Calculate agreement score
    agreement_score = len(agreeing) / len(all_opinions)
    
    # Determine consensus level
    if agreement_score == 1.0:
        consensus_level = ConsensusLevel.UNANIMOUS
        confidence_boost = +0.10
    elif agreement_score >= 0.75:
        consensus_level = ConsensusLevel.STRONG
        confidence_boost = +0.05
    elif agreement_score >= 0.60:
        consensus_level = ConsensusLevel.MODERATE
        confidence_boost = 0.0
    elif agreement_score >= 0.50:
        consensus_level = ConsensusLevel.WEAK
        confidence_penalty = -0.05
    else:
        consensus_level = ConsensusLevel.DIVIDED
        confidence_penalty = -0.15
    
    # Adjust confidence
    adjusted_confidence = primary_confidence + confidence_adjustment
    
    # Regime-specific recommendations
    if current_regime == "panic" and disagreement > 0:
        recommendation = "SKIP_TRADE"  # No disagreement in panic
    elif consensus_level == ConsensusLevel.UNANIMOUS:
        recommendation = "TAKE_TRADE"
    elif consensus_level in [ConsensusLevel.STRONG, ConsensusLevel.MODERATE]:
        recommendation = "TAKE_TRADE"
    elif consensus_level == ConsensusLevel.WEAK:
        recommendation = "REDUCE_SIZE"
    else:
        recommendation = "SKIP_TRADE"
    
    return ConsensusResult(
        primary_direction=primary_direction,
        consensus_level=consensus_level,
        agreement_score=agreement_score,
        adjusted_confidence=adjusted_confidence,
        recommendation=recommendation,
        warnings=[...],  # Disagreement warnings, etc.
    )
```

**Consensus Thresholds by Regime:**
```
TREND Regime:
- Need TREND engine + 1 other agreement
- RANGE disagreement = warning (counter-trend forming)
- BREAKOUT agreement = strong boost

RANGE Regime:
- Need RANGE engine + 1 other agreement
- TREND disagreement = warning (trend emerging)
- SWEEP agreement = strong boost

PANIC Regime:
- Need 3+ engines unanimous
- Any disagreement = skip trade
- Very conservative thresholds
```

### Step 3: Confidence Scoring & Calibration

**Class:** `ConfidenceScorer` (`models/confidence_scorer.py`)

**Purpose:** Regime-aware, multi-factor confidence estimation

```python
def calculate_confidence(
    sample_count,          # Historical examples
    best_score,            # Best action score
    runner_up_score,       # Second-best action
    pattern_similarity,    # Match with patterns (0-1)
    pattern_reliability,   # Historical success rate
    regime_match,          # Does regime match historical?
    regime_confidence,     # Regime detection confidence
    current_regime,        # "trend"/"range"/"panic"
    features_embedding,    # Feature vector for similarity
    memory_store,          # Historical pattern store
    symbol                 # Trading symbol
):
    # 1. Sample-based confidence (sigmoid function)
    # More historical examples = more confidence
    sample_confidence = sigmoid(sample_count, threshold=20)
    # Example: 5 samples = 38%, 20 samples = 50%, 100 samples = 88%
    
    # 2. Score separation (clear winner)
    # Difference between best and runner-up actions
    score_separation = best_score - runner_up_score
    separation_confidence = min(score_separation * 2, 1.0)
    
    # 3. Pattern similarity
    # Find similar historical patterns
    similar_patterns = memory_store.find_similar(
        features_embedding, similarity_threshold=0.7, topk=5
    )
    
    if similar_patterns:
        pattern_similarity = sum(p.similarity for p in similar_patterns) / len(similar_patterns)
        pattern_reliability = sum(p.win_rate for p in similar_patterns) / len(similar_patterns)
    else:
        pattern_similarity = 0.5
        pattern_reliability = 0.5
    
    # 4. Regime alignment
    regime_bonus = 0.0 if regime_match else -0.05
    
    # 5. Combine factors
    base_confidence = (
        0.3 * sample_confidence +
        0.3 * separation_confidence +
        0.2 * pattern_similarity +
        0.2 * pattern_reliability +
        regime_bonus
    )
    
    # 6. Apply regime-specific thresholds
    regime_threshold = {
        "trend": 0.50,
        "range": 0.55,
        "panic": 0.65,
        "unknown": 0.60,
    }[current_regime]
    
    # 7. Decision
    if base_confidence >= regime_threshold:
        decision = "trade"
    else:
        decision = "skip"
    
    return ConfidenceResult(
        confidence=base_confidence,
        factors=ConfidenceFactors(...),
        decision=decision,
        reason=f"Confidence {base_confidence:.2f} vs threshold {regime_threshold}"
    )
```

### Step 4: Portfolio Intelligence & Position Sizing

**Components:**
1. **Portfolio Optimizer** (`portfolio/optimizer.py`)
   - Solves: max(portfolio Sharpe) subject to constraints
   - Uses: Correlation matrix, expected returns, volatility
   - Output: Allocation weights per symbol

2. **Dynamic Position Sizer** (`portfolio/position_sizer.py`)
   - Kelly Criterion: f* = (bp - q) / b, where:
     - f* = optimal fraction of capital
     - b = odds (expected return / risk)
     - p = win probability
     - q = loss probability
   - Volatility Scaling: size *= (target_vol / realized_vol)
   - Output: Position size in GBP

3. **Comprehensive Risk Manager** (`portfolio/risk_manager.py`)
   - Tracks portfolio risk metrics
   - Enforces drawdown limits
   - Manages correlation concentration
   - Circuit breakers for systemic risk

**Position Sizing Algorithm:**

```python
def calculate_position_size(
    symbol,
    confidence,              # 0.52 to 1.0
    volatility,              # Realized volatility
    stop_loss_bps,           # 50-100 bps
    expected_return_bps,     # 10-30 bps
    current_price,           # USDT price
    account_size_gbp,        # Total capital
    regime
):
    # 1. Risk/reward ratio
    risk_bps = stop_loss_bps
    reward_bps = expected_return_bps
    rr_ratio = reward_bps / risk_bps  # Should be >= 1.0
    
    # 2. Volatility adjustment
    base_volatility = 0.15  # 15% target
    vol_adjustment = base_volatility / volatility
    
    # 3. Confidence scaling
    # Higher confidence = larger position
    # 0.52 -> 1x, 0.65 -> 2x, 0.80 -> 3x
    confidence_multiplier = (confidence - 0.50) * 10
    
    # 4. Regime adjustment
    regime_factors = {
        "trend": 1.2,   # More aggressive
        "range": 1.0,   # Normal
        "panic": 0.6,   # Conservative
        "unknown": 0.8,
    }
    regime_factor = regime_factors[regime]
    
    # 5. Kelly criterion variant
    # f* = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
    win_rate = 0.55  # Historical from meta-engine
    win_size = expected_return_bps
    loss_size = stop_loss_bps
    kelly_fraction = (win_rate * win_size - (1 - win_rate) * loss_size) / win_size
    
    # 6. Combine factors
    base_size_gbp = 500  # Base position
    adjusted_size = (
        base_size_gbp *
        kelly_fraction *
        vol_adjustment *
        confidence_multiplier *
        regime_factor
    )
    
    # 7. Apply constraints
    max_size_pct = 0.05  # Max 5% per position
    max_size_gbp = account_size_gbp * max_size_pct
    
    final_size = min(adjusted_size, max_size_gbp)
    
    return PositionSizeRecommendation(
        symbol=symbol,
        size_gbp=final_size,
        kelly_fraction=kelly_fraction,
        vol_adjustment=vol_adjustment,
        confidence_multiplier=confidence_multiplier,
    )
```

### Step 5: Exit Strategy Determination

**Dynamic Exit System:**

```python
class ExitStrategyDeterminer:
    def calculate_exits(
        self,
        entry_signal,
        current_regime,
        hold_duration_minutes,
        unrealized_pnl_bps
    ):
        # 1. Regime-aware stop loss
        regime_stops = {
            "trend": 50,     # Tight stop, follow trend
            "range": 100,    # Wide stop, range expectations
            "panic": 150,    # Very wide, chaos conditions
        }
        stop_loss_bps = regime_stops[current_regime]
        
        # 2. Adaptive take profit
        if entry_signal.confidence > 0.75:
            take_profit_bps = 30  # Very confident
        elif entry_signal.confidence > 0.65:
            take_profit_bps = 20  # Confident
        else:
            take_profit_bps = 15  # Moderate
        
        # 3. Time-based exit (max hold)
        max_hold_minutes = {
            "trend": 240,     # 4 hours, ride the wave
            "range": 120,     # 2 hours, mean reversion is quick
            "panic": 30,      # 30 min, get out fast in chaos
        }[current_regime]
        
        if hold_duration_minutes > max_hold_minutes:
            return ExitSignal(action="EXIT", reason="Max hold exceeded")
        
        # 4. Trailing stop
        if unrealized_pnl_bps > 15:
            trail_amount = unrealized_pnl_bps * 0.5  # Trail by 50% of gain
            return ExitSignal(action="SET_TRAIL", trail_bps=trail_amount)
        
        return ExitSignal(
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            max_hold_minutes=max_hold_minutes,
        )
```

---

## MACHINE LEARNING SYSTEM

### PPO (Proximal Policy Optimization) Agent

**File:** `agents/rl_agent.py`

**Architecture:**
```
Input State (80-100 features)
    ↓
Linear Layer (80 → 256 neurons, ReLU)
    ↓
Policy Head              Value Head
  ↓                        ↓
Linear (256 → 10 actions) Linear (256 → 1 value)
    ↓                        ↓
Softmax (action probs)    Tanh (value estimate)
    ↓                        ↓
Action (0-9)             State value
```

**Discrete Actions (10 total):**
```python
class TradingAction(Enum):
    DO_NOTHING = 0
    ENTER_LONG_SMALL = 1   # 0.5x position
    ENTER_LONG_NORMAL = 2  # 1.0x position
    ENTER_LONG_LARGE = 3   # 1.5x position
    EXIT_POSITION = 4
    HOLD_POSITION = 5
    SCRATCH = 6            # Fast exit (minimize loss)
    ADD_GRID = 7           # Add to position (DCA)
    SCALE_OUT = 8          # Partial exit
    TRAIL_RUNNER = 9       # Activate trailing stop
```

**PPO Training Loop:**

```python
class RLTradingAgent:
    def __init__(self, state_dim, hidden_dim, device):
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 10 actions
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=0.0003  # Learning rate
        )
    
    def train_epoch(self, batch_states, batch_actions, batch_returns, 
                    batch_advantages, old_log_probs):
        """Single PPO epoch"""
        
        # Forward pass
        policy_logits = self.policy_net(batch_states)
        values = self.value_net(batch_states).squeeze()
        probs = torch.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        
        # Compute new log probs
        new_log_probs = dist.log_prob(batch_actions)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * (values - batch_returns).pow(2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + value_loss - 0.01 * entropy
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
```

**PPO Hyperparameters:**
```python
PPOConfig(
    learning_rate=0.0003,    # Adam learning rate
    gamma=0.99,              # Discount factor (0.99 = very future-focused)
    clip_epsilon=0.2,        # PPO clipping range (0.2 = ±20%)
    entropy_coef=0.01,       # Entropy bonus for exploration
    n_epochs=10,             # Epochs per batch
    batch_size=64,           # Samples per batch
)
```

**Training Process:**

1. **Collect Experience:** Run agent in environment, collect (state, action, reward, next_state)
2. **Compute Returns:** Discounted future rewards using gamma
3. **Compute Advantages:** GAE (Generalized Advantage Estimation)
4. **Update Policy:** PPO clipped objective
5. **Update Value:** TD(λ) targets
6. **Repeat:** 10 epochs per batch

### Experience Replay Buffer with Curriculum Learning

**Class:** `ExperienceReplayBuffer`

**Purpose:** Regime-aware sampling for curriculum learning

```python
class ExperienceReplayBuffer:
    def __init__(self, capacity=10000, regime_focus_weight=0.7):
        self.buffer = deque(maxlen=capacity)
        self.regime_focus_weight = regime_focus_weight  # 70% current regime
    
    def sample(self, count, current_regime, use_regime_weighting=True):
        """Sample with regime-weighted prioritization"""
        
        if not use_regime_weighting or current_regime is None:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), count, replace=False)
        else:
            # Weighted sampling: 70% from current regime, 30% from others
            current_regime_experiences = [
                i for i, exp in enumerate(self.buffer)
                if exp["regime"] == current_regime
            ]
            
            other_regime_experiences = [
                i for i, exp in enumerate(self.buffer)
                if exp["regime"] != current_regime
            ]
            
            # Calculate counts
            n_current = int(count * self.regime_focus_weight)
            n_other = count - n_current
            
            # Sample
            if len(current_regime_experiences) > 0:
                current_samples = np.random.choice(
                    current_regime_experiences, n_current, replace=True
                )
            else:
                current_samples = []
            
            if len(other_regime_experiences) > 0:
                other_samples = np.random.choice(
                    other_regime_experiences, n_other, replace=True
                )
            else:
                other_samples = []
            
            indices = np.concatenate([current_samples, other_samples])
        
        # Stack into batches
        return {
            "states": torch.stack([self.buffer[i]["state"] for i in indices]),
            "actions": torch.stack([self.buffer[i]["action"] for i in indices]),
            "log_probs": torch.stack([self.buffer[i]["log_prob"] for i in indices]),
            "advantages": torch.stack([self.buffer[i]["advantage"] for i in indices]),
            "returns": torch.stack([self.buffer[i]["return"] for i in indices]),
        }
```

### Memory & Pattern Learning

**Class:** `MemoryStore`

**Purpose:** Persistent pattern database for similarity-based learning

```python
class MemoryStore:
    def __init__(self, dsn, embedding_dim=128):
        self.db = PostgresConnection(dsn)
        self.embedding_dim = embedding_dim
        # Table: historical_patterns(id, features_embedding, win_rate, profit_bps, regime, etc.)
    
    def store_pattern(self, features_embedding, trade_result):
        """Store pattern and outcome"""
        pattern = Pattern(
            id=uuid4(),
            features_embedding=features_embedding,
            win_rate=trade_result.get("won"),
            profit_bps=trade_result.get("profit_bps"),
            regime=trade_result.get("regime"),
            timestamp=datetime.now(),
        )
        self.db.insert("historical_patterns", pattern)
    
    def find_similar(self, features_embedding, similarity_threshold=0.7, topk=5):
        """Find similar historical patterns using cosine similarity"""
        # Query: Get patterns with cosine_similarity > 0.7
        # ORDER BY similarity DESC LIMIT 5
        similar_patterns = self.db.query("""
            SELECT *,
                   1 - (features_embedding <-> %s) as similarity
            FROM historical_patterns
            WHERE 1 - (features_embedding <-> %s) > %s
            ORDER BY similarity DESC
            LIMIT %s
        """, features_embedding, features_embedding, similarity_threshold, topk)
        
        return similar_patterns
    
    def get_pattern_win_rate(self, pattern_cluster):
        """Get win rate for pattern cluster"""
        win_rate = self.db.query("""
            SELECT AVG(CASE WHEN won THEN 1 ELSE 0 END) as win_rate
            FROM historical_patterns
            WHERE pattern_cluster = %s
        """, pattern_cluster)[0]["win_rate"]
        
        return win_rate
```

### Validation & Backtesting

**Walk-Forward Testing:**

```python
def walk_forward_validation(
    historical_data,
    train_days=60,
    test_days=10,
    min_test_trades=100
):
    """
    Walk-forward analysis:
    - Train on 60 days
    - Test on 10 days
    - Slide forward 10 days
    - Repeat until end of data
    """
    
    results = []
    start_date = historical_data[0].timestamp
    end_date = historical_data[-1].timestamp
    
    current_date = start_date
    while current_date + timedelta(days=train_days + test_days) <= end_date:
        # Split
        train_end = current_date + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        train_data = [d for d in historical_data
                     if current_date <= d.timestamp < train_end]
        test_data = [d for d in historical_data
                    if train_end <= d.timestamp < test_end]
        
        if len(test_data) < min_test_trades:
            current_date += timedelta(days=test_days)
            continue
        
        # Train
        model = train_model(train_data)
        
        # Test
        metrics = backtest(model, test_data)
        results.append({
            "train_period": (current_date, train_end),
            "test_period": (train_end, test_end),
            "metrics": metrics,
        })
        
        current_date += timedelta(days=test_days)
    
    return results
```

**Mandatory Out-of-Sample Gates:**

```python
def apply_mandatory_oos_gates(walk_forward_results):
    """
    Reject models that don't pass strict OOS criteria.
    """
    
    all_oos_metrics = [r["metrics"] for r in walk_forward_results]
    
    # Gate 1: Minimum OOS Sharpe
    oos_sharpe = np.mean([m["sharpe"] for m in all_oos_metrics])
    if oos_sharpe < 1.0:
        return False, "OOS Sharpe < 1.0"
    
    # Gate 2: Minimum OOS win rate
    oos_win_rate = np.mean([m["win_rate"] for m in all_oos_metrics])
    if oos_win_rate < 0.55:
        return False, "OOS win rate < 55%"
    
    # Gate 3: Train-test gap
    train_sharpe = np.mean([m.get("train_sharpe", 1.5) for m in all_oos_metrics])
    gap = train_sharpe - oos_sharpe
    if gap > 0.3 * train_sharpe:
        return False, "Train-test gap too large"
    
    # Gate 4: Minimum OOS trades
    total_oos_trades = sum(len(m["trades"]) for m in all_oos_metrics)
    if total_oos_trades < 500:
        return False, "Insufficient OOS trades"
    
    # Gate 5: Cross-window stability
    sharpe_std = np.std([m["sharpe"] for m in all_oos_metrics])
    if sharpe_std > 0.2:
        return False, "Sharpe unstable across windows"
    
    return True, "All gates passed"
```

---

## CONFIGURATION & SETTINGS

**Primary Config File:** `src/cloud/training/config/settings.py`

**Pydantic BaseSettings Structure:**

```python
class EngineSettings(BaseSettings):
    # Database
    postgres: PostgresSettings
    
    # Exchange
    exchange: ExchangeSettings
    
    # Training
    training: TrainingSettings
        ├── window_days: int = 150
        ├── walk_forward: WalkForwardSettings
        │   ├── train_days: int = 20
        │   ├── test_days: int = 5
        │   └── min_trades: int = 300
        ├── rl_agent: RLAgentSettings
        │   ├── learning_rate: float = 0.0003
        │   ├── gamma: float = 0.99
        │   ├── clip_epsilon: float = 0.2
        │   ├── entropy_coef: float = 0.01
        │   ├── n_epochs: int = 10
        │   ├── batch_size: int = 64
        │   ├── hidden_dim: int = 256
        │   └── state_dim: int = 80
        ├── shadow_trading: ShadowTradingSettings
        │   ├── position_size_gbp: float = 1000.0
        │   ├── stop_loss_bps: int = 50
        │   ├── take_profit_bps: int = 20
        │   └── min_confidence_threshold: float = 0.52
        ├── memory: MemorySettings
        │   ├── vector_dim: int = 128
        │   └── similarity_threshold: float = 0.7
        └── validation: ValidationSettings
            ├── mandatory_oos: MandatoryOOSSettings
            ├── overfitting_detection: OverfittingDetectionSettings
            ├── data_validation: DataValidationSettings
            └── stress_testing: StressTestingSettings
    
    # Costs
    costs: CostSettings
        ├── target_net_bps: int = 15
        ├── taker_buffer_bps: int = 9
        ├── default_fee_bps: float = 8.0
        └── volatility_slippage_multiplier: float = 0.25
    
    # Cloud
    s3: S3Settings
    ray: RaySettings
    
    # Monitoring
    notifications: NotificationSettings
    dropbox: DropboxSettings
    universe: UniverseSettings
```

**Environment Variable Overrides:**
```bash
# Exchange
export EXCHANGE_PRIMARY=binance
export EXCHANGE_API_KEY=...
export EXCHANGE_API_SECRET=...

# Database
export DATABASE_DSN=postgresql://user:pass@host/db

# Dropbox
export DROPBOX_ACCESS_TOKEN=sl.u.xxx

# Telegram
export TELEGRAM_BOT_TOKEN=8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0
export TELEGRAM_CHAT_ID=-123456789

# Ray
export RAY_ADDRESS=ray://127.0.0.1:10001

# S3
export S3_BUCKET=huracan-engine
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

---

## DATA FLOW & PROCESSING

### End-to-End Data Flow

```
1. DAILY STARTUP (02:00 UTC)
   ├── Load configuration
   ├── Initialize services
   └── Create dated Dropbox folder
                │
2. UNIVERSE SELECTION
   ├── Query exchange for top 20 coins by liquidity
   ├── Filter by spread, ADV, trading volume
   └── Return selected symbols list [BTC, ETH, ...]
                │
3. HISTORICAL DATA LOADING (per symbol)
   ├── CandleDataLoader.get_candles(symbol, lookback_days=150)
   │   ├── Query local cache (data/candles/{symbol}.parquet)
   │   ├── If missing, download from exchange (Binance, Kraken, etc.)
   │   ├── Validate data (missing candles, outliers, gaps)
   │   └── Store in local cache and PostgreSQL
   │
   ├── Data Structure (one row per candle):
   │   ├── timestamp (UTC)
   │   ├── open, high, low, close (USDT)
   │   ├── volume (base asset)
   │   ├── quote_volume (USDT)
   │   └── trades (tick count)
   │
   └── Output: Polars DataFrame (150 days x 8 cols)
                │
4. FEATURE ENGINEERING
   ├── FeatureRecipe.calculate_features(candle_df)
   │
   ├── Technical Indicators (30+ features):
   │   ├── trend_strength (normalized EMA differential)
   │   ├── adx (trend strength 0-100)
   │   ├── rsi_14 (momentum 0-100)
   │   ├── atr (volatility)
   │   ├── bb_upper, bb_lower, bb_width (volatility bands)
   │   ├── sma_50, sma_200 (moving averages)
   │   ├── ema_12, ema_26, macd, signal, histogram
   │   ├── volatility_regime (short / long vol ratio)
   │   └── compression_score (range tightness)
   │
   ├── Microstructure Features:
   │   ├── micro_score (overall quality)
   │   ├── uptick_ratio (buy vs sell volume)
   │   ├── spread_bps (bid-ask)
   │   ├── vol_jump_z (volume z-score)
   │   └── price_position_in_range (0-1)
   │
   ├── Pattern Features:
   │   ├── ignition_score (breakout strength)
   │   ├── breakout_quality
   │   ├── pullback_depth
   │   ├── mean_revert_bias
   │   ├── kurtosis (fat tails)
   │   └── skewness
   │
   └── Output: Polars DataFrame (150 days x 80-100 cols)
                │
5. REGIME DETECTION
   ├── RegimeDetector.detect_regime(features_df)
   │
   ├── Calculate regime features:
   │   ├── atr_pct, volatility_ratio, volatility_percentile
   │   ├── adx, trend_strength, ema_slope, momentum_slope
   │   ├── kurtosis, skewness
   │   └── compression_score, bb_width_pct
   │
   ├── Score each regime:
   │   ├── trend_score = -0.15 + 1.8*trend_str + 0.035*adx + ...
   │   ├── range_score = 0.1 + 1.2*compression + ...
   │   └── panic_score = -1.0 + 1.4*vol_ratio + ...
   │
   └── Output: MarketRegime + confidence score (0-1)
                │
6. SHADOW TRADING (simulate every possible trade)
   ├── For each candle i (except last 10):
   │   ├── Generate signal from 23 engines
   │   ├── If signal confidence > 0.52:
   │   │   ├── Execute shadow trade at candle[i].close
   │   │   ├── Track unrealized PnL until exit
   │   │   ├── Exit when: stop hit, TP hit, or max hold
   │   │   ├── Record trade: (entry_time, exit_time, pnl_bps)
   │   │   └── Add to memory store with features
   │   │
   │   └── → Output: Trade list [trade1, trade2, ...]
                │
7. ANALYTICS & LEARNING
   ├── WinAnalyzer: Analyze winning trades
   │   ├── Common features of winners
   │   ├── Best regimes
   │   ├── Best engine combinations
   │   └── Average hold time, profit distribution
   │
   ├── LossAnalyzer: Understand failures
   │   ├── Common setup errors
   │   ├── False signals
   │   ├── Bad regimes to trade in
   │   └── Correlation with macro events
   │
   ├── PatternMatcher: Store & retrieve patterns
   │   ├── Embed features into 128-dim vector
   │   ├── Store in PostgreSQL with outcomes
   │   ├── Retrieve similar patterns for future trades
   │   └── Win rate for each pattern cluster
   │
   └── PostExitTracker: What happens after exit?
       ├── Track price 5min, 30min, 2h after exit
       ├── Calculate "opportunity cost" if trade was exited early
       └── Feedback to improve exits
                │
8. RL AGENT TRAINING
   ├── RLTradingAgent.train_on_symbol()
   │
   ├── Build training batches from:
   │   ├── Recent trades (from replay buffer)
   │   ├── Regime-weighted sampling (70% current, 30% other)
   │   └── Pattern-similar trades (curriculum learning)
   │
   ├── Compute advantages & returns:
   │   ├── Advantage = Q(s,a) - V(s)
   │   ├── Return = discounted future reward
   │   └── Use GAE (Generalized Advantage Estimation)
   │
   ├── PPO update (n_epochs=10):
   │   ├── Forward: features → policy net → action probs
   │   ├── Compute policy loss (PPO clipped objective)
   │   ├── Compute value loss (TD error)
   │   ├── Backward: gradients
   │   ├── Optimizer step (Adam)
   │   └── Repeat 10 times
   │
   └── Output: Trained policy network + value network
                │
9. WALK-FORWARD VALIDATION
   ├── For each historical window:
   │   ├── Train on days 1-60
   │   ├── Test on days 60-70 (out-of-sample)
   │   ├── Compute metrics (sharpe, win_rate, profit_factor)
   │   └── Slide forward 10 days
   │
   ├── Aggregate OOS metrics:
   │   ├── Average OOS Sharpe
   │   ├── Average OOS win rate
   │   ├── Sharpe consistency across windows
   │   └── Train-test gap
   │
   └── Apply gates:
       ├── Gate 1: OOS Sharpe >= 1.0
       ├── Gate 2: OOS win rate >= 55%
       ├── Gate 3: Train-test gap <= 30%
       ├── Gate 4: Min 500 OOS trades
       └── Gate 5: Sharpe std <= 0.2
                │
10. PORTFOLIO OPTIMIZATION
    ├── PortfolioOptimizer.optimize(signals, correlations)
    │
    ├── Input: 20 signals (one per symbol)
    │   ├── Each signal: direction, confidence, expected_return
    │   └── Correlation matrix between symbols
    │
    ├── Solve quadratic program:
    │   ├── Maximize: Expected return - λ * volatility
    │   ├── Subject to:
    │   │   ├── Sum of weights = 1.0
    │   │   ├── Each weight in [0, 0.1] (no short, max 10%)
    │   │   ├── Correlation concentration <= 0.7
    │   │   └── Single position max 5% of portfolio
    │   └──Output: Allocation weights [0.05, 0.03, ..., 0.02]
    │
    └── DynamicPositionSizer.calculate_position_size()
        ├── For each symbol:
        │   ├── Kelly fraction = (win_rate * win_size - loss_rate * loss_size) / win_size
        │   ├── Volatility adjustment = target_vol / realized_vol
        │   ├── Confidence multiplier = (confidence - 0.50) * 10
        │   ├── Regime adjustment (trend: 1.2x, range: 1.0x, panic: 0.6x)
        │   ├── Combine: base * kelly * vol_adj * conf_mult * regime_factor
        │   └── Cap at 5% per position
        │
        └── Output: Position size in GBP for each symbol
                │
11. MODEL PERSISTENCE & REGISTRATION
    ├── Save model artifacts:
    │   ├── policy_net.pth (PyTorch model weights)
    │   ├── value_net.pth (value network)
    │   ├── features_metadata.json (feature names, scaling)
    │   ├── model_metrics.json (performance stats)
    │   └── hyperparameters.yaml (config used)
    │
    ├── Register in PostgreSQL:
    │   ├── INSERT INTO models (id, symbol, created_at, metrics, ...)
    │   ├── INSERT INTO model_versions (...)
    │   └── Mark as "PUBLISHED" or "SHADOW"
    │
    ├── Publish to S3:
    │   ├── s3://huracan-engine/baselines/{symbol}/{date}/model.tar.gz
    │   └── Also upload: metrics, features, hyperparams
    │
    └── Export to Dropbox:
        ├── /Runpodhuracan/{YYYY-MM-DD}/models/
        ├── /Runpodhuracan/data/candles/ (shared, persistent)
        └── /Runpodhuracan/{YYYY-MM-DD}/exports/
                │
12. MONITORING & NOTIFICATIONS
    ├── ComprehensiveTelegramMonitor:
    │   ├── System startup notification
    │   ├── Pre-training health check
    │   ├── Post-training health check
    │   ├── Error notifications
    │   └── System shutdown notification
    │
    ├── HealthMonitor:
    │   ├── Check exchange connectivity
    │   ├── Check database connectivity
    │   ├── Check storage (S3, Dropbox)
    │   ├── Monitor system resources
    │   └── Alert on anomalies
    │
    ├── LearningTracker:
    │   ├── Log all patterns learned
    │   ├── Track engine performance
    │   ├── Record feature importance
    │   └── Archive to Dropbox
    │
    └── DropboxSync (running in background):
        ├── Sync logs every 5 minutes
        ├── Sync models every 30 minutes
        ├── Sync learning data every 10 minutes
        ├── Sync historical data every 1 hour
        └── Restore data cache on startup
                │
13. COMPLETION
    ├── Export logs
    ├── Final Dropbox sync
    ├── Send completion notification
    └── Exit cleanly
```

---

## DEPLOYMENT & INFRASTRUCTURE

### Supported Deployment Environments

**1. RunPod GPU Instance**
- GPU: RTX 4090 (24GB VRAM)
- CPU: 32-core
- RAM: 256GB
- Network: 10Gbps

**2. Local Development**
- CPU-only training (slow)
- Single symbol at a time

**3. Cloud (AWS/GCP)**
- Ray cluster for distributed training
- S3 for artifact storage
- RDS PostgreSQL

**4. RunPod Interruptible Instances**
- Lower cost, preemptible
- Checkpoint/resume capability

### Deployment Checklist

```
Pre-Deployment:
□ Set all required environment variables
□ Configure PostgreSQL DSN
□ Create S3 bucket and IAM credentials
□ Set up Telegram bot and get chat ID
□ Generate Dropbox access token
□ Configure exchange API credentials
□ Download historical data (24+ hours for candles)
□ Run health checks (all services must be accessible)

Deployment:
□ Start daily schedule (APScheduler)
□ Monitor first run with Telegram alerts
□ Verify models being saved to S3
□ Confirm Dropbox sync is working
□ Check database registrations
□ Review learning metrics in Telegram

Post-Deployment:
□ Monitor performance continuously
□ Set up alerts for failures
□ Regular backups of PostgreSQL
□ Archive old models to cold storage
```

---

## CONCLUSION

The Huracan Trading Engine is a production-grade system designed for institutional-quality algorithmic trading. It combines:

- **Machine Learning:** PPO reinforcement learning with curriculum learning
- **Domain Expertise:** 23 specialized alpha engines capturing different market inefficiencies
- **Risk Management:** Portfolio optimization, position sizing, regime-aware stops
- **Validation:** Walk-forward testing, mandatory out-of-sample gates, stress testing
- **Monitoring:** Real-time health checks, Telegram alerts, comprehensive logging
- **Scalability:** Ray distributed training, parallel engine execution

The system trains daily to adapt to changing market conditions while maintaining strict validation standards to prevent overfitting. It prioritizes Sharpe ratio, win rate stability, and real-world cost awareness.

---

## APPENDIX: Key Metrics & Benchmarks

### Target Performance Metrics
- **Out-of-Sample Sharpe:** >= 1.0
- **Win Rate:** >= 55%
- **Profit Factor:** >= 2.0
- **Max Drawdown:** <= 15%
- **Trades per day:** 5-50 (depending on regime)
- **Average hold time:** 30 min - 4 hours

### Computational Requirements
- **Training time:** 30-60 min per symbol (RTX 4090)
- **Feature calculation:** <100ms per candle
- **Signal generation:** <50ms (all 23 engines in parallel)
- **Daily full run:** 2-4 hours (20 symbols)

### Database Storage
- **Historical candles:** ~2GB per 100 symbols per year
- **Trade records:** ~100MB per symbol per year
- **Pattern embeddings:** ~50MB per symbol per year
- **Total: ~200GB for 5 years of data on 20 symbols

