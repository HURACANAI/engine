-- Memory database schema for self-learning trade analysis
-- Simplified version without pgvector (can add later)

-- Core trade memory: every historical and live trade
CREATE TABLE IF NOT EXISTS trade_memory (
    trade_id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,

    -- Entry data
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_features JSONB NOT NULL,  -- All 50+ features at entry
    entry_embedding_json JSONB,     -- Store as JSON array for now

    -- Position details
    position_size_gbp DECIMAL(12, 2) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('LONG', 'SHORT')),

    -- Exit data
    exit_timestamp TIMESTAMP WITH TIME ZONE,
    exit_price DECIMAL(20, 8),
    exit_reason VARCHAR(50),  -- 'TAKE_PROFIT', 'STOP_LOSS', 'TIMEOUT', 'MODEL_SIGNAL'
    hold_duration_minutes INTEGER,

    -- Performance
    gross_profit_bps DECIMAL(10, 2),
    net_profit_gbp DECIMAL(12, 2),
    fees_gbp DECIMAL(12, 2),
    slippage_bps DECIMAL(10, 2),

    -- Market context
    market_regime VARCHAR(20),  -- 'LOW_VOL', 'HIGH_VOL', 'TRENDING', 'RANGING'
    volatility_bps DECIMAL(10, 2),
    spread_at_entry_bps DECIMAL(10, 2),

    -- Outcome classification
    is_winner BOOLEAN,
    win_quality VARCHAR(20),  -- 'OPTIMAL', 'EARLY_EXIT', 'LUCKY', 'MARGINAL'

    -- Model metadata
    model_version VARCHAR(50),
    model_confidence DECIMAL(5, 4),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Post-exit analysis: what happened AFTER we exited
CREATE TABLE IF NOT EXISTS post_exit_tracking (
    tracking_id BIGSERIAL PRIMARY KEY,
    trade_id BIGINT REFERENCES trade_memory(trade_id),

    -- Track price movement after exit
    price_1h_later DECIMAL(20, 8),
    price_4h_later DECIMAL(20, 8),
    price_24h_later DECIMAL(20, 8),

    -- Missed opportunity analysis
    max_price_reached DECIMAL(20, 8),
    max_price_time_minutes INTEGER,
    min_price_reached DECIMAL(20, 8),
    min_price_time_minutes INTEGER,

    -- Counterfactual profits
    missed_profit_bps DECIMAL(10, 2),  -- If we held to optimal exit
    optimal_exit_time_minutes INTEGER,

    -- Learning insights
    should_have_held_longer BOOLEAN,
    should_have_exited_earlier BOOLEAN,
    insight_summary TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Win analysis: deep dive on successful trades
CREATE TABLE IF NOT EXISTS win_analysis (
    analysis_id BIGSERIAL PRIMARY KEY,
    trade_id BIGINT REFERENCES trade_memory(trade_id),

    -- What drove the win?
    top_contributing_features JSONB,  -- {feature_name: importance_score}
    pattern_frequency INTEGER,  -- How often we've seen this pattern
    historical_win_rate DECIMAL(5, 4),  -- Win rate for similar patterns

    -- Quality assessment
    statistical_significance DECIMAL(5, 4),  -- p-value
    skill_vs_luck_score DECIMAL(5, 4),  -- 0=luck, 1=skill

    -- Execution quality
    entry_quality VARCHAR(20),  -- 'EXCELLENT', 'GOOD', 'FAIR'
    exit_quality VARCHAR(20),

    -- Reproducibility
    pattern_strength DECIMAL(5, 4),
    confidence_for_future DECIMAL(5, 4),

    insights TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Loss analysis: root cause analysis for failures
CREATE TABLE IF NOT EXISTS loss_analysis (
    analysis_id BIGSERIAL PRIMARY KEY,
    trade_id BIGINT REFERENCES trade_memory(trade_id),

    -- Root causes
    primary_failure_reason VARCHAR(100),
    misleading_features JSONB,  -- Features that gave false signal

    -- What went wrong?
    regime_changed BOOLEAN,
    stop_too_tight BOOLEAN,
    stop_too_wide BOOLEAN,
    adverse_selection BOOLEAN,  -- Got filled at bad price
    insufficient_confirmation BOOLEAN,
    news_event_impact BOOLEAN,

    -- Severity
    loss_severity VARCHAR(20),  -- 'MINOR', 'MODERATE', 'SEVERE'
    preventable BOOLEAN,

    -- Lessons learned
    corrective_action TEXT,
    pattern_to_avoid JSONB,
    confidence_penalty DECIMAL(5, 4),

    insights TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pattern library: clusters of similar setups with outcomes
CREATE TABLE IF NOT EXISTS pattern_library (
    pattern_id BIGSERIAL PRIMARY KEY,
    pattern_name VARCHAR(100),
    pattern_embedding_json JSONB,  -- Store as JSON array for now

    -- Pattern characteristics
    feature_signature JSONB,
    market_regime VARCHAR(20),

    -- Historical performance
    total_occurrences INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4),
    avg_profit_bps DECIMAL(10, 2),
    avg_hold_minutes INTEGER,

    -- Statistical metrics
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown_bps DECIMAL(10, 2),
    profit_factor DECIMAL(10, 4),

    -- Confidence scoring
    reliability_score DECIMAL(5, 4),  -- How consistent is this pattern?
    sample_size_adequate BOOLEAN,

    -- Optimal parameters learned
    optimal_position_size_multiplier DECIMAL(5, 2),
    optimal_exit_threshold_bps DECIMAL(10, 2),

    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model performance tracking over time
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,

    -- Daily metrics
    trades_total INTEGER,
    trades_won INTEGER,
    trades_lost INTEGER,
    win_rate DECIMAL(5, 4),

    -- P&L
    total_profit_gbp DECIMAL(12, 2),
    avg_profit_per_trade_gbp DECIMAL(12, 2),
    largest_win_gbp DECIMAL(12, 2),
    largest_loss_gbp DECIMAL(12, 2),

    -- Risk metrics
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    max_drawdown_gbp DECIMAL(12, 2),

    -- Learning metrics
    patterns_learned INTEGER,
    insights_generated INTEGER,
    strategy_updates INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_version, evaluation_date)
);

-- Indices for fast queries
CREATE INDEX IF NOT EXISTS idx_trade_memory_symbol ON trade_memory(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_memory_entry_ts ON trade_memory(entry_timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_memory_winner ON trade_memory(is_winner);
CREATE INDEX IF NOT EXISTS idx_trade_memory_regime ON trade_memory(market_regime);

CREATE INDEX IF NOT EXISTS idx_pattern_library_win_rate ON pattern_library(win_rate DESC);

CREATE INDEX IF NOT EXISTS idx_post_exit_trade ON post_exit_tracking(trade_id);
CREATE INDEX IF NOT EXISTS idx_win_analysis_trade ON win_analysis(trade_id);
CREATE INDEX IF NOT EXISTS idx_loss_analysis_trade ON loss_analysis(trade_id);
