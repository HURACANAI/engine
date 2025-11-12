-- MODEL REGISTRY SCHEMA
-- Schema for ModelRegistry service in src/cloud/training/services/model_registry.py
-- Version: 1.0.0

-- Enable UUID extension (if using UUIDs)
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Models table: Stores model metadata and configuration
CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    kind VARCHAR(50) NOT NULL,  -- 'baseline' or 'candidate'
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    s3_path TEXT,
    params JSONB,
    features JSONB,
    notes TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model metrics table: Stores performance metrics for each model
-- Drop existing table if it has wrong schema (from Brain Library)
DROP TABLE IF EXISTS model_metrics CASCADE;

CREATE TABLE model_metrics (
    model_id VARCHAR(255) PRIMARY KEY,
    sharpe REAL,
    profit_factor REAL,
    hit_rate REAL,
    max_dd_bps REAL,
    pnl_bps REAL,
    trades_oos INTEGER,
    turnover REAL,
    fee_bps REAL,
    spread_bps REAL,
    slippage_bps REAL,
    total_costs_bps REAL,
    validation_window TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Publish log table: Tracks publish/reject events for models
CREATE TABLE IF NOT EXISTS publish_log (
    model_id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    published BOOLEAN NOT NULL,
    reason TEXT,
    at TIMESTAMP WITH TIME ZONE NOT NULL,
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Training runs table: Tracks training run metadata and results
CREATE TABLE IF NOT EXISTS training_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50),  -- 'success', 'failed', 'pending'
    metrics JSONB,
    model_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol);
CREATE INDEX IF NOT EXISTS idx_models_kind ON models(kind);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_symbol_created ON models(symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_sharpe ON model_metrics(sharpe DESC);

CREATE INDEX IF NOT EXISTS idx_publish_log_symbol ON publish_log(symbol);
CREATE INDEX IF NOT EXISTS idx_publish_log_published ON publish_log(published);
CREATE INDEX IF NOT EXISTS idx_publish_log_at ON publish_log(at DESC);

CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_start_time ON training_runs(start_time DESC);

-- Updated timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER model_metrics_updated_at
    BEFORE UPDATE ON model_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Comments
COMMENT ON TABLE models IS 'Model metadata and configuration storage';
COMMENT ON TABLE model_metrics IS 'Performance metrics for each trained model';
COMMENT ON TABLE publish_log IS 'Publish/reject event log for models';
COMMENT ON TABLE training_runs IS 'Training run metadata and results tracking';

COMMENT ON COLUMN models.model_id IS 'Unique model identifier (UUID)';
COMMENT ON COLUMN models.kind IS 'Model kind: baseline (published) or candidate (rejected)';
COMMENT ON COLUMN models.params IS 'Model hyperparameters (JSON)';
COMMENT ON COLUMN models.features IS 'Feature metadata (JSON)';
COMMENT ON COLUMN model_metrics.sharpe IS 'Sharpe ratio (risk-adjusted returns)';
COMMENT ON COLUMN model_metrics.profit_factor IS 'Profit factor (gross profit / gross loss)';
COMMENT ON COLUMN model_metrics.hit_rate IS 'Hit rate (win rate) as decimal (0.0-1.0)';
COMMENT ON COLUMN model_metrics.max_dd_bps IS 'Maximum drawdown in basis points';
COMMENT ON COLUMN model_metrics.pnl_bps IS 'Profit and loss in basis points';
COMMENT ON COLUMN model_metrics.trades_oos IS 'Number of out-of-sample trades';
COMMENT ON COLUMN model_metrics.validation_window IS 'Validation window description (e.g., "2025-06-15 to 2025-11-12")';

