-- UNIFIED MODEL REGISTRY SCHEMA
-- Single source of truth for all model tracking
-- Version: 1.0.0

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Model status enum
CREATE TYPE model_status AS ENUM ('publish', 'shadow', 'reject', 'pending');

-- Main models table
CREATE TABLE IF NOT EXISTS unified_models (
    -- Primary identification
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Lineage and dependencies
    feature_set_id VARCHAR(100),  -- Links to feature_store
    run_manifest_id VARCHAR(100), -- Links to run_manifests
    parent_model_id UUID REFERENCES unified_models(model_id),  -- Model genealogy

    -- Gate evaluation results
    gate_verdict JSONB,  -- Full gate evaluation results
    meta_weight FLOAT CHECK (meta_weight >= 0 AND meta_weight <= 1),
    publish_status model_status DEFAULT 'pending',

    -- Performance metrics (denormalized for fast queries)
    sharpe_ratio FLOAT,
    max_drawdown_pct FLOAT,
    brier_score FLOAT,
    win_rate FLOAT,
    trades_oos INTEGER,

    -- Artifacts
    s3_artifacts_uri TEXT,
    onnx_path TEXT,
    metadata_path TEXT,
    model_params JSONB,
    feature_metadata JSONB,

    -- Lifecycle
    published_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    deprecated_reason TEXT,

    -- Unique constraint
    UNIQUE(symbol, version)
);

-- Feature sets table (linked to feature store)
CREATE TABLE IF NOT EXISTS feature_sets (
    feature_set_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    feature_names TEXT[],
    feature_versions JSONB,  -- {feature_name: version}
    feature_checksums JSONB,  -- For reproducibility
    deprecated BOOLEAN DEFAULT FALSE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    deprecated_reason TEXT
);

-- Run manifests table (reproducibility tracking)
CREATE TABLE IF NOT EXISTS run_manifests (
    run_manifest_id VARCHAR(100) PRIMARY KEY,
    run_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Git info
    git_commit VARCHAR(40),
    git_branch VARCHAR(100),
    git_dirty BOOLEAN,

    -- Configuration
    settings_hash VARCHAR(64),  -- SHA256 of settings.yaml
    settings_snapshot JSONB,     -- Full settings for this run

    -- Data
    dataset_checksums JSONB,     -- {symbol: sha256}
    symbol_list TEXT[],

    -- Environment
    python_version VARCHAR(20),
    package_versions JSONB,      -- {package: version}
    ray_cluster_info JSONB,

    -- Results summary
    models_trained INTEGER,
    models_published INTEGER,
    models_shadowed INTEGER,
    models_rejected INTEGER,

    -- Full manifest
    manifest_data JSONB           -- Complete manifest
);

-- Model performance history (time series)
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID REFERENCES unified_models(model_id) ON DELETE CASCADE,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Performance in Hamilton (live)
    live_trades INTEGER,
    live_pnl_gbp FLOAT,
    live_sharpe FLOAT,
    live_win_rate FLOAT,

    -- Execution quality
    avg_slippage_bps FLOAT,
    avg_timing_ms FLOAT,
    fills_partial INTEGER,
    fills_complete INTEGER,

    -- Regime breakdown
    regime VARCHAR(50),
    regime_pnl_gbp FLOAT,
    regime_trades INTEGER
);

-- Gate evaluation history
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID REFERENCES unified_models(model_id) ON DELETE CASCADE,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    gate_config_version VARCHAR(20),
    passed_gates TEXT[],
    failed_gates TEXT[],
    warnings TEXT[],

    final_verdict model_status,
    meta_weight FLOAT,

    -- Full results
    evaluation_data JSONB
);

-- Operator actions log
CREATE TABLE IF NOT EXISTS operator_actions (
    id BIGSERIAL PRIMARY KEY,
    action_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    operator VARCHAR(100),
    action_type VARCHAR(50),  -- blocklist, shadow, retrain, cooldown, etc.

    target_type VARCHAR(20),  -- model, symbol, engine
    target_id TEXT,

    reason TEXT NOT NULL,
    parameters JSONB,

    -- Reversal tracking
    reversed BOOLEAN DEFAULT FALSE,
    reversed_at TIMESTAMP WITH TIME ZONE,
    reversed_by VARCHAR(100),
    reversal_reason TEXT
);

-- Indexes for fast queries
CREATE INDEX idx_models_symbol ON unified_models(symbol);
CREATE INDEX idx_models_status ON unified_models(publish_status);
CREATE INDEX idx_models_created ON unified_models(created_at DESC);
CREATE INDEX idx_models_version ON unified_models(symbol, version);
CREATE INDEX idx_models_feature_set ON unified_models(feature_set_id);
CREATE INDEX idx_models_run_manifest ON unified_models(run_manifest_id);

CREATE INDEX idx_performance_model ON model_performance(model_id, recorded_at DESC);
CREATE INDEX idx_performance_regime ON model_performance(regime);

CREATE INDEX idx_gate_evals_model ON gate_evaluations(model_id, evaluated_at DESC);

CREATE INDEX idx_operator_actions_time ON operator_actions(action_at DESC);
CREATE INDEX idx_operator_actions_target ON operator_actions(target_type, target_id);

-- Views for common queries

-- Active publishable models
CREATE OR REPLACE VIEW v_publishable_models AS
SELECT
    m.*,
    f.feature_names,
    rm.git_commit,
    rm.run_date
FROM unified_models m
LEFT JOIN feature_sets f ON m.feature_set_id = f.feature_set_id
LEFT JOIN run_manifests rm ON m.run_manifest_id = rm.run_manifest_id
WHERE m.publish_status = 'publish'
  AND m.deprecated_at IS NULL
ORDER BY m.created_at DESC;

-- Shadow models for monitoring
CREATE OR REPLACE VIEW v_shadow_models AS
SELECT
    m.*,
    f.feature_names,
    rm.git_commit
FROM unified_models m
LEFT JOIN feature_sets f ON m.feature_set_id = f.feature_set_id
LEFT JOIN run_manifests rm ON m.run_manifest_id = rm.run_manifest_id
WHERE m.publish_status = 'shadow'
  AND m.deprecated_at IS NULL
ORDER BY m.created_at DESC;

-- Model lineage (parents and children)
CREATE OR REPLACE VIEW v_model_lineage AS
WITH RECURSIVE lineage AS (
    -- Base case: models with no parents
    SELECT
        model_id,
        symbol,
        version,
        parent_model_id,
        0 AS generation,
        ARRAY[model_id] AS lineage_path
    FROM unified_models
    WHERE parent_model_id IS NULL

    UNION ALL

    -- Recursive case: children
    SELECT
        m.model_id,
        m.symbol,
        m.version,
        m.parent_model_id,
        l.generation + 1,
        l.lineage_path || m.model_id
    FROM unified_models m
    JOIN lineage l ON m.parent_model_id = l.model_id
)
SELECT * FROM lineage;

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER unified_models_updated_at
    BEFORE UPDATE ON unified_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Comments
COMMENT ON TABLE unified_models IS 'Single source of truth for all model tracking and versioning';
COMMENT ON TABLE feature_sets IS 'Feature set versioning with checksums for reproducibility';
COMMENT ON TABLE run_manifests IS 'Complete reproducibility tracking for every training run';
COMMENT ON TABLE model_performance IS 'Time series of model performance in production';
COMMENT ON TABLE gate_evaluations IS 'History of all gate evaluations';
COMMENT ON TABLE operator_actions IS 'Audit log of all manual operator interventions';
