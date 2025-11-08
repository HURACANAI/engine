-- Model Lineage & Rollback Schema
--
-- Tracks model ancestry and enables safe rollback.

-- Model Lineage Table
-- Tracks parent-child relationships and changes between model versions
CREATE TABLE IF NOT EXISTS model_lineage (
    lineage_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES unified_models(model_id),
    parent_model_id TEXT REFERENCES unified_models(model_id),
    symbol TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Change tracking
    change_type TEXT NOT NULL CHECK (change_type IN (
        'initial',
        'hyperparameter_tuning',
        'architecture_change',
        'feature_update',
        'data_expansion',
        'training_method',
        'calibration',
        'bugfix',
        'experiment'
    )),
    changes JSONB,  -- What changed from parent
    reason TEXT,    -- Why this change was made

    -- Performance comparison
    parent_sharpe REAL,
    current_sharpe REAL,
    sharpe_improvement REAL,

    -- Metadata
    created_by TEXT,
    tags TEXT[],

    -- Indexes
    UNIQUE(model_id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_lineage_model_id ON model_lineage(model_id);
CREATE INDEX IF NOT EXISTS idx_lineage_parent_model_id ON model_lineage(parent_model_id);
CREATE INDEX IF NOT EXISTS idx_lineage_symbol ON model_lineage(symbol);
CREATE INDEX IF NOT EXISTS idx_lineage_created_at ON model_lineage(created_at DESC);


-- Rollback History Table
-- Records all model rollbacks for audit and analysis
CREATE TABLE IF NOT EXISTS rollback_history (
    rollback_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Models
    from_model_id TEXT NOT NULL REFERENCES unified_models(model_id),
    to_model_id TEXT NOT NULL REFERENCES unified_models(model_id),
    symbol TEXT NOT NULL,

    -- Context
    reason TEXT NOT NULL,
    triggered_by TEXT NOT NULL CHECK (triggered_by IN ('operator', 'automated')),

    -- Performance (if available)
    from_model_live_sharpe REAL,
    to_model_live_sharpe REAL,

    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verification_time TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rollback_symbol ON rollback_history(symbol);
CREATE INDEX IF NOT EXISTS idx_rollback_timestamp ON rollback_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rollback_from_model ON rollback_history(from_model_id);
CREATE INDEX IF NOT EXISTS idx_rollback_to_model ON rollback_history(to_model_id);


-- View: Model Lineage Tree
-- Provides hierarchical view of model ancestry
CREATE OR REPLACE VIEW vw_model_lineage_tree AS
WITH RECURSIVE lineage_tree AS (
    -- Base case: models with no parent (roots)
    SELECT
        lineage_id,
        model_id,
        parent_model_id,
        symbol,
        created_at,
        change_type,
        sharpe_improvement,
        0 AS depth,
        model_id::TEXT AS path
    FROM model_lineage
    WHERE parent_model_id IS NULL

    UNION ALL

    -- Recursive case: children
    SELECT
        ml.lineage_id,
        ml.model_id,
        ml.parent_model_id,
        ml.symbol,
        ml.created_at,
        ml.change_type,
        ml.sharpe_improvement,
        lt.depth + 1,
        lt.path || ' → ' || ml.model_id
    FROM model_lineage ml
    INNER JOIN lineage_tree lt ON ml.parent_model_id = lt.model_id
    WHERE lt.depth < 20  -- Prevent infinite loops
)
SELECT * FROM lineage_tree
ORDER BY symbol, created_at;


-- View: Recent Rollbacks
-- Shows recent rollback activity for monitoring
CREATE OR REPLACE VIEW vw_recent_rollbacks AS
SELECT
    rollback_id,
    timestamp,
    symbol,
    from_model_id,
    to_model_id,
    reason,
    triggered_by,
    verified,
    verification_time,
    from_model_live_sharpe,
    to_model_live_sharpe,
    CASE
        WHEN verified THEN '✅ Verified'
        ELSE '⏳ Pending'
    END AS status
FROM rollback_history
ORDER BY timestamp DESC
LIMIT 100;


-- View: Lineage Performance Summary
-- Summarizes performance across model lineages
CREATE OR REPLACE VIEW vw_lineage_performance AS
SELECT
    symbol,
    model_id,
    parent_model_id,
    change_type,
    current_sharpe,
    sharpe_improvement,
    created_at,
    RANK() OVER (
        PARTITION BY symbol
        ORDER BY current_sharpe DESC NULLS LAST
    ) AS sharpe_rank
FROM model_lineage
WHERE current_sharpe IS NOT NULL
ORDER BY symbol, sharpe_rank;


-- Function: Get Model Ancestors
-- Returns all ancestor model IDs for a given model
CREATE OR REPLACE FUNCTION get_model_ancestors(
    p_model_id TEXT,
    p_max_depth INTEGER DEFAULT 10
)
RETURNS TABLE (
    ancestor_model_id TEXT,
    depth INTEGER,
    change_type TEXT,
    sharpe_improvement REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE ancestors AS (
        -- Base: Direct parent
        SELECT
            parent_model_id,
            1 AS depth,
            change_type,
            sharpe_improvement
        FROM model_lineage
        WHERE model_id = p_model_id
        AND parent_model_id IS NOT NULL

        UNION ALL

        -- Recursive: Parent's parent
        SELECT
            ml.parent_model_id,
            a.depth + 1,
            ml.change_type,
            ml.sharpe_improvement
        FROM model_lineage ml
        INNER JOIN ancestors a ON ml.model_id = a.ancestor_model_id
        WHERE ml.parent_model_id IS NOT NULL
        AND a.depth < p_max_depth
    )
    SELECT
        ancestor_model_id,
        depth,
        change_type,
        sharpe_improvement
    FROM ancestors
    ORDER BY depth;
END;
$$ LANGUAGE plpgsql;


-- Function: Get Model Descendants
-- Returns all descendant model IDs (children, grandchildren, etc.)
CREATE OR REPLACE FUNCTION get_model_descendants(
    p_model_id TEXT,
    p_max_depth INTEGER DEFAULT 10
)
RETURNS TABLE (
    descendant_model_id TEXT,
    depth INTEGER,
    change_type TEXT,
    sharpe_improvement REAL
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE descendants AS (
        -- Base: Direct children
        SELECT
            model_id,
            1 AS depth,
            change_type,
            sharpe_improvement
        FROM model_lineage
        WHERE parent_model_id = p_model_id

        UNION ALL

        -- Recursive: Children's children
        SELECT
            ml.model_id,
            d.depth + 1,
            ml.change_type,
            ml.sharpe_improvement
        FROM model_lineage ml
        INNER JOIN descendants d ON ml.parent_model_id = d.descendant_model_id
        WHERE d.depth < p_max_depth
    )
    SELECT
        descendant_model_id,
        depth,
        change_type,
        sharpe_improvement
    FROM descendants
    ORDER BY depth;
END;
$$ LANGUAGE plpgsql;


-- Function: Find Best Ancestor
-- Returns best performing ancestor for rollback
CREATE OR REPLACE FUNCTION find_best_ancestor(
    p_model_id TEXT,
    p_max_depth INTEGER DEFAULT 5,
    p_min_sharpe REAL DEFAULT 0.5
)
RETURNS TEXT AS $$
DECLARE
    v_best_model_id TEXT;
BEGIN
    SELECT ancestor_model_id INTO v_best_model_id
    FROM get_model_ancestors(p_model_id, p_max_depth) a
    INNER JOIN model_lineage ml ON ml.model_id = a.ancestor_model_id
    WHERE ml.current_sharpe IS NOT NULL
    AND ml.current_sharpe >= p_min_sharpe
    ORDER BY ml.current_sharpe DESC
    LIMIT 1;

    RETURN v_best_model_id;
END;
$$ LANGUAGE plpgsql;


-- Comments
COMMENT ON TABLE model_lineage IS 'Tracks model ancestry and evolution over time';
COMMENT ON TABLE rollback_history IS 'Records all model rollbacks for audit and analysis';

COMMENT ON COLUMN model_lineage.changes IS 'JSONB describing what changed from parent (e.g., {"learning_rate": {"old": 0.001, "new": 0.0005}})';
COMMENT ON COLUMN model_lineage.sharpe_improvement IS 'current_sharpe - parent_sharpe';

COMMENT ON FUNCTION get_model_ancestors IS 'Returns all ancestor model IDs for a given model';
COMMENT ON FUNCTION get_model_descendants IS 'Returns all descendant model IDs (children, grandchildren, etc.)';
COMMENT ON FUNCTION find_best_ancestor IS 'Returns best performing ancestor for safe rollback';
