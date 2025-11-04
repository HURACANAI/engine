-- Migration: Add regime detection and confidence scoring columns
-- Date: November 4, 2025
-- Purpose: Support regime-aware trading and confidence-based filtering

BEGIN;

-- Add regime columns to trade_memory
ALTER TABLE trade_memory
ADD COLUMN IF NOT EXISTS regime VARCHAR(20),
ADD COLUMN IF NOT EXISTS regime_confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS decision_reason TEXT;

-- Add comments for documentation
COMMENT ON COLUMN trade_memory.regime IS 'Market regime at trade time (trend/range/panic)';
COMMENT ON COLUMN trade_memory.regime_confidence IS 'Confidence in regime detection (0-1)';
COMMENT ON COLUMN trade_memory.confidence IS 'Overall confidence in trade decision (0-1)';
COMMENT ON COLUMN trade_memory.decision_reason IS 'Human-readable explanation for trade decision';

-- Add regime columns to pattern_library
ALTER TABLE pattern_library
ADD COLUMN IF NOT EXISTS best_regime VARCHAR(20),
ADD COLUMN IF NOT EXISTS regime_trend_win_rate DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS regime_range_win_rate DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS regime_panic_win_rate DECIMAL(5,4);

-- Add comments
COMMENT ON COLUMN pattern_library.best_regime IS 'Regime where this pattern performs best';
COMMENT ON COLUMN pattern_library.regime_trend_win_rate IS 'Win rate in TREND regime';
COMMENT ON COLUMN pattern_library.regime_range_win_rate IS 'Win rate in RANGE regime';
COMMENT ON COLUMN pattern_library.regime_panic_win_rate IS 'Win rate in PANIC regime';

-- Create index for regime-based queries
CREATE INDEX IF NOT EXISTS idx_trade_memory_regime ON trade_memory(regime);
CREATE INDEX IF NOT EXISTS idx_trade_memory_confidence ON trade_memory(confidence);
CREATE INDEX IF NOT EXISTS idx_pattern_library_best_regime ON pattern_library(best_regime);

-- Create view for regime performance analysis
CREATE OR REPLACE VIEW regime_performance AS
SELECT
    regime,
    COUNT(*) as total_trades,
    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as winning_trades,
    AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate,
    AVG(net_profit_gbp) as avg_profit_gbp,
    AVG(confidence) as avg_confidence
FROM trade_memory
WHERE regime IS NOT NULL
GROUP BY regime
ORDER BY win_rate DESC;

-- Create view for confidence calibration
CREATE OR REPLACE VIEW confidence_calibration AS
SELECT
    CASE
        WHEN confidence < 0.6 THEN '0.5-0.6'
        WHEN confidence < 0.7 THEN '0.6-0.7'
        WHEN confidence < 0.8 THEN '0.7-0.8'
        WHEN confidence < 0.9 THEN '0.8-0.9'
        ELSE '0.9-1.0'
    END as confidence_bin,
    COUNT(*) as trade_count,
    AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as actual_win_rate,
    AVG(confidence) as avg_confidence_in_bin
FROM trade_memory
WHERE confidence IS NOT NULL
GROUP BY confidence_bin
ORDER BY confidence_bin;

COMMIT;

-- Verify migration
SELECT 'Migration complete! Added columns:' as message;
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'trade_memory'
  AND column_name IN ('regime', 'regime_confidence', 'confidence', 'decision_reason')
ORDER BY column_name;
