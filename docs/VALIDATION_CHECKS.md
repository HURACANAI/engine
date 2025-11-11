# Validation Checks Documentation

This document describes all validation checks implemented in the training pipeline to ensure data quality and prevent errors.

## Validation Check Summary

All validation checks are now **PASSING** ✅

### 1. ✅ Close Column Exists Validation

**Location**: `src/cloud/training/services/orchestration.py` (lines 838-883)

**Checks**:
- Verifies `close` column exists in feature_frame
- Handles case mismatches (e.g., "Close" → "close")
- Checks if close column was lost during feature engineering
- Provides detailed error messages with available columns

**Error Message**:
```
Close column missing from feature frame - FeatureRecipe may have dropped it
```

### 2. ✅ Close Column Data Quality Validation

**Location**: `src/cloud/training/services/orchestration.py` (lines 885-937)
**Location**: `src/cloud/training/services/labeling.py` (lines 30-58)

**Checks**:
- Verifies no NaN/null values in close column
- Verifies no zero values in close column (division by zero risk)
- Logs detailed statistics (min, max, null count, zero count)

**Error Messages**:
```
Close column has NaN values - cannot compute labels
Found X null/NaN values in 'close' column
Found X zero values in 'close' column (division by zero risk)
```

### 3. ✅ Costs Validation

**Location**: `src/cloud/training/services/orchestration.py` (lines 719-737)

**Checks**:
- Verifies `costs.total_costs_bps` is a valid number (int or float)
- Verifies costs are non-negative
- Logs detailed cost breakdown (fee, spread, slippage)

**Error Message**:
```
Invalid costs.total_costs_bps: X (type: Y)
```

### 4. ✅ Labeling Result Validation

**Location**: `src/cloud/training/services/labeling.py` (lines 68-94)
**Location**: `src/cloud/training/services/orchestration.py` (lines 943-1008)

**Checks**:
- Verifies `net_edge_bps` column exists after labeling
- Verifies no NaN values in `net_edge_bps` after `drop_nulls()`
- Validates sample values are not all NaN
- Checks Polars statistics before Pandas conversion
- Compares Polars vs Pandas value counts

**Error Messages**:
```
net_edge_bps column not found after labeling
Found X null values in 'net_edge_bps' after drop_nulls
All X net_edge_bps values are NaN in Polars
Sample check failed: All sampled net_edge_bps values are NaN/invalid
```

### 5. ✅ Polars to Pandas Conversion Validation

**Location**: `src/cloud/training/services/orchestration.py` (lines 1019-1043)

**Checks**:
- Verifies `net_edge_bps` column exists after conversion
- Compares valid value counts between Polars and Pandas
- Logs conversion statistics

**Error/Warning Messages**:
```
net_edge_bps column missing after Polars to Pandas conversion
Value count mismatch between Polars and Pandas - investigating
```

## Test Results

All validation checks have been tested and are working correctly:

```
✅ PASS: Valid data processed successfully
✅ PASS: Correctly detected missing close column
✅ PASS: Correctly detected NaN values in close column
✅ PASS: Correctly detected zero values in close column
✅ PASS: Costs are valid
✅ PASS: Conversion preserved all values correctly
```

## Implementation Details

### LabelBuilder Validation

The `LabelBuilder.build()` method now includes:
1. Close column existence check
2. Close column null/zero value check
3. Net edge calculation validation
4. Post-calculation null check
5. Sample value validation

### Orchestration Validation

The `_train_symbol()` function now includes:
1. Close column verification before labeling
2. Costs validation before use
3. Close column statistics logging
4. Polars statistics logging before conversion
5. Pandas conversion verification
6. Value count comparison

## Error Handling

All validation failures result in:
- Clear, descriptive error messages
- Detailed logging with context
- Early return with appropriate `TrainingTaskResult`
- Proper error propagation

## Future Improvements

Potential enhancements:
1. Add validation for feature column data types
2. Add validation for timestamp column continuity
3. Add validation for volume data quality
4. Add validation for price data sanity (e.g., high > low)
5. Add validation for cost model parameters

