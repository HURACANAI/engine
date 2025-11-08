# Deployment Instructions

## ✅ Fixes Have Been Committed and Pushed

All fixes have been committed and pushed to GitHub. The server needs to pull the latest code.

## Critical Fixes Included

### 1. ✅ Polars `cumsum()` Error (FIXED)
**File**: `src/shared/features/recipe.py:827-829`
**Change**: Changed from `.cumsum()` method to `pl.cumsum()` function
**Impact**: Training will now complete without `AttributeError`

### 2. ✅ Dropbox Path Normalization (FIXED)
**File**: `src/cloud/training/integrations/dropbox_sync.py`
**Changes**:
- Added `use_dated_folder` parameter to `upload_file()` and `sync_directory()`
- Fixed double-normalization issue
- Files will now go to `/Runpodhuracan/2025-11-08/logs/...` instead of `/Runpodhuracan/logs/...`

### 3. ✅ Telegram Health Check Error (FIXED)
**File**: `src/cloud/training/monitoring/enhanced_health_check.py:917`
**Change**: Changed from `settings.notifications.telegram` to `settings.notifications.telegram_enabled`
**Impact**: `/health` and `/status` Telegram commands will now work

### 4. ✅ Download Progress Tracking (NEW)
**File**: `src/cloud/training/monitoring/download_progress_tracker.py` (new)
**File**: `src/cloud/training/monitoring/telegram_command_handler.py`
**Feature**: Added `/download` command to view download progress
**Impact**: You can now use `/download` in Telegram to see progress of downloading historical data

## Deployment Steps (On Server)

1. **Pull latest code**:
   ```bash
   cd /workspace/engine
   git pull origin main
   ```

2. **Verify changes**:
   ```bash
   git log --oneline -1
   # Should show: "Fix Polars cumsum error, Dropbox path normalization..."
   ```

3. **Restart the engine**:
   ```bash
   ./run.sh
   ```

## What Will Work After Deployment

✅ **Training will complete** - No more `cumsum()` errors  
✅ **Dropbox files in dated folders** - Files will be organized as `/Runpodhuracan/2025-11-08/logs/...`  
✅ **Telegram `/health` command** - Will work without errors  
✅ **Telegram `/status` command** - Will work without errors  
✅ **Telegram `/download` command** - New command to view download progress  

## Verification

After deploying, check:

1. **Training logs** - Should not show `cumsum()` errors
2. **Dropbox** - Files should be in dated folders like `/Runpodhuracan/2025-11-08/...`
3. **Telegram** - Try `/health` command - should work without errors
4. **Telegram** - Try `/download` command - should show download progress

## Current Issues (Expected)

⚠️ **Database connection errors** - Expected if PostgreSQL is not running. This is non-critical if you're not using the database.

## Files Changed

- `src/shared/features/recipe.py` - Fixed Polars cumsum
- `src/cloud/training/integrations/dropbox_sync.py` - Fixed path normalization
- `src/cloud/training/monitoring/enhanced_health_check.py` - Fixed Telegram settings access
- `src/cloud/training/monitoring/telegram_command_handler.py` - Added `/download` command
- `src/cloud/training/monitoring/download_progress_tracker.py` - New file for progress tracking
- `src/cloud/training/services/orchestration.py` - Added progress tracking integration
- `src/cloud/training/pipelines/daily_retrain.py` - Integrated progress tracking
- `src/cloud/training/pipelines/feature_manager.py` - New file for feature management

