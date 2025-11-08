# Dropbox Sync Fix Summary

## Issues Identified

1. **Files uploaded to wrong location**: Files are being uploaded to `/Runpodhuracan/logs/...` instead of `/Runpodhuracan/2025-11-08/logs/...` (dated folder structure)

2. **Path normalization issue**: When `sync_directory` normalizes the `remote_dir` and then constructs file paths, `upload_file` was normalizing again, potentially causing path confusion.

3. **Telegram health check error**: The health check was trying to access `settings.notifications.telegram` instead of `settings.notifications.telegram_enabled` (this was already fixed in the code but the server might have old code).

## Fixes Applied

### 1. Fixed Path Normalization in `upload_file`
- Added `use_dated_folder` parameter to `upload_file` method
- When `use_dated_folder=False`, the path is assumed to be already fully normalized and used as-is
- Added logging to track path normalization

### 2. Fixed Path Normalization in `sync_directory`
- Added `use_dated_folder` parameter to `sync_directory` method
- When constructing file paths from normalized `remote_dir`, pass `use_dated_folder=False` to `upload_file` to prevent double-normalization
- Added logging to show normalized remote directory paths

### 3. Added Debug Logging
- Log normalized paths in `sync_directory` to help debug path issues
- Log dated folder status in `upload_file` to verify it's set correctly

## How Files Should Be Organized

After the fix, files should be organized as:
```
/Runpodhuracan/YYYY-MM-DD/
  ├── logs/
  │   ├── engine_monitoring_*.log
  │   └── ...
  ├── models/
  │   └── ...
  ├── learning/
  │   └── ...
  ├── monitoring/
  │   └── ...
  ├── exports/
  │   └── ...
  └── data/
      └── candles/
          └── ...
```

## Next Steps

1. **Pull latest code** on the server to get these fixes
2. **Restart the engine** to apply the changes
3. **Check logs** for `sync_directory_normalized_path` entries to verify paths are being normalized correctly
4. **Verify Dropbox** - files should now appear in dated folders like `/Runpodhuracan/2025-11-08/logs/...`

## Debugging

If files still don't appear in dated folders:
1. Check logs for `sync_directory_normalized_path` - verify `dated_folder` is set (not "not_set")
2. Check logs for `file_uploaded` - verify `remote_path` includes the dated folder
3. Verify `_dated_folder` is set during initialization by checking `dropbox_dated_folder_created` log entries

## Files Modified

- `src/cloud/training/integrations/dropbox_sync.py`
  - `upload_file`: Added `use_dated_folder` parameter and improved path handling
  - `sync_directory`: Added `use_dated_folder` parameter and fixed double-normalization issue
  - Added debug logging for path normalization

