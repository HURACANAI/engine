# Dropbox Local Testing Guide

This guide helps you test Dropbox integration locally to diagnose authentication and file storage issues.

## Quick Start

### Option 1: Comprehensive Test (Recommended)

This tests everything - token loading, authentication, file operations, folder operations, and the DropboxSync class:

```bash
cd "/Users/haq/ENGINE (HF1) Crsor/engine"
python3 scripts/test_dropbox_local.py
```

### Option 2: Engine Integration Test

This tests Dropbox using the exact same code path as the engine:

```bash
cd "/Users/haq/ENGINE (HF1) Crsor/engine"
python3 scripts/test_dropbox_engine_integration.py
```

### Option 3: Simple Token Test

This is a simpler test that just validates the token:

```bash
cd "/Users/haq/ENGINE (HF1) Crsor/engine"
python3 scripts/test_dropbox_token.py
```

## Prerequisites

1. **Install Dropbox package:**
   ```bash
   pip install dropbox
   ```

2. **Set your Dropbox token:**
   
   Option A: Set environment variable (recommended):
   ```bash
   export DROPBOX_ACCESS_TOKEN="sl.your_token_here"
   ```
   
   Option B: Update `src/cloud/training/config/settings.py`:
   ```python
   class DropboxSettings(BaseModel):
       access_token: Optional[str] = "sl.your_token_here"
   ```

## Understanding Test Results

### ✅ All Tests Pass
If all tests pass, Dropbox integration is working correctly. The engine should be able to use Dropbox.

### ❌ Authentication Failed
**Symptoms:**
- `AuthError` exception
- "Token authentication failed" message

**Possible causes:**
1. Token is expired
2. Token was revoked
3. Token doesn't have required permissions
4. Token is for a different app

**Fix:**
1. Go to https://www.dropbox.com/developers/apps
2. Select your app (or create a new one)
3. Go to "Permissions" tab
4. Enable:
   - `files.content.write` - Write files
   - `files.metadata.read` - Read file metadata
5. Generate a new access token
6. Copy the full token (should start with `sl.` and be 1000+ characters)
7. Update the token in environment variable or settings.py

### ❌ File Operations Failed
**Symptoms:**
- `ApiError` when uploading/downloading files
- "File operation failed" message

**Possible causes:**
1. Token doesn't have `files.content.write` permission
2. Token doesn't have `files.content.read` permission
3. App folder access is restricted

**Fix:**
1. Go to Dropbox App Console → Permissions tab
2. Enable `files.content.write` and `files.content.read`
3. Generate a new token (old tokens don't get new permissions)

### ❌ Folder Operations Failed
**Symptoms:**
- Cannot create folders
- Cannot list folders
- "Folder operations failed" message

**Possible causes:**
1. Token doesn't have `files.metadata.read` permission
2. App folder doesn't exist and cannot be created

**Fix:**
1. Go to Dropbox App Console → Permissions tab
2. Enable `files.metadata.read`
3. Generate a new token

### ❌ Token Format Invalid
**Symptoms:**
- "Token should start with 'sl.'" error
- Token seems too short

**Possible causes:**
1. Token was truncated when copying
2. Token includes extra whitespace or quotes
3. Wrong token type (using App Key/Secret instead of Access Token)

**Fix:**
1. Make sure you're using an Access Token (not App Key/Secret)
2. Copy the entire token (should be 1000+ characters)
3. Token should start with `sl.`
4. Remove any extra whitespace, newlines, or quotes

## Getting a New Dropbox Token

1. **Go to Dropbox App Console:**
   - Visit: https://www.dropbox.com/developers/apps
   - Sign in with your Dropbox account

2. **Find Your App:**
   - Look for your app (or create a new one)
   - Click on your app to open its settings

3. **Check Permissions:**
   - Go to "Permissions" tab
   - Ensure these are enabled:
     - ✅ `files.content.write`
     - ✅ `files.metadata.read`
   - Click "Submit" to save

4. **Generate Access Token:**
   - Go to "Settings" tab
   - Scroll down to "OAuth 2" section
   - Find "Generated access token"
   - Click "Generate" button
   - **Copy the entire token** (it will be very long, 1000+ characters)
   - The token should start with `sl.`

5. **Update Token:**
   ```bash
   export DROPBOX_ACCESS_TOKEN="sl.your_new_token_here"
   ```
   
   Or update `src/cloud/training/config/settings.py`

## Running Tests with Different Token Sources

The engine checks for token in this order:
1. Environment variable `DROPBOX_ACCESS_TOKEN`
2. Settings file `settings.dropbox.access_token`

You can test with different sources:

```bash
# Test with environment variable
export DROPBOX_ACCESS_TOKEN="sl.your_token"
python3 scripts/test_dropbox_local.py

# Test with settings file (unset env var first)
unset DROPBOX_ACCESS_TOKEN
python3 scripts/test_dropbox_local.py
```

## Common Issues

### Issue: "dropbox package not installed"
**Fix:**
```bash
pip install dropbox
```

### Issue: "Failed to import engine code"
**Fix:**
- Make sure you're running from the project root
- Make sure PYTHONPATH includes the project root:
  ```bash
  export PYTHONPATH="/Users/haq/ENGINE (HF1) Crsor/engine:$PYTHONPATH"
  ```

### Issue: Token works in test but not in engine
**Possible causes:**
1. Engine is using a different token source
2. Engine has different environment variables
3. Engine code path is different

**Fix:**
- Use `test_dropbox_engine_integration.py` which uses the exact same code path
- Check that both use the same token source (env var or settings file)

## Debugging Tips

1. **Enable verbose logging:**
   - The test scripts use structlog, so you can add more logging if needed

2. **Check token in code:**
   - Add print statements to see what token is being used
   - Check token length and prefix

3. **Test token directly:**
   ```python
   import dropbox
   dbx = dropbox.Dropbox("your_token_here")
   account = dbx.users_get_current_account()
   print(account.email)
   ```

4. **Check Dropbox app settings:**
   - Make sure app type is "App folder" (not "Full Dropbox")
   - Check that permissions are enabled
   - Verify app folder name matches your settings

## Next Steps

Once tests pass:
1. Try running the engine: `python3 -m src.cloud.training.pipelines.daily_retrain`
2. Check logs for Dropbox-related messages
3. Verify files are being uploaded to Dropbox
4. Check Dropbox app folder to see if dated folders are created

## Support

If tests fail and you can't figure out why:
1. Run the comprehensive test and save the output
2. Check Dropbox App Console for app status
3. Verify token permissions in Dropbox App Console
4. Try generating a new token with all required permissions



