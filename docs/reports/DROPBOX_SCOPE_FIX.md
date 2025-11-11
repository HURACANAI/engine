# Dropbox Scope Fix Guide

## Problem

Your Dropbox app is missing required permissions (scopes). The error messages indicate:
- Missing `files.content.write` scope (needed for uploads)
- Missing `files.metadata.read` scope (needed for listing files)

## Solution

You need to enable these scopes in your Dropbox App Console:

### Step 1: Go to Dropbox App Console

1. Go to: https://www.dropbox.com/developers/apps
2. Find your app (ID: 7988481) or create a new one
3. Click on your app to open settings

### Step 2: Enable Required Scopes

1. Go to the **"Permissions"** tab
2. Under **"Scopes"**, enable:
   - ✅ `files.content.write` - **REQUIRED** for uploading files
   - ✅ `files.metadata.read` - **REQUIRED** for reading/listing files
   - ✅ `files.content.read` - Recommended for downloading files

### Step 3: Generate New Access Token

After enabling scopes, you need a new access token:

1. Go to the **"Settings"** tab
2. Scroll down to **"OAuth 2"** section
3. Under **"Generated access token"**, click **"Generate"**
4. Copy the **entire token** (it will be 1000+ characters long)
5. The token should start with `sl.`

### Step 4: Update Token in Settings

Update the token in `src/cloud/training/config/settings.py`:

```python
access_token: Optional[str] = os.getenv("DROPBOX_ACCESS_TOKEN") or "YOUR_NEW_TOKEN_HERE"
```

Or set it as an environment variable:
```bash
export DROPBOX_ACCESS_TOKEN="YOUR_NEW_TOKEN_HERE"
```

### Step 5: Test Connection

Run this to test:
```bash
python -c "
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import os
os.environ['HURACAN_ENV'] = 'local'

settings = EngineSettings.load()
sync = DropboxSync(
    access_token=settings.dropbox.access_token,
    app_folder='Runpodhuracan',
    enabled=True,
)
print('✅ Dropbox connected successfully!')
"
```

## Important Notes

1. **Token Length**: Valid Dropbox tokens are typically **1000+ characters** long
2. **Token Format**: Should start with `sl.`
3. **Scopes Required**: Make sure all required scopes are enabled before generating token
4. **App Type**: If using an "App folder" type app, files will be in `/Apps/Runpodhuracan/` folder

## Quick Fix Script

After updating your token, run:
```bash
python scripts/upload_local_candles_to_dropbox.py
```

This will upload all locally downloaded candle data to Dropbox.

