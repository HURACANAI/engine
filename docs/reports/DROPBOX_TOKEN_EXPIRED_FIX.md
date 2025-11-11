# Dropbox Token Expired - How to Fix

## Problem

You're seeing this error:
```
AuthError('expired_access_token', None)
```

This means your Dropbox access token has expired and needs to be regenerated.

## Solution

### Step 1: Generate a New Token

1. Go to the [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Sign in with your Dropbox account
3. Find your app (or create a new one if needed)
4. Click on your app
5. Scroll down to "OAuth 2" section
6. Under "Generated access token", click "Generate" or "Refresh"
7. Copy the new token

### Step 2: Update the Token

You have two options:

#### Option A: Environment Variable (Recommended)
```bash
export DROPBOX_ACCESS_TOKEN="your_new_token_here"
```

#### Option B: Settings File
Update the token in your settings file (usually `config/settings.yaml` or environment-specific config).

### Step 3: Restart the Engine

After updating the token, restart the engine:
```bash
./run.sh
```

## Verification

After restarting, you should see:
```
✅ Dropbox folder created: /Runpodhuracan/2025-01-08/
```

If you still see errors, check:
- Token format (should start with `sl.`)
- Token permissions (needs `files.content.write` and `files.metadata.read`)
- Token hasn't been revoked

## Token Permissions Required

Make sure your Dropbox app has these permissions enabled:
- ✅ `files.content.write` - Write files to Dropbox
- ✅ `files.metadata.read` - Read file metadata

You can check/update permissions in the [Dropbox App Console](https://www.dropbox.com/developers/apps) under "Permissions" tab.

## Notes

- Dropbox access tokens can expire (typically after 4 hours for short-lived tokens, or never for long-lived tokens)
- If you're using a short-lived token, consider generating a long-lived token or setting up token refresh
- The engine will continue running without Dropbox sync if the token is invalid (non-fatal error)

## Need Help?

If you continue to have issues:
1. Check the Dropbox App Console for any error messages
2. Verify your app has the correct permissions
3. Try generating a completely new token
4. Check the engine logs for more detailed error messages

