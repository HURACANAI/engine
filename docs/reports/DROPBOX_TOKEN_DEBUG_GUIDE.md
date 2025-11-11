# Dropbox Token Debugging Guide

## Why a Token Works Locally But Not in RunPod

There are several reasons why a Dropbox token might work locally but fail in RunPod:

### 1. **Different Token Sources**
The engine checks for tokens in this order:
1. `DROPBOX_ACCESS_TOKEN` environment variable (highest priority)
2. Settings file token (`settings.dropbox.access_token`)
3. Hardcoded fallback token (lowest priority)

**Check which token is being used:**
```bash
# In RunPod, check if environment variable is set
echo $DROPBOX_ACCESS_TOKEN

# Check token length and prefix
echo $DROPBOX_ACCESS_TOKEN | head -c 30
```

### 2. **Token Expiration Timing**
Dropbox access tokens can expire. If you:
- Used the token locally successfully
- Then RunPod tried to use it later
- The token may have expired in between

**Dropbox tokens typically expire after:**
- Short-lived tokens: 4 hours
- Long-lived tokens: Never (unless revoked)

### 3. **Token Was Regenerated**
If you or someone else regenerated the token in the Dropbox App Console:
- The old token becomes invalid immediately
- All environments using the old token will fail
- Only the new token will work

### 4. **Different Token Values**
The token in RunPod might be different from your local token:
- RunPod environment variable might have an old/expired token
- Settings file in RunPod might have a different token
- Hardcoded fallback might be different

### 5. **Multiple Token Usage**
If the same token is used from multiple places simultaneously:
- Dropbox might rate-limit or revoke it
- This is more common with short-lived tokens

## How to Debug

### Step 1: Check Token Source
The engine now logs which token source is being used. Look for:
```json
{"event": "dropbox_token_source", "source": "environment_variable", ...}
```

Possible sources:
- `environment_variable` - Using `DROPBOX_ACCESS_TOKEN` env var
- `settings_file` - Using token from settings file
- `hardcoded_fallback` - Using hardcoded token (⚠️ warning)

### Step 2: Compare Token Prefixes
Compare the token prefix between local and RunPod:

**Local:**
```bash
# Check what token prefix you're using locally
echo $DROPBOX_ACCESS_TOKEN | head -c 30
```

**RunPod:**
Check the logs for:
```json
{"token_prefix": "sl.u.AGE0b2yjwoCn0RRDbBC4b5h6O", ...}
```

If the prefixes are different, you're using different tokens!

### Step 3: Check Token Length
Token lengths should match:
- Valid Dropbox tokens are typically 1000+ characters
- Logs show: `"token_length": 1309`

### Step 4: Verify Token in Dropbox Console
1. Go to https://www.dropbox.com/developers/apps
2. Select your app
3. Check the "Generated access token" section
4. See if the token matches what you're using
5. Check if it shows as "active" or "expired"

### Step 5: Test Token Directly
Test the token to see if it's actually expired:

```python
import dropbox

token = "your_token_here"
dbx = dropbox.Dropbox(token)
try:
    account = dbx.users_get_current_account()
    print(f"Token is valid! Account: {account.email}")
except dropbox.exceptions.AuthError as e:
    print(f"Token is expired/invalid: {e}")
```

## Solutions

### Solution 1: Use Environment Variable (Recommended)
Set the token in RunPod environment:
```bash
export DROPBOX_ACCESS_TOKEN="your_new_token_here"
```

### Solution 2: Generate a New Token
1. Go to https://www.dropbox.com/developers/apps
2. Select your app
3. Scroll to "Generated access token"
4. Click "Generate" or "Refresh"
5. Copy the new token
6. Update `DROPBOX_ACCESS_TOKEN` in RunPod

### Solution 3: Check RunPod Environment Variables
In RunPod, check if there's an environment variable set that's overriding your token:
```bash
# List all environment variables containing DROPBOX
env | grep -i dropbox

# Check the actual value
echo "Token length: ${#DROPBOX_ACCESS_TOKEN}"
echo "Token prefix: ${DROPBOX_ACCESS_TOKEN:0:30}"
```

### Solution 4: Use Long-Lived Token
Make sure you're generating a **long-lived** token (doesn't expire):
- In Dropbox App Console, look for "Token expiration" setting
- Set it to "No expiration" if possible
- Or use OAuth flow to get refresh tokens

## Common Issues

### Issue: Token Works Locally But Not in RunPod
**Cause:** Different token sources or token expired between runs
**Fix:** Ensure RunPod has the same token as local, or generate a new token

### Issue: Token Expires Frequently
**Cause:** Using short-lived tokens (4-hour expiry)
**Fix:** Generate a long-lived token or set up token refresh

### Issue: Token Invalid Immediately After Generation
**Cause:** Token format issue or app permissions
**Fix:** 
- Check token starts with `sl.`
- Verify app has correct permissions (`files.content.write`, `files.metadata.read`)
- Regenerate token

## Prevention

1. **Use Environment Variables:** Always use `DROPBOX_ACCESS_TOKEN` environment variable
2. **Long-Lived Tokens:** Generate tokens with "No expiration"
3. **Don't Hardcode:** Remove hardcoded tokens from code
4. **Monitor Expiry:** Check token status regularly in Dropbox Console
5. **One Token Per Environment:** Use different tokens for local vs production

## Quick Fix

If you need to fix it right now:

1. **Generate new token:**
   - Go to https://www.dropbox.com/developers/apps
   - Generate a new access token
   - Copy it

2. **Set in RunPod:**
   ```bash
   export DROPBOX_ACCESS_TOKEN="your_new_token_here"
   ```

3. **Verify:**
   ```bash
   # Check it's set
   echo $DROPBOX_ACCESS_TOKEN | head -c 30
   
   # Restart engine
   ./run.sh
   ```

4. **Check logs:**
   Look for: `"dropbox_token_source": "environment_variable"` and `"dropbox_sync_initialized"`

