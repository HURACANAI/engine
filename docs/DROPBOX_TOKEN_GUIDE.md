# Dropbox Access Token Guide

## Quick Steps to Generate Access Token

You have **App Key** and **App Secret**, but we need an **Access Token** for the API.

### Option 1: Generate from Dropbox App Console (RECOMMENDED - EASIEST)

1. **Go to Dropbox App Console:**
   - Visit: https://www.dropbox.com/developers/apps
   - Sign in with your Dropbox account

2. **Find Your App:**
   - Look for your app (or create a new one if needed)
   - Click on your app to open its settings

3. **Generate Access Token:**
   - Go to the **"Settings"** tab
   - Scroll down to the **"OAuth 2"** section
   - Find **"Generated access token"**
   - Click the **"Generate"** button
   - **Copy the entire token** (it will be very long, 1000+ characters)
   - The token should start with `sl.`

4. **Update the Token in Code:**
   - Update `src/cloud/training/config/settings.py`:
     ```python
     class DropboxSettings(BaseModel):
         access_token: Optional[str] = "YOUR_NEW_TOKEN_HERE"
     ```
   - Update `src/cloud/training/pipelines/daily_retrain.py`:
     ```python
     dropbox_token_raw = settings.dropbox.access_token or "YOUR_NEW_TOKEN_HERE"
     ```

5. **Verify Token:**
   - Run the engine and check logs for `dropbox_token_received`
   - Should see `dropbox_sync_initialized` with your email

### Option 2: Use OAuth2 Flow (More Complex)

If you prefer OAuth2 flow, you'll need to:
1. Open browser to authorize: https://www.dropbox.com/oauth2/authorize?client_id=yxnputg7g9kijch&response_type=code&token_access_type=offline
2. Get authorization code
3. Exchange code for access token using App Key and App Secret
4. This requires a web server to handle the callback

**Recommendation:** Use Option 1 - it's much simpler!

## Troubleshooting

### Token Invalid Error
- **Cause:** Token expired, revoked, or incorrect
- **Fix:** Generate a new token from App Console

### Token Format Error
- **Cause:** Token doesn't start with `sl.`
- **Fix:** Make sure you copied the entire token (it's very long)

### Token Too Short
- **Cause:** Token was truncated when copying
- **Fix:** Copy the entire token (should be 1000+ characters)

### Authentication Failed
- **Cause:** Token doesn't have required permissions
- **Fix:** 
  1. Go to App Console → Permissions tab
  2. Enable `files.content.write` and `files.content.read`
  3. Generate a new token

## Your App Credentials

- **App Key:** `yxnputg7g9kijch`
- **App Secret:** `8llmdzmxj5hw6i8`
- **App Folder:** `Runpodhuracan`

## Important Notes

- Access tokens can expire or be revoked
- If token stops working, generate a new one
- Keep your App Secret secure (don't commit to public repos)
- The access token is what the code uses, not App Key/Secret








